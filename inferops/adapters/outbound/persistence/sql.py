"""Transactional PostgreSQL persistence with a SQLite development adapter.

The legacy tables remain projections, so the designated compatibility release can read data
created by the new engine. No module-level client, configuration or lock exists.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    and_,
    func,
    or_,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine


class SqlRepository:
    def __init__(self, url: str, *, max_pending_outbox: int | None = 100_000):
        if max_pending_outbox is not None and max_pending_outbox < 0:
            raise ValueError("Outbox capacity cannot be negative")
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif url.startswith("sqlite://"):
            url = url.replace("sqlite://", "sqlite+aiosqlite://", 1)
        self.url = url
        self.max_pending_outbox = max_pending_outbox
        self._engine: AsyncEngine | None = None
        self._write_lock = asyncio.Lock()
        self.metadata = MetaData()
        self.schema_versions = Table(
            "inferops_schema_versions", self.metadata,
            Column("version", Integer, primary_key=True),
            Column("description", String, nullable=False),
            Column("applied_at", DateTime(timezone=True), server_default=func.now()),
        )
        payload_type = JSONB().with_variant(JSON(), "sqlite")
        self.events = Table(
            "events", self.metadata,
            Column("id", String, primary_key=True),
            Column("pipeline", String, index=True),
            Column("type", String, index=True),
            Column("entity_id", String, index=True),
            Column("ts_ms", BigInteger, index=True),
            Column("payload", payload_type),
            Column("created_at", DateTime(timezone=True), server_default=func.now()),
        )
        self.incidents = Table(
            "alerts", self.metadata,
            Column("id", String, primary_key=True),
            Column("type", String, index=True),
            Column("entity_id", String, index=True),
            Column("severity", String, index=True),
            Column("dedupe_key", String, index=True),
            Column("status", String, default="open"),
            Column("ts_ms", BigInteger, index=True),
            Column("payload", payload_type),
            Column("created_at", DateTime(timezone=True), server_default=func.now()),
        )
        self.jobs = Table(
            "jobs", self.metadata,
            Column("id", String, primary_key=True),
            Column("pipeline", String, index=True),
            Column("status", String, index=True),
            Column("result", payload_type, nullable=True),
            Column("created_at", DateTime(timezone=True), server_default=func.now()),
        )
        # Keep the existing jobs projection readable by the rollback release.
        # Input is immutable and survives queued-job restart and explicit retry.
        self.job_requests = Table(
            "inferops_job_requests", self.metadata,
            Column("job_id", String, primary_key=True),
            Column("input", payload_type, nullable=False),
            Column("created_at", DateTime(timezone=True), server_default=func.now()),
        )
        self.sessions = Table(
            "inferops_sessions", self.metadata,
            Column("id", String, primary_key=True),
            Column("owner", String, nullable=False, index=True),
            Column("status", String, nullable=False, index=True),
            Column("document", payload_type, nullable=False),
            Column("created_at", DateTime(timezone=True), server_default=func.now()),
        )
        self.evidence = Table(
            "inferops_evidence", self.metadata,
            Column("id", String, primary_key=True),
            Column("session_id", String, nullable=False, index=True),
            Column("state", String, nullable=False, index=True),
            Column("document", payload_type, nullable=False),
            Column("created_at", DateTime(timezone=True), server_default=func.now()),
        )
        self.outbox = Table(
            "inferops_outbox", self.metadata,
            Column("id", String, primary_key=True),
            Column("topic", String, nullable=False),
            Column("key", String, nullable=True),
            Column("payload", payload_type, nullable=False),
            Column("status", String, nullable=False, default="pending", index=True),
            Column("owner", String, nullable=True),
            Column("generation", Integer, nullable=False, default=0),
            Column("attempts", Integer, nullable=False, default=0),
            Column("available_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
            Column("lease_until", DateTime(timezone=True), nullable=True),
            Column("error", String, nullable=True),
            Column("created_at", DateTime(timezone=True), server_default=func.now()),
        )
        self.inbox = Table(
            "inferops_inbox", self.metadata,
            Column("event_id", String, primary_key=True),
            Column("consumer", String, primary_key=True),
            Column("created_at", DateTime(timezone=True), server_default=func.now()),
        )

    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            raise RuntimeError("Repository has not been started")
        return self._engine

    async def start(self) -> None:
        if self._engine is not None:
            return
        kwargs = {"connect_args": {"timeout": 30}} if self.url.startswith("sqlite") else {}
        engine = create_async_engine(self.url, pool_pre_ping=True, **kwargs)
        try:
            async with engine.begin() as conn:
                if engine.dialect.name == "sqlite":
                    await conn.execute(text("PRAGMA journal_mode=WAL"))
                    await conn.execute(text("PRAGMA busy_timeout=30000"))
                await conn.run_sync(self.metadata.create_all)
                # Version 1 is entirely additive. Record it only after every
                # table was created successfully in this startup transaction.
                await conn.execute(self._insert(self.schema_versions, {
                    "version": 1, "description": "Sessions, evidence, delivery and immutable job requests",
                }, dialect=engine.dialect.name))
        except BaseException:
            await engine.dispose()
            raise
        self._engine = engine

    async def close(self) -> None:
        engine, self._engine = self._engine, None
        if engine is not None:
            await engine.dispose()

    def _insert(self, table: Table, row: dict[str, Any], *, dialect: str | None = None):
        if (dialect or self.engine.dialect.name) == "sqlite":
            from sqlalchemy.dialects.sqlite import insert
        else:
            from sqlalchemy.dialects.postgresql import insert
        return insert(table).values(**row).on_conflict_do_nothing()

    @staticmethod
    async def _now(conn: AsyncConnection) -> datetime:
        # Scheduling survives restarts because its clock belongs to the database.
        value = await conn.scalar(select(func.now()))
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value

    @staticmethod
    def _legacy_row(table: Table, row: dict[str, Any]) -> dict[str, Any]:
        fields = set(table.c.keys())
        result = {key: value for key, value in row.items() if key in fields}
        payload = dict(result.get("payload") or {})
        payload.update({key: value for key, value in row.items() if key not in fields})
        result["payload"] = payload
        return result

    @staticmethod
    def _outbox_row(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": row["id"], "topic": row["topic"], "key": row.get("key"),
            "payload": row.get("payload", {}), "status": "pending",
            "generation": 0, "attempts": 0,
        }

    async def save_session(self, row: dict[str, Any]) -> dict[str, Any]:
        document = {"status": "open", "generation": 1, "owner_generation": 1, **row}
        values = {
            "id": document["id"], "owner": document["owner"],
            "status": document["status"], "document": document,
        }
        async with self._write_lock, self.engine.begin() as conn:
            await conn.execute(self._insert(self.sessions, values))
            # A repeated save cannot steal ownership or reopen an interrupted
            # session. Return durable state so callers can detect fencing.
            return await conn.scalar(select(self.sessions.c.document).where(
                self.sessions.c.id == document["id"],
            ))

    async def close_session(self, session_id: str, status: str = "closed") -> None:
        async with self._write_lock, self.engine.begin() as conn:
            query = select(self.sessions.c.document).where(self.sessions.c.id == session_id).with_for_update()
            document = await conn.scalar(query)
            if document is not None:
                document = {**document, "status": status, "state": status,
                            "closed_at": (await self._now(conn)).isoformat()}
                await conn.execute(self.sessions.update().where(self.sessions.c.id == session_id).values(
                    status=status, document=document,
                ))

    async def interrupt_sessions(self, owner: str) -> int:
        """Interrupt sessions owned by a specific process (compatibility API)."""
        return await self._interrupt_sessions(self.sessions.c.owner == owner)

    async def interrupt_other_sessions(self, current_owner: str) -> int:
        """Fence prior process owners on startup of the single-owner server.

        Ownership generations are separate from media timeline segments. Increasing
        them invalidates an interrupted owner's generation without reconstructing
        any algorithm state. This operation assumes one application owner process.
        """
        if not current_owner:
            raise ValueError("A current process owner is required")
        return await self._interrupt_sessions(self.sessions.c.owner != current_owner)

    async def _interrupt_sessions(self, owner_condition) -> int:
        async with self._write_lock, self.engine.begin() as conn:
            rows = (await conn.execute(select(self.sessions).where(
                owner_condition, self.sessions.c.status == "open",
            ).with_for_update())).mappings().all()
            closed_at = (await self._now(conn)).isoformat()
            for row in rows:
                await conn.execute(self.sessions.update().where(self.sessions.c.id == row["id"]).values(
                    status="interrupted", document={
                        **row["document"], "status": "interrupted", "state": "interrupted",
                        "generation": row["document"].get("generation", 1) + 1,
                        "owner_generation": row["document"].get("owner_generation", 1) + 1,
                        "closed_at": closed_at,
                    },
                ))
            return len(rows)

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        async with self.engine.connect() as conn:
            return await conn.scalar(select(self.sessions.c.document).where(self.sessions.c.id == session_id))

    async def accept(
        self, events: list[dict[str, Any]], incidents: list[dict[str, Any]],
        evidence: list[dict[str, Any]], outbox: list[dict[str, Any]],
        *, inbox: tuple[str, str] | None = None,
    ) -> bool:
        async with self._write_lock, self.engine.begin() as conn:
            if inbox is not None:
                event_id, consumer = inbox
                reserved = await conn.execute(self._insert(self.inbox, {
                    "event_id": event_id, "consumer": consumer,
                }))
                if reserved.rowcount != 1:
                    return False
            if outbox:
                await self._lock_outbox_admission(conn)
            await self._check_outbox_capacity(conn, {row["id"] for row in outbox})
            for table, rows in ((self.events, events), (self.incidents, incidents)):
                for row in rows:
                    await conn.execute(self._insert(table, self._legacy_row(table, row)))
            for row in evidence:
                document = {"state": "Pending", "revision": 0, **row}
                await conn.execute(self._insert(self.evidence, {
                    "id": document["id"], "session_id": document["session_id"],
                    "state": document["state"], "document": document,
                }))
            for row in outbox:
                await conn.execute(self._insert(self.outbox, self._outbox_row(row)))
            return True

    async def list_events(self, pipeline: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        stmt = select(self.events).order_by(self.events.c.ts_ms.desc()).limit(max(0, limit))
        if pipeline is not None:
            stmt = stmt.where(self.events.c.pipeline == pipeline)
        async with self.engine.connect() as conn:
            return [dict(row) for row in (await conn.execute(stmt)).mappings()]

    async def list_incidents(self, limit: int = 100) -> list[dict[str, Any]]:
        stmt = select(self.incidents).order_by(self.incidents.c.ts_ms.desc()).limit(max(0, limit))
        async with self.engine.connect() as conn:
            return [dict(row) for row in (await conn.execute(stmt)).mappings()]

    async def get_evidence(self, evidence_id: str) -> dict[str, Any] | None:
        async with self.engine.connect() as conn:
            return await conn.scalar(select(self.evidence.c.document).where(self.evidence.c.id == evidence_id))

    async def list_evidence(
        self, session_id: str | None = None, limit: int = 100,
    ) -> list[dict[str, Any]]:
        stmt = select(self.evidence.c.document).order_by(
            self.evidence.c.created_at.desc(), self.evidence.c.id,
        ).limit(max(0, limit))
        if session_id is not None:
            stmt = stmt.where(self.evidence.c.session_id == session_id)
        async with self.engine.connect() as conn:
            return list((await conn.execute(stmt)).scalars())

    async def update_evidence(self, evidence_id: str, **changes: Any) -> dict[str, Any] | None:
        if "id" in changes or "session_id" in changes:
            raise ValueError("Evidence identity and session ownership are immutable")
        async with self._write_lock, self.engine.begin() as conn:
            # All admission writers lock the outbox before individual records;
            # this keeps duplicate acceptance and evidence updates deadlock-free.
            await self._lock_outbox_admission(conn)
            document = await conn.scalar(select(self.evidence.c.document).where(
                self.evidence.c.id == evidence_id,
            ).with_for_update())
            if document is None:
                return None
            previous_document = document
            next_revision = document.get("revision", 0) + 1
            await self._check_outbox_capacity(conn, {f"evidence:{evidence_id}:{next_revision}"})
            document = {
                **document, **changes, "revision": next_revision,
                "updated_at": (await self._now(conn)).isoformat(),
            }
            await conn.execute(self.evidence.update().where(self.evidence.c.id == evidence_id).values(
                state=document["state"], document=document,
            ))
            await conn.execute(self._insert(self.outbox, self._outbox_row({
                "id": f"evidence:{evidence_id}:{document['revision']}",
                "topic": "evidence.updated", "key": evidence_id, "payload": document,
            })))
            # Maintain readable evidence links for the compatibility release.
            incident_id = document.get("incident_id") or document.get("alert_id")
            uris = self._artifact_uris(document.get("artifacts"))
            uris.update(document.get("evidence_uris") or [])
            deleted = document.get("deleted") or document["state"] == "Deleted"
            removed_uris = set()
            if deleted:
                removed_uris.update(self._artifact_uris(previous_document.get("artifacts")))
                removed_uris.update(self._artifact_uris(document.get("deleted_artifacts")))
                removed_uris.update(previous_document.get("evidence_uris") or [])
                removed_uris.update(uris)
                uris.clear()
            if incident_id:
                payload = await conn.scalar(select(self.incidents.c.payload).where(
                    self.incidents.c.id == incident_id,
                ).with_for_update())
                if payload is not None:
                    payload = {
                        **payload, "evidence_id": evidence_id, "evidence_state": document["state"],
                        "evidence_uris": sorted((set(payload.get("evidence_uris") or []) | uris) - removed_uris),
                    }
                    if deleted:
                        payload["evidence_deleted"] = True
                    await conn.execute(self.incidents.update().where(
                        self.incidents.c.id == incident_id,
                    ).values(payload=payload))
            return document

    @staticmethod
    def _artifact_uris(artifacts) -> set[str]:
        items = artifacts.values() if isinstance(artifacts, dict) else (artifacts or [])
        uris = set()
        for artifact in items:
            uri = artifact.get("uri") if isinstance(artifact, dict) else artifact
            if isinstance(uri, str):
                uris.add(uri)
        return uris

    async def _lock_outbox_admission(self, conn: AsyncConnection) -> None:
        if self.max_pending_outbox is not None and self.engine.dialect.name == "postgresql":
            # Serialize short admission transactions across repository instances.
            # Capturing/encoding/network operations never hold this lock.
            name = self.engine.dialect.identifier_preparer.format_table(self.outbox)
            await conn.execute(text(f"LOCK TABLE {name} IN SHARE ROW EXCLUSIVE MODE"))

    async def _check_outbox_capacity(self, conn: AsyncConnection, message_ids: set[str]) -> None:
        if self.max_pending_outbox is None or not message_ids:
            return
        existing_ids = set((await conn.execute(select(self.outbox.c.id).where(
            self.outbox.c.id.in_(message_ids),
        ))).scalars())
        new_count = len(message_ids - existing_ids)
        if not new_count:
            return
        pending_count = await conn.scalar(select(func.count()).select_from(self.outbox).where(
            self.outbox.c.status == "pending",
        ))
        if pending_count + new_count > self.max_pending_outbox:
            raise RuntimeError("transactional outbox capacity exhausted")

    async def diagnostics(self) -> dict[str, Any]:
        """Small aggregate snapshots for health/metrics without loading documents."""
        async with self.engine.connect() as conn:
            now = await self._now(conn)
            pending = await conn.scalar(select(func.count()).select_from(self.outbox).where(
                self.outbox.c.status == "pending",
            ))
            oldest = await conn.scalar(select(func.min(self.outbox.c.created_at)).where(
                self.outbox.c.status == "pending",
            ))
            expired = await conn.scalar(select(func.count()).select_from(self.outbox).where(
                self.outbox.c.status == "pending", self.outbox.c.lease_until <= now,
            ))
            failures = await conn.scalar(select(func.count()).select_from(self.outbox).where(
                self.outbox.c.status == "pending", self.outbox.c.error.is_not(None),
            ))
            evidence = dict((await conn.execute(select(
                self.evidence.c.state, func.count(),
            ).group_by(self.evidence.c.state))).all())
            sessions = dict((await conn.execute(select(
                self.sessions.c.status, func.count(),
            ).group_by(self.sessions.c.status))).all())
        if oldest is not None:
            oldest = oldest.replace(tzinfo=timezone.utc) if oldest.tzinfo is None else oldest
        age = max(0.0, (now - oldest).total_seconds()) if oldest else 0.0
        return {"pending_outbox": pending, "oldest_outbox_age_seconds": age,
                "expired_outbox_leases": expired, "outbox_failures": failures,
                "evidence_states": evidence,
                "session_states": sessions, "active_sessions": sessions.get("open", 0)}

    async def claim_outbox(
        self, owner: str, limit: int = 16, lease_s: float = 30,
    ) -> list[dict[str, Any]]:
        if not owner or lease_s <= 0:
            raise ValueError("An owner and positive lease duration are required")
        async with self._write_lock, self.engine.begin() as conn:
            now = await self._now(conn)
            eligible = and_(
                self.outbox.c.status == "pending", self.outbox.c.available_at <= now,
                or_(self.outbox.c.lease_until.is_(None), self.outbox.c.lease_until <= now),
            )
            stmt = select(self.outbox.c.id).where(eligible).order_by(
                self.outbox.c.created_at, self.outbox.c.id,
            ).limit(max(0, limit)).with_for_update(skip_locked=True)
            ids = list((await conn.execute(stmt)).scalars())
            claimed = []
            for message_id in ids:
                # The predicate is repeated for SQLite and independent repositories.
                row = (await conn.execute(self.outbox.update().where(
                    self.outbox.c.id == message_id, eligible,
                ).values(
                    owner=owner, generation=self.outbox.c.generation + 1,
                    attempts=self.outbox.c.attempts + 1, lease_until=now + timedelta(seconds=lease_s),
                ).returning(self.outbox))).mappings().first()
                if row is not None:
                    claimed.append(dict(row))
            return claimed

    def _claim_identity(self, message_id: str, owner: str, generation: int):
        return and_(
            self.outbox.c.id == message_id, self.outbox.c.owner == owner,
            self.outbox.c.generation == generation, self.outbox.c.status == "pending",
        )

    async def ack_outbox(self, message_id: str, owner: str, generation: int) -> bool:
        async with self._write_lock, self.engine.begin() as conn:
            now = await self._now(conn)
            result = await conn.execute(self.outbox.update().where(
                self._claim_identity(message_id, owner, generation),
                self.outbox.c.lease_until > now,
            ).values(status="delivered", owner=None, lease_until=None, error=None))
            return result.rowcount == 1

    async def retry_outbox(
        self, message_id: str, owner: str, generation: int, error: str, delay_s: float,
    ) -> bool:
        async with self._write_lock, self.engine.begin() as conn:
            now = await self._now(conn)
            result = await conn.execute(self.outbox.update().where(
                self._claim_identity(message_id, owner, generation),
                self.outbox.c.lease_until > now,
            ).values(
                owner=None, lease_until=None, error=error[:4096],
                available_at=now + timedelta(seconds=max(0, delay_s)),
            ))
            return result.rowcount == 1

    async def record_inbox(self, event_id: str, consumer: str) -> bool:
        """Reserve an event for consumers whose durable effect is this insertion.

        Consumers with additional SQL effects must use ``consume`` to put those
        writes in the same transaction; reserving before a remote effect is unsafe.
        """
        async with self._write_lock, self.engine.begin() as conn:
            result = await conn.execute(self._insert(self.inbox, {"event_id": event_id, "consumer": consumer}))
            return result.rowcount == 1

    async def consume(self, event_id: str, consumer: str, effect) -> bool:
        """Apply an async SQL effect and inbox reservation in one transaction."""
        async with self._write_lock, self.engine.begin() as conn:
            result = await conn.execute(self._insert(self.inbox, {"event_id": event_id, "consumer": consumer}))
            if result.rowcount != 1:
                return False
            await effect(conn)
            return True

    async def create_job(self, row: dict[str, Any]) -> None:
        async with self._write_lock, self.engine.begin() as conn:
            values = {key: value for key, value in row.items() if key in self.jobs.c}
            result = await conn.execute(self._insert(self.jobs, values))
            if result.rowcount == 1:
                await conn.execute(self._insert(self.job_requests, {
                    "job_id": row["id"], "input": row.get("input") or {},
                }))

    async def update_job(self, job_id: str, *, status: str, result: dict[str, Any] | None = None) -> None:
        values: dict[str, Any] = {"status": status}
        if result is not None:
            values["result"] = result
        async with self._write_lock, self.engine.begin() as conn:
            await conn.execute(self.jobs.update().where(self.jobs.c.id == job_id).values(**values))

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        async with self.engine.connect() as conn:
            row = (await conn.execute(self._job_query().where(self.jobs.c.id == job_id))).mappings().first()
            return self._job_document(row) if row is not None else None

    async def list_jobs(self, limit: int = 100) -> list[dict[str, Any]]:
        async with self.engine.connect() as conn:
            rows = (await conn.execute(self._job_query().order_by(
                self.jobs.c.created_at.desc(), self.jobs.c.id,
            ).limit(max(0, limit)))).mappings()
            return [self._job_document(row) for row in rows]

    async def list_queued_jobs(self, limit: int = 64) -> list[dict[str, Any]]:
        """Recover oldest admitted work without a recent-completed-job cutoff."""
        async with self.engine.connect() as conn:
            rows = (await conn.execute(self._job_query().where(
                self.jobs.c.status == "queued",
            ).order_by(self.jobs.c.created_at, self.jobs.c.id).limit(max(0, limit)))).mappings()
            return [self._job_document(row) for row in rows]

    def _job_query(self):
        return select(self.jobs, self.job_requests.c.input).select_from(
            self.jobs.outerjoin(self.job_requests, self.jobs.c.id == self.job_requests.c.job_id),
        )

    @staticmethod
    def _job_document(row) -> dict[str, Any]:
        return {**dict(row), "input": row["input"] or {}}

    async def interrupt_jobs(self) -> int:
        async with self._write_lock, self.engine.begin() as conn:
            rows = (await conn.execute(select(self.jobs).where(
                self.jobs.c.status.in_(["running", "processing"]),
            ).with_for_update())).mappings().all()
            for row in rows:
                await conn.execute(self.jobs.update().where(self.jobs.c.id == row["id"]).values(
                    status="failed", result={**(row["result"] or {}), "error": "process_interrupted"},
                ))
            return len(rows)
