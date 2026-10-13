"""Generic Postgres DAO: events, jobs, models, alerts tables (JSONB payloads)."""
from __future__ import annotations

import asyncio
from typing import Any

from sqlalchemy import JSON, BigInteger, Column, DateTime, MetaData, String, Table, func, select, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from inferops.core.config import settings

metadata = MetaData()

events_t = Table(
    "events", metadata,
    Column("id", String, primary_key=True),
    Column("pipeline", String, index=True),
    Column("type", String, index=True),
    Column("entity_id", String, index=True),
    Column("ts_ms", BigInteger, index=True),
    Column("payload", JSONB().with_variant(JSON(), "sqlite")),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
)

alerts_t = Table(
    "alerts", metadata,
    Column("id", String, primary_key=True),
    Column("type", String, index=True),
    Column("entity_id", String, index=True),
    Column("severity", String, index=True),
    Column("dedupe_key", String, index=True),
    Column("status", String, default="open"),
    Column("ts_ms", BigInteger, index=True),
    Column("payload", JSONB().with_variant(JSON(), "sqlite")),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
)

jobs_t = Table(
    "jobs", metadata,
    Column("id", String, primary_key=True),
    Column("pipeline", String, index=True),
    Column("status", String, index=True),
    Column("result", JSONB().with_variant(JSON(), "sqlite"), nullable=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
)

_engine: AsyncEngine | None = None
_write_lock = asyncio.Lock()


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        connect_args = {"timeout": 30} if settings.pg.url.startswith("sqlite") else {}
        _engine = create_async_engine(
            settings.pg.url,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
    return _engine


async def init_schema() -> None:
    e = get_engine()
    async with e.begin() as conn:
        if settings.pg.url.startswith("sqlite"):
            await conn.execute(text("PRAGMA journal_mode=WAL"))
            await conn.execute(text("PRAGMA busy_timeout=30000"))
        await conn.run_sync(metadata.create_all)


async def insert_event(row: dict[str, Any]) -> None:
    await _insert_with_retry(events_t, row)


async def insert_alert(row: dict[str, Any]) -> None:
    # A rule can enable both a Postgres sink and engine persistence. Both paths
    # write the same incident ID, so retries must be idempotent.
    await _insert_with_retry(alerts_t, row, ignore_duplicate=True)


async def attach_alert_evidence(alert_id: str, uris: list[str]) -> None:
    """Link completed evidence to the persisted incident without a new alert."""
    async with _write_lock, get_engine().begin() as conn:
        row = (await conn.execute(select(alerts_t.c.payload).where(alerts_t.c.id == alert_id))).first()
        if row is not None:
            payload = dict(row.payload or {})
            payload["evidence_uris"] = sorted(set(payload.get("evidence_uris", []) + list(uris)))
            await conn.execute(alerts_t.update().where(alerts_t.c.id == alert_id).values(payload=payload))


async def create_job(row: dict[str, Any]) -> None:
    await _insert_with_retry(jobs_t, row)


async def update_job(job_id: str, *, status: str, result: dict[str, Any] | None = None) -> None:
    e = get_engine()
    values: dict[str, Any] = {"status": status}
    if result is not None:
        values["result"] = result
    for attempt in range(5):
        try:
            async with _write_lock, e.begin() as conn:
                await conn.execute(jobs_t.update().where(jobs_t.c.id == job_id).values(**values))
            return
        except OperationalError as exc:
            if "database is locked" not in str(exc).lower() or attempt == 4:
                raise
            await asyncio.sleep(0.2 * (attempt + 1))


async def get_job(job_id: str) -> dict[str, Any] | None:
    e = get_engine()
    stmt = select(jobs_t).where(jobs_t.c.id == job_id)
    async with e.begin() as conn:
        row = (await conn.execute(stmt)).mappings().first()
    return dict(row) if row else None


async def list_jobs(limit: int = 100) -> list[dict[str, Any]]:
    e = get_engine()
    stmt = select(jobs_t).order_by(jobs_t.c.created_at.desc()).limit(limit)
    async with e.begin() as conn:
        rows = (await conn.execute(stmt)).mappings().all()
    return [dict(r) for r in rows]


async def _insert_with_retry(table: Table, row: dict[str, Any], *, ignore_duplicate: bool = False) -> None:
    e = get_engine()
    statement = table.insert().values(**row)
    if ignore_duplicate:
        if e.dialect.name == "sqlite":
            from sqlalchemy.dialects.sqlite import insert
        else:
            from sqlalchemy.dialects.postgresql import insert
        statement = insert(table).values(**row).on_conflict_do_nothing(index_elements=["id"])
    for attempt in range(5):
        try:
            async with _write_lock, e.begin() as conn:
                await conn.execute(statement)
            return
        except OperationalError as exc:
            if "database is locked" not in str(exc).lower() or attempt == 4:
                raise
            await asyncio.sleep(0.2 * (attempt + 1))


async def list_events(pipeline: str | None = None, limit: int = 100) -> list[dict]:
    e = get_engine()
    stmt = select(events_t).order_by(events_t.c.ts_ms.desc()).limit(limit)
    if pipeline:
        stmt = stmt.where(events_t.c.pipeline == pipeline)
    async with e.begin() as conn:
        rows = (await conn.execute(stmt)).mappings().all()
    return [dict(r) for r in rows]


async def list_alerts(limit: int = 100) -> list[dict]:
    e = get_engine()
    stmt = select(alerts_t).order_by(alerts_t.c.ts_ms.desc()).limit(limit)
    async with e.begin() as conn:
        rows = (await conn.execute(stmt)).mappings().all()
    return [dict(r) for r in rows]
