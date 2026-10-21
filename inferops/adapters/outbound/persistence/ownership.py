"""PostgreSQL leases and record scopes, checked under a lock in every write.

Version 2 adds tables only; legacy rows must be explicitly assigned while old
owners are stopped. Durable ownership uses PostgreSQL database-time leases.
"""
from __future__ import annotations

import re
import uuid
from datetime import timedelta

from sqlalchemy import func, select, text

from inferops.adapters.outbound.persistence.sql import SqlRepository
from inferops.ports.ownership import OwnershipLost


class ScopedSqlRepository(SqlRepository):
    def __init__(self, url, *, scope, lease_s=20, instance=None, **kwargs):
        if not url.startswith("postgresql"):
            raise ValueError("distributed ownership requires PostgreSQL")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", scope) or lease_s <= 0:
            raise ValueError("scope and a positive ownership lease are required")
        super().__init__(url, **kwargs)
        self.scope, self.instance = scope, instance or uuid.uuid4().hex
        self.lease_s, self.generation = lease_s, 0

    async def start(self):
        if self._engine is not None:
            return
        await super().start()
        try:
            async with self.engine.begin() as conn:
                await conn.execute(text("SELECT pg_advisory_xact_lock(731914215)"))
                # No automatic conversion of legacy records into somebody's shard.
                for table, kind in ((self.sessions, "session"), (self.evidence, "evidence"), (self.jobs, "job")):
                    legacy = await conn.scalar(select(table.c.id).where(~table.c.id.in_(
                        select(self.record_scopes.c.id).where(self.record_scopes.c.kind == kind))).limit(1))
                    if legacy:
                        raise ValueError("legacy unscoped records exist; stop old owners and explicitly assign "
                                         "them with inferops ownership adopt-legacy before distributed startup")
                await conn.execute(self._insert(self.leases, {
                    "scope": self.scope, "instance": "", "generation": 0,
                    "expires_at": func.clock_timestamp(),
                }))
                row = (await conn.execute(select(self.leases).where(
                    self.leases.c.scope == self.scope).with_for_update())).mappings().one()
                now = await conn.scalar(select(func.clock_timestamp()))
                if row["expires_at"] > now:
                    raise OwnershipLost(f"scope {self.scope} already has a live owner")
                self.generation = row["generation"] + 1
                await conn.execute(self.leases.update().where(self.leases.c.scope == self.scope).values(
                    instance=self.instance, generation=self.generation,
                    expires_at=now + timedelta(seconds=self.lease_s)))
        except BaseException:
            await super().close()
            raise

    async def _fence(self, conn):
        row = (await conn.execute(select(self.leases).where(
            self.leases.c.scope == self.scope).with_for_update())).mappings().first()
        now = await conn.scalar(select(func.clock_timestamp()))
        if (row is None or row["instance"] != self.instance or row["generation"] != self.generation
                or row["expires_at"] <= now):
            raise OwnershipLost(f"ownership lost for scope {self.scope}")

    async def renew_ownership(self):
        async with self._transaction() as conn:
            await conn.execute(self.leases.update().where(self.leases.c.scope == self.scope).values(
                expires_at=func.clock_timestamp() + timedelta(seconds=self.lease_s)))

    async def release_ownership(self):
        if self._engine is not None:
            async with self.engine.begin() as conn:
                await conn.execute(self.leases.update().where(
                    self.leases.c.scope == self.scope, self.leases.c.instance == self.instance,
                    self.leases.c.generation == self.generation).values(expires_at=func.clock_timestamp()))

    async def close(self):
        try:
            await self.release_ownership()
        finally:
            await super().close()

    async def _record_owner(self, conn, kind, record_id, *, create=False):
        if create:
            await conn.execute(self._insert(self.record_scopes, {
                "kind": kind, "id": record_id, "scope": self.scope}))
        owner = await conn.scalar(select(self.record_scopes.c.scope).where(
            self.record_scopes.c.kind == kind, self.record_scopes.c.id == record_id))
        if owner != self.scope:
            raise OwnershipLost(f"{kind} is not owned by scope {self.scope}")

    def _scope_condition(self, kind, column):
        return column.in_(select(self.record_scopes.c.id).where(
            self.record_scopes.c.kind == kind, self.record_scopes.c.scope == self.scope))

    async def assert_evidence_owner(self, evidence_id):
        async with self._transaction() as conn:
            await self._record_owner(conn, "evidence", evidence_id)

    @property
    def artifact_prefix(self):
        return f"owners/{self.scope}/{self.generation}"


async def adopt_legacy(repository, scope, *, owners_stopped=False):
    """Explicit additive adoption, without moving spools or rewriting artifacts."""
    if not owners_stopped or not repository.maintenance or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", scope):
        raise ValueError("maintenance mode, an explicit scope and --owners-stopped are required")
    async with repository.engine.begin() as conn:
        await conn.execute(text("SELECT pg_advisory_xact_lock(731914215)"))
        active = await conn.scalar(select(repository.leases.c.scope).where(
            repository.leases.c.expires_at > func.clock_timestamp()).limit(1))
        if active:
            raise OwnershipLost("stop all distributed owners before legacy adoption")
        counts = {}
        for table, kind in ((repository.sessions, "session"), (repository.evidence, "evidence"),
                            (repository.jobs, "job")):
            ids = list((await conn.execute(select(table.c.id).where(~table.c.id.in_(
                select(repository.record_scopes.c.id).where(repository.record_scopes.c.kind == kind))))).scalars())
            for record_id in ids:
                await conn.execute(repository._insert(repository.record_scopes, {
                    "kind": kind, "id": record_id, "scope": scope}))
            counts[kind] = len(ids)
        return counts
