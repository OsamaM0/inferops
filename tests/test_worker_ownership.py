"""Real PostgreSQL: duplicate owners, stale generations and foreign recovery."""
import asyncio
import os
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import create_async_engine

from inferops.adapters.outbound.persistence.ownership import ScopedSqlRepository, adopt_legacy
from inferops.adapters.outbound.persistence.sql import SqlRepository
from inferops.ports.ownership import OwnershipLost


@pytest_asyncio.fixture
async def owners():
    url = os.environ.get("INFEROPS_TEST_PG_URL")
    if not url:
        pytest.skip("requires disposable INFEROPS_TEST_PG_URL")
    admin = create_async_engine(url.replace("postgresql://", "postgresql+asyncpg://", 1))
    schema = f"inferops_ownership_{uuid4().hex}"
    repositories = []

    def factory(scope=None, **kwargs):
        repo = ScopedSqlRepository(url, scope=scope, **kwargs) if scope else SqlRepository(url, **kwargs)
        for table in repo.metadata.tables.values():
            table.schema = schema
        repositories.append(repo)
        return repo

    async with admin.begin() as conn:
        await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    try:
        yield factory
    finally:
        await asyncio.gather(*(repo.close() for repo in repositories), return_exceptions=True)
        async with admin.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin.dispose()


async def test_duplicate_owner_is_rejected_and_foreign_sessions_survive_recovery(owners):
    a, b = owners("camera-a"), owners("camera-b")
    await asyncio.gather(a.start(), b.start())
    await a.save_session({"id": "a", "owner": "instance-a"})
    await b.save_session({"id": "b", "owner": "instance-b"})
    with pytest.raises(OwnershipLost, match="live owner"):
        await owners("camera-a").start()
    await a.interrupt_other_sessions("restart-a")
    assert (await a.get_session("a"))["status"] == "interrupted"
    assert (await a.get_session("b"))["status"] == "open"
    with pytest.raises(OwnershipLost):
        await a.close_session("b")
    await a.claim_camera("front")
    with pytest.raises(OwnershipLost):
        await b.claim_camera("front")


async def test_expired_generation_cannot_accept_update_or_release_new_owner(owners):
    old = owners("camera-a")
    await old.start()
    await old.save_session({"id": "session", "owner": "old"})
    await old.accept([], [], [{"id": "evidence", "session_id": "session"}], [])
    # Deterministic expired lease; the production claim still uses database time.
    async with old.engine.begin() as conn:
        await conn.execute(old.leases.update().values(expires_at=func.clock_timestamp()))
    new = owners("camera-a")
    await new.start()
    assert new.generation == old.generation + 1
    for operation in [old.accept([], [], [], []), old.update_evidence("evidence", state="Saved"),
                      old.close_session("session"), old.renew_ownership()]:
        with pytest.raises(OwnershipLost):
            await operation
    await old.release_ownership()
    await new.renew_ownership()
    assert new.artifact_prefix != old.artifact_prefix
    await new.update_evidence("evidence", state="Incomplete")


async def test_scoped_jobs_evidence_pagination_and_api_only_relay(owners):
    a, b, api = owners("camera-a"), owners("camera-b"), owners("api")
    await asyncio.gather(a.start(), b.start(), api.start())
    for repo, label in [(a, "a"), (b, "b")]:
        await repo.save_session({"id": label, "owner": label})
        await repo.accept([], [], [{"id": f"{label}{n}", "session_id": label} for n in range(5)],
                          [{"id": label, "topic": "incident.accepted", "key": label, "payload": {}}])
        await repo.create_job({"id": label, "status": "running"})
    assert await a.interrupt_jobs() == 1
    assert (await a.get_job("b"))["status"] == "running"
    assert [row["id"] async for row in a.recovery_evidence(page_size=2)] == [f"a{n}" for n in range(5)]
    with pytest.raises(OwnershipLost):
        await a.update_evidence("b1", state="Failed")
    with pytest.raises(PermissionError):
        await a.claim_outbox("worker")
    assert len(await api.claim_outbox("relay")) == 2
    async with api.engine.connect() as conn:
        versions = list((await conn.execute(select(api.schema_versions.c.version))).scalars())
    assert sorted(versions) == [1, 2, 3]


def test_non_postgres_urls_cannot_enable_distributed_ownership():
    with pytest.raises(ValueError, match="PostgreSQL"):
        ScopedSqlRepository("mysql://localhost/test", scope="api")


async def test_legacy_pending_records_require_explicit_adoption_and_preserve_references(owners):
    legacy = owners()
    await legacy.start()
    await legacy.save_session({"id": "legacy-session", "owner": "old-process"})
    await legacy.accept([], [], [{"id": "pending", "session_id": "legacy-session", "state": "Pending",
                                  "spool_path": "/old/spool/pending", "artifacts": {"snapshot": {"key": "old/key"}}}], [])
    await legacy.create_job({"id": "legacy-job", "status": "running"})
    original = await legacy.get_evidence("pending")
    # Recreate the previous additive schema version, with live legacy documents.
    async with legacy.engine.begin() as conn:
        for table in (legacy.record_scopes, legacy.leases, legacy.delivery_receipts):
            await conn.run_sync(table.drop)
        await conn.execute(legacy.schema_versions.delete().where(legacy.schema_versions.c.version > 1))
    await legacy.close()
    with pytest.raises(ValueError, match="legacy unscoped"):
        await owners("camera-a").start()
    maintenance = owners(maintenance=True)
    await maintenance.start()
    with pytest.raises(ValueError, match="owners-stopped"):
        await adopt_legacy(maintenance, "camera-a")
    assert await adopt_legacy(maintenance, "camera-a", owners_stopped=True) == {"session": 1, "evidence": 1, "job": 1}
    assert await adopt_legacy(maintenance, "camera-a", owners_stopped=True) == {"session": 0, "evidence": 0, "job": 0}
    await maintenance.close()
    new = owners("camera-a")
    await new.start()
    assert [row["id"] async for row in new.recovery_evidence()] == ["pending"]
    assert await new.get_evidence("pending") == original
