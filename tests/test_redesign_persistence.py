"""Transactions, owner fencing, recovery and legacy-reader contracts.

PostgreSQL runs use an isolated disposable schema, enabled explicitly with
INFEROPS_TEST_PG_URL. SQLite runs require no external database.
"""
from __future__ import annotations

import asyncio
import os
from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import func, select, text
from sqlalchemy.exc import StatementError
from sqlalchemy.ext.asyncio import create_async_engine

from inferops.adapters.outbound.persistence.sql import SqlRepository


@pytest_asyncio.fixture(params=["sqlite", "postgresql"])
async def repository(request, tmp_path):
    schema = None
    admin = None
    if request.param == "postgresql":
        url = os.environ.get("INFEROPS_TEST_PG_URL")
        if not url:
            pytest.skip("set INFEROPS_TEST_PG_URL for disposable PostgreSQL schema tests")
        if not url.startswith("postgresql"):
            pytest.skip("INFEROPS_TEST_PG_URL must select PostgreSQL")
        normalized = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        admin = create_async_engine(normalized)
        schema = f"inferops_test_{uuid4().hex}"
        async with admin.begin() as conn:
            await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    else:
        url = f"sqlite+aiosqlite:///{tmp_path / 'records.db'}"
    repo = SqlRepository(url)
    if schema:
        for table in repo.metadata.tables.values():
            table.schema = schema
    await repo.start()
    try:
        yield repo
    finally:
        await repo.close()
        if admin is not None:
            async with admin.begin() as conn:
                await conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
            await admin.dispose()


def acceptance():
    return (
        [{"id": "event", "pipeline": "driver", "type": "fatigue", "entity_id": "1",
          "ts_ms": 1000, "payload": {"confidence": 0.9}, "session_id": "session"}],
        [{"id": "incident", "type": "fatigue", "severity": "high", "ts_ms": 1000,
          "payload": {"evidence_uris": ["s3://evidence/old.jpg"]}, "session_id": "session"}],
        [{"id": "capture", "session_id": "session", "incident_id": "incident"}],
        [{"id": "event:incident", "topic": "incident.accepted", "key": "incident",
          "payload": {"incident_id": "incident"}}],
    )


async def count(repo, table):
    async with repo.engine.connect() as conn:
        return await conn.scalar(select(func.count()).select_from(table))


async def test_acceptance_atomic_and_duplicate_ids_immutable(repository):
    rows = acceptance()
    broken = [*rows[:3], [{**rows[3][0], "payload": {"bad": object()}}]]
    with pytest.raises(StatementError):
        await repository.accept(*broken)
    for table in (repository.events, repository.incidents, repository.evidence, repository.outbox):
        assert await count(repository, table) == 0
    await repository.accept(*rows)
    changed = acceptance()
    changed[0][0]["payload"] = {"confidence": 0.1}
    changed[2][0]["state"] = "Saved"
    await repository.accept(*changed)
    for table in (repository.events, repository.incidents, repository.evidence, repository.outbox):
        assert await count(repository, table) == 1
    assert (await repository.list_events())[0]["payload"] == {
        "confidence": 0.9, "session_id": "session",
    }
    assert (await repository.get_evidence("capture"))["state"] == "Pending"


async def test_session_restart_fences_prior_process_owner(repository):
    await repository.save_session({"id": "prior", "owner": "old", "state": "active", "generation": 3})
    await repository.save_session({"id": "current", "owner": "new", "state": "active"})
    await repository.save_session({"id": "already_closed", "owner": "old", "status": "closed"})
    assert await repository.interrupt_other_sessions("new") == 1
    assert await repository.interrupt_other_sessions("new") == 0
    old = await repository.get_session("prior")
    assert old["state"] == old["status"] == "interrupted"
    assert old["generation"] == 4
    assert old["closed_at"]
    assert (await repository.get_session("current"))["status"] == "open"
    await repository.close_session("current")
    assert (await repository.get_session("current"))["state"] == "closed"


async def test_outbox_leases_generation_and_restart_delivery(repository):
    await repository.accept(*acceptance())
    first = (await repository.claim_outbox("worker-a", lease_s=60))[0]
    assert first["generation"] == first["attempts"] == 1
    assert await repository.claim_outbox("worker-b") == []
    assert not await repository.ack_outbox(first["id"], "worker-b", first["generation"])
    async with repository.engine.begin() as conn:
        now = await repository._now(conn)
        await conn.execute(repository.outbox.update().values(lease_until=now - timedelta(seconds=1)))
    # An expired worker cannot acknowledge or reschedule even before takeover.
    assert not await repository.ack_outbox(first["id"], "worker-a", 1)
    assert not await repository.retry_outbox(first["id"], "worker-a", 1, "late", 0)
    second = (await repository.claim_outbox("worker-b"))[0]
    assert second["generation"] == second["attempts"] == 2
    assert not await repository.ack_outbox(first["id"], "worker-a", 1)
    assert not await repository.retry_outbox(first["id"], "worker-a", 1, "late", 0)
    assert await repository.retry_outbox(second["id"], "worker-b", 2, "broker down", 60)
    assert await repository.claim_outbox("worker-c") == []
    async with repository.engine.begin() as conn:
        now = await repository._now(conn)
        await conn.execute(repository.outbox.update().values(available_at=now - timedelta(seconds=1)))
    await repository.close()
    await repository.start()
    third = (await repository.claim_outbox("worker-c"))[0]
    assert third["generation"] == third["attempts"] == 3
    assert await repository.ack_outbox(third["id"], "worker-c", 3)
    assert not await repository.ack_outbox(third["id"], "worker-c", 3)
    assert await repository.claim_outbox("worker-d") == []


async def test_inbox_effect_is_transactional_and_idempotent(repository):
    async def fail_after_write(conn):
        await conn.execute(repository.events.insert().values(id="effect", pipeline="driver"))
        raise RuntimeError("crashed before transaction commit")

    with pytest.raises(RuntimeError, match="before transaction commit"):
        await repository.consume("message", "consumer", fail_after_write)
    assert await count(repository, repository.inbox) == 0
    assert await count(repository, repository.events) == 0

    async def durable_effect(conn):
        await conn.execute(repository.events.insert().values(id="effect", pipeline="driver"))

    assert await repository.consume("message", "consumer", durable_effect)
    assert not await repository.consume("message", "consumer", durable_effect)
    assert await count(repository, repository.inbox) == 1
    assert await count(repository, repository.events) == 1


async def test_new_records_remain_readable_with_legacy_projections(repository, monkeypatch):
    await repository.accept(*acceptance())
    await repository.update_evidence("capture", state="Saved", artifacts={
        "snapshot": {"uri": "s3://evidence/new.jpg"},
        "clip": {"uri": "s3://evidence/new.mp4"},
    })
    await repository.update_evidence("capture", state="Incomplete", reason="coverage_missing")
    async with repository.engine.begin() as conn:
        legacy_incident = dict((await conn.execute(select(repository.incidents))).mappings().first())
    assert legacy_incident["payload"]["evidence_uris"] == [
        "s3://evidence/new.jpg", "s3://evidence/new.mp4", "s3://evidence/old.jpg",
    ]
    assert legacy_incident["payload"]["evidence_state"] == "Incomplete"
    assert legacy_incident["payload"]["evidence_id"] == "capture"
    async with repository.engine.begin() as conn:
        event = (await conn.execute(select(repository.events))).mappings().first()
    assert event["payload"]["session_id"] == "session"
    await repository.create_job({"id": "job", "pipeline": "driver", "status": "queued",
                                 "input": {"uri": "input.mp4"}, "result": None})
    async with repository.engine.begin() as conn:
        job = (await conn.execute(select(repository.jobs.c.id, repository.jobs.c.status))).mappings().first()
    assert job["status"] == "queued"


async def test_evidence_identity_and_session_cannot_be_reassigned(repository):
    await repository.accept(*acceptance())
    for changes in ({"id": "replacement"}, {"session_id": "other"}):
        with pytest.raises(ValueError, match="immutable"):
            await repository.update_evidence("capture", **changes)
    assert (await repository.get_evidence("capture"))["revision"] == 0


async def test_job_input_is_atomic_immutable_and_survives_restart(repository):
    row = {"id": "job", "pipeline": "driver", "status": "queued", "result": None,
           "input": {"uri": "input.mp4", "stream_id": "external-name", "options": {"sample": 2}}}
    with pytest.raises(StatementError):
        await repository.create_job({**row, "input": {"invalid": object()}})
    assert await repository.get_job("job") is None
    assert await count(repository, repository.job_requests) == 0
    await repository.create_job(row)
    await repository.create_job({**row, "status": "failed", "input": {"uri": "wrong.mp4"}})
    await repository.close()
    await repository.start()
    restored = await repository.get_job("job")
    assert restored["status"] == "queued"
    assert restored["input"] == row["input"]
    assert (await repository.list_jobs())[0]["input"] == row["input"]
    await repository.update_job("job", status="running", result={"progress": 5})
    assert await repository.interrupt_jobs() == 1
    assert await repository.interrupt_jobs() == 0
    failed = await repository.get_job("job")
    assert failed["status"] == "failed"
    assert failed["result"] == {"progress": 5, "error": "process_interrupted"}
    assert failed["input"] == row["input"]


async def test_job_service_recovers_and_retries_stored_input(repository):
    from inferops.application.jobs import JobService

    await repository.create_job({"id": "queued", "pipeline": "driver", "status": "queued",
                                 "input": {"uri": "queued.mp4"}})
    await repository.create_job({"id": "interrupted", "pipeline": "driver", "status": "running",
                                 "input": {"uri": "running.mp4"}})
    application = SimpleNamespace(repository=repository, definition=SimpleNamespace(pipelines={"driver": object()}))
    service = JobService(SimpleNamespace(application=application))
    scheduled = []
    service._schedule = lambda *args: scheduled.append(args)
    await service.start()
    assert scheduled == [("queued", "driver", {"uri": "queued.mp4"})]
    retry = await service.retry("interrupted")
    assert retry["job_id"] != "interrupted"
    assert scheduled[-1][2] == {"uri": "running.mp4", "retry_of": "interrupted"}
    assert (await repository.get_job(retry["job_id"]))["input"] == scheduled[-1][2]
    assert (await repository.get_job("interrupted"))["input"] == {"uri": "running.mp4"}


async def test_legacy_job_without_new_input_is_readable(repository):
    async with repository.engine.begin() as conn:
        await conn.execute(repository.jobs.insert().values(id="legacy", pipeline="driver", status="succeeded"))
    assert (await repository.get_job("legacy"))["input"] == {}
    assert (await repository.list_jobs())[0]["input"] == {}


async def test_schema_version_marker_survives_idempotent_startup(repository):
    async with repository.engine.connect() as conn:
        versions = (await conn.execute(select(repository.schema_versions))).mappings().all()
    assert [row["version"] for row in versions] == [1]
    assert versions[0]["applied_at"]
    await repository.close()
    await repository.start()
    assert await count(repository, repository.schema_versions) == 1


async def test_schema_failure_does_not_record_migration(tmp_path, monkeypatch):
    repo = SqlRepository(f"sqlite+aiosqlite:///{tmp_path / 'failed-start.db'}")
    create_all = repo.metadata.create_all

    def create_then_fail(connection):
        create_all(connection)
        raise RuntimeError("injected migration failure")

    monkeypatch.setattr(repo.metadata, "create_all", create_then_fail)
    with pytest.raises(RuntimeError, match="migration failure"):
        await repo.start()
    assert repo._engine is None
    # SQLite can retain DDL after rollback, but an unapplied version must never
    # appear successful. PostgreSQL rolls back the transaction's DDL as well.
    engine = create_async_engine(repo.url)
    try:
        async with engine.connect() as conn:
            assert await conn.scalar(select(func.count()).select_from(repo.schema_versions)) == 0
    finally:
        await engine.dispose()
    monkeypatch.setattr(repo.metadata, "create_all", create_all)
    await repo.start()
    try:
        assert await count(repo, repo.schema_versions) == 1
    finally:
        await repo.close()


async def test_outbox_capacity_rolls_back_acceptance_and_evidence_update(repository):
    repository.max_pending_outbox = 1
    assert await repository.accept(*acceptance())
    # Replaying stable IDs is permitted even when pending capacity is full.
    assert await repository.accept(*acceptance())
    different = acceptance()
    different[0][0]["id"] = "event-2"
    different[1][0]["id"] = "incident-2"
    different[2][0]["id"] = "capture-2"
    different[3][0]["id"] = "event:incident-2"
    with pytest.raises(RuntimeError, match="outbox capacity exhausted"):
        await repository.accept(*different)
    for table in (repository.events, repository.incidents, repository.evidence, repository.outbox):
        assert await count(repository, table) == 1
    with pytest.raises(RuntimeError, match="outbox capacity exhausted"):
        await repository.update_evidence("capture", state="Saved", artifacts={"snapshot": {"uri": "new.jpg"}})
    assert (await repository.get_evidence("capture"))["state"] == "Pending"
    assert (await repository.get_evidence("capture"))["revision"] == 0
    claim = (await repository.claim_outbox("relay"))[0]
    assert await repository.ack_outbox(claim["id"], "relay", claim["generation"])
    assert (await repository.update_evidence("capture", state="Saved"))["state"] == "Saved"
    assert (await repository.diagnostics())["pending_outbox"] == 1


async def test_acceptance_inbox_atomicity_and_duplicate_no_effect(repository):
    broken = acceptance()
    broken[3][0]["payload"] = {"bad": object()}
    with pytest.raises(StatementError):
        await repository.accept(*broken, inbox=("source-event", "pipeline"))
    assert await count(repository, repository.inbox) == 0
    assert await repository.accept(*acceptance(), inbox=("source-event", "pipeline"))
    duplicate = acceptance()
    duplicate[0][0]["id"] = "unwanted-effect"
    assert not await repository.accept(*duplicate, inbox=("source-event", "pipeline"))
    assert await count(repository, repository.inbox) == 1
    assert await count(repository, repository.events) == 1


async def test_acceptance_capacity_failure_releases_inbox_reservation(repository):
    repository.max_pending_outbox = 0
    with pytest.raises(RuntimeError, match="outbox capacity exhausted"):
        await repository.accept(*acceptance(), inbox=("source-event", "pipeline"))
    assert await count(repository, repository.inbox) == 0
    assert await count(repository, repository.events) == 0
    repository.max_pending_outbox = 1
    assert await repository.accept(*acceptance(), inbox=("source-event", "pipeline"))


async def test_deleted_evidence_hides_legacy_links_and_retains_unrelated_artifacts(repository):
    await repository.accept(*acceptance())
    artifacts = {"snapshot": {"uri": "s3://evidence/new.jpg"}, "clip": {"uri": "s3://evidence/new.mp4"}}
    await repository.update_evidence("capture", state="Saved", artifacts=artifacts)
    await repository.update_evidence("capture", state="Failed", deleted=True,
                                     deleted_artifacts=artifacts, artifacts={}, deletion_pending=True)
    incident = (await repository.list_incidents())[0]
    assert incident["payload"]["evidence_uris"] == ["s3://evidence/old.jpg"]
    assert incident["payload"]["evidence_deleted"] is True
    await repository.update_evidence("capture", deletion_pending=False)
    assert (await repository.list_incidents())[0]["payload"]["evidence_uris"] == ["s3://evidence/old.jpg"]
    assert (await repository.get_evidence("capture"))["deleted_artifacts"] == artifacts


async def test_duplicate_save_cannot_reopen_fenced_session(repository):
    original = {"id": "session", "owner": "old", "generation": 1, "owner_generation": 1}
    await repository.save_session(original)
    await repository.interrupt_other_sessions("new")
    fenced = await repository.save_session({**original, "owner": "new", "state": "active"})
    assert fenced["owner"] == "old"
    assert fenced["state"] == "interrupted"
    assert fenced["owner_generation"] == 2
    assert fenced == await repository.get_session("session")


async def test_diagnostics_report_aggregates_without_document_loading(repository):
    await repository.accept(*acceptance())
    await repository.save_session({"id": "active", "owner": "worker"})
    await repository.save_session({"id": "closed", "owner": "worker"})
    await repository.close_session("closed")
    diagnostic = await repository.diagnostics()
    assert diagnostic["pending_outbox"] == 1
    assert diagnostic["evidence_states"] == {"Pending": 1}
    assert diagnostic["session_states"] == {"open": 1, "closed": 1}
    assert diagnostic["active_sessions"] == 1
    assert diagnostic["oldest_outbox_age_seconds"] >= 0
    assert diagnostic["expired_outbox_leases"] == diagnostic["outbox_failures"] == 0


async def test_job_cancelled_before_task_body_starts_is_persisted(repository, monkeypatch):
    from inferops.application.jobs import JobService

    await repository.create_job({"id": "queued", "pipeline": "driver", "status": "queued",
                                 "input": {"uri": "queued.mp4"}})
    original_get = repository.get_job
    initial = await original_get("queued")
    first = True

    async def immediate_first_lookup(job_id):
        nonlocal first
        if first:
            first = False
            return initial
        return await original_get(job_id)

    monkeypatch.setattr(repository, "get_job", immediate_first_lookup)
    application = SimpleNamespace(repository=repository, definition=SimpleNamespace(pipelines={"driver": object()}))
    service = JobService(SimpleNamespace(application=application))
    service._schedule("queued", "driver", initial["input"])
    task = service.tasks["queued"]
    cancelled = await service.cancel("queued")
    assert task.cancelled()
    assert cancelled["status"] == "cancelled"
    assert (await original_get("queued"))["status"] == "cancelled"
    await asyncio.sleep(0)  # Allow the task bookkeeping callback to finish.


async def test_queued_job_recovery_ignores_more_recent_completed_jobs(repository):
    await repository.create_job({"id": "oldest", "pipeline": "driver", "status": "queued",
                                 "input": {"uri": "old.mp4"}})
    await repository.create_job({"id": "second", "pipeline": "driver", "status": "queued",
                                 "input": {"uri": "second.mp4"}})
    async with repository.engine.begin() as conn:
        now = await repository._now(conn)
        await conn.execute(repository.jobs.update().where(repository.jobs.c.id == "oldest").values(
            created_at=now - timedelta(hours=2),
        ))
        await conn.execute(repository.jobs.update().where(repository.jobs.c.id == "second").values(
            created_at=now - timedelta(hours=1),
        ))
    for number in range(4):
        await repository.create_job({"id": f"completed-{number}", "pipeline": "driver", "status": "succeeded"})
    assert len(await repository.list_jobs(limit=2)) == 2
    queued = await repository.list_queued_jobs(limit=1)
    assert queued[0]["id"] == "oldest"
    assert queued[0]["input"] == {"uri": "old.mp4"}
    assert [row["id"] for row in await repository.list_queued_jobs(limit=2)] == ["oldest", "second"]
    assert await repository.list_queued_jobs(limit=0) == []
