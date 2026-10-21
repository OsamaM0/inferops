import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from inferops.adapters.outbound.media.owned_spool import OwnedSpool
from inferops.adapters.outbound.persistence.sql import SqlRepository
from inferops.domain.evidence_budget import evidence_budget
from inferops.testing.postgres import test_database_url
from tests.test_recoverable_evidence import record, service


def test_twelve_camera_budget_uses_high_percentile_and_overhead():
    result = evidence_budget(cameras=12, jpeg_p95_bytes=250 * 1024, width=1920, height=1080)
    assert result["global_history_bytes"] == 368640000
    assert result["max_active_captures"] == 36
    assert result["burst_material_bytes"] == 1105920000


@pytest.mark.parametrize("fault", [None, "storage", "quota"])
async def test_180_accepted_capture_requests_all_reach_truthful_terminal_outcomes(tmp_path, fault):
    overrides = {"max_active_captures": 256, "workers": 2}
    if fault == "quota":
        overrides["max_spool_bytes"] = 1000
    evidence, repository, storage = service(tmp_path, **overrides)
    storage.fail_clip = fault == "storage"
    try:
        for camera in range(60):
            session = f"camera-{camera}"
            for seq in range(2):
                await evidence.observe(session, "t1", seq * 100_000_000, seq, b"jpeg")
            for index in range(3):
                value = record(repository, f"c{camera}-incident-{index}", session)
                await evidence.request(value)
            await evidence.observe(session, "t1", 200_000_000, 2, b"jpeg")
        await asyncio.wait_for(evidence.wait_idle(), 60)
        assert len(repository.records) == 180
        assert all(row["state"] in {"Saved", "Incomplete", "Failed"} for row in repository.records.values())
        for row in repository.records.values():
            if row["state"] == "Saved":
                assert fault is None
                assert not row["errors"]
                for artifact in row["artifacts"].values():
                    assert await storage.verify(artifact["uri"], artifact["sha256"])
            else:
                assert row["errors"]
    finally:
        await evidence.close()


async def test_live_spool_owner_blocks_recovery_until_native_operations_drain(tmp_path):
    old, new = OwnedSpool(tmp_path, disk_reserve_bytes=0), OwnedSpool(tmp_path, disk_reserve_bytes=0)
    await old.start()
    try:
        with pytest.raises(RuntimeError, match="live process owner"):
            await new.start()
    finally:
        await old.close()
    await new.start()
    await new.close()


async def test_retention_excludes_pending_delivery_and_checks_again_inside_tombstone_transaction(tmp_path):
    repository = SqlRepository(test_database_url(tmp_path))
    await repository.start()
    before = datetime.now(timezone.utc) + timedelta(days=1)
    try:
        await repository.accept([], [], [
            {"id": "saved", "session_id": "s1", "state": "Saved"},
            {"id": "pending", "session_id": "s2"},
            {"id": "delivery", "session_id": "s3", "state": "Saved"}],
            [{"id": "message", "key": "s3", "topic": "incident.accepted", "payload": {}}])
        assert [row["id"] for row in await repository.retention_candidates(before)] == ["saved"]
        # A delivery intent arrives after the dry run. The mutation must refuse it.
        await repository.accept([], [], [], [{"id": "later", "key": "s1", "topic": "incident.accepted", "payload": {}}])
        with pytest.raises(ValueError, match="no longer eligible"):
            await repository.update_evidence("saved", deleted=True, _retention_before=before)
        assert not (await repository.get_evidence("saved")).get("deleted")
    finally:
        await repository.close()


async def test_outbox_cleanup_uses_delivery_age_and_never_prunes_inbox_or_pending_work(tmp_path):
    repository = SqlRepository(test_database_url(tmp_path))
    await repository.start()
    try:
        messages = [{"id": name, "key": "s1", "topic": "event", "payload": {}} for name in ["one", "two"]]
        await repository.accept([], [], [], messages, inbox=("input", "consumer"))
        claims = await repository.claim_outbox("relay", limit=1)
        assert await repository.ack_outbox(claims[0]["id"], "relay", claims[0]["generation"])
        before = datetime.now(timezone.utc) + timedelta(days=1)
        preview = await repository.prune_delivered_outbox(before)
        assert preview["dry_run"] and preview["message_ids"] == ["one"]
        assert (await repository.prune_delivered_outbox(before, dry_run=False))["message_ids"] == ["one"]
        assert not (await repository.prune_delivered_outbox(before, dry_run=False))["message_ids"]
        assert not await repository.record_inbox("input", "consumer")
        assert (await repository.diagnostics())["pending_outbox"] == 1
    finally:
        await repository.close()
