"""Alerts and state changes require positive observations across media time."""
import pytest

from inferops.domain.fusion import FusionEngine
from inferops.domain.fusion.spec import FusionSpec
from inferops.domain.types import StageItem


def fusion():
    return FusionEngine(FusionSpec(
        name="driver", signals=["phone"], windows={"phone": {"duration_ms": 5000}},
        extract={"phone": "payload.get('phone', 0)"},
        rules=[{"name": "phone_use", "when": "phone.latest() > .5", "confirm_ms": 3000,
                "min_samples": 3, "max_gap_ms": 1500, "cooldown_ms": 30000}],
        state_machine={"initial": "normal", "states": ["normal", "unsafe"], "transitions": [
            {"from_state": "normal", "to_state": "unsafe", "when": "phone.latest() > .5",
             "confirm_ms": 3000, "min_samples": 3, "max_gap_ms": 1500}]}))


async def observe(engine, timestamp, value=1, camera="a"):
    return await engine.ingest_item(StageItem(stream_id=camera, ts_ms=timestamp, payload={"phone": value}))


async def test_alert_and_state_wait_three_seconds_and_are_isolated_per_camera():
    engine = fusion()
    for timestamp in (0, 500, 1000, 2000, 2999):
        assert not await observe(engine, timestamp)
    assert not await observe(engine, 3000, camera="b")
    assert [event.type for event in await observe(engine, 3000)] == ["phone_use", "STATE_unsafe"]
    assert not await observe(engine, 3500)


@pytest.mark.parametrize("interrupt", ["negative", "gap", "rewind"])
async def test_spikes_missing_observations_and_rewinds_restart_confirmation(interrupt):
    engine = fusion()
    assert not await observe(engine, 0)
    assert not await observe(engine, 1000)
    if interrupt == "negative":
        assert not await observe(engine, 1500, 0)
        start = 2000
    elif interrupt == "gap":
        start = 4000
    else:
        start = 0
    for timestamp in (start, start + 1000, start + 2000):
        assert not await observe(engine, timestamp)
    assert {event.type for event in await observe(engine, start + 3000)} == {"phone_use", "STATE_unsafe"}


async def test_checkpoint_rolls_back_pending_confirmation_and_duplicate_times_do_not_count():
    engine = fusion()
    assert not await observe(engine, 0)
    checkpoint = engine.checkpoint()
    assert not await observe(engine, 1000)
    engine.restore(checkpoint)
    assert not await observe(engine, 3000)
    assert not await observe(engine, 3000)
    assert not await observe(engine, 4000)
    assert not await observe(engine, 5000)
    assert await observe(engine, 6000)


def test_critical_transition_cannot_use_old_dwell_to_bypass_fresh_confirmation():
    from inferops.domain.fusion.spec import StateMachineSpec
    from inferops.domain.fusion.state_machine import StateMachine

    machine = StateMachine(StateMachineSpec(initial="unsafe", states=["unsafe", "critical"], transitions=[{
        "from_state": "unsafe", "to_state": "critical", "when": "unsafe",
        "min_dwell_ms": 8000, "confirm_ms": 3000, "min_samples": 3, "max_gap_ms": 1500}]))
    for timestamp in range(0, 8000, 1000):
        assert machine.evaluate("driver", timestamp, {"unsafe": True})[0] is None
    for timestamp in (12000, 13000, 14000):
        assert machine.evaluate("driver", timestamp, {"unsafe": True})[0] is None
    assert machine.evaluate("driver", 15000, {"unsafe": True})[0] == "critical"
    assert machine.confirmed["driver"]["observed_ms"] == 3000


async def test_only_a_confirmed_behavior_creates_a_durable_incident_and_evidence(tmp_path):
    import numpy as np
    import yaml

    from inferops import Engine
    from inferops.adapters.outbound.persistence.sql import SqlRepository
    from inferops.adapters.outbound.storage.filesystem import FilesystemStorage
    from inferops.domain.evidence import EvidenceConfig
    from inferops.domain.processing import MediaFrame
    from inferops.testing.postgres import test_database_url
    from tests.test_redesign_engine import project

    root = project(tmp_path)
    (root / "features.py").write_text(
        "from inferops.adapters.outbound.stages.base import Stage\n"
        "class Features(Stage):\n    async def process(self, item):\n        return item\n")
    (root / "configs/fusion").mkdir()
    (root / "configs/fusion/driver.yaml").write_text(yaml.safe_dump(fusion().spec.model_dump(exclude={"state_machine"})))
    path = root / "configs/pipelines/test.yaml"
    pipeline = yaml.safe_load(path.read_text())
    pipeline["stages"].insert(1, {"name": "confirm", "stage": "fuse", "params": {"fusion": "driver"}})
    path.write_text(yaml.safe_dump(pipeline))
    engine = Engine.from_project(root, repository=SqlRepository(test_database_url(tmp_path)),
                                 storage=FilesystemStorage(tmp_path / "objects"), evidence=True,
                                 spool_path=tmp_path / "spool", evidence_config=EvidenceConfig(
                                     pre_seconds=0, post_seconds=0, disk_reserve_bytes=0, retry_delays=()))
    async with engine:
        session = await engine.open_session("test", stream_name="driver")
        image = np.zeros((32, 32, 3), np.uint8)
        for sequence, (timestamp, value) in enumerate(((0, 1), (500, 0), (1000, 1), (2000, 1), (3000, 1))):
            await session.process(MediaFrame(image, timestamp * 1_000_000, sequence, payload={"phone": value}))
            assert not await engine.application.repository.list_incidents()
            assert not await engine.application.repository.list_evidence()
        outputs = await session.process(MediaFrame(image, 4_000_000_000, 5, payload={"phone": 1}))
        assert len(outputs[0].payload["alerts"]) == 1
        assert outputs[0].payload["alerts"][0]["payload"]["confirmation"]["observed_ms"] == 3000
        await session.close()
        await engine.application.evidence.wait_idle()
        assert len(await engine.application.repository.list_incidents()) == 1
        [evidence] = await engine.application.repository.list_evidence()
        assert evidence["state"] == "Saved"
        for asset in evidence["artifacts"].values():
            assert await engine.application.evidence.storage.verify(asset["uri"], asset["sha256"])
