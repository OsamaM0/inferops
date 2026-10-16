"""Persistence is owned by application transactions, never domain decisions."""

import re
from pathlib import Path

import numpy as np
import yaml

from inferops import Engine
from inferops.adapters.outbound.messaging.delivery import DeliveryHub
from inferops.adapters.outbound.persistence.sql import SqlRepository
from inferops.domain.fusion import FusionEngine
from inferops.domain.fusion.spec import FusionSpec
from inferops.domain.processing import MediaFrame
from inferops.domain.types import StageItem
from tests.test_redesign_engine import project

REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = REPO_ROOT / "inferops/adapters/inbound/http/static/dashboard.js"


def _fusion_spec():
    return FusionSpec(
        name="unit_fusion",
        signals=["score"],
        extract={"score": "payload.get('score')"},
        windows={"score": {"kind": "rolling_time", "duration_ms": 5000, "reducers": ["latest"]}},
        rules=[{"name": "hot", "when": "score.latest() > 0.5", "emit": {"type": "hot"}}],
    )


def application(tmp_path):
    root = project(tmp_path)
    folder = root / "configs/fusion"
    folder.mkdir()
    (folder / "unit.yaml").write_text(yaml.safe_dump(_fusion_spec().model_dump(mode="json")))
    path = root / "configs/pipelines/test.yaml"
    data = yaml.safe_load(path.read_text())
    data["stages"].insert(1, {"name": "fusion", "stage": "fuse", "params": {"fusion": "unit_fusion"}})
    path.write_text(yaml.safe_dump(data))
    return Engine.from_project(
        root, repository=SqlRepository(f"sqlite+aiosqlite:///{tmp_path}/events.sqlite")
    )


async def test_fusion_events_are_persisted_under_the_pipeline_name(tmp_path):
    selected = application(tmp_path)
    async with selected:
        session = await selected.open_session("test", stream_name="veh-1")
        await session.process(MediaFrame(np.zeros((8, 8, 3), np.uint8), 0, 0, payload={"score": 0.9}))
        rows = await selected.application.repository.list_events("test")
        assert len(rows) == 1 and rows[0]["type"] == "hot"
        assert rows[0]["pipeline"] == "test" and rows[0]["entity_id"] == session.id
        assert rows[0]["id"] and rows[0]["payload"]["stream_name"] == "veh-1"


async def test_domain_fusion_has_no_persistence_side_effects():
    fusion = FusionEngine(_fusion_spec())
    events = await fusion.ingest_item(StageItem(stream_id="veh", ts_ms=0, payload={"score": 0.9}))
    assert events and events[0].type == "hot"
    assert not hasattr(fusion, "repository")


async def test_persistence_failure_preserves_calculations_but_is_visible(tmp_path, monkeypatch):
    selected = application(tmp_path)
    async with selected:
        session = await selected.open_session("test")

        async def failed(*args, **kwargs):
            raise OSError("database is down")

        monkeypatch.setattr(selected.application.repository, "accept", failed)
        out = (
            await session.process(MediaFrame(np.zeros((8, 8, 3), np.uint8), 0, 0, payload={"score": 0.9}))
        )[0]
        assert out.payload["fusion_events"]
        assert out.payload["incident_error"]["code"] == "persistence_unavailable"
        assert out.payload["alerts"] == []


async def test_failed_incident_acceptance_can_be_retried_without_duplicate_effects(tmp_path, monkeypatch):
    selected = application(tmp_path)
    async with selected:
        session = await selected.open_session("test")
        original = selected.application.repository.accept

        async def failed(*args, **kwargs):
            raise OSError("database is down")

        monkeypatch.setattr(selected.application.repository, "accept", failed)
        frame = MediaFrame(np.zeros((8, 8, 3), np.uint8), 0, 0, payload={"score": 0.9})
        assert not (await session.process(frame))[0].payload["alerts"]
        monkeypatch.setattr(selected.application.repository, "accept", original)
        assert (await session.process(frame))[0].payload["alerts"]
        await session.process(frame)
        assert len(await selected.application.repository.list_incidents()) == 1


# ── dashboard WebSocket contract ─────────────────────────────────────────────


def test_every_dashboard_socket_handles_binary_frames():
    """sink_ws always calls send_bytes, so a text-only reader drops messages."""
    html = DASHBOARD.read_text(encoding="utf-8")

    sockets = re.findall(r"new WebSocket\(", html)
    assert sockets, "expected the dashboard to open WebSockets"
    binary_type = re.findall(r"binaryType\s*=\s*[\"']arraybuffer[\"']", html)
    assert len(binary_type) >= len(sockets), (
        f"{len(sockets)} sockets opened but only {len(binary_type)} set "
        "binaryType='arraybuffer'; a Blob reaches JSON.parse and the payload is lost"
    )

    # All three readers must go through the one shared decoder so they cannot
    # drift apart again the way the alert socket did.
    assert "async function decodeSocketPayload" in html
    assert len(re.findall(r"decodeSocketPayload\(event\.data\)", html)) == len(sockets)

    # And no reader may parse event.data directly any more.
    assert "JSON.parse(event.data)" not in html


async def test_ws_hub_bounds_notifications_and_unsubscribes():
    import asyncio

    hub = DeliveryHub()
    queue = asyncio.Queue(maxsize=1)
    hub.subscribe("alerts.session", queue)
    await hub.publish("alerts.session", {"id": "first"})
    await hub.publish("alerts.session", {"id": "latest"})
    assert queue.qsize() == 1 and (await queue.get())["id"] == "latest"
    hub.unsubscribe("alerts.session", queue)
    assert not hub.subscribers
