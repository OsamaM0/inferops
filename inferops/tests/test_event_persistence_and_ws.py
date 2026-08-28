"""Regressions for defects found by driving the dashboard in a real browser.

1. Fusion events were emitted but never stored, so GET /v1/events was
   permanently empty and the dashboard's Events panel could never populate —
   while alerts persisted automatically. Events are now stored symmetrically,
   under the *pipeline* name the endpoint filters on.
2. The server fans out binary WebSocket frames, but the dashboard's alert
   socket parsed `event.data` as text, so every pushed alert was dropped with
   only a console warning.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from inferops.alerts.engine import AlertEngine
from inferops.core.types import StageItem
from inferops.fusion.engine import FusionEngine
from inferops.fusion.spec import FusionSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD = REPO_ROOT / "frontend" / "dashboard.html"


def _fusion_spec(**overrides) -> FusionSpec:
    data = {
        "name": "unit_fusion",
        "signals": ["score"],
        "windows": {"score": {"kind": "rolling_time", "duration_ms": 5000,
                              "reducers": ["latest", "max"]}},
        "extract": {"score": "payload.get('score')"},
        "rules": [{"name": "hot", "when": "score.latest() > 0.5",
                   "emit": {"type": "hot"}}],
    }
    data.update(overrides)
    return FusionSpec(**data)


@pytest.mark.asyncio
async def test_fusion_events_are_persisted_under_the_pipeline_name():
    from inferops.adapters.postgres_dao import init_schema, list_events

    await init_schema()
    engine = FusionEngine(_fusion_spec())
    engine.spec.outputs.events_topic = None

    item = StageItem(stream_id="veh-1", ts_ms=9_000_000, payload={"score": 0.9})
    events = await engine.ingest_item(item, pipeline="my_pipeline")
    assert events, "fusion rule should have fired"

    stored = await list_events("my_pipeline", 50)
    types = {row["type"] for row in stored}
    assert "hot" in types, f"event was not persisted; got {types}"

    # The endpoint filters on the pipeline name, not the fusion spec name.
    row = next(r for r in stored if r["type"] == "hot")
    assert row["pipeline"] == "my_pipeline"
    assert row["entity_id"] == "veh-1"
    assert row["id"], "events need a stable id to be stored and de-duplicated"


@pytest.mark.asyncio
async def test_fusion_persistence_can_be_disabled():
    from inferops.adapters.postgres_dao import init_schema, list_events

    await init_schema()
    engine = FusionEngine(_fusion_spec(name="quiet_fusion", persistence_enabled=False))
    engine.spec.outputs.events_topic = None

    item = StageItem(stream_id="veh-2", ts_ms=9_100_000, payload={"score": 0.9})
    events = await engine.ingest_item(item, pipeline="quiet_pipeline")
    assert events, "rule should still fire"
    assert await list_events("quiet_pipeline", 50) == []


@pytest.mark.asyncio
async def test_fusion_persistence_failure_never_breaks_the_pipeline(monkeypatch):
    """A database outage must degrade to a warning, not drop the frame."""
    engine = FusionEngine(_fusion_spec(name="failing_fusion"))
    engine.spec.outputs.events_topic = None

    async def boom(_row):
        raise RuntimeError("database is down")

    import inferops.adapters.postgres_dao as dao

    monkeypatch.setattr(dao, "insert_event", boom)
    FusionEngine._persist_warned = False

    item = StageItem(stream_id="veh-3", ts_ms=9_200_000, payload={"score": 0.9})
    events = await engine.ingest_item(item, pipeline="failing_pipeline")
    assert events, "events must still reach the caller when persistence fails"
    assert FusionEngine._persist_warned, "silent data loss: nothing was logged"


@pytest.mark.asyncio
async def test_alert_persistence_failure_warns_rather_than_hiding(monkeypatch):
    """This was log.debug, so a dead database lost every alert silently."""
    import inferops.adapters.postgres_dao as dao

    async def boom(_row):
        raise RuntimeError("database is down")

    monkeypatch.setattr(dao, "insert_alert", boom)
    AlertEngine._persist_warned = False

    from inferops.alerts.spec import AlertSpec

    engine = AlertEngine(AlertSpec(
        name="unit_alerts",
        severity={"levels": [{"name": "low", "rank": 1}], "default": "low"},
        sinks=[],
    ))
    item = StageItem(stream_id="veh-4", ts_ms=9_300_000)
    item.payload["fusion_events"] = [
        {"type": "hot", "entity_id": "veh-4", "ts_ms": 9_300_000, "payload": {}}
    ]
    alerts = await engine.ingest_item(item)
    assert alerts, "alerts must still reach sinks when persistence fails"
    assert AlertEngine._persist_warned, "silent data loss: nothing was logged"


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


def test_ws_hub_sends_binary_frames():
    """Pins the server side of the contract the dashboard test above assumes."""
    from inferops.pipeline.stages import sink_ws

    source = Path(sink_ws.__file__).read_text(encoding="utf-8")
    assert "send_bytes" in source
    assert "send_text" not in source
