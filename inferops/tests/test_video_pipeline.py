"""End-to-end: drive /workspace/train/test.mp4 through the platform.

Exercises decode_any -> infer (synthetic DETR backend) -> RFDetr postprocess
-> fusion (driver_fusion) -> alerts (driver_alerts).
"""
from pathlib import Path

import pytest

from inferops.alerts.store import autoload as autoload_alerts
from inferops.core.types import StageItem, now_ms
from inferops.fusion.store import autoload as autoload_fusion
from inferops.pipeline.store import autoload as autoload_pipelines
from inferops.pipeline.store import get_pipeline_store
from inferops.registry.store import autoload as autoload_models

VIDEO = Path("/workspace/train/test.mp4")


@pytest.mark.asyncio
@pytest.mark.skipif(not VIDEO.exists(), reason="test video missing")
async def test_video_drives_pipeline_and_emits_alert():
    await autoload_models()
    await autoload_pipelines()
    await autoload_fusion()
    await autoload_alerts()

    runner = await get_pipeline_store().get_runner("driver_behaviour")
    item = StageItem(stream_id="cam-test", ts_ms=now_ms(),
                     payload={"video_path": str(VIDEO)})
    outs = await runner.run_one(item)
    assert outs, "expected at least one frame produced"

    alerts: list = []
    events: list = []
    for it in outs:
        alerts.extend(it.payload.get("alerts", []))
        events.extend(it.payload.get("fusion_events", []))

    assert events, "fusion should have emitted at least one event"
    assert alerts, "alert engine should have emitted at least one alert"
    assert any(a["type"] == "phone_use" and a["severity"] == "medium" for a in alerts)
