"""End-to-end: drive the sample dashcam clip through the driver pipeline.

Exercises decode_any -> infer (real RF-DETR ONNX weights) -> RFDetr postprocess
-> real face mesh -> fusion (driver_fusion) -> alerts (driver_alerts).
"""
from pathlib import Path

import pytest

from inferops.api.main import create_app
from inferops.core.domain import load_domain_manifest
from inferops.core.types import StageItem, now_ms
from inferops.pipeline.store import get_pipeline_store
from inferops.testing.fixtures import find_sample_video

DOMAIN_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = load_domain_manifest(DOMAIN_ROOT)
VIDEO = find_sample_video(
    DOMAIN_ROOT / (MANIFEST.scenario.sample_video or "missing.mp4"),
    Path("/workspace/research/driver_training/test.mp4"),
)


@pytest.mark.asyncio
@pytest.mark.skipif(VIDEO is None, reason="no sample video found (set INFEROPS_TEST_VIDEO)")
async def test_video_drives_pipeline_and_emits_alert():
    app = create_app()
    async with app.router.lifespan_context(app):
        runner = await get_pipeline_store().get_runner(MANIFEST.scenario.pipeline)
        item = StageItem(stream_id="cam-test", ts_ms=now_ms(),
                         payload={"video_path": str(VIDEO)})
        outs = await runner.run_one(item)
    assert outs, "expected at least one frame produced"
    assert all(it.payload["model_selection"]["used"] == "rfdetr_driver_onnx" for it in outs)

    alerts: list = []
    events: list = []
    for it in outs:
        alerts.extend(it.payload.get("alerts", []))
        events.extend(it.payload.get("fusion_events", []))

    assert events, "fusion should have emitted at least one event"
    assert alerts, "alert engine should have emitted at least one alert"
    assert any(a["severity"] in {"low", "medium", "high"} for a in alerts)
