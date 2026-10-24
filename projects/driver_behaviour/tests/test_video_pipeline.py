"""End-to-end: drive the sample dashcam clip through the driver pipeline.

Exercises decode_any -> infer (real RF-DETR ONNX weights) -> RFDetr postprocess
-> real face mesh -> fusion (driver_fusion) -> alerts (driver_alerts).
"""
from pathlib import Path

import pytest

from inferops.adapters.inbound.http.main import create_app
from inferops.adapters.outbound.projects.manifest import load_domain_manifest
from inferops.testing.fixtures import find_sample_video

DOMAIN_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = load_domain_manifest(DOMAIN_ROOT)
VIDEO = find_sample_video(
    DOMAIN_ROOT / (MANIFEST.scenario.sample_video or "missing.mp4"),
    Path("/workspace/research/driver_training/test.mp4"),
)


@pytest.mark.asyncio
@pytest.mark.skipif(VIDEO is None, reason="no sample video found (set INFEROPS_TEST_VIDEO)")
async def test_video_drives_real_pipeline_and_only_emits_confirmed_incidents():
    app = create_app()
    async with app.router.lifespan_context(app):
        result = await app.state.engine.run(MANIFEST.scenario.pipeline, VIDEO, stream_name="cam-test")
    assert result["frames"] == 187
    assert "rfdetr_driver_onnx" in result["models_used"]
    # This sample has no ground-truth promise of unsafe behavior. A short,
    # intermittent detection must not be forced into an incident to pass QA.
    for alert in result["alerts"]:
        confirmed = alert["payload"]["confirmation"]
        assert confirmed["required_ms"] == 3000
        assert confirmed["observed_ms"] >= 3000 and confirmed["samples"] >= 3
        assert alert["type"] not in {"STATE_NORMAL", "STATE_RECOVERED"}
