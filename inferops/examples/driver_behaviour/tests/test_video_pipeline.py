"""End-to-end: drive a sample video through the platform.

Exercises decode_any -> infer (synthetic DETR backend) -> RFDetr postprocess
-> fusion (driver_fusion) -> alerts (driver_alerts).
"""
import os
from pathlib import Path

import pytest

from inferops.alerts.store import autoload as autoload_alerts
from inferops.core.types import StageItem, now_ms
from inferops.fusion.store import autoload as autoload_fusion
from inferops.pipeline.store import autoload as autoload_pipelines
from inferops.pipeline.store import get_pipeline_store
from inferops.registry.store import autoload as autoload_models

# The clip was hardcoded to /workspace/train/test.mp4, an absolute path that
# existed on one machine only, so this end-to-end test silently skipped
# everywhere else — including CI. Discover it relative to the checkout instead,
# keeping the old path as a last resort and INFEROPS_TEST_VIDEO as an override.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]   # tests/ -> project root
_REPO_ROOT = _PROJECT_ROOT.parent                     # -> git repo root


def _find_video() -> Path | None:
    override = os.environ.get("INFEROPS_TEST_VIDEO")
    candidates = [Path(override)] if override else []
    candidates += [
        _REPO_ROOT / "train" / "test.mp4",
        _PROJECT_ROOT / "train" / "test.mp4",
        Path("/workspace/train/test.mp4"),
    ]
    return next((c for c in candidates if c.is_file()), None)


VIDEO = _find_video()


@pytest.mark.asyncio
@pytest.mark.skipif(VIDEO is None, reason="no sample video found (set INFEROPS_TEST_VIDEO)")
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
