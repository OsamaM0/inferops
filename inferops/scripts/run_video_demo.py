"""Run the driver-behaviour video pipeline against a real .mp4 file.

End-to-end exercise of: decode_video -> infer (synthetic DETR) ->
postprocess_rfdetr -> FusionEngine -> AlertEngine.

Usage:
    python scripts/run_video_demo.py /workspace/train/test.mp4
"""
from __future__ import annotations

import asyncio
import os
import sys
from collections import Counter
from pathlib import Path

# make sure we resolve configs/ relative to the repo
REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))  # so `examples.*` is importable
os.environ.setdefault("INFEROPS_TRANSPORT_KIND", "redpanda")
os.environ.setdefault("INFEROPS_PG_URL", "sqlite+aiosqlite:///./.demo.db")
os.environ.setdefault("INFEROPS_OBS_LOG_JSON", "false")

from inferops.alerts.store import autoload as autoload_alerts  # noqa: E402
from inferops.core.types import StageItem, now_ms  # noqa: E402
from inferops.fusion.store import autoload as autoload_fusion  # noqa: E402
from inferops.pipeline.store import autoload as autoload_pipelines  # noqa: E402
from inferops.pipeline.store import get_pipeline_store  # noqa: E402
from inferops.registry.store import autoload as autoload_models  # noqa: E402


async def main(video_path: str, pipeline: str = "driver_video") -> int:
    if not Path(video_path).exists():
        print(f"error: file not found: {video_path}", file=sys.stderr)
        return 2

    await autoload_models()
    await autoload_pipelines()
    await autoload_fusion()
    await autoload_alerts()

    store = get_pipeline_store()
    runner = await store.get_runner(pipeline)

    item = StageItem(stream_id="cam-demo", ts_ms=now_ms(),
                     payload={"video_path": video_path})
    print(f"[demo] running '{pipeline}' against {video_path}")
    outs = await runner.run_one(item)

    frames = len(outs)
    fusion_events: list[dict] = []
    alerts: list[dict] = []
    det_counts: Counter[str] = Counter()
    for it in outs:
        for ev in it.payload.get("fusion_events", []):
            fusion_events.append(ev)
        for a in it.payload.get("alerts", []):
            alerts.append(a)
        for r in it.results:
            for d in (r.detections or []):
                det_counts[d.label] += 1

    print(f"[demo] frames processed: {frames}")
    print(f"[demo] detections by class: {dict(det_counts)}")
    print(f"[demo] fusion events: {len(fusion_events)}")
    for ev in fusion_events[:5]:
        print(f"   - {ev.get('type')} @ ts={ev.get('ts_ms')} severity={ev.get('severity')}")
    print(f"[demo] alerts emitted: {len(alerts)}")
    for a in alerts[:5]:
        print(f"   - {a.get('type')} sev={a.get('severity')} dedupe={a.get('dedupe_key')}")
    return 0 if alerts else 1


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "/workspace/train/test.mp4"
    pipeline_name = sys.argv[2] if len(sys.argv) > 2 else "driver_video"
    sys.exit(asyncio.run(main(video, pipeline_name)))
