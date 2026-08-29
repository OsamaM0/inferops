"""Run one pipeline against a video file and print a summary.

Works for any domain pack: select it with INFEROPS_ACTIVE_DOMAIN (or run via
``make``), then::

    INFEROPS_ACTIVE_DOMAIN=driver_behaviour python scripts/run_video_demo.py ../train/test.mp4 driver_video

With no pipeline argument the first pipeline of the active config surface is
used. Exits 0 when at least one alert was emitted, 1 otherwise.
"""
from __future__ import annotations

import asyncio
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)   # configs/ and model_repository/ are resolved relative to the project
os.environ.setdefault("INFEROPS_TRANSPORT_KIND", "inproc")
os.environ.setdefault("INFEROPS_PG_URL", "sqlite+aiosqlite:///./.demo.db")
os.environ.setdefault("INFEROPS_OBS_LOG_JSON", "false")

from inferops.alerts.store import autoload as autoload_alerts  # noqa: E402
from inferops.core.types import StageItem, now_ms  # noqa: E402
from inferops.fusion.store import autoload as autoload_fusion  # noqa: E402
from inferops.pipeline.store import autoload as autoload_pipelines  # noqa: E402
from inferops.pipeline.store import get_pipeline_store  # noqa: E402
from inferops.registry.store import autoload as autoload_models  # noqa: E402


async def main(video_path: str, pipeline: str | None = None) -> int:
    if not Path(video_path).exists():
        print(f"error: file not found: {video_path}", file=sys.stderr)
        return 2

    await autoload_models()
    await autoload_pipelines()
    await autoload_fusion()
    await autoload_alerts()

    store = get_pipeline_store()
    if pipeline is None:
        specs = store.list()
        if not specs:
            print("error: no pipelines loaded (set INFEROPS_ACTIVE_DOMAIN)", file=sys.stderr)
            return 2
        pipeline = specs[0].name
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
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(asyncio.run(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)))
