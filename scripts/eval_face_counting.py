"""Evaluate face counting on real videos.

Runs each video through the ``face_counting`` pipeline in-process and prints,
per video, a JSON summary: the final counters (``unique_faces``, ``line_in``,
``line_out``, zone counts), how many track IDs were created, how many lost
faces were re-linked by appearance, embedder calls and latency. With
``--expected`` / ``--expected-in`` it compares against a ground-truth count
and exits non-zero on a mismatch.

    python scripts/eval_face_counting.py clip.mp4 --runtime cuda --expected 12 \
        --line "0,0.55,1,0.55" --output annotated.mp4

``--line x1,y1,x2,y2`` and ``--zone x1,y1,x2,y2,x3,y3,...`` (normalised) apply
to every video, overriding the project's counting geometry. ``--every-n 3``
processes every third frame, e.g. 10 fps from a 30 fps clip.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np


def _floats(text: str, minimum: int, what: str) -> list[list[float]]:
    values = [float(v) for v in text.replace(" ", "").split(",") if v]
    if len(values) % 2 or len(values) < 2 * minimum:
        raise argparse.ArgumentTypeError(f"{what} needs at least {minimum} x,y pairs")
    return [values[i:i + 2] for i in range(0, len(values), 2)]


def counting_override(args) -> dict | None:
    geometry = {}
    if args.line:
        geometry["line"] = {"points": _floats(args.line, 2, "--line")[:2], "invert": args.invert}
    if args.zone:
        geometry["zone"] = {"points": _floats(args.zone, 3, "--zone")}
    return {"default": geometry} if geometry else None


def counting_stage(session):
    for _, kind, step in session.steps:
        if kind == "stage" and hasattr(step, "default_geometry") and hasattr(step, "camera_geometry"):
            return step
    raise SystemExit("the pipeline has no face counting stage (select the face_tracker model)")


def percentile(values, q):
    return round(float(np.percentile(values, q)), 2) if values else None


async def evaluate(args, videos: list[Path]) -> list[dict]:
    from inferops import Engine
    from inferops.adapters.outbound.projects.loader import ProjectPluginResolver

    options = {"triton_url": args.triton_url} if args.runtime == "triton" else {}
    engine = Engine.from_project(args.project, runtime=args.runtime, **options)
    annotator = None
    if args.output:
        resolver = ProjectPluginResolver(Path(args.project).resolve(), "eval-face-counting")
        annotator = resolver.load_object(".plugins.frame_annotator:FaceCountAnnotator")()
    override = counting_override(args)
    reports = []
    async with engine:
        for video in videos:
            session = await engine.open_session(args.pipeline, stream_name=f"eval-{video.stem}")
            stage = counting_stage(session)
            if override is not None:
                from importlib import import_module
                parse = import_module(type(stage).__module__.rsplit(".", 1)[0] + ".counting").parse_counting
                stage.default_geometry, stage.camera_geometry = parse(override)
            target = None
            if args.output:
                output = Path(args.output)
                target = output if output.suffix == ".mp4" and len(videos) == 1 else output / f"{video.stem}.counted.mp4"
                target.parent.mkdir(parents=True, exist_ok=True)
            writer = None
            track_ids, relinked, embedded, frame_ms, stage_ms = set(), 0, 0, [], []
            last: dict = {}
            frames = 0
            started = time.perf_counter()
            try:
                async for frame in engine.application.source_factory(video, every_n=args.every_n):
                    t0 = time.perf_counter()
                    items = await session.process(frame)
                    frame_ms.append((time.perf_counter() - t0) * 1000)
                    for item in items:
                        payload = item.payload
                        for track in payload.get("face_tracks") or []:
                            track_ids.add(track["track_id"])
                            relinked += bool(track.get("relinked"))
                        for result in item.results:
                            embedded += int((result.raw or {}).get("embedded", 0))
                        timing = item.trace.get("stage_ms.face_tracker")
                        if timing:
                            stage_ms.append(float(timing))
                        last = {k: payload[k] for k in ("unique_faces", "faces_in_frame", "line_in", "line_out",
                                                        "zone_entries", "zone_occupancy") if k in payload}
                        image = payload.get("image_np")
                        if target is not None and image is not None:
                            if writer is None:
                                h, w = image.shape[:2]
                                fps = max(1.0, 30.0 / args.every_n)
                                writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                            writer.write(annotator(image.copy(), item))
                    frames += 1
                    if args.max_frames and frames >= args.max_frames:
                        break
            finally:
                await session.close()
                if writer is not None:
                    writer.release()
            elapsed = time.perf_counter() - started
            report = {
                "video": str(video), "frames": frames, **last,
                "track_ids_created": len(track_ids), "relinked_detections": relinked,
                "faces_embedded": embedded,
                "stage_ms": {"p50": percentile(stage_ms, 50), "p95": percentile(stage_ms, 95)},
                "frame_ms": {"p50": percentile(frame_ms, 50), "p95": percentile(frame_ms, 95),
                             "mean": round(statistics.fmean(frame_ms), 2) if frame_ms else None},
                "throughput_fps": round(frames / elapsed, 1) if elapsed else None,
                "annotated": str(target) if target else None,
            }
            checks = {}
            if args.expected is not None:
                checks["unique_faces"] = {"expected": args.expected, "got": last.get("unique_faces", 0)}
            if args.expected_in is not None:
                checks["line_in"] = {"expected": args.expected_in, "got": last.get("line_in", 0)}
            if checks:
                report["checks"] = checks
                report["passed"] = all(abs(c["got"] - c["expected"]) <= args.tolerance for c in checks.values())
            reports.append(report)
            print(json.dumps(report), flush=True)
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", type=Path, nargs="+")
    parser.add_argument("--runtime", choices=["cpu", "cuda", "tensorrt", "triton"], default="cuda")
    parser.add_argument("--triton-url", default="localhost:8001")
    parser.add_argument("--project", type=Path, default=Path("projects/face_counting"))
    parser.add_argument("--pipeline", default="face_counting")
    parser.add_argument("--every-n", type=int, default=1, help="process every n-th frame")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--line", help="counting line x1,y1,x2,y2 (normalised)")
    parser.add_argument("--invert", action="store_true", help="swap IN and OUT for --line")
    parser.add_argument("--zone", help="zone polygon x1,y1,x2,y2,x3,y3,... (normalised)")
    parser.add_argument("--expected", type=int, help="ground-truth unique people")
    parser.add_argument("--expected-in", type=int, help="ground-truth line entries")
    parser.add_argument("--tolerance", type=int, default=0)
    parser.add_argument("--output", help="annotated .mp4 (one video) or a directory")
    args = parser.parse_args()
    missing = [str(v) for v in args.videos if not v.is_file()]
    if missing:
        raise SystemExit(f"videos not found: {missing}")
    reports = asyncio.run(evaluate(args, args.videos))
    if any(r.get("passed") is False for r in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
