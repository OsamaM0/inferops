"""GPU serving ceiling: both project models, without decode/Haar/evidence.

Results here are explicitly not full camera-pipeline capacity. Independent
streams send distinct precomputed image tensors, sequentially through both
models. --fps 0 supplies requests as quickly as the backend can handle them.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from benchmark_capacity import comparison_models, gpu_snapshot, sha256

from inferops import Engine


async def measure(args):
    cv2.setNumThreads(1)
    root = Path(args.output) / time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    root.mkdir(parents=True, exist_ok=True)
    report = {"scope": "model_inference_only", "runtime": args.runtime, "completed": False,
              "cameras": args.counts, "requested_fps": args.fps or None,
              "model_concurrency": args.model_concurrency, "cases": [],
              "tf32_override": os.getenv("NVIDIA_TF32_OVERRIDE"),
              "limitations": ["Not full camera capacity: excludes per-frame decode, Haar, tracking, fusion, evidence and previews",
                              "Resolution is source image size; both model tensor sizes remain fixed"]}
    (root / "report.json").write_text(json.dumps(report, indent=2))
    engine = Engine.from_project(args.project, runtime=args.runtime,
                                 model_concurrency=args.model_concurrency,
                                 triton_url=args.triton_url, triton_concurrency=args.triton_concurrency)
    comparison_models(engine, args.model_repository)
    await engine.start()
    try:
        logical = ["rfdetr_driver", "face_mesh"]
        backends = [await engine.application.models.get_backend(n) for n in logical]
        if args.runtime == "cuda" and any(b.device != "gpu" for b in backends):
            raise RuntimeError("CUDA provider required; no CPU model fallback")
        capture = cv2.VideoCapture(args.video)
        ok, image = capture.read()
        capture.release()
        if not ok:
            raise ValueError("Cannot read representative source image")
        report.update(hardware=gpu_snapshot(), source_shape=list(image.shape[:2]),
                      model_artifacts={n: sha256(b.spec.artifact.path) for n, b in zip(logical, backends, strict=True)},
                      providers={n: b.device_info for n, b in zip(logical, backends, strict=True)})
        feeds = []
        for backend in backends:
            spec = backend.spec.inputs[0]
            height, width = spec.shape[-2:]
            rgb = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (width, height))
            array = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255
            feeds.append([{spec.name: np.ascontiguousarray(array * (.7 + i / max(1, args.counts) * .3))}
                          for i in range(args.counts)])
        async def statistics():
            return {name: await backend.statistics() for name, backend in zip(logical, backends, strict=True)} if args.runtime == "triton" else {}
        before = await statistics()
        start = time.monotonic()
        deadline = start + args.warmup + args.seconds
        counts, latencies, errors = [0] * args.counts, [], []

        async def stream(index):
            tick = start
            try:
                while time.monotonic() < deadline:
                    if args.fps:
                        await asyncio.sleep(max(0, tick - time.monotonic()))
                    began = time.monotonic()
                    for model_index, backend in enumerate(backends):
                        result = await backend.infer(feeds[model_index][index])
                        if any(value.shape[0] != 1 or not np.isfinite(value).all() for value in result.values()):
                            raise ValueError("Invalid/nonfinite response or request separation")
                    now = time.monotonic()
                    if start + args.warmup <= now <= deadline:
                        counts[index] += 1
                        latencies.append((now - began) * 1000)
                    tick = max(tick + 1 / args.fps, now) if args.fps else now
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")

        await asyncio.gather(*(stream(i) for i in range(args.counts)))
        rates = [round(c / args.seconds, 3) for c in counts]
        report["cases"].append({"cameras": args.counts, "seconds": args.seconds,
                                 "warmup_seconds": args.warmup, "per_camera_fps": rates,
                                 "total_fps": round(sum(rates), 3), "errors": errors,
                                 "p95_ms": round(float(np.percentile(latencies, 95)), 1) if latencies else None,
                                 "p99_ms": round(float(np.percentile(latencies, 99)), 1) if latencies else None,
                                 "triton_statistics_before": before, "triton_statistics_after": await statistics(),
                                 "passed": not errors and bool(rates) and min(rates) > 0
                                           and (not args.fps or min(rates) >= args.fps * .95)})
        report["completed"] = True
    finally:
        (root / "report.json").write_text(json.dumps(report, indent=2))
        await engine.close()
    print(f"REPORT={root / 'report.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=["cuda", "triton"], required=True)
    parser.add_argument("--project", default="projects/driver_behaviour")
    parser.add_argument("--model-repository", required=True)
    parser.add_argument("--video", default=".run/capacity/fixtures/driver-1080p-15.mp4")
    parser.add_argument("--counts", type=int, default=16)
    parser.add_argument("--fps", type=float, default=0)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--warmup", type=float, default=5)
    parser.add_argument("--model-concurrency", type=int, default=1)
    parser.add_argument("--triton-concurrency", type=int, default=64)
    parser.add_argument("--triton-url", default="127.0.0.1:8001")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.counts < 1 or args.fps < 0 or args.seconds < 5:
        parser.error("Invalid stream count/FPS/duration")
    asyncio.run(measure(args))
