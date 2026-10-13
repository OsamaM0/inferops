"""Validate real driver inference and the offline pipeline in an isolated runtime.

Install inferops[server,cpu|cuda|tensorrt] and aiosqlite first. Start a local
moto_server and pass its URL via --s3-endpoint; no production services are used.
The JSON report includes actual profiled node providers, not just provider names.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=["cpu", "cuda", "tensorrt"], required=True)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1] / "projects/driver_behaviour")
    parser.add_argument("--video", type=Path, default=Path(__file__).resolve().parents[1] / "research/driver_training/test.mp4")
    parser.add_argument("--s3-endpoint", required=True, help="Local disposable S3 test server")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def configure(args):
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    env = {
        "INFEROPS_DOMAIN_ROOT": str(args.project.resolve()),
        "INFEROPS_PROJECT_ROOT": "",
        "INFEROPS_ACTIVE_DOMAIN": "driver_behaviour",
        "INFEROPS_CONFIGS_DIR": str(args.project.resolve().parents[1] / "configs"),
        "INFEROPS_EXECUTION_MODE": args.runtime,
        "INFEROPS_MODEL_REPOSITORY": "model_repository",
        "INFEROPS_FORCE_CPU": "1" if args.runtime == "cpu" else "0",
        "INFEROPS_TRANSPORT_KIND": "inproc",
        "INFEROPS_PG_URL": f"sqlite+aiosqlite:///{args.output / 'validation.db'}",
        "INFEROPS_S3_ENDPOINT": args.s3_endpoint,
        "INFEROPS_S3_PUBLIC_URL": args.s3_endpoint,
        "INFEROPS_S3_BUCKET": f"inferops-validation-{args.runtime}",
        "INFEROPS_S3_ACCESS_KEY": "testing",
        "INFEROPS_S3_SECRET_KEY": "testing",
        "INFEROPS_OBS_LOG_LEVEL": "WARNING",
        "INFEROPS_OBS_OTEL_ENDPOINT": "",
        "INFEROPS_PREWARM_MODELS": "",
        "INFEROPS_PREWARM_PIPELINES": "",
        "INFEROPS_TRT_CACHE_DIR": str(args.output / "trt-cache"),
        "AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing",
    }
    os.environ.update(env)


def enable_profiles(output):
    import onnxruntime as ort

    original = ort.SessionOptions
    count = 0

    def options():
        nonlocal count
        count += 1
        settings = original()
        settings.enable_profiling = True
        settings.profile_file_prefix = str(output / f"ort-{count}")
        return settings

    ort.SessionOptions = options


async def validate(args, report):
    import cv2
    import numpy as np
    import onnxruntime as ort
    from sqlalchemy import func, select

    from inferops.adapters.postgres_dao import alerts_t, events_t, get_engine
    from inferops.api.main import create_app
    from inferops.core.plugins import instantiate
    from inferops.core.types import Frame, StageItem, now_ms
    from inferops.pipeline.store import get_pipeline_store
    from inferops.registry import get_registry

    expected = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider",
                "tensorrt": "TensorrtExecutionProvider"}[args.runtime]
    report["onnxruntime"] = ort.__version__
    video = cv2.VideoCapture(str(args.video.resolve()))
    expected_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    report["input_fps"] = video.get(cv2.CAP_PROP_FPS)
    video.release()
    assert expected_frames > 0, "sample video could not be opened"
    report["available_providers"] = ort.get_available_providers()
    report["packages"] = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
                          if dist.metadata["Name"].lower().startswith(("onnxruntime", "tensorrt", "nvidia-", "inferops"))}
    app = create_app()
    async with app.router.lifespan_context(app):
        registry = get_registry()
        started = time.perf_counter()
        for name in ("rfdetr_driver", "face_mesh"):
            print(f"{args.runtime}: loading {name}", flush=True)
            backend = await registry.get_backend(name)
            providers = backend._sess.get_providers()
            assert expected in providers, f"{name}: expected {expected}, got {providers}"
        report["load_seconds"] = time.perf_counter() - started
        runner = await get_pipeline_store().get_runner("driver_behaviour")
        print(f"{args.runtime}: running all {expected_frames} video frames", flush=True)
        started = time.perf_counter()
        outputs = await runner.run_one(StageItem(stream_id=f"validation-{args.runtime}", ts_ms=now_ms(),
                                                 payload={"video_path": str(args.video.resolve())}))
        elapsed = time.perf_counter() - started
        report.update(
            frames=len(outputs), pipeline_seconds=elapsed, pipeline_fps=len(outputs) / elapsed,
            detections=sum(len(result.detections or []) for item in outputs for result in item.results),
            landmarks_frames=sum(len(item.payload.get("face_landmarks") or []) >= 468 for item in outputs),
            face_sources=dict(Counter(item.payload.get("face_source", "none") for item in outputs)),
            fusion_events=sum(len(item.payload.get("fusion_events", [])) for item in outputs),
            alerts=sum(len(item.payload.get("alerts", [])) for item in outputs),
            alert_types=sorted({alert["type"] for item in outputs for alert in item.payload.get("alerts", [])}),
        )
        assert len(outputs) == expected_frames, f"expected all {expected_frames} sample frames, got {len(outputs)}"
        assert report["detections"] > 0
        assert report["landmarks_frames"] > 0, "face mesh produced no valid landmark frames"
        assert report["fusion_events"] > 0 and report["alerts"] > 0
        report["models"] = {}
        for name, backend in registry.loaded_backends().items():
            profile = Path(backend._sess.end_profiling())
            nodes = [event for event in json.loads(profile.read_text()) if event.get("cat") == "Node"]
            provider_counts = Counter(event.get("args", {}).get("provider") for event in nodes)
            provider_counts.pop(None, None)
            report["models"][name] = {
                "device": backend.device, "device_info": backend.device_info,
                "executed_node_events": dict(provider_counts), "profile": str(profile),
            }
            assert provider_counts[expected] > 0, f"{name}: no profiled execution on {expected}"
        # Compare the detector's raw outputs on one identical real frame across runtimes.
        first = outputs[0]
        spec = registry.get_spec("rfdetr_driver")
        pre = instantiate(spec.preprocess.plugin, spec.preprocess.params)
        image = first.payload["image_np"]
        feeds = await pre(Frame(stream_id="comparison", ts_ms=0, image=image,
                                width=image.shape[1], height=image.shape[0]), spec)
        backend = await registry.get_backend("rfdetr_driver")
        await backend.infer(feeds)
        timings = []
        for _ in range(10):
            started = time.perf_counter()
            raw = await backend.infer(feeds)
            timings.append((time.perf_counter() - started) * 1000)
        report["detector_warm_latency_ms_median"] = float(np.median(timings))
        np.savez_compressed(args.output / "detector_outputs.npz", **raw)
        async with get_engine().connect() as connection:
            report["persisted_events"] = await connection.scalar(select(func.count()).select_from(events_t))
            report["persisted_alerts"] = await connection.scalar(select(func.count()).select_from(alerts_t))
        assert report["persisted_events"] > 0 and report["persisted_alerts"] > 0
        import boto3
        s3 = boto3.client("s3", endpoint_url=args.s3_endpoint, region_name="us-east-1")
        objects = s3.list_objects_v2(Bucket=f"inferops-validation-{args.runtime}").get("Contents", [])
        report["evidence_objects"] = len(objects)
        report["evidence_keys"] = [obj["Key"] for obj in objects]
        assert report["evidence_objects"] > 0, "no evidence objects written"
    await get_engine().dispose()


def main():
    args = arguments()
    configure(args)
    enable_profiles(args.output)
    report = {"runtime": args.runtime, "project": str(args.project.resolve()), "video": str(args.video.resolve()),
              "storage": "local Moto S3 + disposable SQLite; in-process transport",
              "status": "failed"}
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True, check=False)
    report["gpu"] = gpu.stdout.strip()
    try:
        asyncio.run(validate(args, report))
        report["status"] = "passed"
    except Exception as exc:
        import traceback
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"])
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: value for key, value in report.items()
                          if key not in {"traceback", "packages", "models", "evidence_keys"}}, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
