"""Measure the actual shared pipeline on paced, independent video streams.

Example: .venv/bin/python scripts/benchmark_capacity.py --runtime cuda --counts 1,2,3,4
No CPU fallback is accepted in CUDA mode. Reports are bounded-duration local-file
tests, not RTSP/network or long-soak certification. Use representative footage.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from collections import Counter
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from inferops import Engine
from inferops.adapters.inbound.registered_sources import registered_frames
from inferops.adapters.inbound.sources import process_live_frames
from inferops.adapters.outbound.media.previews import render_preview
from inferops.adapters.outbound.observability.capacity_profiles import (
    service_profile,
    software_fingerprint,
    workload_fingerprint,
)
from inferops.adapters.outbound.observability.hardware import filesystem_type
from inferops.adapters.outbound.persistence.source_catalog import MemorySourceStore, PostgresSourceStore
from inferops.adapters.outbound.persistence.sql import SqlRepository
from inferops.adapters.outbound.storage.filesystem import FilesystemStorage
from inferops.application.incidents import wire_result
from inferops.domain.evidence import EvidenceConfig
from inferops.domain.sources import SourceSpec


def gpu_snapshot():
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,memory.used,memory.free,utilization.gpu,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=4,
        )
        rows = []
        for line in raw.strip().splitlines():
            name, uuid, total, used, free, utilization, driver = [s.strip() for s in line.split(",")]
            rows.append(
                dict(
                    name=name,
                    uuid=uuid,
                    total_mib=float(total),
                    used_mib=float(used),
                    free_mib=float(free),
                    utilization_percent=float(utilization),
                    driver=driver,
                )
            )
        return {"available": True, "gpus": rows}
    except Exception:
        return {"available": False, "gpus": []}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def comparison_models(engine, repository):
    """Point this benchmark at the exact serving graphs without editing a project."""
    if not repository:
        return
    from dataclasses import replace
    from types import MappingProxyType

    definition = engine.application.definition
    specs = {}
    for name, source in definition.models.items():
        spec = source.model_copy(deep=True)
        model_name = spec.artifact.triton_model_name
        if not model_name and spec.artifact.path:
            model_name = Path(spec.artifact.path).parent.parent.name
        candidate = Path(repository).resolve() / str(model_name) / "1/model.onnx"
        if spec.artifact.path:
            if not candidate.is_file():
                raise ValueError(f"Explicit comparison repository is missing {candidate}")
            spec.artifact.path = str(candidate)
            spec.artifact.sha256 = sha256(candidate)
            spec.artifact.source = None
        specs[name] = spec
    updated = replace(definition, models=MappingProxyType(specs))
    engine.application.definition = updated
    engine.application.models.definition = updated


async def measure(args):
    import cv2

    cv2.setNumThreads(args.opencv_threads)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    root = Path(args.output) / stamp
    root.mkdir(parents=True, exist_ok=True)
    catalog = (PostgresSourceStore(args.database_url, preview_root=root / "previews")
               if args.repository == "postgresql" else MemorySourceStore())
    await catalog.start()
    engine = Engine.from_project(
        args.project,
        runtime=args.runtime,
        repository=SqlRepository(args.database_url)
        if args.repository == "postgresql"
        else None,
        evidence=args.evidence,
        storage=FilesystemStorage(root / "evidence") if args.evidence else None,
        spool_path=root / "spool" if args.evidence else None,
        evidence_config=EvidenceConfig() if args.evidence else None,
        triton_url=args.triton_url,
        triton_concurrency=args.triton_concurrency,
        model_concurrency=args.model_concurrency,
    )
    comparison_models(engine, args.model_repository)
    await engine.start()
    definition = engine.application.definition
    models = {}
    for name, spec in definition.models.items():
        artifact = Path(spec.artifact.path or "")
        if artifact.is_file():
            models[name] = {"sha256": sha256(artifact), "bytes": artifact.stat().st_size}
    required = {
        s.params["model"] for s in definition.pipelines[args.pipeline].stages if s.params.get("model")
    }
    for name in sorted(required):
        backend = await engine.application.models.get_backend(name)
        if args.runtime == "cuda" and backend.device != "gpu":
            raise RuntimeError(f"{name} did not activate CUDA; refusing a misleading GPU benchmark")
    providers = {
        name: backend.device_info for name, backend in engine.application.models.loaded_backends().items()
    }
    model_calls = Counter()
    for name, backend in engine.application.models.loaded_backends().items():
        original = backend._backend.infer

        async def counted(inputs, original=original, name=name):
            model_calls[name] += 1
            return await original(inputs)

        backend._backend.infer = counted
    report = dict(
        schema_version=1,
        created_at=time.time(),
        hardware={
            "hostname": platform.node(),
            "cpu_count": os.cpu_count(),
            "cpu_model": next(
                (
                    line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                platform.processor(),
            ),
            "ram_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
            "data_filesystem": filesystem_type(root),
            **gpu_snapshot(),
        },
        data_path=str(root.resolve()),
        runtime=args.runtime,
        pipeline=args.pipeline,
        opencv_threads=args.opencv_threads,
        project=str(Path(args.project).resolve()),
        model_artifacts=models,
        workload_fingerprint=workload_fingerprint(engine.application),
        software_fingerprint=software_fingerprint(),
        service_profile=service_profile(engine.application),
        providers=providers,
        triton_url=args.triton_url if args.runtime == "triton" else None,
        model_repository=args.model_repository,
        model_concurrency=args.model_concurrency,
        cudnn_max_workspace=int(os.getenv("INFEROPS_CUDNN_MAX_WORKSPACE", "0")),
        tf32_override=os.getenv("NVIDIA_TF32_OVERRIDE"),
        evidence_enabled=args.evidence,
        evidence_policy=asdict(engine.application.evidence.config) if args.evidence else None,
        preview_enabled=args.preview,
        evidence_fps=args.evidence_fps,
        fixture={
            "path": str(Path(args.video).resolve()),
            "sha256": sha256(args.video),
            "provenance": args.provenance,
        },
        source_kind="paced_local_video",
        completed=False,
        limitations=[
            "Repeated local footage; not independent real RTSP cameras",
            "Short sweep, not a stability/incident burst qualification",
            "Image quality/accuracy was not measured; model tensor sizes stay fixed",
        ],
        cases=[],
    )

    async def telemetry(stop, readings):
        while not stop.is_set():
            snap = await asyncio.to_thread(gpu_snapshot)
            readings.append(snap)
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), 1)

    async def triton_statistics():
        if args.runtime != "triton":
            return {}
        return {name: await backend.statistics() for name, backend in engine.application.models.loaded_backends().items()}

    try:
        for size in args.resolutions.split(","):
            width, height = map(int, size.split("x"))
            for fps in map(float, args.fps.split(",")):
                for count in map(int, args.counts.split(",")):
                    sessions, tasks, samples, counts, errors, states = [], [], [], [0] * count, [], []
                    stop = asyncio.Event()
                    gpu = []
                    sampler = asyncio.create_task(telemetry(stop, gpu))
                    start = time.monotonic()
                    cpu_start = time.process_time()
                    calls_start = model_calls.copy()
                    triton_before = await triton_statistics()
                    ready = asyncio.Event()
                    clock = {}
                    for i in range(count):
                        sid = f"bench-{count}-{i}"
                        spec = SourceSpec(
                            camera_id=sid,
                            name=sid,
                            kind="video",
                            uri=args.video,
                            pipeline=args.pipeline,
                            inference_fps=fps,
                            evidence_fps=args.evidence_fps,
                            max_width=width,
                            max_height=height,
                            loop=True,
                        )
                        session = await engine.open_session(args.pipeline, stream_name=sid)
                        sessions.append(session)
                        state = {"decoded": 0, "dropped": 0, "processing_dropped": 0}
                        states.append(state)

                        async def run(
                            index=i,
                            session=session,
                            spec=spec,
                            state=state,
                            counts=counts,
                            samples=samples,
                            errors=errors,
                            ready=ready,
                            clock=clock,
                        ):
                            await ready.wait()
                            start, deadline = clock["start"], clock["deadline"]
                            preview_at = 0

                            def on_drop():
                                state["processing_dropped"] += 1

                            async def on_result(outputs, latency):
                                nonlocal preview_at
                                now = time.monotonic()
                                if now >= start + args.warmup and now <= deadline:
                                    counts[index] += 1
                                    samples.append(latency * 1000)
                                if outputs:
                                    processed_image = outputs[-1].payload.get("image_np")
                                    if processed_image is not None:
                                        state["processing_height"], state["processing_width"] = processed_image.shape[:2]
                                    await catalog.save_result(
                                        spec.camera_id,
                                        session.id,
                                        {
                                            "camera_id": spec.camera_id,
                                            "session_id": session.id,
                                            "results": [wire_result(item) for item in outputs],
                                        },
                                    )
                                    if args.preview and now >= preview_at:
                                        jpeg = await asyncio.to_thread(
                                            render_preview,
                                            outputs[-1].payload["image_np"],
                                            outputs[-1],
                                            spec,
                                            session.annotator,
                                        )
                                        await catalog.save_preview(spec.camera_id, jpeg)
                                        preview_at = now + 0.5

                            try:
                                await process_live_frames(
                                    session, registered_frames(spec, state), on_result=on_result, on_drop=on_drop,
                                    inference_fps=spec.inference_fps,
                                )
                            except asyncio.CancelledError:
                                raise
                            except Exception as exc:
                                errors.append(type(exc).__name__ + ": " + str(exc)[:200])

                        tasks.append(asyncio.create_task(run()))
                    # Open every session before starting the shared timed window.
                    start = time.monotonic()
                    deadline = start + args.warmup + args.seconds
                    clock.update(start=start, deadline=deadline)
                    ready.set()
                    await asyncio.sleep(max(0, deadline - time.monotonic()))
                    report["in_progress_case"] = {
                        "cameras": count, "width": width, "height": height, "inference_fps": fps,
                        "per_camera_fps": [round(n / args.seconds, 3) for n in counts],
                        "total_fps": round(sum(counts) / args.seconds, 3),
                        "p95_ms": round(float(np.percentile(samples, 95)), 1) if samples else None,
                        "errors": errors.copy(), "passed": False,
                        "valid": False, "reason": "Measurement window finished; teardown not yet verified",
                    }
                    (root / "report.json").write_text(json.dumps(report, indent=2))
                    for task in tasks:
                        task.cancel()
                    _done, pending = await asyncio.wait(tasks, timeout=15)
                    if pending:
                        for task in asyncio.all_tasks():
                            task.print_stack()
                        raise RuntimeError("stream cleanup exceeded 15 seconds; this case is invalid")
                    for session in sessions:
                        await session.close()
                    if engine.application.evidence:
                        await engine.application.evidence.wait_idle()
                    stop.set()
                    await sampler
                    rates = [round(n / args.seconds, 3) for n in counts]
                    all_gpu = [g for s in gpu for g in s.get("gpus", [])]
                    p95 = float(np.percentile(samples, 95)) if samples else None
                    case = dict(
                        width=width,
                        height=height,
                        inference_fps=fps,
                        evidence_fps=args.evidence_fps,
                        cameras=count,
                        seconds=args.seconds,
                        warmup_seconds=args.warmup,
                        per_camera_fps=rates,
                        total_fps=round(sum(rates), 3),
                        p95_ms=round(p95, 1) if p95 is not None else None,
                        p99_ms=round(float(np.percentile(samples, 99)), 1) if samples else None,
                        gpu_used_peak_mib=max((g["used_mib"] for g in all_gpu), default=None),
                        gpu_utilization_mean=float(np.mean([g["utilization_percent"] for g in all_gpu]))
                        if all_gpu
                        else None,
                        rss_peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                        decoded=sum(s["decoded"] for s in states),
                        actual_processing_shapes=sorted({(s.get("processing_width"), s.get("processing_height"))
                                                         for s in states if s.get("processing_width")}),
                        actual_source_shapes=sorted({(s.get("source_width"), s.get("source_height"))
                                                     for s in states if s.get("source_width")}),
                        cameras_with_results=sum(rate > 0 for rate in rates),
                        errors=errors,
                        cpu_core_seconds=time.process_time() - cpu_start,
                        model_calls=dict(model_calls - calls_start),
                        triton_statistics_before=triton_before,
                        triton_statistics_after=await triton_statistics(),
                        dropped_before_processing=[state["processing_dropped"] for state in states],
                        dropped_at_capture=[state["dropped"] for state in states],
                        passed=bool(
                            rates
                            and min(rates) >= fps * 0.95
                            and p95 is not None
                            and p95 <= 1000
                            and not errors
                        ),
                    )
                    report["cases"].append(case)
                    report.pop("in_progress_case", None)
                    (root / "report.json").write_text(json.dumps(report, indent=2))
                    print(json.dumps(case), flush=True)
                    if not case["passed"] and not args.keep_going:
                        break
    finally:
        await engine.close()
        await catalog.close()
    report["completed"] = True
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(f"REPORT={root / 'report.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="projects/driver_behaviour")
    parser.add_argument("--pipeline", default="driver_behaviour")
    parser.add_argument("--runtime", choices=["cpu", "cuda", "triton"], default="cuda")
    parser.add_argument("--triton-url", default="127.0.0.1:8001")
    parser.add_argument("--triton-concurrency", type=int, default=64)
    parser.add_argument("--model-concurrency", type=int, default=1)
    parser.add_argument("--model-repository", help="Use exact qualified ONNX bytes for both runtime comparisons")
    parser.add_argument("--opencv-threads", type=int, default=1)
    parser.add_argument("--repository", choices=["postgresql", "memory"], default="postgresql")
    parser.add_argument("--video", default="research/driver_training/test.mp4")
    parser.add_argument("--resolutions", default="1920x1080")
    parser.add_argument("--fps", default="1,5,10")
    parser.add_argument("--counts", default="1,2,3,4")
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--warmup", type=float, default=5)
    parser.add_argument("--evidence", action="store_true")
    parser.add_argument("--evidence-fps", type=float, help="Independent evidence observation rate; empty follows inference")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument(
        "--keep-going", action="store_true", help="Continue higher counts after a profile fails"
    )
    parser.add_argument("--provenance", default="Repository sample, unchanged")
    parser.add_argument("--output", default=".run/capacity")
    parser.add_argument("--database-url", help="Dedicated PostgreSQL benchmark database; never use a live server database")
    args = parser.parse_args()
    if args.repository == "postgresql" and not args.database_url:
        parser.error("--database-url is required for PostgreSQL benchmarks")
    if args.seconds < 5 or args.warmup < 0:
        parser.error("seconds must be >=5 and warmup nonnegative")
    if args.evidence_fps is not None and not 0 < args.evidence_fps <= 120:
        parser.error("evidence-fps must be in (0, 120]")
    asyncio.run(measure(args))


if __name__ == "__main__":
    main()
