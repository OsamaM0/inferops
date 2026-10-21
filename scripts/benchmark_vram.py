"""Isolated stream ramps with external VRAM/RAM telemetry and failure records.

No dummy VRAM reservation is used. Each sample runs in a child process so CUDA
OOM, allocation caching and decoder teardown cannot poison subsequent cases.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_capacity import gpu_snapshot


def available_ram_mib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024
    return None


def stop_child(process):
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_case(args, root, index, width, height, fps, count):
    folder = root / f"{index:03d}-{width}x{height}-{fps:g}fps-{count}streams"
    folder.mkdir()
    output = folder / "pipeline"
    command = [sys.executable, "scripts/benchmark_model_load.py" if args.models_only else "scripts/benchmark_capacity.py",
               "--runtime", args.runtime, "--project", args.project,
               "--model-repository", args.model_repository, "--video", args.video,
               "--counts", str(count), "--fps", str(fps), "--seconds", str(args.seconds),
               "--warmup", str(args.warmup), "--model-concurrency", str(args.model_concurrency),
               "--triton-concurrency", str(args.triton_concurrency), "--triton-url", args.triton_url,
               "--output", str(output)]
    if not args.models_only:
        command += ["--resolutions", f"{width}x{height}", "--evidence", "--preview", "--keep-going",
                    "--provenance", "Repeated scaled/pillarboxed driver sample, H264 1080p 15 FPS; source quality/accuracy unmeasured"]
    record = {"scope": "model_inference_only" if args.models_only else "full_pipeline",
              "runtime": args.runtime, "profile": args.profile, "width_limit": width,
              "height_limit": height, "requested_fps": fps or None, "requested_streams": count,
              "model_concurrency": args.model_concurrency, "workspace": args.workspace,
              "tf32_override": os.getenv("NVIDIA_TF32_OVERRIDE"),
              "command": command, "readings": [], "status": "running"}
    start = time.monotonic()
    process = None
    try:
        with (folder / "worker.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, "INFEROPS_CUDNN_MAX_WORKSPACE": str(args.workspace)})
            while process.poll() is None:
                snapshot = gpu_snapshot()
                reading = {"elapsed_s": round(time.monotonic() - start, 2),
                           "ram_available_mib": available_ram_mib(), **snapshot}
                record["readings"].append(reading)
                (folder / "telemetry.json").write_text(json.dumps(record, indent=2))
                if reading["ram_available_mib"] is not None and reading["ram_available_mib"] < args.ram_reserve_mib:
                    record["status"] = "ram_reserve_guard"
                    break
                if any(g["free_mib"] < args.gpu_reserve_mib for g in snapshot.get("gpus", [])):
                    record["status"] = "gpu_reserve_guard"
                    break
                if time.monotonic() - start > args.timeout:
                    record["status"] = "timeout"
                    break
                time.sleep(.5)
            stop_child(process)
            record["exit_code"] = process.returncode
    finally:
        if process is not None:
            stop_child(process)
    artifacts = list(output.glob("*/report.json"))
    if artifacts:
        record["worker_report"] = json.loads(artifacts[-1].read_text())
        record["result"] = (record["worker_report"]["cases"][-1] if record["worker_report"].get("cases")
                            else record["worker_report"].get("in_progress_case"))
    if record["status"] == "running":
        text = (folder / "worker.log").read_text(errors="replace")
        record["status"] = "completed" if record.get("exit_code") == 0 else (
            "gpu_oom" if any(term in text.lower() for term in ("out of memory", "cudaerrormemoryallocation", "failed to allocate memory"))
            else "failed")
        if record["status"] != "completed":
            record["error_tail"] = text[-2500:]
    errors = " ".join(record.get("result", {}).get("errors", [])) if record.get("result") else ""
    if record["status"] == "completed" and any(term in errors.lower() for term in
                                               ("out of memory", "cudaerrormemoryallocation", "failed to allocate memory")):
        record["status"] = "gpu_oom"
    devices = [g for r in record["readings"] for g in r.get("gpus", [])]
    record.update(elapsed_s=round(time.monotonic() - start, 2),
                  gpu_used_peak_mib=max((g["used_mib"] for g in devices), default=None),
                  gpu_free_min_mib=min((g["free_mib"] for g in devices), default=None),
                  gpu_utilization_peak=max((g["utilization_percent"] for g in devices), default=None),
                  ram_available_min_mib=min((r["ram_available_mib"] for r in record["readings"] if r["ram_available_mib"] is not None), default=None),
                  near_full_allocatable_vram=any(g["used_mib"] / max(1, g["used_mib"] + g["free_mib"]) >= .95 for g in devices))
    (folder / "case.json").write_text(json.dumps(record, indent=2))
    summary = {k: record[k] for k in ("scope", "runtime", "profile", "requested_streams", "requested_fps",
                                    "width_limit", "height_limit", "status", "gpu_used_peak_mib", "gpu_free_min_mib",
                                    "near_full_allocatable_vram")}
    summary["result"] = {k: record["result"].get(k) for k in ("per_camera_fps", "total_fps", "p95_ms", "passed", "errors")} if record.get("result") else None
    print(json.dumps(summary), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=["cuda", "triton"], required=True)
    parser.add_argument("--profile", default="default")
    parser.add_argument("--project", default="projects/driver_behaviour")
    parser.add_argument("--model-repository", default=".run/triton/checkpoint-models")
    parser.add_argument("--video", default=".run/capacity/fixtures/driver-1080p-15.mp4")
    parser.add_argument("--resolutions", default="1920x1080,1280x720,640x360")
    parser.add_argument("--fps", default="5")
    parser.add_argument("--counts", default="1,4,8,16,32")
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--warmup", type=float, default=5)
    parser.add_argument("--timeout", type=float, default=150)
    parser.add_argument("--model-concurrency", type=int, default=1)
    parser.add_argument("--workspace", type=int, choices=[0, 1], default=0)
    parser.add_argument("--triton-concurrency", type=int, default=64)
    parser.add_argument("--triton-url", default="127.0.0.1:8001")
    parser.add_argument("--ram-reserve-mib", type=int, default=1024)
    parser.add_argument("--gpu-reserve-mib", type=int, default=128)
    parser.add_argument("--models-only", action="store_true")
    parser.add_argument("--output", default=".run/vram")
    args = parser.parse_args()
    if not args.models_only and any(float(v) <= 0 for v in args.fps.split(",")):
        parser.error("Full pipeline requires positive FPS; zero means unpaced model-only load")
    root = Path(args.output) / (time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:6])
    root.mkdir(parents=True)
    report = {"settings": vars(args), "cases": [], "completed": False,
              "target": "95% of allocatable VRAM through real inference; retain 128 MiB free. No dummy reservations.",
              "limitations": ["Repeated local video, not independent RTSP cameras", "Full-pipeline and model-only results are separate"]}
    path = root / "report.json"
    try:
        for size in args.resolutions.split(","):
            width, height = map(int, size.split("x"))
            for fps in map(float, args.fps.split(",")):
                for count in map(int, args.counts.split(",")):
                    case = run_case(args, root, len(report["cases"]), width, height, fps, count)
                    report["cases"].append(case)
                    path.write_text(json.dumps(report, indent=2))
                    if case["status"] in {"ram_reserve_guard", "gpu_reserve_guard", "gpu_oom", "timeout"}:
                        break
        report["completed"] = True
    finally:
        path.write_text(json.dumps(report, indent=2))
    print(f"VRAM_REPORT={path}", flush=True)


if __name__ == "__main__":
    main()
