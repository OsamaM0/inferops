"""Configurable real RTSP acquisition soak, with controlled publisher recovery.

This capture-only mode measures acquisition; benchmark_capacity.py measures the
full pipeline and benchmark_model_load.py measures serving. Never add their FPS.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inferops.adapters.outbound.media.camera import capture_frames
from inferops.adapters.outbound.observability.hardware import hardware_snapshot
from inferops.domain.sources import SourceSpec


async def run(args):
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    root = Path(args.output) / f"{stamp}-{os.getpid()}"
    root.mkdir(parents=True)
    config = root / "mediamtx.yml"
    config.write_text(f"logLevel: warn\nrtspAddress: 127.0.0.1:{args.port}\nrtspTransports: [tcp]\n"
                      "rtmp: no\nhls: no\nwebrtc: no\nsrt: no\nmoq: no\npaths:\n  all_others:\n")
    processes, logs, tasks, stats = [], [], [], []
    publishers = []

    def launch(command, log_name):
        log = (root / log_name).open("ab")
        logs.append(log)
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        processes.append(process)
        return process

    async def terminate(process):
        if process.poll() is None:
            process.terminate()
            try:
                await asyncio.wait_for(asyncio.to_thread(process.wait), 8)
            except TimeoutError:
                process.kill()
                await asyncio.to_thread(process.wait)

    def publisher(index):
        return launch([args.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "warning", "-re",
                       "-stream_loop", "-1", "-i", str(args.video.resolve()), "-map", "0:v:0", "-an",
                       "-c:v", "copy", "-f", "rtsp", "-rtsp_transport", "tcp",
                       f"rtsp://127.0.0.1:{args.port}/camera-{index:04d}"], f"publisher-{index}.log")

    manifest = {"mode": "capture_only", "cameras": args.cameras, "seconds": args.seconds,
                "hardware": hardware_snapshot(root),
                "reader_sha256": hashlib.sha256(Path(sys.modules[capture_frames.__module__].__file__).read_bytes()).hexdigest(),
                "sample_fps": args.fps, "width": args.width, "height": args.height,
                "video": str(args.video.resolve()), "video_sha256": hashlib.sha256(args.video.read_bytes()).hexdigest(),
                "restart_camera": 0 if args.outage_seconds else None,
                "outage_seconds": args.outage_seconds, "recovery_limit_seconds": args.recovery_limit,
                "python": sys.version, "started_at": stamp, "mediamtx": args.mediamtx,
                "ffmpeg": subprocess.check_output([args.ffmpeg, "-version"], text=True).splitlines()[0],
                "video_metadata": json.loads(subprocess.check_output(["ffprobe", "-v", "quiet", "-show_streams",
                    "-show_format", "-of", "json", str(args.video)], text=True)),
                "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    interruption = {}
    measurement_started = asyncio.Event()
    began = time.monotonic()

    async def consume(index):
        row = {"camera_id": f"load-{index}", "decoded": 0, "dropped": 0, "samples": 0,
               "first_frame_seconds": None, "last_frame_seconds": None, "max_capture_age_ms": 0}
        stats.append(row)
        spec = SourceSpec(camera_id=row["camera_id"], name="load", kind="rtsp", pipeline="capture-only",
                          uri=f"rtsp://127.0.0.1:{args.port}/camera-{index:04d}", timeout_s=2,
                          inference_fps=args.fps, max_width=args.width, max_height=args.height)
        frames = capture_frames(spec, row)
        try:
            async for frame in frames:
                now = time.monotonic()
                row["samples"] += 1
                row["first_frame_seconds"] = row["first_frame_seconds"] or now - began
                row["last_frame_seconds"] = now - began
                age_ms = (time.monotonic_ns() - frame.payload["_captured_monotonic_ns"]) / 1e6
                row["max_capture_age_ms"] = max(row["max_capture_age_ms"], age_ms)
                if (index == 0 and interruption.get("reachable_at") and "recovered_seconds" not in interruption
                        and frame.payload["connection_generation"] > interruption["old_generation"]):
                    interruption["recovered_seconds"] = now - interruption["reachable_at"]
        finally:
            await frames.aclose()

    async def fault():
        if args.outage_seconds:
            await measurement_started.wait()
            await asyncio.sleep(args.seconds / 3)
            interruption["old_generation"] = stats[0].get("reconnects", 0) + 1
            await terminate(publishers[0])
            await asyncio.sleep(args.outage_seconds)
            publishers[0] = publisher(0)
            interruption["reachable_at"] = time.monotonic()

    resource_samples = []
    errors = []
    try:
        server = launch([args.mediamtx, str(config.resolve())], "mediamtx.log")
        await asyncio.sleep(1)
        if server.poll() is not None:
            raise RuntimeError("MediaMTX failed; inspect mediamtx.log")
        for index in range(args.cameras):
            publishers.append(publisher(index))
            tasks.append(asyncio.create_task(consume(index)))
            await asyncio.sleep(args.stagger_ms / 1000)
        tasks.append(asyncio.create_task(fault()))
        warmup = time.monotonic() + args.warmup_seconds
        while time.monotonic() < warmup:
            if len(stats) == args.cameras and all(row["samples"] > 0 for row in stats):
                break
            await asyncio.sleep(.5)
        if len(stats) != args.cameras or not all(row["samples"] for row in stats):
            raise RuntimeError("some cameras failed startup before the warmup deadline")
        before = [row["samples"] for row in stats]
        measurement = time.monotonic()
        measurement_started.set()
        while time.monotonic() - measurement < args.seconds:
            await asyncio.sleep(min(5, args.seconds))
            status = Path("/proc/self/status").read_text().splitlines()
            rss = next(line.split()[1] for line in status if line.startswith("VmRSS:"))
            threads = next(line.split()[1] for line in status if line.startswith("Threads:"))
            resource_samples.append({"seconds": time.monotonic() - measurement,
                                     "rss_kib": int(rss), "threads": int(threads)})
            (root / "progress.json").write_text(json.dumps({"sources": stats, "resources": resource_samples[-1]}))
        duration = time.monotonic() - measurement
        for row, count in zip(stats, before, strict=True):
            row["sample_fps"] = (row["samples"] - count) / duration
    except BaseException as error:
        errors.append(type(error).__name__)
        raise
    finally:
        closing = time.monotonic()
        for task in tasks:
            task.cancel()
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        errors.extend(type(error).__name__ for error in outcomes
                      if isinstance(error, BaseException) and not isinstance(error, asyncio.CancelledError))
        close_seconds = time.monotonic() - closing
        await asyncio.gather(*(terminate(process) for process in processes))
        for log in logs:
            log.close()
        # The intentionally disconnected camera has a separate recovery gate.
        healthy = [row for row in stats if not args.outage_seconds or row["camera_id"] != "load-0"]
        missing = [row["camera_id"] for row in stats if row["samples"] == 0]
        slow = [row["camera_id"] for row in healthy if row.get("sample_fps", 0) < args.fps * .95]
        recovered = not args.outage_seconds or interruption.get("recovered_seconds", float("inf")) <= args.recovery_limit
        plateau = (len(resource_samples) < 4 or resource_samples[-1]["rss_kib"] <=
                   max(row["rss_kib"] for row in resource_samples[:len(resource_samples)//2]) * 1.10)
        passed = not errors and not missing and not slow and recovered and plateau and close_seconds <= 10
        report = {"manifest": manifest, "passed": passed, "cameras": stats, "resources": resource_samples,
                  "missing": missing, "slow": slow, "errors": errors, "recovery": interruption,
                  "rss_plateau": plateau, "close_seconds": close_seconds,
                  "reader_threads_remaining": [thread.name for thread in __import__('threading').enumerate()
                                               if thread.name.startswith("rtsp-load-")]}
        report["passed"] &= not report["reader_threads_remaining"]
        (root / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({"report": str(root / "report.json"), "passed": report["passed"]}), flush=True)
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--cameras", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=3600)
    parser.add_argument("--warmup-seconds", type=float, default=30)
    parser.add_argument("--fps", type=float, default=5)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--outage-seconds", type=float, default=5)
    parser.add_argument("--recovery-limit", type=float, default=15)
    parser.add_argument("--stagger-ms", type=float, default=100)
    parser.add_argument("--port", type=int, default=18554)
    parser.add_argument("--output", default=".run/load")
    parser.add_argument("--mediamtx", default=shutil.which("mediamtx") or ".run/tools/mediamtx/mediamtx")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    if args.cameras < 1 or args.seconds < 1 or args.fps <= 0 or args.outage_seconds < 0:
        parser.error("positive count/duration/rate and a nonnegative outage are required")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
