"""Verify a complete real video, actual execution providers and saved evidence."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import av
import onnxruntime as ort

from inferops import Engine
from inferops.adapters.outbound.persistence.sql import SqlRepository
from inferops.adapters.outbound.storage.filesystem import FilesystemStorage
from inferops.domain.evidence import EvidenceConfig


async def verify(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    profiles = output / "profiles"
    profiles.mkdir(exist_ok=True)
    original_options = ort.SessionOptions
    profile_sequence = 0

    def options():
        nonlocal profile_sequence
        profile_sequence += 1
        value = original_options()
        value.enable_profiling = True
        value.profile_file_prefix = str(profiles / f"model-{profile_sequence}")
        return value

    ort.SessionOptions = options
    repository = SqlRepository(f"sqlite+aiosqlite:///{output / 'records.sqlite'}")
    storage = FilesystemStorage(output / "artifacts")
    selected = Engine.from_project(
        args.project,
        runtime=args.runtime,
        repository=repository,
        storage=storage,
        evidence=True,
        spool_path=output / "spool",
        evidence_config=EvidenceConfig(disk_reserve_bytes=0, capture_timeout_seconds=1800),
        backend_options={"threads": 4},
    )
    with av.open(str(args.video)) as source:
        expected_frames = sum(1 for _ in source.decode(video=0))
    started = time.monotonic()
    provider_reports = {}
    async with selected:
        summary = await selected.run(args.pipeline, args.video, stream_name="real-video-verification")
        (output / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
        assert summary["frames"] == expected_frames, summary
        assert summary["detections_by_class"], "No real detections"
        assert len(selected.application.models.loaded_backends()) >= 2, "Both real models must execute"
        assert summary["incident_count"] > 0, "No incidents generated from sample video"
        assert summary["evidence"], "No evidence requests persisted"
        for name, wrapper in selected.application.models.loaded_backends().items():
            backend = wrapper._backend
            path = Path(backend._sess.end_profiling())
            trace = json.loads(path.read_text())
            counts = Counter(
                event.get("args", {}).get("provider")
                for event in trace
                if event.get("cat") == "Node" and event.get("args", {}).get("provider")
            )
            required = "CUDAExecutionProvider" if args.runtime == "cuda" else "CPUExecutionProvider"
            assert counts[required] > 0, f"{name} did not execute {required}: {counts}"
            provider_reports[name] = {
                "active": backend.device_info["active_providers"],
                "executed_nodes": dict(counts),
                "profile": str(path),
                "inference_calls": sum(event.get("name") == "model_run" for event in trace),
            }
        checked = 0
        for row in summary["evidence"]:
            assert row["state"] in {"Saved", "Incomplete"}, row
            assert set(row["artifacts"]) == {"snapshot", "clip"}, row
            for kind, artifact in row["artifacts"].items():
                assert await storage.verify(artifact["uri"], artifact["sha256"])
                if kind == "clip":
                    with av.open(str(storage._path(artifact["uri"]))) as clip:
                        assert next(clip.decode(video=0), None) is not None
                checked += 1
            assert await storage.verify(row["manifest_uri"], row["manifest_sha256"])
            checked += 1
    report = {
        "status": "passed",
        "runtime": args.runtime,
        "video": str(args.video.resolve()),
        "video_sha256": hashlib.sha256(args.video.read_bytes()).hexdigest(),
        "elapsed_seconds": time.monotonic() - started,
        "expected_frames": expected_frames,
        "providers": provider_reports,
        "verified_artifacts": checked,
        "evidence_states": dict(Counter(row["state"] for row in summary["evidence"])),
        "summary": summary,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "summary"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--project", type=Path, default=Path("projects/driver_behaviour"))
    parser.add_argument("--pipeline", default="driver_behaviour")
    parser.add_argument("--video", type=Path, default=Path("research/driver_training/test.mp4"))
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(verify(parser.parse_args()))
