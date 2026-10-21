"""Official model-only Perf Analyzer sweep using representative project tensors.

Install the standalone tool separately; this never runs alongside the application
benchmark automatically. Serving requests are not camera pipeline FPS.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inferops.adapters.outbound.projects.loader import ProjectLoader
from inferops.domain.types import Frame


async def run(args):
    import cv2
    import tritonclient.grpc.aio as grpc
    root = args.output / time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    root.mkdir(parents=True)
    definition = ProjectLoader(args.project, mode="triton", triton_url=args.url).load()
    spec = definition.models[definition.resolve_model(args.model)]
    capture = cv2.VideoCapture(str(args.video))
    try:
        ok, image = capture.read()
    finally:
        capture.release()
    if not ok:
        raise ValueError("input has no decodable representative frame")
    if spec.preprocess:
        preprocess = definition.object_loader(spec.preprocess.plugin)(**spec.preprocess.params)
        input_preparation = spec.preprocess.model_dump(mode="json")
    else:
        from inferops.adapters.outbound.inference.plugins.resize_normalize import ResizeNormalize
        preprocess = ResizeNormalize(normalize=False)
        input_preparation = "full image resize and [0,1] scaling; no face localization in model-only mode"
    feeds = await preprocess(Frame(stream_id="benchmark", ts_ms=0, seq=0, image=image,
                                   width=image.shape[1], height=image.shape[0]), spec)
    inputs = root / "inputs.json"
    inputs.write_text(json.dumps({"data": [{name: {"b64": base64.b64encode(value.tobytes()).decode()}
                                          for name, value in feeds.items()}]}))
    result = subprocess.run([str(args.perf_binary), "--version"], check=True, capture_output=True, text=True)
    tool_version = (result.stdout + result.stderr).strip()
    report = {"scope": "model_only", "model": spec.name, "video": str(args.video),
              "input_preparation": input_preparation,
              "tool_version": tool_version, "input_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
              "client_version": version("tritonclient"), "cases": [],
              "limits": "Excludes decoding, temporal state, SQL and evidence; not camera capacity"}
    client = grpc.InferenceServerClient(args.url)
    try:
        report["server"] = await client.get_server_metadata(as_json=True)
        report["configuration"] = await client.get_model_config(spec.artifact.triton_model_name, as_json=True)
        for count in args.concurrency:
            command = [str(args.perf_binary), "-m", spec.artifact.triton_model_name, "-i", "grpc", "-u", args.url,
                       "--async", "--concurrency-range", f"{count}:{count}:1", "--measurement-interval", "2000",
                       "--max-trials", "5", "--percentile", "95", "--input-data", str(inputs.resolve()),
                       "-f", str((root / f"concurrency-{count}.csv").resolve()), "--verbose-csv"]
            for name, value in feeds.items():
                command.extend(["--shape", f"{name}:{','.join(str(dim) for dim in value.shape[1:])}"])
            before = await client.get_inference_statistics(spec.artifact.triton_model_name, as_json=True)
            with (root / f"concurrency-{count}.log").open("w") as log:
                completed = await asyncio.to_thread(subprocess.run, command, stdout=log, stderr=subprocess.STDOUT, timeout=180)
            after = await client.get_inference_statistics(spec.artifact.triton_model_name, as_json=True)
            report["cases"].append({"concurrency": count, "returncode": completed.returncode,
                                    "command": command, "statistics_before": before, "statistics_after": after})
            (root / "report.json").write_text(json.dumps(report, indent=2))
    finally:
        await client.close()
    print(root / "report.json")
    return 0 if all(case["returncode"] == 0 for case in report["cases"]) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="projects/driver_behaviour")
    parser.add_argument("--model", default="rfdetr_driver")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--url", default="127.0.0.1:8001")
    parser.add_argument("--concurrency", type=lambda value: [int(item) for item in value.split(",")], default=[1, 2, 4, 8])
    parser.add_argument("--perf-binary", type=Path, default=Path(".run/tools/perf-env/bin/perf_analyzer"))
    parser.add_argument("--output", type=Path, default=Path(".run/perf"))
    arguments = parser.parse_args()
    if not arguments.concurrency or min(arguments.concurrency) < 1:
        parser.error("concurrency must contain positive counts")
    raise SystemExit(asyncio.run(run(arguments)))
