"""Verify a running Triton server: original parity, response isolation and batching."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import onnxruntime as ort
import tritonclient.grpc.aio as grpc

from inferops.adapters.outbound.inference.checkpoint_export import compare_detections
from inferops.adapters.outbound.projects.loader import ProjectLoader
from inferops.domain.types import Frame


async def verify(args):
    definition = ProjectLoader(args.project, mode="triton", triton_url=args.url).load()
    client = grpc.InferenceServerClient(args.url)
    report = {
        "at": time.time(),
        "server": await client.get_server_metadata(as_json=True),
        "models": [],
        "passed": False,
    }
    capture = cv2.VideoCapture(args.video)
    ok, image = capture.read()
    capture.release()
    if not ok:
        raise ValueError("Representative image/video could not be decoded")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    try:
        for logical in ("face_mesh", "rfdetr_driver"):
            spec = definition.models[definition.resolve_model(logical)]
            backend = definition.backend_factory(spec, definition.source_dirs.get(spec.name))
            await backend.load(spec)
            try:
                reference_path = (Path(args.reference_repository) / spec.artifact.triton_model_name / "1/model.onnx"
                                  if args.reference_repository else Path(spec.artifact.path))
                reference = ort.InferenceSession(str(reference_path), sess_options=options,
                                                 providers=["CPUExecutionProvider"])
                meta = reference.get_inputs()[0]
                height, width = meta.shape[-2:]
                value = (
                    cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (width, height))
                    .transpose(2, 0, 1)[None]
                    .astype(np.float32)
                    / 255
                )
                feeds = [{meta.name: np.ascontiguousarray(value * (0.5 + i / 16))} for i in range(8)]
                singles = [await backend.infer(feed) for feed in feeds]
                raw_parity = True
                foreground_counts = []
                for feed, actual in zip(feeds, singles, strict=True):
                    original = reference.run(None, feed)
                    for out, expected in zip(reference.get_outputs(), original, strict=True):
                        if logical == "rfdetr_driver":
                            raw_parity &= bool(np.allclose(actual[out.name], expected, rtol=1e-3, atol=1e-4))
                        else:
                            np.testing.assert_allclose(actual[out.name], expected, rtol=1e-3, atol=1e-4)
                    if logical == "rfdetr_driver":
                        foreground_counts.append(compare_detections([actual[o.name] for o in reference.get_outputs()], original,
                                                                     classes=len(spec.label_list())))
                before = await backend.statistics()
                for _ in range(4):
                    concurrent = await asyncio.gather(*(backend.infer(feed) for feed in feeds))
                    for baseline, result in zip(singles, concurrent, strict=True):
                        for name in baseline:
                            assert baseline[name].shape == result[name].shape
                            if logical != "rfdetr_driver":
                                np.testing.assert_allclose(result[name], baseline[name], rtol=1e-3, atol=1e-4)
                        if logical == "rfdetr_driver":
                            compare_detections([result[o.name] for o in reference.get_outputs()],
                                               [baseline[o.name] for o in reference.get_outputs()], classes=len(spec.label_list()))
                if logical == "rfdetr_driver" and not sum(foreground_counts):
                    raise AssertionError("No foreground detections; provide a representative input")
                if spec.postprocess:
                    post = definition.object_loader(spec.postprocess.plugin)(**spec.postprocess.params)
                    frame = Frame(
                        stream_id="batch-verification",
                        ts_ms=0,
                        seq=0,
                        image=image,
                        width=image.shape[1],
                        height=image.shape[0],
                    )
                    for result in concurrent:
                        await post(result, frame, spec)
                after = await backend.statistics()
                old = before["versions"][0]
                new = after["versions"][0]
                inferences = new["inference_count"] - old["inference_count"]
                executions = new["execution_count"] - old["execution_count"]
                average = inferences / executions if executions else None
                if after["dynamic_batching"] and (average is None or average <= 1):
                    raise AssertionError(f"{logical}: no actual server batching observed")
                report["models"].append(
                    {
                        "model": logical,
                        "reference_cpu_parity": True,
                        "comparison": "Foreground detections at 0.35" if logical == "rfdetr_driver" else "Raw tensors",
                        "raw_reference_parity": raw_parity,
                        "foreground_counts": foreground_counts,
                        "reference_path": str(reference_path),
                        "per_request_postprocessor": bool(spec.postprocess),
                        "concurrent_response_isolation": True,
                        "inference_count": inferences,
                        "execution_count": executions,
                        "average_execution_batch_size": average,
                        "max_batch_size": after["max_batch_size"],
                        "device_info": backend.device_info,
                    }
                )
            finally:
                await backend.close()
    finally:
        await client.close()
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2))
    report["passed"] = True
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="127.0.0.1:8001")
    parser.add_argument("--project", default="projects/driver_behaviour")
    parser.add_argument("--video", default="research/driver_training/test.mp4")
    parser.add_argument("--output", default=".run/triton/verification.json")
    parser.add_argument("--reference-repository", help="Repository containing the exact ONNX weights served remotely")
    asyncio.run(verify(parser.parse_args()))
