"""Prepare the MediaPipe face-blendshapes ONNX artifact.

Downloads a pinned ONNX conversion of MediaPipe's ``face_blendshapes.tflite``
(Apache-2.0; verified to match the official TFLite within 2e-6), checks its
SHA-256 and rewrites only the graph interface:

* input  ``serving_default_input_points:0`` [1, 146, 2] -> ``points``
* output ``StatefulPartitionedCall:0``      [52]        -> ``blendshapes`` [1, 52]

The leading batch axis lets Triton batch-qualify it like every other model.
Usage: ``python install.py --output <model.onnx>`` (run by project prepare).
"""
from __future__ import annotations

import argparse
import hashlib
import tempfile
import urllib.request
from pathlib import Path

URL = ("https://huggingface.co/fernandotonon/QtMeshEditor-faceblendshapes-onnx/resolve/"
       "495864b0bae88b0b926b8da8cb85f9022ceaa402/face_blendshapes.onnx")
SHA256 = "37cf54d3ccf671f8936427437a1a0b627799e16f20439de3cc35aa189ad40c66"
SOURCE_INPUT, SOURCE_OUTPUT = "serving_default_input_points:0", "StatefulPartitionedCall:0"
INPUT, OUTPUT = "points", "blendshapes"


def rewrite(source: Path, target: Path) -> None:
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(str(source))
    graph = model.graph
    for node in graph.node:
        node.input[:] = [INPUT if name == SOURCE_INPUT else name for name in node.input]
    (graph_input,) = [item for item in graph.input if item.name == SOURCE_INPUT]
    graph_input.name = INPUT
    axes = helper.make_tensor("blendshapes_batch_axis", TensorProto.INT64, [1], [0])
    graph.initializer.append(axes)
    graph.node.append(helper.make_node("Unsqueeze", [SOURCE_OUTPUT, axes.name], [OUTPUT], name="add_batch_axis"))
    del graph.output[:]
    graph.output.append(helper.make_tensor_value_info(OUTPUT, TensorProto.FLOAT, [1, 52]))
    if max(opset.version for opset in model.opset_import if opset.domain in {"", "ai.onnx"}) < 13:
        raise ValueError("expected opset >= 13 (Unsqueeze with axes input)")
    onnx.checker.check_model(model)
    onnx.save(model, str(target))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="blendshapes-", dir=args.output.parent) as directory:
        downloaded = Path(directory) / "source.onnx"
        request = urllib.request.Request(URL, headers={"User-Agent": "InferOps/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            downloaded.write_bytes(response.read())
        digest = hashlib.sha256(downloaded.read_bytes()).hexdigest()
        if digest != SHA256:
            raise SystemExit(f"face_blendshapes: SHA-256 mismatch ({digest})")
        staged = Path(directory) / "model.onnx"
        rewrite(downloaded, staged)
        staged.replace(args.output)


if __name__ == "__main__":
    main()
