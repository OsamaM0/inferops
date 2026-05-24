"""Generate a tiny ONNX classifier for local smoke tests (no GPU needed)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build_model(out_path: Path, num_classes: int = 10) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # GlobalAveragePool over (N,3,32,32) -> (N,3,1,1) -> Flatten -> Gemm(3 -> num_classes)
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 3, 32, 32])
    out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, num_classes])

    rng = np.random.default_rng(0)
    W = numpy_helper.from_array(rng.standard_normal((num_classes, 3)).astype("float32"), name="W")
    B = numpy_helper.from_array(np.zeros(num_classes, dtype="float32"), name="B")

    n1 = helper.make_node("GlobalAveragePool", ["input"], ["gap"])
    n2 = helper.make_node("Flatten", ["gap"], ["flat"], axis=1)
    n3 = helper.make_node("Gemm", ["flat", "W", "B"], ["logits"], transB=1)

    graph = helper.make_graph([n1, n2, n3], "tiny", [inp], [out], initializer=[W, B])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("model_repository/tiny_classifier/1/model.onnx")
    build_model(target)
