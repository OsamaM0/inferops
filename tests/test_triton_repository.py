import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from inferops.adapters.outbound.inference.triton_repository import prepare_repository


def model(path, *, separable=True):
    source = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 2, 2])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 2, 2])
    node = (
        helper.make_node("Add", ["input", "bias"], ["output"])
        if separable
        else helper.make_node("ReduceMean", ["input"], ["output"], axes=[0], keepdims=1)
    )
    graph = helper.make_graph(
        [node],
        "test",
        [source],
        [output],
        [numpy_helper.from_array(np.ones((1, 3, 2, 2), dtype=np.float32), "bias")] if separable else [],
    )
    artifact = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(artifact, path)


def test_only_independent_batch_outputs_are_admitted(tmp_path):
    source = tmp_path / "source"
    model(source / "good" / "1" / "model.onnx")
    model(source / "mixed" / "1" / "model.onnx", separable=False)
    before = {p: p.read_bytes() for p in source.rglob("*.onnx")}
    output = tmp_path / "serving"
    report = prepare_repository(source, output, max_batch_size=4)
    good, bad = report["models"]
    assert good["max_batch_size"] == 4
    assert [c["batch"] for c in good["qualification"]["checks"]] == [1, 2, 4]
    assert bad["max_batch_size"] == 0 and "batching_disabled_reason" in bad
    config = (output / "good" / "config.pbtxt").read_text()
    assert "dynamic_batching" in config and "timeout_action: REJECT" in config
    assert "dims: [3, 2, 2]" in config
    assert "dynamic_batching" not in (output / "mixed" / "config.pbtxt").read_text()
    assert all(p.read_bytes() == data for p, data in before.items())
    assert (output / "mixed" / "1" / "model.onnx").read_bytes() == next(
        data for p, data in before.items() if "mixed" in p.parts
    )


def test_cannot_overwrite_source_models(tmp_path):
    with pytest.raises(ValueError, match="separate"):
        prepare_repository(tmp_path, tmp_path / "generated")


def test_already_dynamic_graph_is_qualified_without_patching_internal_shapes(tmp_path):
    source = tmp_path / "source"
    path = source / "dynamic" / "1" / "model.onnx"
    model(path)
    graph = onnx.load(path)
    for value in [*graph.graph.input, *graph.graph.output]:
        value.type.tensor_type.shape.dim[0].dim_param = "batch"
    onnx.save(graph, path)
    report = prepare_repository(source, tmp_path / "serving")
    assert report["models"][0]["max_batch_size"] == 4
    config = (tmp_path / "serving/dynamic/config.pbtxt").read_text()
    assert report["models"][0]["serving_sha256"] in config
