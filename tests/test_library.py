"""Service-free API, model composition, and generic detector contracts."""
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from inferops import InferenceSession, Model, ModelSpec
from inferops.adapters.outbound.inference.plugins.rfdetr_decode import RFDetrDecode
from inferops.adapters.outbound.inference.plugins.yolo_decode import YoloDecode
from inferops.adapters.outbound.projects.packages import adapter_root, create_project
from inferops.domain.types import Frame
from inferops.ports.inference import InferenceBackend


class IdentityBackend(InferenceBackend):
    loads = 0
    closes = 0

    async def load(self, spec):
        self.spec = spec
        type(self).loads += 1

    async def infer(self, inputs):
        return inputs

    async def close(self):
        type(self).closes += 1


def model_spec(name="identity"):
    return ModelSpec(name=name, backend=f"{__name__}:IdentityBackend", inputs=[{"name": "images"}],
                     preprocess={"plugin": "inferops.adapters.outbound.inference.plugins.resize_normalize:ResizeNormalize",
                                 "params": {"size": 2}})


async def test_session_runs_models_and_releases_resources():
    IdentityBackend.loads = IdentityBackend.closes = 0
    async with InferenceSession({"one": Model(model_spec("one")), "two": Model(model_spec("two"))}) as session:
        results = await session.predict(np.ones((4, 4, 3), dtype=np.uint8), parallel=True)
        assert list(results) == ["one", "two"]
        assert results["one"].model == "one"
        assert np.asarray(results["two"].raw["images"]).shape == (1, 3, 2, 2)
        await asyncio.gather(*(session.models["one"].infer({"images": np.zeros(1)}) for _ in range(3)))
        assert IdentityBackend.loads == 2
    assert IdentityBackend.closes == 2


async def test_project_model_selection_and_plugin_scope(tmp_path):
    import yaml

    from inferops.adapters.outbound.projects import manifest as domain
    root = tmp_path / "sample_adapter"
    folder = root / "models" / "detector"
    folder.mkdir(parents=True)
    (root / "project.yaml").write_text("name: sample_adapter\n")
    spec = model_spec().model_dump(mode="json")
    spec.pop("name")
    (folder / "model.yaml").write_text(yaml.safe_dump({"name": "detector", "defaults": spec,
                                                         "variants": {"local": {}}}))
    async with InferenceSession.from_project(root, models=["detector_local"]) as session:
        result = await session.predict(np.zeros((4, 4, 3), dtype=np.uint8))
        assert result["detector_local"].model == "detector_local"
    assert not hasattr(domain, "_active_package")


def test_installed_adapter_discovery_and_workspace_copy(tmp_path, monkeypatch):
    import inferops.adapters.outbound.projects.packages as adapters
    root = tmp_path / "adapter"
    root.mkdir()
    (root / "project.yaml").write_text("name: adapter\n")
    (root / "__init__.py").write_text("")
    (root / "model_repository").mkdir()
    import types
    module = types.ModuleType("adapter")
    module.__file__ = str(root / "__init__.py")
    entry = SimpleNamespace(name="adapter", load=lambda: module)
    monkeypatch.setattr(adapters.importlib.metadata, "entry_points", lambda **kw: [entry])
    assert adapter_root("adapter") == root
    copied = create_project("adapter", tmp_path / "workspace")
    assert (copied / "project.yaml").is_file()
    assert not (copied / "model_repository").exists()
    with pytest.raises(ValueError, match="Python identifier"):
        create_project("adapter", tmp_path / "invalid-project")
    with pytest.raises(FileExistsError):
        create_project("adapter", copied)


async def test_rfdetr_uses_sigmoid_and_explicit_box_format():
    spec = ModelSpec(name="detector", inputs=[], labels=["person", "car"])
    frame = Frame(stream_id="test", ts_ms=0, width=100, height=100)
    result = await RFDetrDecode(conf_threshold=0.7)(
        {"dets": np.array([[[.5, .5, .2, .4]]]), "labels": np.array([[[2., 1.]]])}, frame, spec)
    assert result.detections[0].label == "person"
    assert result.detections[0].score == pytest.approx(0.880797)
    assert result.detections[0].box == pytest.approx((.4, .3, .6, .7))


async def test_yolo_keeps_overlapping_different_classes():
    spec = ModelSpec(name="yolo", inputs=[], labels=["person", "car"])
    frame = Frame(stream_id="test", ts_ms=0, width=100, height=100)
    raw = {"output": np.array([[[50, 50, 20, 20, .9, .1], [50, 50, 20, 20, .1, .8]]])}
    result = await YoloDecode()(raw, frame, spec)
    assert [det.label for det in result.detections] == ["person", "car"]
    assert len((await YoloDecode(class_agnostic=True)(raw, frame, spec)).detections) == 1


async def test_real_onnx_inference(tmp_path):
    from onnx import TensorProto, helper, save
    path = tmp_path / "identity.onnx"
    graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "identity",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])])
    exported = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    exported.ir_version = 9
    save(exported, path)
    spec = ModelSpec(name="onnx", inputs=[{"name": "x", "shape": [1, 2]}], artifact={"path": str(path)})
    async with Model(spec) as model:
        x = np.array([[1., 2.]], dtype=np.float32)
        np.testing.assert_array_equal((await model.infer({"x": x}))["y"], x)


async def test_relative_backend_is_bound_to_its_project(tmp_path):
    from inferops.adapters.outbound.projects import manifest as domain

    root = tmp_path / "backend_adapter"
    root.mkdir()
    (root / "__init__.py").write_text("")
    (root / "project.yaml").write_text("name: backend_adapter\n")
    (root / "backend.py").write_text(
        "from inferops.ports.inference import InferenceBackend\n"
        "class Backend(InferenceBackend):\n"
        "    async def load(self, spec): self.spec = spec\n"
        "    async def infer(self, inputs): return inputs\n")
    models = root / "models"
    models.mkdir()
    (models / "identity.yaml").write_text(
        "name: identity\nbackend: .backend:Backend\ninputs: [{name: x}]\n")
    async with InferenceSession.from_project(root) as session:
        assert not hasattr(domain, "_active_package")
        assert session.models["identity"].spec.backend.startswith("_inferops_project_")
        assert session.models["identity"].spec.backend.endswith(".backend:Backend")
        x = np.ones((1, 2), dtype=np.float32)
        np.testing.assert_array_equal((await session.models["identity"].infer({"x": x}))["x"], x)


def test_tensorrt_requires_provider(monkeypatch):
    from inferops.adapters.outbound.inference import tensorrt_backend
    from inferops.domain.errors import BackendError

    monkeypatch.setattr(tensorrt_backend, "available_providers", lambda: ("CPUExecutionProvider",))
    with pytest.raises(BackendError, match="TensorRT EP unavailable"):
        tensorrt_backend.TensorRTBackend()._provider_resolution(model_spec())


async def test_project_handles_share_one_backend_for_logical_and_variant_names(tmp_path):
    import yaml

    root = tmp_path / "shared"
    (root / "models").mkdir(parents=True)
    (root / "project.yaml").write_text("name: shared\n")
    defaults = model_spec().model_dump(mode="json")
    defaults.pop("name")
    (root / "models/identity.yaml").write_text(yaml.safe_dump({
        "name": "identity", "defaults": defaults, "variants": {"onnx": {}}}))
    IdentityBackend.loads = IdentityBackend.closes = 0
    async with InferenceSession.from_project(root, models=["identity", "identity_onnx"]) as session:
        results = await session.predict(np.ones((4, 4, 3), dtype=np.uint8), parallel=True)
        assert list(results) == ["identity", "identity_onnx"]
        assert {result.model for result in results.values()} == {"identity_onnx"}
        assert IdentityBackend.loads == 1
    assert IdentityBackend.closes == 1


async def test_public_model_close_waits_for_inflight_inference():
    entered, release = asyncio.Event(), asyncio.Event()

    class BlockingBackend(IdentityBackend):
        closed = False

        async def infer(self, inputs):
            entered.set()
            await release.wait()
            assert not self.closed
            return inputs

        async def close(self):
            self.closed = True

    backend = BlockingBackend()
    model = Model(model_spec(), backend_factory=lambda spec, base: backend)
    prediction = asyncio.create_task(model.infer({"images": np.zeros(1)}))
    await entered.wait()
    closing = asyncio.create_task(model.close())
    await asyncio.sleep(0)
    assert not closing.done() and not backend.closed
    release.set()
    await prediction
    await closing
    assert backend.closed
