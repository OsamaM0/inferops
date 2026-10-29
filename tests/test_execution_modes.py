"""Strict backend choice and runtime startup."""
from pathlib import Path

import pytest

from inferops.adapters.inbound.http.main import create_app
from inferops.adapters.outbound.inference.onnxruntime_backend import OnnxRuntimeBackend
from inferops.adapters.outbound.projects.loader import ProjectLoader
from inferops.adapters.outbound.projects.settings import ConfigStack, get_settings
from inferops.application.models import ModelPool
from inferops.domain.errors import BackendError
from inferops.domain.models import ModelSpec
from inferops.testing import TINY_MODEL_PATH

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,model,face", [
    ("cpu", "rfdetr_driver_onnx", "blinkcnn_onnx"),
    ("cuda", "rfdetr_driver_onnx", "blinkcnn_onnx"),
    ("tensorrt", "rfdetr_driver_tensorrt", "blinkcnn_tensorrt"),
    ("triton", "rfdetr_driver_triton", "blinkcnn_triton"),
    ])
async def test_production_modes_do_not_fall_back_to_synthetic(monkeypatch, mode, model, face):
    monkeypatch.setattr(get_settings(), "execution_mode", mode)
    stack = ConfigStack(ROOT / "configs", active_domain="driver_behaviour")
    definition = ProjectLoader(stack.domain_root, mode=mode).load()
    stages = {s.name: s for s in definition.pipelines["driver_behaviour"].stages}
    assert stages["detect"].params["model"] == model
    assert stages["eyes"].params["model"] == face
    assert stages["head_pose"].params["model"] == face.replace("blinkcnn", "head_pose_360")
    assert stages["eye_landmarks"].params["model"] == face.replace("blinkcnn", "landmarks_106")
    assert stages["detect_face"].params["model"] == model.replace("rfdetr_driver", "scrfd")
    assert stages["detect_face"].params["verification_model"] is None
    assert "face_landmarker" not in stages
    assert stages["detect"].params["fallback_model"] is None
    assert stages["eyes"].params["fallback_model"] is None
    registry = ModelPool(definition)
    if mode in {"cpu", "cuda"}:
        assert registry.get_spec(model).device.preferred == ("cpu" if mode == "cpu" else "gpu")


@pytest.mark.asyncio
async def test_cuda_mode_rejects_a_cpu_session(monkeypatch):
    monkeypatch.setattr(get_settings(), "execution_mode", "cuda")
    monkeypatch.setenv("INFEROPS_FORCE_CPU", "1")
    spec = ModelSpec(name="cuda_required", backend="onnxruntime", device={"preferred": "gpu"},
                     artifact={"path": str(TINY_MODEL_PATH)},
                     inputs=[{"name": "input", "shape": [1, 3, 32, 32]}])
    with pytest.raises(BackendError, match="requires an active CUDA"):
        await OnnxRuntimeBackend().load(spec)


@pytest.mark.asyncio
async def test_explicit_mode_prewarms_the_active_pipeline_without_driver_names(monkeypatch):
    monkeypatch.setattr(get_settings(), "execution_mode", "cpu")
    monkeypatch.delenv("INFEROPS_PREWARM_MODELS", raising=False)
    app = create_app()
    async with app.router.lifespan_context(app):
        models = app.state.engine.application.models.loaded_backends()
        assert set(models) == {"tiny_classifier"}
        assert models["tiny_classifier"].device == "cpu"
