"""Strict backend choice and the same-origin frontend contract."""
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from inferops.api.main import create_app
from inferops.core.config import ConfigStack, settings
from inferops.core.errors import BackendError
from inferops.pipeline.store import PipelineStore
from inferops.registry.model_spec import ModelSpec
from inferops.registry.store import ModelRegistry
from inferops.runtimes.onnxruntime_backend import OnnxRuntimeBackend

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,model,face", [
    ("cpu", "rfdetr_driver_onnx", "face_mesh_onnx"),
    ("cuda", "rfdetr_driver_onnx", "face_mesh_onnx"),
    ("tensorrt", "rfdetr_driver_tensorrt", "face_mesh_tensorrt"),
    ("triton", "rfdetr_driver_triton", "face_mesh_triton"),
    ("deepstream", "rfdetr_driver_triton", "face_mesh_triton"),
])
async def test_production_modes_do_not_fall_back_to_synthetic(monkeypatch, mode, model, face):
    monkeypatch.setattr(settings, "execution_mode", mode)
    stack = ConfigStack(ROOT / "configs", active_domain="driver_behaviour")
    store = PipelineStore()
    await store.reload_from(stack.spec_dir("pipelines"))
    stages = {s.name: s for s in store.get("driver_behaviour").stages}
    assert stages["detect"].params["model"] == model
    assert stages["face_mesh"].params["model"] == face
    assert stages["detect"].params["fallback_model"] is None
    assert stages["face_mesh"].params["fallback_model"] is None
    registry = ModelRegistry()
    await registry.reload_from(stack.spec_dir("models"))
    if mode in {"cpu", "cuda"}:
        assert registry.get_spec(model).device.preferred == ("cpu" if mode == "cpu" else "gpu")


@pytest.mark.asyncio
async def test_frontend_is_served_beside_api_docs():
    async with AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://test") as client:
        response = await client.get("/frontend/")
        assert response.status_code == 200
        assert "Analyze all frames and save evidence" in response.text
        assert (await client.get("/")).json()["frontend"] == "/frontend/"


@pytest.mark.asyncio
async def test_cuda_mode_rejects_a_cpu_session(monkeypatch):
    monkeypatch.setattr(settings, "execution_mode", "cuda")
    monkeypatch.setenv("INFEROPS_FORCE_CPU", "1")
    spec = ModelSpec(name="cuda_required", backend="onnxruntime", device={"preferred": "gpu"},
                     artifact={"path": str(ROOT / "model_repository/tiny_classifier/1/model.onnx")},
                     inputs=[{"name": "input", "shape": [1, 3, 32, 32]}])
    with pytest.raises(BackendError, match="requires an active CUDA"):
        await OnnxRuntimeBackend().load(spec)


@pytest.mark.asyncio
async def test_explicit_mode_prewarms_the_active_pipeline_without_driver_names(monkeypatch):
    from inferops.registry import get_registry
    monkeypatch.setattr(settings, "execution_mode", "cpu")
    monkeypatch.delenv("INFEROPS_PREWARM_MODELS", raising=False)
    app = create_app()
    async with app.router.lifespan_context(app):
        models = get_registry().loaded_backends()
        assert set(models) == {"tiny_classifier"}
        assert models["tiny_classifier"].device == "cpu"
