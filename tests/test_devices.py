"""Device selection, execution-provider resolution, and ONNX backend behaviour.

None of this was covered before: the ONNX Runtime backend had a typo
(``ExecutionMode.ORT_PARALELL``) that raised AttributeError on *every* model
load, and no test ever created a real session. These tests pin down the paths
that matter on both GPU and CPU hosts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from inferops.core import devices
from inferops.core.devices import (
    available_providers,
    cuda_execution_provider_available,
    describe_runtime,
    resolve_onnx_providers,
)
from inferops.core.errors import BackendError
from inferops.registry.model_spec import ModelSpec
from inferops.runtimes import build_backend
from inferops.runtimes.base import InferenceBackend
from inferops.runtimes.onnxruntime_backend import OnnxRuntimeBackend
from inferops.testing.make_tiny_model import build_model

ROOT = Path(__file__).resolve().parents[1]
TINY = ROOT / "model_repository" / "tiny_classifier" / "1" / "model.onnx"

HAS_CUDA = cuda_execution_provider_available()


def _clear_device_caches() -> None:
    devices.available_providers.cache_clear()
    devices._cuda_device_count.cache_clear()


@pytest.fixture(autouse=True)
def _reset_caches():
    _clear_device_caches()
    yield
    _clear_device_caches()


@pytest.fixture(scope="module", autouse=True)
def _tiny_model_exists():
    if not TINY.exists():
        build_model(TINY)
    return TINY


def _tiny_spec(preferred: str = "cpu", gpu_id: int = 0, warmup: int = 0) -> ModelSpec:
    return ModelSpec(
        name="tiny_device_test",
        backend="onnxruntime",
        artifact={"path": str(TINY)},
        device={"preferred": preferred, "gpu_id": gpu_id},
        inputs=[{"name": "input", "shape": [1, 3, 32, 32], "dtype": "float32"}],
        outputs=[{"name": "logits", "shape": [1, 10], "dtype": "float32", "layout": "ND"}],
        warmup={"iterations": warmup},
    )


# -- provider resolution ------------------------------------------------------

def test_cpu_request_never_asks_for_cuda():
    res = resolve_onnx_providers("cpu")
    assert res.providers == ["CPUExecutionProvider"]
    assert res.effective_device == "cpu"
    assert res.downgraded is False


def test_gpu_request_downgrades_cleanly_when_no_cuda(monkeypatch):
    """A gpu-preferring spec must not explode on a CPU-only host."""
    monkeypatch.setattr(devices, "available_providers", lambda: ("CPUExecutionProvider",))
    res = resolve_onnx_providers("gpu", gpu_id=0)
    assert res.providers == ["CPUExecutionProvider"]
    assert res.effective_device == "cpu"
    assert res.requested_device == "gpu"
    assert res.downgraded is True
    assert "CUDAExecutionProvider" in res.reason


def test_gpu_request_downgrades_when_provider_present_but_no_device(monkeypatch):
    """A GPU wheel on a GPU-less host lists CUDA but cannot use it."""
    monkeypatch.setattr(
        devices, "available_providers",
        lambda: ("CUDAExecutionProvider", "CPUExecutionProvider"),
    )
    monkeypatch.setattr(devices, "_cuda_device_count", lambda: 0)
    res = resolve_onnx_providers("gpu")
    assert res.effective_device == "cpu"
    assert res.downgraded is True
    assert "no CUDA device" in res.reason


def test_gpu_id_is_honoured(monkeypatch):
    """DeviceSpec.gpu_id used to be silently dropped, pinning every model to GPU 0."""
    monkeypatch.setattr(
        devices, "available_providers",
        lambda: ("CUDAExecutionProvider", "CPUExecutionProvider"),
    )
    monkeypatch.setattr(devices, "_cuda_device_count", lambda: 4)
    res = resolve_onnx_providers("gpu", gpu_id=3)
    assert res.effective_device == "gpu"
    assert res.downgraded is False
    name, options = res.providers[0]
    assert name == "CUDAExecutionProvider"
    assert options["device_id"] == 3
    assert res.providers[-1] == "CPUExecutionProvider", "CPU must stay as the last resort"


def test_force_cpu_env_overrides_gpu_preference(monkeypatch):
    monkeypatch.setattr(
        devices, "available_providers",
        lambda: ("CUDAExecutionProvider", "CPUExecutionProvider"),
    )
    monkeypatch.setattr(devices, "_cuda_device_count", lambda: 1)
    monkeypatch.setenv("INFEROPS_FORCE_CPU", "1")
    res = resolve_onnx_providers("gpu")
    assert res.effective_device == "cpu"
    assert res.downgraded is True
    assert "FORCE_CPU" in res.reason


def test_available_providers_never_raises(monkeypatch):
    """onnxruntime-gpu raises ImportError on CPU hosts; callers must not care."""
    import builtins

    real_import = builtins.__import__

    def _boom(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("libcudart.so.13: cannot open shared object file")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _boom)
    _clear_device_caches()
    assert available_providers() == ()
    res = resolve_onnx_providers("gpu")
    assert res.effective_device == "cpu"
    assert res.downgraded is True


def test_describe_runtime_reports_the_host_truthfully():
    info = describe_runtime()
    assert set(info) >= {
        "onnxruntime_version", "available_providers",
        "cuda_available", "cuda_device_count", "force_cpu",
    }
    assert isinstance(info["cuda_available"], bool)
    assert info["cuda_available"] == (
        info["cuda_device_count"] > 0
        and "CUDAExecutionProvider" in info["available_providers"]
        and not info["force_cpu"]
    )


# -- real ONNX sessions -------------------------------------------------------

@pytest.mark.asyncio
async def test_onnx_backend_loads_and_infers_on_cpu():
    """Regression: ExecutionMode.ORT_PARALELL made every load raise AttributeError."""
    backend = OnnxRuntimeBackend()
    await backend.load(_tiny_spec("cpu"))
    try:
        assert backend.device == "cpu"
        assert backend.device_info["downgraded"] is False
        assert "CPUExecutionProvider" in backend.device_info["active_providers"]

        out = await backend.infer({"input": np.zeros((1, 3, 32, 32), dtype=np.float32)})
        assert set(out) == {"logits"}
        assert out["logits"].shape == (1, 10)
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_onnx_backend_gpu_preference_still_runs_on_cpu_host():
    """A `preferred: gpu` spec must degrade to CPU, not take the service down."""
    backend = OnnxRuntimeBackend()
    await backend.load(_tiny_spec("gpu"))
    try:
        out = await backend.infer({"input": np.zeros((1, 3, 32, 32), dtype=np.float32)})
        assert out["logits"].shape == (1, 10)
        assert backend.device_info["requested"] == "gpu"
        if not HAS_CUDA:
            assert backend.device == "cpu"
            assert backend.device_info["downgraded"] is True
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_onnx_backend_is_deterministic():
    """Sequential execution mode is chosen so identical input gives identical output."""
    backend = OnnxRuntimeBackend()
    await backend.load(_tiny_spec("cpu"))
    try:
        rng = np.random.default_rng(0)
        feed = {"input": rng.standard_normal((1, 3, 32, 32)).astype(np.float32)}
        first = await backend.infer(feed)
        second = await backend.infer(feed)
        np.testing.assert_array_equal(first["logits"], second["logits"])
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_warmup_runs_against_a_real_session():
    backend = OnnxRuntimeBackend()
    await backend.load(_tiny_spec("cpu", warmup=2))
    try:
        await backend.warmup()   # must not raise on a real graph
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_missing_input_reports_which_input_is_missing():
    backend = OnnxRuntimeBackend()
    await backend.load(_tiny_spec("cpu"))
    try:
        with pytest.raises(BackendError) as exc:
            await backend.infer({"not_the_input": np.zeros((1, 3, 32, 32), dtype=np.float32)})
        assert "input" in str(exc.value)
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_infer_before_load_is_a_backend_error():
    with pytest.raises(BackendError):
        await OnnxRuntimeBackend().infer({"input": np.zeros((1, 3, 32, 32), dtype=np.float32)})


@pytest.mark.asyncio
async def test_close_releases_the_session_and_is_idempotent():
    backend = OnnxRuntimeBackend()
    await backend.load(_tiny_spec("cpu"))
    await backend.close()
    await backend.close()          # second close must not raise
    assert backend._sess is None
    with pytest.raises(BackendError):
        await backend.infer({"input": np.zeros((1, 3, 32, 32), dtype=np.float32)})


@pytest.mark.asyncio
async def test_missing_artifact_lists_every_location_tried():
    backend = OnnxRuntimeBackend()
    spec = _tiny_spec("cpu")
    spec.artifact.path = "definitely/not/here.onnx"
    with pytest.raises(BackendError) as exc:
        await backend.load(spec)
    message = str(exc.value)
    assert "not found" in message and "definitely/not/here.onnx" in message


@pytest.mark.asyncio
async def test_synthetic_backend_needs_no_device_or_weights():
    """The CPU/no-GPU demo path must work with nothing installed."""
    detector_spec = ModelSpec(name="synth", backend="synthetic", inputs=[],
                              labels=["a", "b", "c", "d", "e", "f"],
                              tags={"queries": "20", "emit": "d=1@0.5,0.5,0.2,0.2; b=3"})
    detector = build_backend(detector_spec)
    await detector.load(detector_spec)
    raw = await detector.infer({})
    assert raw["logits"].shape == (1, 20, 6)
    assert raw["boxes"].shape == (1, 20, 4)
    # Frame 1: "d" fires every frame in slot 0; "b" (every 3rd) does not yet.
    assert raw["logits"][0, 0].argmax() == 3
    assert np.allclose(raw["boxes"][0, 0], [0.5, 0.5, 0.2, 0.2])
    assert raw["logits"][0, 1].argmax() != 1
    await detector.infer({})
    raw3 = await detector.infer({})
    assert raw3["logits"][0, 1].argmax() == 1
    await detector.close()


@pytest.mark.asyncio
async def test_synthetic_backend_rejects_unknown_labels():
    spec = ModelSpec(name="synth", backend="synthetic", inputs=[], labels=["a"],
                     tags={"emit": "zzz=2"})
    backend = build_backend(spec)
    with pytest.raises(ValueError, match="zzz"):
        await backend.load(spec)


class _EchoBackend(InferenceBackend):
    """Minimal plugin backend a domain pack might ship."""

    async def load(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.device = "cpu"
        self.device_info = {"requested": spec.device.preferred, "effective": "cpu",
                            "downgraded": False, "reason": "echo", "active_providers": []}

    async def infer(self, inputs):
        return {"echo": np.ones((1, 2), dtype=np.float32)}


@pytest.mark.asyncio
async def test_backend_can_be_a_plugin_reference():
    """`backend:` accepts module:Class so packs add backends without editing core."""
    spec = ModelSpec(name="plug", backend=f"{__name__}:_EchoBackend", inputs=[])
    backend = build_backend(spec)
    assert isinstance(backend, _EchoBackend)
    await backend.load(spec)
    assert (await backend.infer({}))["echo"].shape == (1, 2)


def test_unknown_backend_names_fail_clearly():
    from inferops.core.errors import BackendError

    with pytest.raises(BackendError, match="unknown backend"):
        build_backend(ModelSpec(name="x", backend="no_such_backend", inputs=[]))
    with pytest.raises(BackendError, match="not an InferenceBackend"):
        build_backend(ModelSpec(name="x", backend="inferops.core.types:Frame", inputs=[]))


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_CUDA, reason="no CUDA device on this host")
async def test_onnx_backend_actually_uses_the_gpu_when_present():
    """Runs only on a GPU host: proves the CUDA EP is genuinely selected."""
    backend = OnnxRuntimeBackend()
    await backend.load(_tiny_spec("gpu"))
    try:
        assert backend.device == "gpu"
        assert backend.device_info["downgraded"] is False
        assert any("CUDA" in ep for ep in backend.device_info["active_providers"])
        out = await backend.infer({"input": np.zeros((1, 3, 32, 32), dtype=np.float32)})
        assert out["logits"].shape == (1, 10)
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_system_status_exposes_the_effective_device():
    """A GPU spec silently degrading to CPU used to be invisible to operators."""
    from httpx import ASGITransport, AsyncClient

    from inferops.api.main import create_app

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac, \
            app.router.lifespan_context(app):
        response = await ac.get("/v1/system/status")
        assert response.status_code == 200, response.text
        runtime = response.json()["runtime"]

    assert set(runtime) >= {
        "onnxruntime_version", "available_providers", "cuda_available",
        "cuda_device_count", "force_cpu", "models", "models_downgraded",
    }
    assert runtime["cuda_available"] == HAS_CUDA
    assert isinstance(runtime["models"], list)
    # Anything that asked for GPU and did not get it must be listed, so a
    # degraded deployment is alertable rather than merely slow.
    for entry in runtime["models"]:
        if entry["downgraded"]:
            assert entry["model"] in runtime["models_downgraded"]
            assert entry["reason"], "a downgrade must say why"


# -- every backend reports its device ----------------------------------------

@pytest.mark.asyncio
async def test_all_backends_report_a_device():
    """The status endpoint returned nulls for every non-ONNX backend.

    `device`/`device_info` now live on the base class so /v1/system/status can
    always answer "what is this model running on?".
    """
    for backend_name in ("synthetic",):
        spec = ModelSpec(name=f"probe_{backend_name}", backend=backend_name, inputs=[],
                         device={"preferred": "gpu"})
        backend = build_backend(spec)
        await backend.load(spec)
        try:
            assert backend.device == "cpu"
            info = backend.device_info
            assert info["requested"] == "gpu"
            assert info["effective"] == "cpu"
            # A GPU-preferring spec on a numpy backend is a genuine downgrade.
            assert info["downgraded"] is True
            assert info["reason"]
        finally:
            await backend.close()


def test_base_backend_defaults_are_not_shared_mutable_state():
    """A shared dict default would leak one model's device info into another."""
    from inferops.runtimes.base import InferenceBackend

    assert InferenceBackend.device == "cpu"
    assert InferenceBackend.device_info == {}


@pytest.mark.asyncio
async def test_system_status_runtime_block_shape():
    """Pins the contract the README documents."""
    from inferops.api.routes.system import _runtime_info

    info = _runtime_info()
    assert set(info) >= {
        "onnxruntime_version", "onnxruntime_import_error", "available_providers",
        "cuda_available", "cuda_device_count", "force_cpu",
        "models", "models_downgraded",
    }
    assert isinstance(info["models"], list)
    assert isinstance(info["models_downgraded"], list)
    for entry in info["models"]:
        assert set(entry) >= {
            "model", "backend", "device", "requested_device",
            "downgraded", "reason", "active_providers",
        }
