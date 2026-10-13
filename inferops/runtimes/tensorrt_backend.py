"""TensorRT acceleration of ONNX graphs through ONNX Runtime's TensorRT EP.

Serialized .plan engines are served through the Triton backend instead.
"""
from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path

from inferops.core.devices import DeviceResolution, available_providers
from inferops.core.errors import BackendError
from inferops.runtimes.onnxruntime_backend import OnnxRuntimeBackend


class TensorRTBackend(OnnxRuntimeBackend):
    async def load(self, spec):
        try:
            ort = import_module("onnxruntime")
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls(directory="")
            # The pip wheel imports tensorrt_libs, loading its shared libraries
            # before ONNX Runtime attempts to resolve libnvinfer.so.10.
            import_module("tensorrt")
        except (ImportError, OSError) as exc:
            raise BackendError(f"TensorRT libraries could not be loaded: {exc}; install 'inferops[tensorrt]'") from exc
        spec = spec.model_copy(deep=True)
        spec.device.preferred = "gpu"
        await super().load(spec)

    def _provider_resolution(self, spec):
        if "TensorrtExecutionProvider" not in available_providers():
            raise BackendError("TensorRT EP unavailable; install 'inferops[tensorrt]' and matching NVIDIA drivers")
        options = {"device_id": spec.device.gpu_id}
        cache = os.environ.get("INFEROPS_TRT_CACHE_DIR")
        if cache:
            path = Path(cache).expanduser().resolve()
            path.mkdir(parents=True, exist_ok=True)
            options.update(trt_engine_cache_enable=True, trt_engine_cache_path=str(path),
                           trt_timing_cache_enable=True)
        return DeviceResolution(
            providers=[("TensorrtExecutionProvider", options),
                       ("CUDAExecutionProvider", {"device_id": spec.device.gpu_id}), "CPUExecutionProvider"],
            effective_device="gpu", requested_device="gpu", gpu_id=spec.device.gpu_id,
            downgraded=False, reason="TensorRT EP with CUDA/CPU fallback for unsupported graph nodes")

    def _validate_providers(self, active):
        if "TensorrtExecutionProvider" not in active:
            raise BackendError("TensorRT provider failed to initialize; check TensorRT/CUDA shared libraries")
