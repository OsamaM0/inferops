"""Device / execution-provider resolution for in-process runtimes.

Previously ``OnnxRuntimeBackend`` built its provider list from
``spec.device.preferred`` alone::

    providers = ["CPUExecutionProvider"]
    if spec.device.preferred == "gpu":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

That had three problems:

1. ``spec.device.gpu_id`` was silently ignored, so every model landed on GPU 0
   regardless of config.
2. ONNX Runtime >= 1.19 *raises* when you name a provider it was not built with
   (or whose CUDA libs are missing) instead of quietly falling back, so a spec
   marked ``preferred: gpu`` hard-failed on a CPU-only host.
3. Nothing recorded which device a model actually ended up on, so a silent
   downgrade to CPU was invisible in logs and metrics.

This module centralises the decision and always reports the outcome.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

__all__ = [
    "DeviceResolution",
    "available_providers",
    "cuda_execution_provider_available",
    "describe_runtime",
    "resolve_onnx_providers",
]


@dataclass(frozen=True)
class DeviceResolution:
    """The outcome of matching a requested device against what the host has."""

    providers: list[Any]
    effective_device: str        # "gpu" | "cpu"
    requested_device: str        # "gpu" | "cpu"
    gpu_id: int
    downgraded: bool             # gpu requested but only cpu available
    reason: str


def _force_cpu_env() -> bool:
    """Allow operators to pin everything to CPU without editing every spec."""
    return os.environ.get("INFEROPS_FORCE_CPU", "").strip().lower() in {"1", "true", "yes"}


@lru_cache(maxsize=1)
def available_providers() -> tuple[str, ...]:
    """Providers ONNX Runtime was built with, or () when ORT is unimportable.

    ``onnxruntime-gpu`` wheels fail to import entirely when their CUDA shared
    objects are absent (``ImportError: libcudart.so.13``), so this must never
    let an exception escape — callers use it to *decide* whether to try CUDA.
    """
    try:
        import onnxruntime as ort
    except Exception:
        return ()
    try:
        return tuple(ort.get_available_providers())
    except Exception:
        return ()


def cuda_execution_provider_available() -> bool:
    """True when ORT can actually construct a CUDA session on this host.

    ``get_available_providers()`` lists providers the wheel was *compiled* with,
    which on a CPU-only host still includes ``CUDAExecutionProvider`` for a GPU
    wheel. The extra device-count probe is what distinguishes "compiled in" from
    "usable", and it is why this returns False on a GPU wheel with no GPU.
    """
    if _force_cpu_env():
        return False
    if "CUDAExecutionProvider" not in available_providers():
        return False
    return _cuda_device_count() > 0


@lru_cache(maxsize=1)
def _cuda_device_count() -> int:
    """Best-effort count of usable CUDA devices.

    Tries the cheap, dependency-free probes first and only then falls back to
    trusting the provider list. Any failure means "assume no GPU" — downgrading
    to CPU is always safe, whereas a false positive crashes session creation.
    """
    # 1. NVML via pynvml (ships with several CUDA wheels).
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            return int(pynvml.nvmlDeviceGetCount())
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass

    # 2. torch, if the project happens to be installed alongside it.
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
        return 0
    except Exception:
        pass

    # 3. Device nodes — present whenever the NVIDIA driver is loaded, including
    #    inside containers started with --gpus.
    try:
        from pathlib import Path

        nodes = [p for p in Path("/dev").glob("nvidia[0-9]*") if p.exists()]
        if nodes:
            return len(nodes)
        if Path("/dev/nvidiactl").exists():
            return 1
    except Exception:
        pass

    return 0


def resolve_onnx_providers(preferred: str, gpu_id: int = 0) -> DeviceResolution:
    """Build the ORT provider list for a requested device.

    Returns CPU-only providers (with ``downgraded=True``) rather than raising
    when GPU was asked for but is unavailable: a slower service beats a service
    that will not start.
    """
    requested = "gpu" if preferred == "gpu" else "cpu"
    cpu_only: list[Any] = ["CPUExecutionProvider"]

    if requested != "gpu":
        return DeviceResolution(
            providers=cpu_only, effective_device="cpu", requested_device="cpu",
            gpu_id=gpu_id, downgraded=False, reason="spec requested cpu",
        )

    if _force_cpu_env():
        return DeviceResolution(
            providers=cpu_only, effective_device="cpu", requested_device="gpu",
            gpu_id=gpu_id, downgraded=True, reason="INFEROPS_FORCE_CPU is set",
        )

    providers = available_providers()
    if not providers:
        return DeviceResolution(
            providers=cpu_only, effective_device="cpu", requested_device="gpu",
            gpu_id=gpu_id, downgraded=True,
            reason="onnxruntime is not importable on this host",
        )
    if "CUDAExecutionProvider" not in providers:
        return DeviceResolution(
            providers=cpu_only, effective_device="cpu", requested_device="gpu",
            gpu_id=gpu_id, downgraded=True,
            reason=f"onnxruntime build has no CUDAExecutionProvider (has: {', '.join(providers)})",
        )
    if _cuda_device_count() <= 0:
        return DeviceResolution(
            providers=cpu_only, effective_device="cpu", requested_device="gpu",
            gpu_id=gpu_id, downgraded=True,
            reason="no CUDA device visible to this process",
        )

    # device_id is what actually honours DeviceSpec.gpu_id.
    cuda_options: dict[str, Any] = {
        "device_id": int(gpu_id),
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_algo_search": "EXHAUSTIVE",
        "do_copy_in_default_stream": True,
    }
    return DeviceResolution(
        providers=[("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"],
        effective_device="gpu", requested_device="gpu", gpu_id=int(gpu_id),
        downgraded=False, reason="CUDAExecutionProvider available",
    )


def describe_runtime() -> dict[str, Any]:
    """Device facts for /v1/system and for startup logs."""
    providers = available_providers()
    try:
        import onnxruntime as ort

        ort_version: str | None = ort.__version__
        ort_import_error: str | None = None
    except Exception as exc:
        ort_version = None
        ort_import_error = f"{type(exc).__name__}: {exc}"

    return {
        "onnxruntime_version": ort_version,
        "onnxruntime_import_error": ort_import_error,
        "available_providers": list(providers),
        "cuda_available": cuda_execution_provider_available(),
        "cuda_device_count": _cuda_device_count(),
        "force_cpu": _force_cpu_env(),
    }
