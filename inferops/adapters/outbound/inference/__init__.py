"""Inference backend factory.

``ModelSpec.backend`` is either a built-in name or a ``module:Class`` plugin
reference (pack-relative ``.plugins.x:Y`` works too), so a domain pack can ship
its own backend without touching this package.
"""
from __future__ import annotations

import inspect
from pathlib import Path

from inferops.adapters.outbound.projects.plugins import load_object
from inferops.domain.errors import BackendError
from inferops.domain.models import ModelSpec
from inferops.ports.inference import InferenceBackend

BUILTIN_BACKENDS: dict[str, str] = {
    "onnxruntime": "inferops.adapters.outbound.inference.onnxruntime_backend:OnnxRuntimeBackend",
    "triton": "inferops.adapters.outbound.inference.triton_backend:TritonBackend",
    "tensorrt": "inferops.adapters.outbound.inference.tensorrt_backend:TensorRTBackend",
    "synthetic": "inferops.adapters.outbound.inference.synthetic_backend:SyntheticDetectorBackend",
}

# Import needed for `inferops preflight` to check a built-in backend's dependency.
BACKEND_DEPENDENCIES: dict[str, str] = {
    "onnxruntime": "onnxruntime",
    "triton": "tritonclient",
    "tensorrt": "onnxruntime",
}


def resolve_backend_class(name: str) -> type[InferenceBackend]:
    ref = BUILTIN_BACKENDS.get(name, name)
    if ref == name and ":" not in name and "." not in name:
        raise BackendError(
            f"unknown backend: {name!r} (built-ins: {', '.join(sorted(BUILTIN_BACKENDS))}; "
            "or give a module:Class plugin reference)"
        )
    try:
        cls = load_object(ref)
    except Exception as exc:
        raise BackendError(f"cannot load backend {name!r}: {exc}") from exc
    if not (inspect.isclass(cls) and issubclass(cls, InferenceBackend)):
        raise BackendError(f"backend {name!r} is not an InferenceBackend subclass")
    return cls


def build_backend(spec: ModelSpec, base_dir: Path | None = None) -> InferenceBackend:
    cls = resolve_backend_class(spec.backend)
    if cls.__init__ is object.__init__:
        return cls()
    params = inspect.signature(cls.__init__).parameters
    if "base_dir" in params or any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return cls(base_dir=base_dir)
    return cls()


__all__ = ["BACKEND_DEPENDENCIES", "BUILTIN_BACKENDS", "InferenceBackend", "build_backend", "resolve_backend_class"]
