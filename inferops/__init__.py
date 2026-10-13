"""InferOps: reusable model inference and multi-model pipelines."""
from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["InferenceSession", "Model", "ModelSpec", "Result", "__version__"]


def __getattr__(name: str):
    if name in {"InferenceSession", "Model"}:
        from inferops import library
        return getattr(library, name)
    if name == "ModelSpec":
        from inferops.registry.model_spec import ModelSpec
        return ModelSpec
    if name == "Result":
        from inferops.core.types import Result
        return Result
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
