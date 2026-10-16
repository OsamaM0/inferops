"""InferOps: reusable model inference and multi-model pipelines."""
from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["Engine", "InferenceSession", "Model", "ModelSpec", "Result", "__version__"]


def __getattr__(name: str):
    if name == "Engine":
        from inferops.public import Engine
        return Engine
    if name in {"InferenceSession", "Model"}:
        from inferops import public
        return getattr(public, name)
    if name == "ModelSpec":
        from inferops.domain.models import ModelSpec
        return ModelSpec
    if name == "Result":
        from inferops.domain.types import Result
        return Result
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
