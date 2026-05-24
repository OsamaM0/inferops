"""TensorRT backend stub (reserved)."""
from __future__ import annotations

from inferops.core.errors import BackendError
from inferops.runtimes.base import InferenceBackend


class TensorRTBackend(InferenceBackend):
    async def load(self, spec) -> None:
        raise BackendError("TensorRT backend not yet implemented; use 'triton' with TRT plan files")

    async def infer(self, inputs):
        raise BackendError("TensorRT backend not yet implemented")
