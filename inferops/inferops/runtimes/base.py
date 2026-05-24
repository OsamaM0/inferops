"""InferenceBackend ABC."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from inferops.registry.model_spec import ModelSpec


class InferenceBackend(ABC):
    spec: ModelSpec | None = None

    @abstractmethod
    async def load(self, spec: ModelSpec) -> None: ...

    @abstractmethod
    async def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...

    async def warmup(self) -> None:
        if self.spec is None:
            return
        feeds = {}
        rng = np.random.default_rng(self.spec.warmup.dummy_input_seed)
        for io in self.spec.inputs:
            shape = [d if d > 0 else 1 for d in io.shape]
            dt = np.dtype(io.dtype)
            if dt.kind == "f":
                feeds[io.name] = rng.standard_normal(size=shape).astype(dt)
            else:
                feeds[io.name] = np.zeros(shape, dtype=dt)
        for _ in range(self.spec.warmup.iterations):
            await self.infer(feeds)

    async def close(self) -> None:
        return None
