from types import SimpleNamespace

import numpy as np
import pytest

from inferops.core.types import StageItem
from inferops.pipeline.stage import StageContext
from inferops.pipeline.stages.infer import Infer


class FakeBackend:
    def __init__(self, name: str, registry: "FakeRegistry") -> None:
        self.name = name
        self.registry = registry

    async def infer(self, feeds):
        self.registry.calls.append(self.name)
        if self.name == "primary" and self.registry.primary_failures_remaining > 0:
            self.registry.primary_failures_remaining -= 1
            raise RuntimeError("primary unavailable")
        return {"out": np.array([1], dtype=np.float32)}


class FakeRegistry:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.primary_failures_remaining = 1

    def get_spec(self, name: str):
        return SimpleNamespace(preprocess=None, postprocess=None)

    async def get_backend(self, name: str):
        return FakeBackend(name, self)


def _item() -> StageItem:
    return StageItem(payload={"feeds": {"input": np.zeros((1, 1), dtype=np.float32)}})


@pytest.mark.asyncio
async def test_infer_retries_primary_after_fallback_backoff_expires():
    registry = FakeRegistry()
    stage = Infer(model="primary", fallback_model="fallback")
    await stage.setup(StageContext(pipeline="pipe", name="infer", registry=registry))

    first = await stage.process(_item())
    assert first.payload["model_selection"] == {"requested": "primary", "used": "fallback"}
    assert registry.calls == ["primary", "fallback"]

    second = await stage.process(_item())
    assert second.payload["model_selection"] == {"requested": "primary", "used": "fallback"}
    assert registry.calls == ["primary", "fallback", "fallback"]

    stage._primary_retry_at = 0.0
    third = await stage.process(_item())
    assert third.payload["model_selection"] == {"requested": "primary", "used": "primary"}
    assert registry.calls == ["primary", "fallback", "fallback", "primary"]
