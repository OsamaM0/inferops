"""Stage ABC + context."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, ClassVar

from inferops.core.types import StageItem
from inferops.obs import STAGE_LATENCY


@dataclass(slots=True)
class StageContext:
    pipeline: str
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    registry: Any = None
    transport: Any = None
    extra: dict[str, Any] = field(default_factory=dict)


class Stage(ABC):
    name: str = "stage"
    inputs: ClassVar[list[str]] = []
    outputs: ClassVar[list[str]] = []

    def __init__(self, **params: Any) -> None:
        self.params = params

    async def setup(self, ctx: StageContext) -> None:
        self.ctx = ctx

    async def teardown(self) -> None:
        return None

    async def finalize(self) -> None:
        """Called once after all items are processed (e.g. flush buffered video clips)."""
        return None

    @abstractmethod
    async def process(self, item: StageItem) -> StageItem | list[StageItem] | None: ...

    async def __call__(self, item: StageItem) -> StageItem | list[StageItem] | None:
        t0 = time.perf_counter()
        result: "StageItem | list[StageItem] | None" = None
        try:
            result = await self.process(item)
        finally:
            elapsed_s = time.perf_counter() - t0
            STAGE_LATENCY.labels(pipeline=self.ctx.pipeline, stage=self.ctx.name).observe(elapsed_s)
            key = f"stage_ms.{self.ctx.name}"
            elapsed_str = f"{elapsed_s * 1000:.1f}"
            if isinstance(result, list):
                for it in result:
                    it.trace[key] = elapsed_str
            elif result is not None:
                result.trace[key] = elapsed_str
        return result


class SourceStage(Stage):
    """Pushes items into the pipeline."""
    @abstractmethod
    async def run(self) -> AsyncIterator[StageItem]: ...

    async def process(self, item: StageItem):
        return item


class SinkStage(Stage):
    """Terminal stage; returns None."""
