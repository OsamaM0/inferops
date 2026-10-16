"""Plugin ABCs."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from inferops.domain.incident_spec import Alert, SinkSpec
from inferops.domain.models import ModelSpec
from inferops.domain.types import Frame, Result, StageItem


class Preprocessor(ABC):
    @abstractmethod
    async def __call__(self, frame: Frame, spec: ModelSpec) -> dict[str, np.ndarray]: ...


class Postprocessor(ABC):
    @abstractmethod
    async def __call__(self, raw: dict[str, np.ndarray], frame: Frame, spec: ModelSpec) -> Result: ...


class Aggregator(ABC):
    @abstractmethod
    async def __call__(self, results: Sequence[Result], ctx: dict[str, Any]) -> StageItem | None: ...


class FrameAnnotator(ABC):
    """Domain-level plugin that draws custom overlays on a BGR frame.

    ``__call__`` receives the frame *copy* (already annotated with generic
    detection boxes by the sink) and the ``StageItem``, and must return the
    (possibly further annotated) frame.
    """
    @abstractmethod
    def __call__(self, frame: np.ndarray, item: StageItem) -> np.ndarray: ...


class ProcessingStep(Protocol):
    async def setup(self, context: Any) -> None: ...
    async def process(self, item: StageItem) -> StageItem | list[StageItem] | None: ...
    async def teardown(self) -> None: ...

class AlertSink:
    def __init__(self, spec: SinkSpec) -> None:
        self.spec = spec
    async def setup(self) -> None: ...
    async def emit(self, alert: Alert) -> None: ...
    async def close(self) -> None: ...
