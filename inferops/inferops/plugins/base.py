"""Plugin ABCs."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np

from inferops.core.types import Frame, Result, StageItem
from inferops.registry.model_spec import ModelSpec


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
