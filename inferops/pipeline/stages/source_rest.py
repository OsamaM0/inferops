"""Sync request source: stage that just passes a user-provided item through."""
from __future__ import annotations

from collections.abc import AsyncIterator

from inferops.core.types import StageItem
from inferops.pipeline.stage import SourceStage


class SourceREST(SourceStage):
    """Used by /v1/infer/{pipeline}: caller supplies StageItem via run_one()."""

    async def run(self) -> AsyncIterator[StageItem]:
        if False:  # pragma: no cover - never iterated for sync calls
            yield  # type: ignore[unreachable]
