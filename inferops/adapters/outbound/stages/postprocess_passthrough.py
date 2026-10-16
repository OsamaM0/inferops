"""Pass-through postprocess (no-op)."""
from __future__ import annotations

from inferops.adapters.outbound.stages.base import Stage


class PostprocessPassthrough(Stage):
    async def process(self, item):
        return item
