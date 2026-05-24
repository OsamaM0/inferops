"""Pass-through postprocess (no-op)."""
from __future__ import annotations

from inferops.pipeline.stage import Stage


class PostprocessPassthrough(Stage):
    async def process(self, item):
        return item
