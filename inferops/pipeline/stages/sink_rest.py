"""Stash results in a shared dict (used by sync REST handler)."""
from __future__ import annotations

from inferops.pipeline.stage import SinkStage


class SinkREST(SinkStage):
    async def process(self, item):
        # The runner returns items; SinkREST simply makes sure it's a no-op terminus.
        return None
