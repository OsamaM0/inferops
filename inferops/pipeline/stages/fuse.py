"""Pipeline stage wrapper around FusionEngine."""
from __future__ import annotations

from inferops.pipeline.stage import Stage


class Fuse(Stage):
    def __init__(self, fusion: str) -> None:
        super().__init__(fusion=fusion)
        self.fusion_name = fusion
        self._engine = None

    async def setup(self, ctx):
        await super().setup(ctx)
        from inferops.fusion.store import get_fusion_store
        self._engine = get_fusion_store().get(self.fusion_name)

    async def process(self, item):
        # Pass the pipeline name so persisted events carry the same identifier
        # sink_postgres uses, which is what GET /v1/events filters on.
        events = await self._engine.ingest_item(item, pipeline=self.ctx.pipeline)
        if events:
            item.payload.setdefault("fusion_events", []).extend(
                [e.model_dump(mode="json") for e in events]
            )
        return item
