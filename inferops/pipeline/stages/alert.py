"""Pipeline stage wrapper around AlertEngine."""
from __future__ import annotations

from inferops.pipeline.stage import Stage


class Alert(Stage):
    def __init__(self, alerts: str) -> None:
        super().__init__(alerts=alerts)
        self.alerts_name = alerts
        self._engine = None

    async def setup(self, ctx):
        await super().setup(ctx)
        from inferops.alerts.store import get_alert_store
        self._engine = get_alert_store().get(self.alerts_name)

    async def process(self, item):
        emitted = await self._engine.ingest_item(item)
        if emitted:
            item.payload.setdefault("alerts", []).extend(
                [a.model_dump(mode="json") for a in emitted]
            )
        return item
