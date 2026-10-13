"""POST each item to a webhook URL."""
from __future__ import annotations

import httpx

from inferops.obs import get_logger
from inferops.pipeline.stage import SinkStage

log = get_logger(__name__)


class SinkWebhook(SinkStage):
    def __init__(self, url: str, timeout_s: float = 5.0) -> None:
        super().__init__(url=url, timeout_s=timeout_s)
        self.url = url
        self.timeout_s = timeout_s
        self._client: httpx.AsyncClient | None = None

    async def setup(self, ctx):
        await super().setup(ctx)
        self._client = httpx.AsyncClient(timeout=self.timeout_s)

    async def process(self, item):
        try:
            await self._client.post(self.url, json=item.model_dump(mode="json"))
        except Exception as e:
            log.warning("webhook_failed", url=self.url, err=str(e))
        return None

    async def teardown(self):
        if self._client is not None:
            await self._client.aclose()
