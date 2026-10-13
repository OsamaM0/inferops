"""Publish StageItem JSON to a Kafka/Redpanda topic."""
from __future__ import annotations

import orjson

from inferops.adapters.transport import get_transport
from inferops.pipeline.stage import SinkStage


class SinkKafka(SinkStage):
    def __init__(self, topic: str, key: str | None = None,
                 ensure_topic: bool = True, partitions: int = 1,
                 retention_ms: int = 86_400_000) -> None:
        super().__init__(topic=topic, key=key, ensure_topic=ensure_topic,
                         partitions=partitions, retention_ms=retention_ms)
        self.topic = topic
        self.key = key
        self.ensure_topic = ensure_topic
        self.partitions = partitions
        self.retention_ms = retention_ms
        self._prod = None

    async def setup(self, ctx):
        await super().setup(ctx)
        t = ctx.transport or get_transport()
        if self.ensure_topic:
            await t.admin_ensure_topic(self.topic, partitions=self.partitions, retention_ms=self.retention_ms)
        self._prod = await t.producer(f"sink-{self.topic}")

    async def process(self, item):
        if self._prod is None:
            return None
        value = orjson.dumps(item.model_dump(mode="json"))
        k = (self.key or item.stream_id or "").encode() if (self.key or item.stream_id) else None
        await self._prod.send(self.topic, value, key=k)
        return None

    async def teardown(self):
        if self._prod is not None:
            await self._prod.flush()
            await self._prod.close()
            self._prod = None
