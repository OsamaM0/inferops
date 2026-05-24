"""Source consuming a Kafka/Redpanda topic via Transport."""
from __future__ import annotations

from collections.abc import AsyncIterator

import orjson

from inferops.adapters.transport import get_transport
from inferops.core.types import StageItem
from inferops.pipeline.stage import SourceStage


class SourceKafka(SourceStage):
    def __init__(self, topic: str, group: str = "inferops") -> None:
        super().__init__(topic=topic, group=group)
        self.topic = topic
        self.group = group
        self._consumer = None

    async def run(self) -> AsyncIterator[StageItem]:
        t = self.ctx.transport or get_transport()
        self._consumer = await t.consumer(self.group)
        await self._consumer.subscribe([self.topic], self.group)
        async for msg in self._consumer.stream():
            try:
                data = orjson.loads(msg.value)
            except Exception:
                continue
            item = StageItem(**data) if "id" in data else StageItem(payload=data)
            yield item
            await self._consumer.commit(msg)

    async def teardown(self) -> None:
        if self._consumer is not None:
            await self._consumer.close()
            self._consumer = None
