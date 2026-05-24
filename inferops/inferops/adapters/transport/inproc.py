"""In-process Transport backed by asyncio.Queue (tests + single-process pipelines)."""
from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import suppress

from inferops.adapters.transport.base import Consumer, Message, Producer, Transport
from inferops.core.config import settings
from inferops.core.errors import PayloadTooLarge


class _InProcProducer:
    def __init__(self, bus: InProcTransport) -> None:
        self._bus = bus

    async def send(self, topic, value, *, key=None, headers=None):
        if len(value) > self._bus.max_value_bytes:
            raise PayloadTooLarge(
                f"value {len(value)}B > limit {self._bus.max_value_bytes}B (use Redis/S3 ref)"
            )
        await self._bus._deliver(Message(topic=topic, value=value, key=key, headers=headers or {}))

    async def flush(self) -> None: ...
    async def close(self) -> None: ...


class _InProcConsumer:
    def __init__(self, bus: InProcTransport) -> None:
        self._bus = bus
        self._queue: asyncio.Queue[Message] = asyncio.Queue(maxsize=10_000)
        self._topics: list[str] = []
        self._group: str = ""
        self._closed = False

    async def subscribe(self, topics, group):
        self._topics = list(topics)
        self._group = group
        for t in topics:
            self._bus._register(t, group, self._queue)

    def stream(self) -> AsyncIterator[Message]:
        async def gen():
            while not self._closed:
                msg = await self._queue.get()
                yield msg
        return gen()

    async def commit(self, msg) -> None: ...
    async def close(self) -> None:
        self._closed = True
        for t in self._topics:
            self._bus._unregister(t, self._group, self._queue)


class InProcTransport(Transport):
    """Round-robin per group, fanout across groups."""

    def __init__(self, max_value_bytes: int | None = None) -> None:
        # topic -> group -> list[Queue]
        self._subs: dict[str, dict[str, list[asyncio.Queue[Message]]]] = defaultdict(lambda: defaultdict(list))
        self._rr: dict[tuple[str, str], int] = defaultdict(int)
        self.max_value_bytes = max_value_bytes or settings.transport.max_value_bytes

    async def producer(self, name: str) -> Producer:
        return _InProcProducer(self)

    async def consumer(self, name: str) -> Consumer:
        return _InProcConsumer(self)

    async def admin_ensure_topic(self, topic, *, partitions=1, retention_ms=86_400_000) -> None: ...

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        self._subs.clear()

    def _register(self, topic, group, q):
        self._subs[topic][group].append(q)

    def _unregister(self, topic, group, q):
        with suppress(ValueError):
            self._subs[topic][group].remove(q)

    async def _deliver(self, msg: Message) -> None:
        for group, queues in list(self._subs.get(msg.topic, {}).items()):
            if not queues:
                continue
            idx = self._rr[(msg.topic, group)] % len(queues)
            self._rr[(msg.topic, group)] = idx + 1
            await queues[idx].put(msg)
