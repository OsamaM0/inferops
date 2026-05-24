"""Redpanda/Kafka transport via aiokafka."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from aiokafka.errors import TopicAlreadyExistsError

from inferops.adapters.transport.base import Consumer, Message, Producer, Transport
from inferops.core.config import settings
from inferops.core.errors import PayloadTooLarge, TransportError
from inferops.obs import TRANSPORT_CONSUME, TRANSPORT_PRODUCE, get_logger

log = get_logger(__name__)


class _RPProducer:
    def __init__(self, p: AIOKafkaProducer, max_bytes: int) -> None:
        self._p = p
        self._max = max_bytes

    async def send(self, topic, value, *, key=None, headers=None):
        if len(value) > self._max:
            TRANSPORT_PRODUCE.labels(topic=topic, result="too_large").inc()
            raise PayloadTooLarge(f"value {len(value)}B > limit {self._max}B (use Redis/S3 ref)")
        hdrs = [(k, v) for k, v in (headers or {}).items()]
        try:
            await self._p.send_and_wait(topic, value=value, key=key, headers=hdrs)
            TRANSPORT_PRODUCE.labels(topic=topic, result="ok").inc()
        except Exception as e:
            TRANSPORT_PRODUCE.labels(topic=topic, result="error").inc()
            raise TransportError(str(e)) from e

    async def flush(self): await self._p.flush()
    async def close(self): await self._p.stop()


class _RPConsumer:
    def __init__(self, bootstrap: str, client_id: str) -> None:
        self._bootstrap = bootstrap
        self._client_id = client_id
        self._c: AIOKafkaConsumer | None = None
        self._group = ""

    async def subscribe(self, topics, group):
        self._group = group
        self._c = AIOKafkaConsumer(
            *topics, bootstrap_servers=self._bootstrap,
            client_id=self._client_id, group_id=group,
            enable_auto_commit=False, auto_offset_reset="earliest",
        )
        await self._c.start()

    def stream(self) -> AsyncIterator[Message]:
        async def gen():
            assert self._c is not None
            async for r in self._c:
                TRANSPORT_CONSUME.labels(topic=r.topic, group=self._group).inc()
                yield Message(
                    topic=r.topic, value=r.value, key=r.key,
                    headers={k: v for k, v in (r.headers or [])},
                    partition=r.partition, offset=r.offset, ts_ms=r.timestamp,
                )
        return gen()

    async def commit(self, msg):
        if self._c is not None:
            await self._c.commit()

    async def close(self):
        if self._c is not None:
            await self._c.stop()
            self._c = None


class RedpandaTransport(Transport):
    def __init__(self, bootstrap: str | None = None, client_id: str | None = None,
                 max_value_bytes: int | None = None) -> None:
        self._bootstrap = bootstrap or settings.transport.bootstrap
        self._client_id = client_id or settings.transport.client_id
        self._max = max_value_bytes or settings.transport.max_value_bytes
        self._producer: AIOKafkaProducer | None = None
        self._lock = asyncio.Lock()

    async def producer(self, name: str) -> Producer:
        async with self._lock:
            if self._producer is None:
                p = AIOKafkaProducer(
                    bootstrap_servers=self._bootstrap,
                    client_id=f"{self._client_id}-{name}",
                    enable_idempotence=True,
                    acks="all",
                    linger_ms=5, max_request_size=2 * 1024 * 1024,
                )
                await p.start()
                self._producer = p
        return _RPProducer(self._producer, self._max)

    async def consumer(self, name: str) -> Consumer:
        return _RPConsumer(self._bootstrap, f"{self._client_id}-{name}")

    async def admin_ensure_topic(self, topic, *, partitions=1, retention_ms=86_400_000):
        admin = AIOKafkaAdminClient(bootstrap_servers=self._bootstrap, client_id=self._client_id)
        await admin.start()
        try:
            with suppress(TopicAlreadyExistsError):
                await admin.create_topics([NewTopic(
                    name=topic, num_partitions=partitions, replication_factor=1,
                    topic_configs={"retention.ms": str(retention_ms)},
                )])
        finally:
            await admin.close()

    async def health(self) -> bool:
        try:
            admin = AIOKafkaAdminClient(bootstrap_servers=self._bootstrap, client_id=self._client_id)
            await admin.start()
            await admin.close()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        if self._producer is not None:
            await self._producer.stop()
            self._producer = None
