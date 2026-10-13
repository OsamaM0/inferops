"""Broker-agnostic transport ABCs."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(slots=True)
class Message:
    topic: str
    value: bytes
    key: bytes | None = None
    headers: dict[str, bytes] = field(default_factory=dict)
    partition: int | None = None
    offset: int | None = None
    ts_ms: int | None = None


class Producer(Protocol):
    async def send(self, topic: str, value: bytes, *,
                   key: bytes | None = None,
                   headers: dict[str, bytes] | None = None) -> None: ...
    async def flush(self) -> None: ...
    async def close(self) -> None: ...


class Consumer(Protocol):
    async def subscribe(self, topics: list[str], group: str) -> None: ...
    def stream(self) -> AsyncIterator[Message]: ...
    async def commit(self, msg: Message) -> None: ...
    async def close(self) -> None: ...


class Transport(ABC):
    @abstractmethod
    async def producer(self, name: str) -> Producer: ...
    @abstractmethod
    async def consumer(self, name: str) -> Consumer: ...
    @abstractmethod
    async def admin_ensure_topic(self, topic: str, *, partitions: int = 1,
                                 retention_ms: int = 86_400_000) -> None: ...
    @abstractmethod
    async def health(self) -> bool: ...
    @abstractmethod
    async def close(self) -> None: ...
