"""Durable ownership of a statically assigned processing scope."""
from __future__ import annotations

from typing import Protocol


class OwnershipLost(RuntimeError):
    """This instance cannot mutate its shard until a new exclusive claim."""


class Ownership(Protocol):
    scope: str
    instance: str
    generation: int

    async def renew_ownership(self) -> None: ...
    async def release_ownership(self) -> None: ...
