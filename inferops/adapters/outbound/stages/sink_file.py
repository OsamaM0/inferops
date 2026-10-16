"""Append each item to a JSON-lines file."""
from __future__ import annotations

import asyncio
from pathlib import Path

import orjson

from inferops.adapters.outbound.stages.base import SinkStage


class SinkFile(SinkStage):
    def __init__(self, path: str) -> None:
        super().__init__(path=path)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def process(self, item):
        async with self._lock:
            with self.path.open("ab") as f:
                f.write(orjson.dumps(item.model_dump(mode="json")) + b"\n")
        return None
