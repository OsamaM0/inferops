"""Folder source: yields a StageItem per image file in a directory."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from inferops.adapters.outbound.stages.base import SourceStage
from inferops.domain.types import StageItem, now_ms


class SourceFolder(SourceStage):
    def __init__(self, path: str, glob: str = "*.jpg", interval_ms: int = 0,
                 stream_id: str = "folder") -> None:
        super().__init__(path=path, glob=glob, interval_ms=interval_ms, stream_id=stream_id)
        self.path = Path(path)
        self.glob = glob
        self.interval_ms = interval_ms
        self.stream_id = stream_id

    async def run(self) -> AsyncIterator[StageItem]:
        for seq, p in enumerate(sorted(self.path.glob(self.glob)), start=1):
            data = p.read_bytes()
            yield StageItem(
                stream_id=self.stream_id, ts_ms=now_ms(), seq=seq,
                payload={"image_bytes": data, "filename": p.name},
            )
            if self.interval_ms:
                await asyncio.sleep(self.interval_ms / 1000)
