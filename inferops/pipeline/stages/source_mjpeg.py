"""MJPEG HTTP source."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import cv2
import httpx
import numpy as np

from inferops.core.types import StageItem, now_ms
from inferops.pipeline.stage import SourceStage


class SourceMJPEG(SourceStage):
    def __init__(self, url: str, stream_id: str = "mjpeg", fps: float = 5.0) -> None:
        super().__init__(url=url, stream_id=stream_id, fps=fps)
        self.url = url
        self.stream_id = stream_id
        self.fps = fps

    async def run(self) -> AsyncIterator[StageItem]:
        period = 1.0 / max(self.fps, 0.1)
        last = 0.0
        seq = 0
        async with httpx.AsyncClient(timeout=None) as client, client.stream("GET", self.url) as r:
            buf = b""
            async for chunk in r.aiter_bytes():
                buf += chunk
                while True:
                    soi = buf.find(b"\xff\xd8")
                    eoi = buf.find(b"\xff\xd9", soi + 2) if soi != -1 else -1
                    if soi == -1 or eoi == -1:
                        break
                    jpg = buf[soi:eoi + 2]
                    buf = buf[eoi + 2:]
                    now = asyncio.get_event_loop().time()
                    if now - last < period:
                        continue
                    last = now
                    arr = np.frombuffer(jpg, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    seq += 1
                    yield StageItem(
                        stream_id=self.stream_id, ts_ms=now_ms(), seq=seq,
                        payload={"image_np": frame, "width": frame.shape[1], "height": frame.shape[0]},
                    )
