"""RTSP source via OpenCV (sync capture in thread)."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import cv2

from inferops.core.types import StageItem, now_ms
from inferops.obs import get_logger
from inferops.pipeline.stage import SourceStage

log = get_logger(__name__)


class SourceRTSP(SourceStage):
    def __init__(self, url: str, stream_id: str = "rtsp", fps: float = 5.0,
                 reconnect_s: float = 2.0) -> None:
        super().__init__(url=url, stream_id=stream_id, fps=fps, reconnect_s=reconnect_s)
        self.url = url
        self.stream_id = stream_id
        self.fps = fps
        self.reconnect_s = reconnect_s

    async def run(self) -> AsyncIterator[StageItem]:
        seq = 0
        period = 1.0 / max(self.fps, 0.1)
        while True:
            cap = await asyncio.to_thread(cv2.VideoCapture, self.url)
            if not cap.isOpened():
                log.warning("rtsp_open_failed", url=self.url)
                await asyncio.sleep(self.reconnect_s)
                continue
            try:
                while True:
                    ok, frame = await asyncio.to_thread(cap.read)
                    if not ok or frame is None:
                        log.warning("rtsp_read_failed", url=self.url)
                        break
                    seq += 1
                    yield StageItem(
                        stream_id=self.stream_id, ts_ms=now_ms(), seq=seq,
                        payload={"image_np": frame, "width": frame.shape[1], "height": frame.shape[0]},
                    )
                    await asyncio.sleep(period)
            finally:
                await asyncio.to_thread(cap.release)
            await asyncio.sleep(self.reconnect_s)
