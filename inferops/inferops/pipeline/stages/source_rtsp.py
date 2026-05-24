"""RTSP source via OpenCV – dedicated reader thread with always-latest frame.

The reader thread captures frames as fast as the camera delivers them and
stores only the **most recent** frame.  The async generator yields that frame
each time a new one arrives, so:
- There is no per-frame thread-pool overhead.
- When inference is slower than the camera, old frames are silently discarded
  (frame-drop), keeping end-to-end latency bounded.
- ``fps > 0`` caps the *producer* side (frames captured per second); use
  ``fps=0`` for fully uncapped capture.
"""
from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator

import cv2
import numpy as np

from inferops.core.types import StageItem, now_ms
from inferops.obs import get_logger
from inferops.pipeline.stage import SourceStage

log = get_logger(__name__)


class SourceRTSP(SourceStage):
    def __init__(self, url: str, stream_id: str = "rtsp", fps: float = 0.0,
                 reconnect_s: float = 2.0) -> None:
        super().__init__(url=url, stream_id=stream_id, fps=fps, reconnect_s=reconnect_s)
        self.url = url
        self.stream_id = stream_id
        self.fps = fps          # 0 = uncapped; >0 = producer-side rate cap
        self.reconnect_s = reconnect_s

    async def run(self) -> AsyncIterator[StageItem]:
        loop = asyncio.get_running_loop()

        # Shared state: always-latest frame.  Assignment is atomic under the GIL
        # in CPython, so no explicit lock is needed.
        _latest: list[np.ndarray | None] = [None]
        _new_frame = asyncio.Event()
        _running = [True]

        def _reader() -> None:
            min_period = (1.0 / self.fps) if self.fps > 0 else 0.0
            while _running[0]:
                cap = cv2.VideoCapture(self.url)
                if not cap.isOpened():
                    log.warning("rtsp_open_failed", url=self.url)
                    time.sleep(self.reconnect_s)
                    continue
                try:
                    last_t = 0.0
                    while _running[0]:
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            log.warning("rtsp_read_failed", url=self.url)
                            break
                        now = time.monotonic()
                        if min_period > 0 and (now - last_t) < min_period:
                            continue   # discard frame: producer-side rate limit
                        last_t = now
                        _latest[0] = frame  # atomic in CPython (GIL protects ptr swap)
                        loop.call_soon_threadsafe(_new_frame.set)
                finally:
                    cap.release()
                if _running[0]:
                    time.sleep(self.reconnect_s)

        t = threading.Thread(target=_reader, daemon=True)
        t.start()

        seq = 0
        try:
            while True:
                await _new_frame.wait()
                _new_frame.clear()
                frame = _latest[0]
                if frame is None:
                    continue
                seq += 1
                yield StageItem(
                    stream_id=self.stream_id, ts_ms=now_ms(), seq=seq,
                    payload={"image_np": frame, "width": frame.shape[1], "height": frame.shape[0]},
                )
        finally:
            _running[0] = False
