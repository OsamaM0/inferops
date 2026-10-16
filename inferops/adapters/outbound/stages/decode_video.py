"""Decode a video file path into per-frame items (one-shot)."""
from __future__ import annotations

import cv2

from inferops.adapters.outbound.stages.base import Stage
from inferops.domain.types import StageItem, now_ms


class DecodeVideo(Stage):
    def __init__(self, every_n: int = 1) -> None:
        super().__init__(every_n=every_n)
        self.every_n = every_n

    async def process(self, item: StageItem):
        path = item.payload.get("video_path")
        if not path:
            return None
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_dt_ms = round(1000.0 / fps)
        base_ms = item.ts_ms or now_ms()
        items: list[StageItem] = []
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if i % self.every_n != 0:
                continue
            it = StageItem(
                stream_id=item.stream_id or "video",
                ts_ms=base_ms + i * frame_dt_ms,
                seq=i,
                payload={"image_np": frame, "width": frame.shape[1], "height": frame.shape[0]},
            )
            items.append(it)
        cap.release()
        return items
