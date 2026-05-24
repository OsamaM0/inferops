"""Universal decode stage: handles all input types in one place.

Priority order:
  1. ``image_np``   — already a numpy frame (from RTSP / MJPEG sources)
  2. ``image_bytes``— raw JPEG/PNG bytes (from /v1/infer image upload)
  3. ``video_path`` — local video file path (from /v1/run upload or JSON body)

For RTSP / MJPEG live sources the stage is a transparent pass-through because
the source stage already places ``image_np`` in the payload.

For video files the stage fans out: it returns a *list* of StageItems (one per
sampled frame) so the rest of the pipeline processes every frame.
"""
from __future__ import annotations

import cv2
import numpy as np

from inferops.core.types import StageItem, now_ms
from inferops.pipeline.stage import Stage


class DecodeAny(Stage):
    """Decode video / image / raw-numpy — pick whichever is present."""

    def __init__(self, every_n: int = 1) -> None:
        super().__init__(every_n=every_n)
        self.every_n = every_n

    async def process(self, item: StageItem):
        # ── 1. already a numpy frame (RTSP / MJPEG source) ─────────────────
        if "image_np" in item.payload:
            img = item.payload["image_np"]
            item.payload.setdefault("width", img.shape[1])
            item.payload.setdefault("height", img.shape[0])
            return item

        # ── 2. raw image bytes (single frame upload) ────────────────────────
        if "image_bytes" in item.payload:
            data = item.payload["image_bytes"]
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return None
            item.payload["image_np"] = img
            item.payload["width"] = img.shape[1]
            item.payload["height"] = img.shape[0]
            return item

        # ── 3. video file path ───────────────────────────────────────────────
        path = item.payload.get("video_path")
        if path:
            return self._decode_video(item, path)

        return None  # nothing to decode

    # ── helpers ──────────────────────────────────────────────────────────────

    def _decode_video(self, item: StageItem, path: str) -> list[StageItem] | None:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_dt_ms = round(1000.0 / fps)
        base_ms = item.ts_ms or now_ms()
        frames: list[StageItem] = []
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if i % self.every_n != 0:
                continue
            frames.append(StageItem(
                stream_id=item.stream_id,
                ts_ms=base_ms + i * frame_dt_ms,
                seq=i,
                payload={
                    "image_np": frame,
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                },
            ))
        cap.release()
        return frames or None
