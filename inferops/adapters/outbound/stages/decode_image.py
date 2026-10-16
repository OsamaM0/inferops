"""Decode bytes -> numpy image. Stores result in payload['image_np']."""
from __future__ import annotations

import cv2
import numpy as np

from inferops.adapters.outbound.stages.base import Stage
from inferops.domain.types import StageItem


class DecodeImage(Stage):
    async def process(self, item: StageItem):
        if "image_np" in item.payload:
            img = item.payload["image_np"]
        else:
            data = item.payload.get("image_bytes")
            if data is None:
                return None
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return None
            item.payload["image_np"] = img
        item.payload["width"] = int(img.shape[1])
        item.payload["height"] = int(img.shape[0])
        return item
