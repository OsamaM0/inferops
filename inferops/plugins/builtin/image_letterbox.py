"""Letterbox resize keeping aspect ratio (for YOLO/DETR style models)."""
from __future__ import annotations

import asyncio

import cv2
import numpy as np

from inferops.core.types import Frame
from inferops.plugins.base import Preprocessor
from inferops.registry.model_spec import ModelSpec


def letterbox(img: np.ndarray, size: int, pad_value: int = 114) -> tuple[np.ndarray, float, tuple[int, int]]:
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    pad_x = (size - nw) // 2
    pad_y = (size - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, r, (pad_x, pad_y)


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Letterbox(Preprocessor):
    def __init__(self, size: int = 640, pad_value: int = 114,
                 normalize: bool = False) -> None:
        self.size = size
        self.pad_value = pad_value
        self.normalize = normalize

    async def __call__(self, frame: Frame, spec: ModelSpec):
        img = frame.image
        if img is None:
            raise ValueError("image required")
        in_spec = spec.inputs[0]
        size = self.size
        pad_value = self.pad_value
        normalize = self.normalize

        def _process() -> tuple[np.ndarray, float, int, int]:
            canvas, r, (pad_x, pad_y) = letterbox(img, size, pad_value)
            if in_spec.color == "RGB":
                canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            arr = canvas.astype(np.float32) / 255.0
            if normalize:
                arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
            if in_spec.layout == "NCHW":
                arr = arr.transpose(2, 0, 1)
            return arr[None, ...].astype(np.float32), r, pad_x, pad_y

        arr, r, pad_x, pad_y = await asyncio.to_thread(_process)
        frame.meta["letterbox"] = {"r": r, "pad_x": pad_x, "pad_y": pad_y, "size": self.size,
                                   "orig_w": img.shape[1], "orig_h": img.shape[0]}
        return {in_spec.name: arr}
