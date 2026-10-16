"""ImageNet normalize (resize + center crop + mean/std)."""
from __future__ import annotations

import cv2
import numpy as np

from inferops.domain.models import ModelSpec
from inferops.domain.types import Frame
from inferops.ports.extensions import Preprocessor

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ImageNetNormalize(Preprocessor):
    def __init__(self, resize: int = 256, crop: int = 224) -> None:
        self.resize = resize
        self.crop = crop

    async def __call__(self, frame: Frame, spec: ModelSpec):
        img = frame.image
        if img is None:
            raise ValueError("image required")
        h, w = img.shape[:2]
        scale = self.resize / min(h, w)
        nh, nw = round(h * scale), round(w * scale)
        img = cv2.resize(img, (nw, nh))
        y0 = (nh - self.crop) // 2
        x0 = (nw - self.crop) // 2
        img = img[y0:y0 + self.crop, x0:x0 + self.crop]
        in_spec = spec.inputs[0]
        if in_spec.color == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        arr = img.astype(np.float32) / 255.0
        arr = (arr - _MEAN) / _STD
        if in_spec.layout == "NCHW":
            arr = arr.transpose(2, 0, 1)
        arr = arr[None, ...]
        return {in_spec.name: arr.astype(np.float32)}
