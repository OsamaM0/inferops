"""Plain stretch-resize preprocessor (no aspect-ratio padding).

This is the correct partner for postprocessors that map normalised box
coordinates straight back onto the original frame — e.g.
``examples.driver_behaviour.plugins.postprocess_rfdetr:RFDetrPostprocess``,
which does::

    x1_px = (cx - w / 2) * frame.width

That mapping is only valid when the whole frame was stretched into the model's
square input. ``image_letterbox:Letterbox`` pads to preserve aspect ratio, so
pairing it with such a postprocessor silently shifts and scales every box.
Use ``Letterbox`` only with decoders that read back ``frame.meta["letterbox"]``.

The target size is taken from the ModelSpec input shape by default, so a single
config entry stays correct if the model is re-exported at another resolution.
"""
from __future__ import annotations

import asyncio

import cv2
import numpy as np

from inferops.core.types import Frame
from inferops.plugins.base import Preprocessor
from inferops.registry.model_spec import ModelSpec

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _size_from_spec(spec: ModelSpec, fallback: int) -> tuple[int, int]:
    """Return (height, width) taken from the first input's static dims."""
    if not spec.inputs:
        return fallback, fallback
    shape = list(spec.inputs[0].shape or [])
    if len(shape) < 4:
        return fallback, fallback
    layout = spec.inputs[0].layout
    h, w = (shape[2], shape[3]) if layout == "NCHW" else (shape[1], shape[2])
    h = h if isinstance(h, int) and h > 0 else fallback
    w = w if isinstance(w, int) and w > 0 else fallback
    return h, w


class ResizeNormalize(Preprocessor):
    """Resize to the model input size, scale to [0, 1], optionally normalise.

    Params:
        size:       explicit square size; ``None`` (default) reads it from the spec
        normalize:  apply ImageNet mean/std after scaling to [0, 1]
        fallback:   size used when neither ``size`` nor the spec provides one
    """

    def __init__(self, size: int | None = None, normalize: bool = False,
                 fallback: int = 640) -> None:
        self.size = size
        self.normalize = normalize
        self.fallback = fallback

    async def __call__(self, frame: Frame, spec: ModelSpec):
        img = frame.image
        if img is None:
            raise ValueError("image required")
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"expected HxWx3 BGR image, got shape {img.shape}")
        in_spec = spec.inputs[0]
        if self.size is not None:
            th = tw = self.size
        else:
            th, tw = _size_from_spec(spec, self.fallback)
        normalize = self.normalize

        def _process() -> np.ndarray:
            resized = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
            if in_spec.color == "RGB":
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            elif in_spec.color == "GRAY":
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)[..., None]
            arr = resized.astype(np.float32) / 255.0
            if normalize and arr.shape[-1] == 3:
                arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
            if in_spec.layout == "NCHW":
                arr = arr.transpose(2, 0, 1)
            return np.ascontiguousarray(arr[None, ...], dtype=np.float32)

        arr = await asyncio.to_thread(_process)
        # Recorded so downstream stages can reason about the mapping. Unlike
        # letterbox there is no padding, so the inverse is a pure per-axis scale.
        frame.meta["resize"] = {
            "mode": "stretch",
            "in_w": tw, "in_h": th,
            "orig_w": int(img.shape[1]), "orig_h": int(img.shape[0]),
        }
        return {in_spec.name: arr}
