"""RF-DETR postprocess — faithful port of the legacy driver-behaviour code.

DETR-family models emit per-query class logits + cxcywh boxes in [0, 1].
Sigmoid the logits, take argmax class+score, threshold, rescale boxes to
original pixel coords, group by class name for the fusion engine.
"""
from __future__ import annotations

import numpy as np

from inferops.core.types import Detection, Frame, Result
from inferops.plugins.base import Postprocessor
from inferops.registry.model_spec import ModelSpec

# Keep the serialized labels compatible with the original driver model output.
CLASS_NAMES = ("DriverBehaviour", "cigarette", "foodItem", "phone", "seatbelt", "wheel")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float32, copy=False)))


def _split_outputs(raw: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    logits = boxes = None
    for arr in raw.values():
        if arr.ndim == 3:
            if arr.shape[-1] == 4 and boxes is None:
                boxes = arr
            elif logits is None:
                logits = arr
    if logits is None or boxes is None:
        vs = list(raw.values())
        logits, boxes = vs[0], vs[1]
    return logits, boxes


def _looks_like_placeholder_output(logits: np.ndarray, boxes: np.ndarray) -> bool:
    """Detect tiny constant demo ONNX models that ignore the input frame."""
    if logits.ndim != 2 or boxes.ndim != 2 or logits.shape[0] < 10:
        return False
    logit_rows = np.unique(np.round(logits, 5), axis=0).shape[0]
    box_rows = np.unique(np.round(boxes, 5), axis=0).shape[0]
    return logit_rows <= 3 and box_rows <= 3


class RFDetrPostprocess(Postprocessor):
    def __init__(self, conf_threshold: float = 0.35,
                 class_names: list[str] | None = None,
                 reject_placeholder_outputs: bool = True) -> None:
        self.conf = conf_threshold
        self.classes = tuple(class_names) if class_names else CLASS_NAMES
        self.reject_placeholder_outputs = reject_placeholder_outputs

    async def __call__(self, raw, frame: Frame, spec: ModelSpec) -> Result:
        logits_b, boxes_b = _split_outputs(raw)
        logits, boxes = logits_b[0], boxes_b[0]
        if self.reject_placeholder_outputs and _looks_like_placeholder_output(logits, boxes):
            return Result(
                kind="detections",
                detections=[],
                raw={"by_class": {}, "warning": "placeholder_model_output"},
                model=spec.name,
            )
        scores = _sigmoid(logits)                 # [Q, C]
        cls_ids = scores.argmax(axis=-1)
        conf = scores[np.arange(len(cls_ids)), cls_ids]
        keep = conf >= self.conf
        cls_ids, conf, bx = cls_ids[keep], conf[keep], boxes[keep]

        ow = max(int(frame.width or 1), 1)
        oh = max(int(frame.height or 1), 1)
        dets: list[Detection] = []
        by_class: dict[str, list[dict]] = {}
        for i in range(len(cls_ids)):
            cx, cy, w, h = bx[i]
            x1 = float(max(0.0, (cx - w / 2)))
            y1 = float(max(0.0, (cy - h / 2)))
            x2 = float(min(1.0, (cx + w / 2)))
            y2 = float(min(1.0, (cy + h / 2)))
            xyxy_pixels = (x1 * ow, y1 * oh, x2 * ow, y2 * oh)
            c = int(cls_ids[i])
            label = self.classes[c] if 0 <= c < len(self.classes) else str(c)
            d = Detection(label=label, score=float(conf[i]), box=(x1, y1, x2, y2),
                          attrs={"orig_w": ow, "orig_h": oh, "xyxy_pixels": xyxy_pixels})
            dets.append(d)
            by_class.setdefault(label, []).append(d.model_dump(mode="json"))

        return Result(kind="detections", detections=dets,
                      raw={"by_class": by_class}, model=spec.name)
