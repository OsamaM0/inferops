"""DETR-style decode: (boxes [N,4] in xyxy normalized, logits [N,C+1] or scores)."""
from __future__ import annotations

import numpy as np

from inferops.core.types import Detection, Frame, Result
from inferops.plugins.base import Postprocessor
from inferops.registry.model_spec import ModelSpec


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class DetrDecode(Postprocessor):
    def __init__(self, conf_threshold: float = 0.5, boxes_out: str = "boxes",
                 logits_out: str = "logits", background_index: int | None = -1) -> None:
        self.conf = conf_threshold
        self.boxes_out = boxes_out
        self.logits_out = logits_out
        self.bg = background_index

    async def __call__(self, raw, frame: Frame, spec: ModelSpec) -> Result:
        boxes = raw.get(self.boxes_out)
        logits = raw.get(self.logits_out)
        if boxes is None or logits is None:
            keys = list(raw.keys())
            # heuristic: 4-channel = boxes
            for k in keys:
                if raw[k].ndim >= 2 and raw[k].shape[-1] == 4 and boxes is None:
                    boxes = raw[k]
                elif logits is None:
                    logits = raw[k]
        if boxes.ndim == 3:
            boxes = boxes[0]
        if logits.ndim == 3:
            logits = logits[0]
        probs = _softmax(logits, axis=-1)
        probs_no_bg = np.delete(probs, self.bg, axis=-1) if self.bg is not None else probs
        scores = probs_no_bg.max(axis=-1)
        cls = probs_no_bg.argmax(axis=-1)
        mask = scores >= self.conf
        labels = spec.label_list()
        dets = []
        for i in np.where(mask)[0]:
            b = boxes[i]
            if b.max() > 1.5:  # likely xyxy in pixels
                ow = frame.width or 1
                oh = frame.height or 1
                b = np.array([b[0] / ow, b[1] / oh, b[2] / ow, b[3] / oh])
            else:
                # cxcywh normalized -> xyxy normalized
                cx, cy, w, h = b
                b = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            cid = int(cls[i])
            dets.append(Detection(
                label=labels[cid] if cid < len(labels) else str(cid),
                score=float(scores[i]),
                box=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                attrs={"class_id": cid},
            ))
        return Result(kind="detections", detections=dets, model=spec.name)
