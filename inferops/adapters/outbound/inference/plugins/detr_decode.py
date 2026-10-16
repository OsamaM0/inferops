"""DETR-style decode: (boxes [N,4] in xyxy normalized, logits [N,C+1] or scores)."""
from __future__ import annotations

import numpy as np

from inferops.domain.models import ModelSpec
from inferops.domain.types import Detection, Frame, Result
from inferops.ports.extensions import Postprocessor


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class DetrDecode(Postprocessor):
    def __init__(self, conf_threshold: float = 0.5, boxes_out: str = "boxes",
                 logits_out: str = "logits", background_index: int | None = -1,
                 activation: str = "softmax", box_format: str = "auto") -> None:
        if activation not in {"softmax", "sigmoid", "none"}:
            raise ValueError("activation must be softmax, sigmoid, or none")
        if box_format not in {"auto", "cxcywh", "xyxy"}:
            raise ValueError("box_format must be auto, cxcywh, or xyxy")
        self.activation = activation
        self.box_format = box_format
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
        if boxes is None or logits is None:
            raise ValueError(f"expected boxes and logits outputs; got {list(raw)}")
        if (boxes.ndim == 3 and boxes.shape[0] != 1) or (logits.ndim == 3 and logits.shape[0] != 1):
            raise ValueError("DETR decoder expects one image per call")
        if boxes.ndim == 3:
            boxes = boxes[0]
        if logits.ndim == 3:
            logits = logits[0]
        if boxes.ndim != 2 or boxes.shape[-1] != 4 or logits.ndim != 2 or len(boxes) != len(logits):
            raise ValueError("expected boxes [Q,4] and logits [Q,C] with matching query counts")
        probs = (_softmax(logits, axis=-1) if self.activation == "softmax" else
                 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80))) if self.activation == "sigmoid" else logits)
        probs_no_bg = np.delete(probs, self.bg, axis=-1) if self.bg is not None else probs
        scores = probs_no_bg.max(axis=-1)
        cls = probs_no_bg.argmax(axis=-1)
        mask = scores >= self.conf
        labels = spec.label_list()
        dets = []
        for i in np.where(mask)[0]:
            b = boxes[i]
            if self.box_format == "auto" and b.max() > 1.5:  # likely xyxy in pixels
                ow = frame.width or 1
                oh = frame.height or 1
                b = np.array([b[0] / ow, b[1] / oh, b[2] / ow, b[3] / oh])
            elif self.box_format != "xyxy":
                # cxcywh normalized -> xyxy normalized
                cx, cy, w, h = b
                b = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            b = np.clip(b, 0.0, 1.0)
            cid = int(cls[i])
            if self.bg is not None:
                bg = self.bg % probs.shape[-1]
                cid += int(cid >= bg)
            dets.append(Detection(
                label=labels[cid] if cid < len(labels) else str(cid),
                score=float(scores[i]),
                box=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                attrs={"class_id": cid},
            ))
        return Result(kind="detections", detections=dets, model=spec.name)
