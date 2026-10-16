"""RF-DETR postprocess — faithful port of the legacy driver-behaviour code.

DETR-family models emit per-query class logits + cxcywh boxes in [0, 1].
Sigmoid the logits, take argmax class+score, threshold, rescale boxes to
original pixel coords, group by class name for the fusion engine.
"""
from __future__ import annotations

import numpy as np

from inferops.domain.models import ModelSpec
from inferops.domain.types import Detection, Frame, Result
from inferops.ports.extensions import Postprocessor

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
                 reject_placeholder_outputs: bool = True,
                 drop_unmapped_classes: bool = True) -> None:
        self.conf = conf_threshold
        self.classes = tuple(class_names) if class_names else None
        self.reject_placeholder_outputs = reject_placeholder_outputs
        # The Triton export emits 7 logit columns for 6 real classes; the extra
        # column is the DETR no-object/background slot. Emitting it produced
        # phantom high-confidence detections labelled "6" on every frame.
        self.drop_unmapped_classes = drop_unmapped_classes

    def _class_names(self, spec: ModelSpec) -> tuple[str, ...]:
        """Explicit param > the spec's own labels > the legacy built-in list.

        Reading ``spec.label_list()`` is what makes ``labels.inline`` in the
        model YAML live configuration rather than documentation.
        """
        if self.classes:
            return self.classes
        from_spec = spec.label_list()
        if from_spec:
            return tuple(from_spec)
        return CLASS_NAMES

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

        classes = self._class_names(spec)
        ow = max(int(frame.width or 1), 1)
        oh = max(int(frame.height or 1), 1)
        dets: list[Detection] = []
        by_class: dict[str, list[dict]] = {}
        for i in range(len(cls_ids)):
            # Background / no-object columns have no label; emitting them as
            # str(index) invented detections the fusion rules then reacted to.
            if self.drop_unmapped_classes and not 0 <= int(cls_ids[i]) < len(classes):
                continue
            cx, cy, w, h = bx[i]
            x1 = float(max(0.0, (cx - w / 2)))
            y1 = float(max(0.0, (cy - h / 2)))
            x2 = float(min(1.0, (cx + w / 2)))
            y2 = float(min(1.0, (cy + h / 2)))
            xyxy_pixels = (x1 * ow, y1 * oh, x2 * ow, y2 * oh)
            c = int(cls_ids[i])
            label = classes[c] if 0 <= c < len(classes) else str(c)
            d = Detection(label=label, score=float(conf[i]), box=(x1, y1, x2, y2),
                          attrs={"orig_w": ow, "orig_h": oh, "xyxy_pixels": xyxy_pixels})
            dets.append(d)
            by_class.setdefault(label, []).append(d.model_dump(mode="json"))

        return Result(kind="detections", detections=dets,
                      raw={"by_class": by_class}, model=spec.name)
