"""Generic YOLO-style decode (xywh + objectness*class)."""
from __future__ import annotations

import numpy as np

from inferops.core.types import Detection, Frame, Result
from inferops.plugins.base import Postprocessor
from inferops.registry.model_spec import ModelSpec


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou < iou_thr]
    return keep


class YoloDecode(Postprocessor):
    def __init__(self, conf_threshold: float = 0.25, nms_iou: float = 0.45, max_det: int = 300,
                 layout: str = "auto", objectness: bool | None = None,
                 class_agnostic: bool = False) -> None:
        if layout not in {"auto", "channels_first", "channels_last"}:
            raise ValueError("layout must be auto, channels_first, or channels_last")
        self.layout = layout
        self.objectness = objectness
        self.class_agnostic = class_agnostic
        self.conf = conf_threshold
        self.iou = nms_iou
        self.max_det = max_det

    async def __call__(self, raw, frame: Frame, spec: ModelSpec) -> Result:
        out = next(iter(raw.values()))
        if out.ndim != 3 or out.shape[0] != 1:
            raise ValueError("YOLO decoder expects [1,N,4+C] or [1,4+C,N] (optional objectness)")
        labels = spec.label_list()
        channels = {4 + len(labels), 5 + len(labels)} if labels else set()
        first = self.layout == "channels_first"
        if self.layout == "auto":
            if out.shape[1] in channels and out.shape[2] not in channels:
                first = True
            elif out.shape[2] in channels and out.shape[1] not in channels:
                first = False
            elif out.shape[1] in channels and out.shape[2] in channels:
                raise ValueError("ambiguous YOLO output layout; specify layout explicitly")
            else:
                first = out.shape[1] < out.shape[2]
        if first:
            out = out.transpose(0, 2, 1)
        preds = out[0]
        boxes_xywh, scores_obj_cls = preds[:, :4], preds[:, 4:]
        has_obj = self.objectness
        if has_obj is None:
            has_obj = bool(labels) and scores_obj_cls.shape[1] == len(labels) + 1
        cls_scores = scores_obj_cls[:, 1:] * scores_obj_cls[:, :1] if has_obj else scores_obj_cls
        if cls_scores.shape[1] == 0:
            raise ValueError("YOLO output has no class scores")
        if labels and cls_scores.shape[1] != len(labels):
            raise ValueError("YOLO class count does not match configured labels")
        class_ids = cls_scores.argmax(axis=1)
        scores = cls_scores.max(axis=1)
        mask = scores >= self.conf
        boxes_xywh, scores, class_ids = boxes_xywh[mask], scores[mask], class_ids[mask]
        if boxes_xywh.size == 0:
            return Result(kind="detections", detections=[], model=spec.name)
        x, y, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        if self.class_agnostic:
            keep = _nms(boxes, scores, self.iou)
        else:
            keep = []
            for cid in np.unique(class_ids):
                indices = np.where(class_ids == cid)[0]
                keep.extend(int(indices[i]) for i in _nms(boxes[indices], scores[indices], self.iou))
            keep.sort(key=lambda i: float(scores[i]), reverse=True)
        keep = keep[:self.max_det]
        lb = frame.meta.get("letterbox") or {}
        r = lb.get("r", 1.0)
        pad_x = lb.get("pad_x", 0)
        pad_y = lb.get("pad_y", 0)
        ow = lb.get("orig_w", frame.width or 1)
        oh = lb.get("orig_h", frame.height or 1)
        labels = spec.label_list()
        dets = []
        for i in keep:
            bx1 = max(0.0, (boxes[i, 0] - pad_x) / max(r, 1e-9) / ow)
            by1 = max(0.0, (boxes[i, 1] - pad_y) / max(r, 1e-9) / oh)
            bx2 = min(1.0, (boxes[i, 2] - pad_x) / max(r, 1e-9) / ow)
            by2 = min(1.0, (boxes[i, 3] - pad_y) / max(r, 1e-9) / oh)
            cid = int(class_ids[i])
            dets.append(Detection(
                label=labels[cid] if cid < len(labels) else str(cid),
                score=float(scores[i]),
                box=(bx1, by1, bx2, by2),
                attrs={"class_id": cid},
            ))
        return Result(kind="detections", detections=dets, model=spec.name)
