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
    def __init__(self, conf_threshold: float = 0.25, nms_iou: float = 0.45, max_det: int = 300) -> None:
        self.conf = conf_threshold
        self.iou = nms_iou
        self.max_det = max_det

    async def __call__(self, raw, frame: Frame, spec: ModelSpec) -> Result:
        out = next(iter(raw.values()))
        # accept (1, N, 5+C) or (1, 5+C, N)
        if out.ndim == 3 and out.shape[1] < out.shape[2]:
            out = out.transpose(0, 2, 1)
        preds = out[0]
        boxes_xywh, scores_obj_cls = preds[:, :4], preds[:, 4:]
        if scores_obj_cls.shape[1] > 1:
            # yolov5: obj * class
            obj = scores_obj_cls[:, 0:1] if scores_obj_cls.shape[1] > len(spec.label_list() or []) else None
            if obj is not None and scores_obj_cls.shape[1] - 1 == len(spec.label_list() or []):
                cls_scores = scores_obj_cls[:, 1:] * obj
            else:
                cls_scores = scores_obj_cls
        else:
            cls_scores = scores_obj_cls
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
        keep = _nms(boxes, scores, self.iou)[: self.max_det]
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
