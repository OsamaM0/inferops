"""YuNet (face_detection_yunet_2023mar) pre/post-processing.

The network takes a 640x640 BGR float image in [0, 255] and emits, for each of
the strides 8/16/32, per-cell class/objectness scores, a box regression and five
keypoints (right eye, left eye, nose, right and left mouth corner in image
order). Decoding follows OpenCV's ``FaceDetectorYN``.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

INPUT_SIZE = 640
STRIDES = (8, 16, 32)


@dataclass(slots=True)
class FaceDetection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in source pixels
    score: float
    keypoints: np.ndarray                    # (5, 2) source pixels

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


def preprocess(image_bgr: np.ndarray, input_name: str = "input",
               size: int = INPUT_SIZE) -> tuple[dict[str, np.ndarray], float]:
    """Letterbox to ``size`` (top-left anchored) and return feeds + scale."""
    h, w = image_bgr.shape[:2]
    scale = size / max(h, w)
    canvas = np.zeros((size, size, 3), np.uint8)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    canvas[:nh, :nw] = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    tensor = canvas.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    return {input_name: tensor}, scale


def decode(outputs: dict[str, np.ndarray], scale: float, *, score_threshold: float = 0.6,
           nms_threshold: float = 0.3, size: int = INPUT_SIZE, top_k: int = 20) -> list[FaceDetection]:
    """Decode raw YuNet outputs into source-pixel detections, best first."""
    boxes: list[list[float]] = []
    scores: list[float] = []
    keypoints: list[np.ndarray] = []
    for stride in STRIDES:
        cells = size // stride
        cls = np.clip(outputs[f"cls_{stride}"].reshape(-1), 0.0, 1.0)
        obj = np.clip(outputs[f"obj_{stride}"].reshape(-1), 0.0, 1.0)
        score = np.sqrt(cls * obj)
        index = np.flatnonzero(score >= score_threshold)
        if index.size == 0:
            continue
        rows, cols = index // cells, index % cells
        reg = outputs[f"bbox_{stride}"].reshape(-1, 4)[index]
        cx = (cols + reg[:, 0]) * stride
        cy = (rows + reg[:, 1]) * stride
        bw = np.exp(reg[:, 2]) * stride
        bh = np.exp(reg[:, 3]) * stride
        kps = outputs[f"kps_{stride}"].reshape(-1, 5, 2)[index]
        kps = (kps + np.stack([cols, rows], axis=1)[:, None, :]) * stride
        kept = score[index]
        for i in range(index.size):
            boxes.append([float(cx[i] - bw[i] / 2) / scale, float(cy[i] - bh[i] / 2) / scale,
                          float(bw[i]) / scale, float(bh[i]) / scale])
            scores.append(float(kept[i]))
            keypoints.append(kps[i] / scale)
    if not boxes:
        return []
    keep = np.asarray(cv2.dnn.NMSBoxes(boxes, scores, score_threshold, nms_threshold, top_k=top_k)).reshape(-1)
    detections = []
    for i in keep:
        x, y, w, h = boxes[int(i)]
        detections.append(FaceDetection((x, y, x + w, y + h), scores[int(i)], keypoints[int(i)]))
    return sorted(detections, key=lambda d: d.score, reverse=True)
