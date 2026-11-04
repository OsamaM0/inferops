"""RetinaFace (yakhyo/retinaface-pytorch ONNX export) pre/post-processing.

The network takes a BGR float image with the per-channel mean (104, 117, 123)
subtracted, NCHW, and returns, for every prior (anchor):

* ``loc``       (1, N, 4)  box offsets (cx, cy, log w, log h) w.r.t. the prior
* ``conf``      (1, N, 2)  softmax [background, face]
* ``landmarks`` (1, N, 10) five keypoint offsets (image-left eye, image-right
  eye, nose, image-left and image-right mouth corner)

Priors follow the reference ``PriorBox``: strides 8/16/32 with two square
anchor sizes per stride, centre-offset form normalised to the input size, and
variances (0.1, 0.2). Decoding returns the same ``FaceDetection`` type as the
YuNet helper, so both detectors are interchangeable.
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from ..face_detector.yunet import FaceDetection

INPUT_SIZE = 640
MEAN_BGR = (104.0, 117.0, 123.0)
MIN_SIZES = ((16, 32), (64, 128), (256, 512))
STEPS = (8, 16, 32)
VARIANCE = (0.1, 0.2)


@lru_cache(maxsize=8)
def priors(height: int, width: int) -> np.ndarray:
    """(N, 4) priors [cx, cy, w, h] normalised to the input, reference order."""
    blocks = []
    for step, sizes in zip(STEPS, MIN_SIZES, strict=True):
        rows, cols = -(-height // step), -(-width // step)  # ceil
        cy, cx = np.meshgrid((np.arange(rows) + 0.5) * step / height,
                             (np.arange(cols) + 0.5) * step / width, indexing="ij")
        cells = np.stack([cx.ravel(), cy.ravel()], axis=1)                 # (cells, 2)
        wh = np.array([[s / width, s / height] for s in sizes])            # (anchors, 2)
        block = np.concatenate([np.repeat(cells, len(sizes), axis=0),
                                np.tile(wh, (len(cells), 1))], axis=1)     # anchor-minor order
        blocks.append(block)
    out = np.concatenate(blocks).astype(np.float32)
    out.setflags(write=False)
    return out


def preprocess(image_bgr: np.ndarray, input_name: str = "input",
               size: int = INPUT_SIZE) -> tuple[dict[str, np.ndarray], float]:
    """Letterbox to ``size`` x ``size`` (top-left anchored) and return feeds + scale.

    The padding takes the mean colour, so it is exactly zero after
    normalisation and does not look like image content.
    """
    h, w = image_bgr.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    canvas = np.empty((size, size, 3), np.float32)
    canvas[:] = MEAN_BGR
    canvas[:nh, :nw] = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas -= np.asarray(MEAN_BGR, np.float32)
    tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[np.newaxis])
    return {input_name: tensor}, scale


def _outputs(raw: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(loc, conf, landmarks) by name, falling back to the last dimension."""
    if {"loc", "conf", "landmarks"} <= set(raw):
        return raw["loc"], raw["conf"], raw["landmarks"]
    by_width = {np.asarray(v).shape[-1]: np.asarray(v) for v in raw.values()}
    try:
        return by_width[4], by_width[2], by_width[10]
    except KeyError as error:
        raise ValueError(f"unexpected RetinaFace outputs: {sorted(raw)}") from error


def decode(outputs: dict[str, np.ndarray], scale: float, *, size: int = INPUT_SIZE,
           score_threshold: float = 0.6, nms_threshold: float = 0.4,
           pre_nms_top_k: int = 5000, top_k: int = 750) -> list[FaceDetection]:
    """Decode raw RetinaFace outputs into source-pixel detections, best first."""
    loc, conf, landmarks = (np.asarray(a, np.float32).reshape(-1, a.shape[-1]) for a in _outputs(outputs))
    anchors = priors(size, size)
    if loc.shape[0] != anchors.shape[0]:
        raise ValueError(f"RetinaFace returned {loc.shape[0]} priors, expected {anchors.shape[0]} for {size}px")
    scores = conf[:, 1]
    keep = np.flatnonzero(scores >= score_threshold)
    if keep.size == 0:
        return []
    keep = keep[np.argsort(-scores[keep], kind="stable")][:pre_nms_top_k]
    prior, offset, marks = anchors[keep], loc[keep], landmarks[keep].reshape(-1, 5, 2)
    centre = prior[:, :2] + offset[:, :2] * VARIANCE[0] * prior[:, 2:]
    wh = prior[:, 2:] * np.exp(offset[:, 2:] * VARIANCE[1])
    pixels = size / scale                                   # normalised input -> source pixels
    xy1 = (centre - wh / 2) * pixels
    box_wh = wh * pixels
    points = (prior[:, None, :2] + marks * VARIANCE[0] * prior[:, None, 2:]) * pixels
    kept_scores = scores[keep]
    rects = [[float(x), float(y), float(bw), float(bh)] for (x, y), (bw, bh) in zip(xy1, box_wh, strict=True)]
    order = np.asarray(cv2.dnn.NMSBoxes(rects, kept_scores.tolist(), score_threshold, nms_threshold,
                                        top_k=top_k)).reshape(-1)
    detections = []
    for i in order:
        x, y, bw, bh = rects[int(i)]
        detections.append(FaceDetection((x, y, x + bw, y + bh), float(kept_scores[int(i)]), points[int(i)]))
    return sorted(detections, key=lambda d: d.score, reverse=True)
