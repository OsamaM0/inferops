"""Five-point face alignment for ArcFace-family embedders (AuraFace).

Faces are warped with a similarity transform (rotation, uniform scale,
translation) that best maps the detector's five keypoints onto the standard
ArcFace 112x112 template, the same alignment the embedder was trained on.
"""
from __future__ import annotations

import cv2
import numpy as np

SIZE = 112
# ArcFace template (x, y): image-left eye, image-right eye, nose,
# image-left mouth corner, image-right mouth corner.
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float64)


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity (Umeyama, no reflection) mapping src -> dst, as 2x3."""
    src = np.asarray(src, np.float64).reshape(-1, 2)
    dst = np.asarray(dst, np.float64).reshape(-1, 2)
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    s_c, d_c = src - mu_s, dst - mu_d
    var_s = float((s_c ** 2).sum()) / len(src)
    if var_s <= 1e-12:
        raise ValueError("degenerate keypoints")
    u, singular, vt = np.linalg.svd(d_c.T @ s_c / len(src))
    reflection = np.diag([1.0, float(np.sign(np.linalg.det(u) * np.linalg.det(vt)) or 1.0)])
    rotation = u @ reflection @ vt
    scale = float((singular * np.diag(reflection)).sum()) / var_s
    translation = mu_d - scale * rotation @ mu_s
    return np.hstack([scale * rotation, translation[:, None]])


def align_face(image_bgr: np.ndarray, keypoints: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Aligned ``size`` x ``size`` BGR crop for five source-pixel keypoints."""
    template = ARCFACE_TEMPLATE * (size / SIZE)
    matrix = similarity_transform(keypoints, template)
    return cv2.warpAffine(image_bgr, matrix, (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def to_tensor(crops: list[np.ndarray]) -> np.ndarray:
    """(N, 3, H, W) float32 RGB scaled to [-1, 1] (``(x - 127.5) / 127.5``)."""
    batch = np.stack([cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for crop in crops]).astype(np.float32)
    batch = (batch - 127.5) / 127.5
    return np.ascontiguousarray(batch.transpose(0, 3, 1, 2))


def frontal(keypoints: np.ndarray, *, tolerance: float = 0.1) -> bool:
    """Roughly frontal: the nose lies between the eyes, below them, above the mouth.

    Strong profiles give unreliable embeddings, so they are not embedded; the
    track keeps its earlier (frontal) vector and still follows by motion.
    """
    k = np.asarray(keypoints, np.float64).reshape(5, 2)
    left_eye, right_eye, nose, left_mouth, right_mouth = k
    span = right_eye[0] - left_eye[0]
    if span <= 1e-6:
        return False
    margin = tolerance * span
    eye_y = (left_eye[1] + right_eye[1]) / 2.0
    mouth_y = (left_mouth[1] + right_mouth[1]) / 2.0
    return bool(left_eye[0] + margin <= nose[0] <= right_eye[0] - margin and eye_y < nose[1] < mouth_y)
