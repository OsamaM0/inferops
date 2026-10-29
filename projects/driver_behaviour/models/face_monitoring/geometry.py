"""Camera-relative head rotation and whole-frame 106-point eye geometry."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

# Corners, upper eyelids, opposite corner, lower eyelids: InsightFace's markup.
IMAGE_LEFT_EYE = (35, 41, 42, 39, 37, 36)
IMAGE_RIGHT_EYE = (89, 95, 96, 93, 91, 90)


def head_tensor(crop):
    image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).resize(
        (256, 256), Image.Resampling.BILINEAR
    )
    pixels = np.asarray(image.crop((16, 16, 240, 240)), np.float32) / 255
    return np.ascontiguousarray(
        ((pixels - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(2, 0, 1)[None], dtype=np.float32
    )


def rotation_to_pose(rotation):
    matrix = np.asarray(rotation, dtype=float)
    if matrix.size != 9:
        return None
    matrix = matrix.reshape(3, 3)
    if (
        not np.isfinite(matrix).all()
        or not np.allclose(matrix.T @ matrix, np.eye(3), atol=0.01)
        or not np.isclose(np.linalg.det(matrix), 1, atol=0.01)
    ):
        return None
    # The forward axis preserves yaw beyond +/-90; Euler yaw would fold there.
    forward = matrix[:, 2]
    return {
        "yaw": -float(np.degrees(np.arctan2(forward[0], forward[2]))),
        "pitch": float(np.degrees(np.arctan2(forward[1], np.hypot(forward[0], forward[2])))),
        "roll": float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))),
    }


def landmark_tensor(image, box):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = np.asarray(box) * [width, height, width, height]
    scale = 192 / (max(x2 - x1, y2 - y1) * 1.5)
    transform = np.asarray(
        [[scale, 0, 96 - (x1 + x2) / 2 * scale], [0, scale, 96 - (y1 + y2) / 2 * scale]], np.float32
    )
    crop = cv2.warpAffine(image, transform, (192, 192))
    # The pinned graph includes its own pixel normalization. Feed RGB 0..255.
    return cv2.dnn.blobFromImage(crop, 1, (192, 192), (0, 0, 0), swapRB=True), transform


def decode_landmarks(raw, transform):
    points = np.asarray(raw, dtype=np.float32)
    if points.size != 212 or not np.isfinite(points).all():
        return None
    points = (points.reshape(106, 2) + 1) * 96
    points = cv2.transform(points[None], cv2.invertAffineTransform(transform))[0]
    return points if np.isfinite(points).all() else None


def eye_inputs(image, points, raw_pose, min_width=6):
    if (
        points is None
        or raw_pose is None
        or not np.isfinite([raw_pose["yaw"], raw_pose["pitch"]]).all()
        or abs(raw_pose["yaw"]) > 40
        or abs(raw_pose["pitch"]) > 35
    ):
        return None
    points = np.asarray(points, dtype=float)
    if points.shape != (106, 2) or not np.isfinite(points).all():
        return None
    height, width = image.shape[:2]
    patches, ears = [], []
    # Anatomical left/right, not the viewer's left/right, for an unmirrored frame.
    for indices in (IMAGE_RIGHT_EYE, IMAGE_LEFT_EYE):
        p1, p2, p3, p4, p5, p6 = points[list(indices)]
        span = float(np.linalg.norm(p4 - p1))
        if span < min_width:
            return None
        u = (p4 - p1) / span
        v = np.asarray([-u[1], u[0]])
        center, side = (p1 + p4) / 2, 1.8 * span
        corners = np.asarray(
            [center + side / 2 * (a * u + b * v) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        )
        if (corners < 0).any() or (corners[:, 0] >= width).any() or (corners[:, 1] >= height).any():
            return None
        affine = np.column_stack((u * side / 64, v * side / 64, center - 31.5 * side / 64 * (u + v))).astype(
            np.float32
        )
        patch = cv2.warpAffine(image, affine, (64, 64), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        patches.append(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).transpose(2, 0, 1))
        ears.append(float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2 * span)))
    # B=2 eyes, T=1 frame. ImageNet normalization is already inside BlinkCNN.
    return np.ascontiguousarray(np.stack(patches)[:, None], dtype=np.float32) / 255, ears


def closure_scores(raw):
    logits = np.asarray(raw, dtype=np.float32)
    if logits.size != 2 or not np.isfinite(logits).all():
        return None
    logits = logits.reshape(2)
    positive = logits >= 0
    scores = np.empty_like(logits)
    scores[positive] = 1 / (1 + np.exp(-logits[positive]))
    exponential = np.exp(logits[~positive])
    scores[~positive] = exponential / (1 + exponential)
    return scores
