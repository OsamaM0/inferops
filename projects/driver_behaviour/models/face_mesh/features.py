"""Driver-behaviour face feature helpers.

Ports the legacy MediaPipe face worker math into the example layer. The stage is
intentionally small: upstream code can either provide precomputed EAR/head-pose
values in ``item.payload['face']`` or provide face-mesh landmarks in
``item.payload['face_landmarks']``.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from inferops.adapters.outbound.stages.base import Stage

LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_MOUTH = 61
RIGHT_MOUTH = 291

EAR_DROWSY_THRESHOLD = 0.21
DISTRACT_YAW_DEG = 28.0
DISTRACT_PITCH_DEG = 28.0


def eye_aspect_ratio(points: np.ndarray, indices: Sequence[int]) -> float:
    """Return EAR using the six-point formula from the legacy face worker."""
    p1, p2, p3, p4, p5, p6 = points[list(indices)]
    a = np.linalg.norm(p2 - p6)
    b = np.linalg.norm(p3 - p5)
    c = np.linalg.norm(p1 - p4)
    return float((a + b) / (2.0 * c + 1e-6))


def head_pose(points: np.ndarray, width: int, height: int) -> dict[str, float]:
    """Return coarse yaw/pitch/roll in degrees from stable landmark geometry."""
    left_eye = points[LEFT_EYE_OUTER]
    right_eye = points[RIGHT_EYE_OUTER]
    nose = points[NOSE_TIP]
    mouth_mid = (points[LEFT_MOUTH] + points[RIGHT_MOUTH]) / 2.0
    eye_mid = (left_eye + right_eye) / 2.0

    left_dist = np.linalg.norm(nose - left_eye)
    right_dist = np.linalg.norm(nose - right_eye)
    # MediaPipe LEFT_EYE_OUTER (33) is the person's anatomical left eye, which
    # appears on the RIGHT side of the image.  When the driver looks to their
    # LEFT the face turns to the camera's right, the nose approaches that eye
    # and left_dist DECREASES while right_dist INCREASES.  Negating the
    # numerator (right_dist - left_dist) makes yaw > 0 = "driver looking left",
    # consistent with the Haar-based yaw below.
    yaw = float(np.clip(
        ((right_dist - left_dist) / (left_dist + right_dist + 1e-6)) * 90.0,
        -45.0,
        45.0,
    ))

    eye_to_nose = float(nose[1] - eye_mid[1])
    nose_to_mouth = float(mouth_mid[1] - nose[1])
    # Calibrated pitch: use the fractional position of the nose between the eye
    # midpoint and the mouth midpoint.  In a neutral, forward-facing head the
    # nose sits at roughly 57 % of that span; subtracting 0.57 centres pitch at
    # zero for a neutral face regardless of camera height.
    # Positive pitch → nose below the neutral point → driver looking DOWN.
    # Negative pitch → nose above the neutral point → driver looking UP.
    total_y = max(abs(eye_to_nose) + abs(nose_to_mouth), 1.0)
    nose_ratio = eye_to_nose / total_y
    pitch = float(np.clip(
        (nose_ratio - 0.57) * 200.0,
        -35.0,
        35.0,
    ))

    roll = float(np.degrees(np.arctan2(
        right_eye[1] - left_eye[1],
        right_eye[0] - left_eye[0],
    )))

    return {"yaw": yaw, "pitch": pitch, "roll": roll}


def behavior_scores_from_face(
    ear_left: float | None,
    ear_right: float | None,
    yaw: float | None,
    pitch: float | None,
    *,
    ear_threshold: float = EAR_DROWSY_THRESHOLD,
    yaw_threshold: float = DISTRACT_YAW_DEG,
    pitch_threshold: float = DISTRACT_PITCH_DEG,
) -> dict[str, float]:
    """Convert face features to the legacy drowsy/distracted signal scores."""
    scores: dict[str, float] = {}
    if ear_left is not None and ear_right is not None:
        ear = (ear_left + ear_right) / 2.0
        scores["drowsy_ear"] = max(0.0, min(1.0, (ear_threshold - ear) / ear_threshold))
    if yaw is not None and pitch is not None:
        yaw_off = max(0.0, abs(yaw) - yaw_threshold) / 30.0
        pitch_off = max(0.0, abs(pitch) - pitch_threshold) / 20.0
        scores["distracted_score"] = min(1.0, max(yaw_off, pitch_off))
    return scores


def _landmarks_to_points(landmarks: Sequence[Any], width: int, height: int) -> np.ndarray:
    points = []
    for landmark in landmarks:
        if isinstance(landmark, dict):
            x, y = landmark.get("x", 0.0), landmark.get("y", 0.0)
        else:
            x, y = landmark[0], landmark[1]
        points.append((float(x) * width, float(y) * height))
    return np.asarray(points, dtype=np.float64)


class FaceFeatures(Stage):
    """Stage that adds drowsy/distraction scores to ``StageItem.payload``."""

    async def process(self, item):
        width = int(item.payload.get("width") or 1)
        height = int(item.payload.get("height") or 1)
        face = item.payload.get("face") or {}

        ear_left = face.get("ear_left")
        ear_right = face.get("ear_right")
        pose = face.get("head_pose") or {}
        yaw = pose.get("yaw")
        pitch = pose.get("pitch")

        landmarks = item.payload.get("face_landmarks")
        if landmarks and len(landmarks) > max(CHIN, RIGHT_EYE_OUTER, RIGHT_MOUTH):
            points = _landmarks_to_points(landmarks, width, height)
            ear_left = eye_aspect_ratio(points, LEFT_EYE)
            ear_right = eye_aspect_ratio(points, RIGHT_EYE)
            pose = head_pose(points, width, height)
            yaw = pose["yaw"]
            pitch = pose["pitch"]

        item.payload.update(behavior_scores_from_face(ear_left, ear_right, yaw, pitch))
        return item
