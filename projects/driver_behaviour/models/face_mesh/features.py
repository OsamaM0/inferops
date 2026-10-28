"""Driver-behaviour face feature helpers.

Ports the legacy MediaPipe face worker math into the example layer. The stage is
intentionally small: upstream code can either provide precomputed EAR/head-pose
values in ``item.payload['face']`` or provide face-mesh landmarks in
``item.payload['face_landmarks']``.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cv2
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

# Six vertices from MediaPipe's canonical_face_model.obj (Apache-2.0), with
# Y/Z reversed to OpenCV camera coordinates: X right, Y down, face toward -Z.
# https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj
POSE_INDICES = (NOSE_TIP, CHIN, LEFT_EYE_OUTER, RIGHT_EYE_OUTER, LEFT_MOUTH, RIGHT_MOUTH)
POSE_MODEL = np.array([
    (0.0, 1.126865, -7.475604), (0.0, 9.403378, -4.264492),
    (-4.445859, -2.663991, -3.173422), (4.445859, -2.663991, -3.173422),
    (-2.456206, 4.342621, -4.283884), (2.456206, 4.342621, -4.283884),
], dtype=np.float64)


def eye_aspect_ratio(points: np.ndarray, indices: Sequence[int]) -> float:
    """Return EAR using the six-point formula from the legacy face worker."""
    p1, p2, p3, p4, p5, p6 = points[list(indices)]
    a = np.linalg.norm(p2 - p6)
    b = np.linalg.norm(p3 - p5)
    c = np.linalg.norm(p1 - p4)
    return float((a + b) / (2.0 * c + 1e-6))


def head_pose(points: np.ndarray, width: int, height: int) -> dict[str, float] | None:
    """Fit a 3D face to pixels; positive yaw=driver left, positive pitch=down.

    Intrinsics are approximated from frame size. A fixed camera needs neutral
    yaw/pitch calibration; these are estimates, not measured gaze angles.
    """
    if width <= 0 or height <= 0 or len(points) <= max(POSE_INDICES):
        return None
    image_points = np.asarray(points[list(POSE_INDICES), :2], dtype=np.float64)
    span = np.ptp(image_points, axis=0)
    if not np.isfinite(image_points).all() or min(span) < 8:
        return None
    focal = float(max(width, height))
    camera = np.array(((focal, 0, width / 2), (0, focal, height / 2), (0, 0, 1)), dtype=np.float64)
    distortion = np.zeros((4, 1), dtype=np.float64)
    try:
        # A six-point DLT initialization can choose a flipped solution on
        # shallow face geometry. SQPnP supplies a stable global initialization.
        ok, rotation, translation = cv2.solvePnP(
            POSE_MODEL, image_points, camera, distortion, flags=cv2.SOLVEPNP_SQPNP)
        if not ok or not np.isfinite(rotation).all() or translation[2, 0] <= 0:
            return None
        rotation, translation = cv2.solvePnPRefineLM(
            POSE_MODEL, image_points, camera, distortion, rotation, translation)
        projected, _ = cv2.projectPoints(POSE_MODEL, rotation, translation, camera, distortion)
        error = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - image_points) ** 2, axis=1))))
        if error / max(span) > 0.12:
            return None
        matrix, _ = cv2.Rodrigues(rotation)
    except cv2.error:
        return None
    sy = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    pitch = float(np.degrees(np.arctan2(matrix[2, 1], matrix[2, 2])))
    yaw = -float(np.degrees(np.arctan2(-matrix[2, 0], sy)))
    roll = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    if abs(pitch) > 85 or abs(yaw) > 85:
        return None
    return {"yaw": yaw, "pitch": pitch, "roll": roll, "reprojection_error_px": error}


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
    """Map configured physical thresholds to the fusion decision score of 0.5."""
    scores: dict[str, float] = {}
    if (ear_left is not None and ear_right is not None
            and np.isfinite([ear_left, ear_right]).all() and min(ear_left, ear_right) >= 0):
        # Require bilateral closure; one closed eye must not imply drowsiness.
        ear = max(ear_left, ear_right)
        if np.isfinite(ear) and ear_threshold > 0:
            scores["drowsy_ear"] = float(np.clip(0.5 + (ear_threshold - ear) / 0.06, 0, 1))
    axes = [(value, threshold) for value, threshold in ((yaw, yaw_threshold), (pitch, pitch_threshold))
            if value is not None and np.isfinite(value) and threshold > 0]
    if axes:
        scores["distracted_score"] = float(max(np.clip(
            0.5 + (abs(value) - threshold) / 20.0, 0, 1) for value, threshold in axes))
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
            yaw = pose["yaw"] if pose else None
            pitch = pose["pitch"] if pose else None

        item.payload.update(behavior_scores_from_face(ear_left, ear_right, yaw, pitch))
        return item
