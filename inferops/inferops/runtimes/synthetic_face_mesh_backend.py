"""Synthetic face mesh backend for tests and CI (no real ONNX weights needed).

Returns anatomically plausible 468-landmark positions for a forward-facing
driver with open eyes.  The landmark layout matches MediaPipe FaceMesh v1/v2.

Configure via ``ModelSpec.tags``:
    drowsy:          "0" or "1"   – eye-closure scenario   (default: "0")
    distracted:      "0" or "1"   – head-turned scenario   (default: "0")
    smoking:         "0" or "1"   – cigarette-near-mouth    (default: "0")
    face_confidence: float string – face-flag output        (default: "0.97")
    seed:            int string   – RNG seed                (default: "0")
"""
from __future__ import annotations

import math

import numpy as np

from inferops.registry.model_spec import ModelSpec
from inferops.runtimes.base import InferenceBackend

# ── MediaPipe 468-landmark canonical positions for 192×192 input ──────────
#
# We place key landmarks accurately enough to exercise the EAR and head-pose
# code paths, then fill the rest with a plausible grid.
#
# Face centre in 192×192: nose ≈ (96, 105), chin ≈ (96, 160)
# Left eye  (image coords):  outer ≈ (62, 82),  inner ≈ (84, 82)
# Right eye (image coords):  inner ≈ (108, 82), outer ≈ (130, 82)
# Mouth left  ≈ (75, 133),  mouth right  ≈ (117, 133)
#
# EAR formula: (|p2−p6| + |p3−p5|) / (2·|p1−p4|)
# Open eyes: vertical ≈ 12 px, horizontal ≈ 22 px → EAR ≈ 0.27 (awake)
# Closed eyes: vertical ≈ 1 px                    → EAR ≈ 0.02 (drowsy)

_OPEN_EYE_V = 12.0   # vertical opening for open eyes
_CLOSED_EYE_V = 1.0  # vertical opening for closed eyes

# Indices used by face_features.py
_LEFT_EYE  = (33, 160, 158, 133, 153, 144)   # p1..p6
_RIGHT_EYE = (362, 385, 387, 263, 373, 380)  # p1..p6

# Nose tip, chin, eye outers, mouth corners used for head pose
_NOSE_TIP        = 1
_CHIN            = 152
_LEFT_EYE_OUTER  = 33
_RIGHT_EYE_OUTER = 263
_LEFT_MOUTH      = 61
_RIGHT_MOUTH     = 291


def _build_neutral_landmarks(
    eye_v: float = _OPEN_EYE_V,
    yaw_deg: float = 0.0,
) -> np.ndarray:
    """Return (468, 3) landmark array for a 192×192 face crop."""
    pts = np.zeros((468, 3), dtype=np.float32)

    # ── Fill with a face-shaped grid ──────────────────────────────────────
    # Simple ellipse distribution so landmarks don't all overlap
    cx, cy = 96.0, 96.0
    for i in range(468):
        angle = (i / 468.0) * 2 * math.pi
        r = 60.0 + 20.0 * math.sin(i * 0.37)
        pts[i] = [cx + r * math.cos(angle), cy + r * math.sin(angle) * 0.75, 0.0]

    # ── Left eye (indices from face_features.py) ──────────────────────────
    # p1=33(outer-L), p4=133(inner-L), p2=160(top-L), p6=144(bot-L),
    # p3=158(top-R), p5=153(bot-R)
    eye_mid_y = 82.0
    pts[33]  = [62.0, eye_mid_y, 0.0]           # p1 outer-left
    pts[133] = [84.0, eye_mid_y, 0.0]           # p4 inner-left
    pts[160] = [68.0, eye_mid_y - eye_v / 2, 0.0]  # p2 top
    pts[144] = [68.0, eye_mid_y + eye_v / 2, 0.0]  # p6 bottom
    pts[158] = [78.0, eye_mid_y - eye_v / 2, 0.0]  # p3 top
    pts[153] = [78.0, eye_mid_y + eye_v / 2, 0.0]  # p5 bottom

    # ── Right eye ─────────────────────────────────────────────────────────
    pts[362] = [130.0, eye_mid_y, 0.0]          # p1 outer-right
    pts[263] = [108.0, eye_mid_y, 0.0]          # p4 inner-right
    pts[385] = [124.0, eye_mid_y - eye_v / 2, 0.0]
    pts[380] = [124.0, eye_mid_y + eye_v / 2, 0.0]
    pts[387] = [114.0, eye_mid_y - eye_v / 2, 0.0]
    pts[373] = [114.0, eye_mid_y + eye_v / 2, 0.0]

    # ── Nose / chin / mouth ───────────────────────────────────────────────
    pts[1]   = [96.0,  105.0, 0.0]    # nose tip
    pts[152] = [96.0,  160.0, 0.0]    # chin
    pts[61]  = [75.0,  133.0, 0.0]    # left mouth corner
    pts[291] = [117.0, 133.0, 0.0]    # right mouth corner
    # Outer-lip ring (rough positions so mouth bbox works)
    pts[0]   = [96.0,  128.0, 0.0]    # upper lip centre
    pts[17]  = [96.0,  140.0, 0.0]    # lower lip centre

    # ── Apply yaw shift (distraction simulation) ──────────────────────────
    if abs(yaw_deg) > 1.0:
        shift = yaw_deg * 0.8  # rough pixel shift per degree
        pts[:, 0] += shift

    return pts


class SyntheticFaceMeshBackend(InferenceBackend):
    """Deterministic face-mesh backend for unit tests and CI."""

    def __init__(self, base_dir=None) -> None:
        self.base_dir = base_dir
        self._drowsy = False
        self._distracted = False
        self._smoking = False
        self._confidence = 0.97

    async def load(self, spec: ModelSpec) -> None:
        self.spec = spec
        tags = spec.tags or {}
        self._drowsy     = tags.get("drowsy", "0") == "1"
        self._distracted = tags.get("distracted", "0") == "1"
        self._smoking    = tags.get("smoking", "0") == "1"
        self._confidence = float(tags.get("face_confidence", "0.97"))

    async def infer(self, inputs: dict) -> dict[str, np.ndarray]:
        eye_v = _CLOSED_EYE_V if self._drowsy else _OPEN_EYE_V
        yaw   = 45.0 if self._distracted else 0.0
        pts   = _build_neutral_landmarks(eye_v=eye_v, yaw_deg=yaw)

        # Output shapes mirror the PINTO FaceMesh ONNX export
        # landmarks: (1, 1, 1404)  face_flag: (1, 1)
        landmarks_flat = pts.reshape(1, 1, 1404)  # type: ignore[misc]
        face_flag = np.array([[self._confidence]], dtype=np.float32)
        return {
            "output_mesh_landmarks": landmarks_flat,
            "output_face_flag": face_flag,
        }

    async def close(self) -> None:
        pass
