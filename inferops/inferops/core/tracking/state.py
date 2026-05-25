"""Constant-velocity Kalman filter for 2-D bounding-box tracking.

State vector (8): [cx, cy, area, aspect, vx, vy, va, vp]
  - cx, cy  : center coordinates (normalized 0..1)
  - area    : bbox area (normalized)
  - aspect  : width / height
  - vx, vy  : center velocity
  - va, vp  : area / aspect velocity

This is a pure-NumPy implementation; no scipy/filterpy dependency. Designed
for low per-track overhead (≪ 50 µs/update on CPU).
"""
from __future__ import annotations

import numpy as np

# State transition: x_{k+1} = F · x_k
_F = np.eye(8, dtype=np.float32)
for _i in range(4):
    _F[_i, _i + 4] = 1.0

# Observation matrix: we measure cx, cy, area, aspect (first four state dims)
_H = np.zeros((4, 8), dtype=np.float32)
for _i in range(4):
    _H[_i, _i] = 1.0


def box_to_z(box: tuple[float, float, float, float]) -> np.ndarray:
    """xyxy → [cx, cy, area, aspect]."""
    x1, y1, x2, y2 = box
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    return np.array([x1 + w / 2, y1 + h / 2, w * h, w / h], dtype=np.float32)


def z_to_box(z: np.ndarray) -> tuple[float, float, float, float]:
    cx, cy, area, aspect = float(z[0]), float(z[1]), max(float(z[2]), 1e-6), max(float(z[3]), 1e-6)
    w = float(np.sqrt(area * aspect))
    h = float(area / w) if w > 0 else 0.0
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


class KalmanBoxState:
    """A single track's Kalman filter state.

    Process / measurement noise constants are tuned for boxes normalised to
    [0, 1] sampled at typical 5–30 FPS video.
    """

    __slots__ = ("x", "P")

    def __init__(self, z0: np.ndarray) -> None:
        self.x = np.zeros(8, dtype=np.float32)
        self.x[:4] = z0
        # Large initial uncertainty on velocity, small on observed position.
        self.P = np.diag(np.array([1e-2, 1e-2, 1e-2, 1e-2,
                                   1.0,  1.0,  1.0,  1.0], dtype=np.float32))

    def predict(self) -> np.ndarray:
        # x = F·x ; P = F·P·F^T + Q
        self.x = _F @ self.x
        q = np.diag(np.array([1e-4, 1e-4, 1e-4, 1e-4,
                              1e-3, 1e-3, 1e-3, 1e-3], dtype=np.float32))
        self.P = _F @ self.P @ _F.T + q
        return self.x[:4].copy()

    def update(self, z: np.ndarray) -> None:
        # Standard Kalman update
        r = np.diag(np.array([1e-3, 1e-3, 1e-2, 1e-2], dtype=np.float32))
        y = z - _H @ self.x
        S = _H @ self.P @ _H.T + r
        K = self.P @ _H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8, dtype=np.float32) - K @ _H) @ self.P
