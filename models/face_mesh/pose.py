"""Head pose, neutral-pose calibration and head-direction classification.

Conventions (camera facing the driver, image not mirrored):

* ``yaw``   > 0: the driver turns to *their* left (nose moves to image right)
* ``pitch`` > 0: the driver looks down
* ``roll``  > 0: the head tilts clockwise in the image

Pose comes from a weighted rigid Procrustes (Umeyama) fit of the FaceMesh 3D
landmarks to MediaPipe's canonical face, using MediaPipe's own landmark basis
weights. It needs no camera intrinsics and stays valid up to profile views.
Everything here is plain numpy, so CPU, CUDA and Triton runs produce identical
decisions from identical landmarks.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

CANONICAL_FACE = Path(__file__).with_name("canonical_face.npz")
DIRECTIONS = ("front", "left", "right", "up", "down", "away")


@dataclass(slots=True)
class HeadPose:
    yaw: float
    pitch: float
    roll: float
    residual: float  # weighted RMS fit error in canonical units (~cm); high = poor mesh

    def as_dict(self) -> dict[str, float]:
        return {"yaw": self.yaw, "pitch": self.pitch, "roll": self.roll,
                "yaw_deg": self.yaw, "pitch_deg": self.pitch, "roll_deg": self.roll}


@cache
def canonical_face() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (canonical basis points, landmark ids, normalised weights)."""
    data = np.load(CANONICAL_FACE)
    ids = data["procrustes_ids"].astype(np.int64)
    weights = data["procrustes_weights"].astype(np.float64)
    return data["vertices"].astype(np.float64)[ids], ids, weights / weights.sum()


def estimate_pose(points: np.ndarray) -> HeadPose:
    """Estimate head pose from FaceMesh landmarks in image pixels.

    ``points`` is (>=468, 3): x right, y down, z away from the camera, all in
    the same pixel scale (FaceMesh z mapped with the crop scale).
    """
    canon, ids, w = canonical_face()
    # Image frame -> canonical frame (x right, y up, z towards the camera).
    observed = np.column_stack([points[ids, 0], -points[ids, 1], -points[ids, 2]]).astype(np.float64)
    canon_c = canon - w @ canon
    observed_c = observed - w @ observed
    cov = (canon_c * w[:, None]).T @ observed_c
    u, s, vt = np.linalg.svd(cov)
    d = np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T)) or 1.0])
    rotation = vt.T @ d @ u.T
    variance = float((canon_c ** 2).sum(axis=1) @ w)
    scale = float(np.trace(np.diag(s) @ d)) / max(variance, 1e-9)
    forward = rotation[:, 2]
    right = rotation[:, 0]
    yaw = math.degrees(math.atan2(forward[0], forward[2]))
    pitch = math.degrees(math.atan2(-forward[1], math.hypot(forward[0], forward[2])))
    roll = math.degrees(math.atan2(-right[1], right[0]))
    fitted = scale * canon_c @ rotation.T
    residual = math.sqrt(float(((fitted - observed_c) ** 2).sum(axis=1) @ w)) / max(abs(scale), 1e-9)
    return HeadPose(yaw, pitch, roll, residual)


class OneEuroFilter:
    """Speed-adaptive low-pass filter (Casiez et al. 2012) for jitter-free angles."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self._value: float | None = None
        self._derivative = 0.0
        self._t: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._value, self._t, self._derivative = None, None, 0.0

    def __call__(self, value: float, t_s: float) -> float:
        if self._value is None or self._t is None or t_s <= self._t or t_s - self._t > 1.0:
            self._value, self._t, self._derivative = value, t_s, 0.0
            return value
        dt = t_s - self._t
        derivative = (value - self._value) / dt
        self._derivative += self._alpha(self.d_cutoff, dt) * (derivative - self._derivative)
        cutoff = self.min_cutoff + self.beta * abs(self._derivative)
        self._value += self._alpha(cutoff, dt) * (value - self._value)
        self._t = t_s
        return self._value


class NeutralPoseCalibrator:
    """Learn where "looking at the road" is for this camera and driver.

    Cab cameras are rarely on the driver's optical axis, so the road-facing
    pose is not yaw = pitch = 0. While the driver is attentive (eyes open, head
    still) samples are collected; after ``warmup_ms`` the neutral pose is their
    median, clamped to ``max_offset`` around the configured prior. Afterwards it
    adapts slowly, and only from samples already close to neutral, so a long
    glance can never become the new "front".
    """

    def __init__(self, *, mode: str = "auto", yaw: float = 0.0, pitch: float = 0.0,
                 warmup_ms: int = 4000, min_samples: int = 20, timeout_ms: int = 12000,
                 max_yaw_offset: float = 30.0, max_pitch_offset: float = 20.0,
                 adapt_band_deg: float = 10.0, adapt_time_constant_s: float = 60.0):
        if mode not in {"auto", "fixed"}:
            raise ValueError("neutral.mode must be 'auto' or 'fixed'")
        self.mode = mode
        self.prior = (float(yaw), float(pitch))
        self.yaw, self.pitch = self.prior
        self.warmup_ms = warmup_ms
        self.min_samples = min_samples
        self.timeout_ms = timeout_ms
        self.max_offset = (max_yaw_offset, max_pitch_offset)
        self.adapt_band = adapt_band_deg
        self.time_constant_s = adapt_time_constant_s
        self.calibrated = mode == "fixed"
        self._samples: list[tuple[float, float]] = []
        self._first_ms: int | None = None
        self._last_ms: int | None = None

    def offset(self, yaw: float, pitch: float) -> tuple[float, float]:
        return yaw - self.yaw, pitch - self.pitch

    def update(self, yaw: float, pitch: float, t_ms: int, *, attentive: bool) -> None:
        if self.mode == "fixed":
            return
        last, self._last_ms = self._last_ms, t_ms
        if not self.calibrated:
            self._first_ms = t_ms if self._first_ms is None else self._first_ms
            if attentive:
                self._samples.append((yaw, pitch))
            elapsed = t_ms - self._first_ms
            # Never fall back to the camera-axis prior: an off-axis camera would
            # then report a permanent left/right. Past the timeout a few
            # attentive frames suffice; until then direction stays undecided.
            enough = self.min_samples if elapsed < self.timeout_ms else min(5, self.min_samples)
            if elapsed >= self.warmup_ms and len(self._samples) >= enough:
                samples = np.asarray(self._samples)
                self.yaw = self._clamp(float(np.median(samples[:, 0])), 0)
                self.pitch = self._clamp(float(np.median(samples[:, 1])), 1)
                self.calibrated = True
                self._samples.clear()
            return
        if not attentive:
            return
        d_yaw, d_pitch = self.offset(yaw, pitch)
        if abs(d_yaw) > self.adapt_band or abs(d_pitch) > self.adapt_band or last is None:
            return
        dt = min(max(t_ms - last, 0), 1000) / 1000.0
        alpha = dt / (self.time_constant_s + dt)
        self.yaw = self._clamp(self.yaw + alpha * d_yaw, 0)
        self.pitch = self._clamp(self.pitch + alpha * d_pitch, 1)

    def _clamp(self, value: float, axis: int) -> float:
        low = self.prior[axis] - self.max_offset[axis]
        return float(np.clip(value, low, self.prior[axis] + self.max_offset[axis]))


@dataclass(slots=True)
class DirectionThresholds:
    yaw_enter: float = 25.0
    yaw_exit: float = 18.0
    down_enter: float = 18.0
    down_exit: float = 12.0
    up_enter: float = 20.0
    up_exit: float = 14.0


class DirectionClassifier:
    """Hysteresis classifier over the offset from the neutral pose."""

    def __init__(self, thresholds: DirectionThresholds | None = None, *, mirror: bool = False):
        self.t = thresholds or DirectionThresholds()
        self.mirror = mirror
        self.state = "front"

    def _exceed(self, direction: str, d_yaw: float, d_pitch: float, entering: bool) -> float:
        t = self.t
        if direction in {"left", "right"}:
            signed = d_yaw if (direction == "left") != self.mirror else -d_yaw
            return signed / (t.yaw_enter if entering else t.yaw_exit)
        if direction == "down":
            return d_pitch / (t.down_enter if entering else t.down_exit)
        return -d_pitch / (t.up_enter if entering else t.up_exit)

    def update(self, d_yaw: float, d_pitch: float, *, gaze_down: bool = False) -> str:
        holding = self.state in {"left", "right", "up", "down"} and not (self.state == "up" and gaze_down)
        if holding and self._exceed(self.state, d_yaw, d_pitch, False) >= 1.0:
            return self.state
        candidates = {name: self._exceed(name, d_yaw, d_pitch, True) for name in ("left", "right", "up", "down")}
        if gaze_down:
            # Eyelids lowered with the eyes looking down: the driver reads a phone
            # or the dash even when the head itself barely pitches.
            candidates["down"] = max(candidates["down"], 1.0)
        best = max(candidates, key=candidates.get)
        self.state = best if candidates[best] >= 1.0 else "front"
        return self.state

    def scores(self, d_yaw: float, d_pitch: float) -> dict[str, float]:
        """Continuous [0, 1] closeness to each direction's entry threshold."""
        return {name: float(np.clip(self._exceed(name, d_yaw, d_pitch, True), 0.0, 1.0))
                for name in ("left", "right", "up", "down")}


class Presence:
    """Remember whether a driver face was seen recently on this stream."""

    def __init__(self, away_max_ms: int = 15000):
        self.away_max_ms = away_max_ms
        self.last_seen_ms: int | None = None
        self.seen_frames: deque[int] = deque(maxlen=8)

    def seen(self, t_ms: int) -> None:
        self.last_seen_ms = t_ms
        self.seen_frames.append(t_ms)

    def lost_for_ms(self, t_ms: int) -> int | None:
        """Milliseconds since an established face was last seen, else None."""
        if self.last_seen_ms is None or len(self.seen_frames) < self.seen_frames.maxlen:
            return None
        elapsed = t_ms - self.last_seen_ms
        return elapsed if 0 <= elapsed <= self.away_max_ms else None
