"""Core multi-object tracking.

This package is **domain-agnostic** — it operates on `Detection` instances
defined in `inferops.core.types` and assigns a stable `track_id` to each
detection's `attrs` dict. Domains plug it in by adding a `tracking:` block
to their pipeline YAML; the runner auto-injects a `track` stage right after
the first `infer` stage.

Default algorithm: BYTETrack-style two-stage IoU matching backed by a
constant-velocity Kalman filter (8-state: [cx, cy, area, aspect, vx, vy,
va, vp]). The Kalman implementation is pure NumPy — no external deps.
"""
from __future__ import annotations

from inferops.core.tracking.factory import TrackingSpec, build_tracker
from inferops.core.tracking.tracker import IoUKalmanTracker, Tracker

__all__ = ["IoUKalmanTracker", "Tracker", "TrackingSpec", "build_tracker"]
