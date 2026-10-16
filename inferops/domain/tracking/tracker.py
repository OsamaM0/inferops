"""IoU + Kalman multi-object tracker (BYTETrack-style two-stage matching).

Each `update()` call returns the list of detections it was given, with
`attrs["track_id"]` populated. Unmatched detections start new tracks; tracks
not seen for `max_age` updates are closed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from inferops.domain.tracking.state import KalmanBoxState, box_to_z, z_to_box
from inferops.domain.types import Detection


class Tracker(Protocol):
    def update(self, detections: list[Detection]) -> list[Detection]: ...
    @property
    def active_count(self) -> int: ...


@dataclass
class _Track:
    track_id: int
    label: str
    state: KalmanBoxState
    hits: int = 1
    age: int = 0
    time_since_update: int = 0
    is_confirmed: bool = False
    history: list[tuple[float, float, float, float]] = field(default_factory=list)


def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Vectorised IoU between two sets of xyxy boxes (Nx4 and Mx4)."""
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)
    a = boxes_a[:, None, :]   # (N, 1, 4)
    b = boxes_b[None, :, :]   # (1, M, 4)
    x1 = np.maximum(a[..., 0], b[..., 0])
    y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2])
    y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[..., 2] - a[..., 0], 0, None) * np.clip(a[..., 3] - a[..., 1], 0, None)
    area_b = np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)
    union = area_a + area_b - inter + 1e-9
    return (inter / union).astype(np.float32)


def _greedy_assign(cost: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Greedy matching: highest-IoU pair first, until none above threshold."""
    pairs: list[tuple[int, int]] = []
    if cost.size == 0:
        return pairs
    c = cost.copy()
    for _ in range(min(c.shape)):
        idx = int(np.argmax(c))
        i, j = divmod(idx, c.shape[1])
        if c[i, j] < threshold:
            break
        pairs.append((i, j))
        c[i, :] = -1.0
        c[:, j] = -1.0
    return pairs


class IoUKalmanTracker:
    """Per-stream BYTETrack-style tracker.

    Args:
        max_age:        Drop a lost track after this many missed updates.
        min_hits:       Promote a track from tentative to confirmed after this
                        many consecutive hits.
        iou_threshold:  Primary (high-conf) association IoU cutoff.
        low_threshold:  Secondary (low-conf) association IoU cutoff (BYTETrack).
        score_high:     Detections with score >= this go to the primary stage.
        score_low:      Detections with score in [low, high) go to the secondary
                        stage; anything below `score_low` is dropped entirely.
        per_class:      If True, only associate detections of the same `label`.
    """

    def __init__(
        self,
        *,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        low_threshold: float = 0.5,
        score_high: float = 0.5,
        score_low: float = 0.1,
        per_class: bool = True,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.low_threshold = low_threshold
        self.score_high = score_high
        self.score_low = score_low
        self.per_class = per_class
        self._tracks: list[_Track] = []
        self._next_id = 1
        self._closed_ids: list[int] = []

    @property
    def active_count(self) -> int:
        return sum(1 for t in self._tracks if t.is_confirmed)

    def pop_closed_ids(self) -> list[int]:
        out, self._closed_ids = self._closed_ids, []
        return out

    def _new_track(self, det: Detection) -> _Track:
        tid = self._next_id
        self._next_id += 1
        return _Track(
            track_id=tid,
            label=det.label,
            state=KalmanBoxState(box_to_z(det.box)),
            is_confirmed=self.min_hits <= 1,
        )

    def _match_subset(
        self,
        track_indices: list[int],
        det_indices: list[int],
        track_boxes: np.ndarray,
        det_boxes: np.ndarray,
        det_labels: list[str],
        threshold: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not track_indices or not det_indices:
            return [], list(track_indices), list(det_indices)
        sub_tracks = track_boxes[track_indices]
        sub_dets = det_boxes[det_indices]
        iou = _iou_matrix(sub_tracks, sub_dets)
        if self.per_class:
            for i, ti in enumerate(track_indices):
                t_label = self._tracks[ti].label
                for j, dj in enumerate(det_indices):
                    if det_labels[dj] != t_label:
                        iou[i, j] = 0.0
        pairs = _greedy_assign(iou, threshold)
        matched_t = {p[0] for p in pairs}
        matched_d = {p[1] for p in pairs}
        return (
            [(track_indices[i], det_indices[j]) for i, j in pairs],
            [ti for k, ti in enumerate(track_indices) if k not in matched_t],
            [dj for k, dj in enumerate(det_indices) if k not in matched_d],
        )

    def update(self, detections: list[Detection]) -> list[Detection]:
        # 1) Kalman predict for every track and gather predicted boxes.
        track_boxes = np.empty((len(self._tracks), 4), dtype=np.float32)
        for i, tr in enumerate(self._tracks):
            tr.state.predict()
            tr.age += 1
            tr.time_since_update += 1
            track_boxes[i] = z_to_box(tr.state.x[:4])

        # 2) Split detections by score.
        det_boxes = np.array(
            [d.box for d in detections] if detections else [], dtype=np.float32,
        ).reshape(-1, 4)
        det_labels = [d.label for d in detections]
        high_idx = [i for i, d in enumerate(detections) if d.score >= self.score_high]
        low_idx = [
            i for i, d in enumerate(detections)
            if self.score_low <= d.score < self.score_high
        ]
        all_track_idx = list(range(len(self._tracks)))

        # 3) First-stage: confirmed/unconfirmed tracks vs high-conf detections.
        matches_a, unmatched_tracks, unmatched_high = self._match_subset(
            all_track_idx, high_idx, track_boxes, det_boxes, det_labels,
            self.iou_threshold,
        )

        # 4) Second-stage (BYTETrack): unmatched tracks vs low-conf detections.
        matches_b, unmatched_tracks, _ = self._match_subset(
            unmatched_tracks, low_idx, track_boxes, det_boxes, det_labels,
            self.low_threshold,
        )

        # 5) Apply matches: update Kalman, tag detection.
        for ti, di in matches_a + matches_b:
            tr = self._tracks[ti]
            d = detections[di]
            tr.state.update(box_to_z(d.box))
            tr.hits += 1
            tr.time_since_update = 0
            tr.label = d.label
            if not tr.is_confirmed and tr.hits >= self.min_hits:
                tr.is_confirmed = True
            d.attrs["track_id"] = tr.track_id

        # 6) Spawn new tracks for unmatched high-conf detections.
        for di in unmatched_high:
            d = detections[di]
            tr = self._new_track(d)
            self._tracks.append(tr)
            d.attrs["track_id"] = tr.track_id

        # 7) Prune dead tracks.
        survivors: list[_Track] = []
        for tr in self._tracks:
            if tr.time_since_update > self.max_age:
                self._closed_ids.append(tr.track_id)
                continue
            survivors.append(tr)
        self._tracks = survivors
        return detections


def make_tracker(algorithm: str = "iou_kalman", **kwargs) -> Tracker:
    if algorithm in {"iou_kalman", "bytetrack"}:
        return IoUKalmanTracker(**kwargs)
    raise ValueError(f"unknown tracking algorithm: {algorithm!r}")
