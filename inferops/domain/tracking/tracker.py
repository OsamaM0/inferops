"""IoU + Kalman multi-object tracker (BYTETrack-style two-stage matching).

Each `update()` call returns the list of detections it was given, with
`attrs["track_id"]` populated. Unmatched detections start new tracks; tracks
not seen for `max_age` updates are closed.

Optional extensions (all off by default, so plain IoU tracking is unchanged):

* **Appearance** (``appearance=True``): detections may carry an L2-normalised
  vector in ``attrs[embedding_key]``. Each track keeps an exponential moving
  average of its vectors (Deep OC-SORT / BoT-SORT style). A pair whose cosine
  similarity is below ``appearance_gate`` cannot match on IoU alone, which
  prevents ID swaps when objects cross. Detections left over after IoU matching
  are compared with *lost* tracks; a cosine similarity of at least
  ``reid_threshold`` (within ``reid_max_distance`` box sizes of where the track
  was last seen) revives the old track ID instead of starting a new one.
* **Time-based aging** (``update(..., ts_ms=...)`` with ``max_lost_ms``): lost
  tracks are kept for a duration rather than a number of updates, so behaviour
  does not depend on the processing frame rate.
* **Exit-aware aging** (``exit_lost_ms``): a track last seen touching the frame
  border (within ``exit_margin``), or predicted to leave through a border it was
  within one box size of, is closed after the shorter ``exit_lost_ms``. Objects that leave the view are
  forgotten quickly; objects lost in the middle of the view (occluded) are kept.
* **Observation-centric recovery** (``observation_recovery=True``, OC-SORT):
  while a track is unobserved its Kalman prediction drifts (a shrinking box
  keeps shrinking), so a reappearing object can miss its own prediction. Left
  over detections get a second IoU match against each lost track's *last
  observed* box, and a recovered track restarts its motion model there.
* **Nearby recovery** (``recovery_distance > 0``): small, fast objects barely
  overlap their own last box after a short gap. A leftover detection of a
  similar size whose centre lies within ``recovery_distance`` box sizes of a
  track lost at most ``recovery_max_ms`` ago takes that track back, unless both
  carry vectors that are clearly different (``appearance_gate``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from inferops.domain.tracking.state import KalmanBoxState, box_to_z, z_to_box
from inferops.domain.types import Detection

Box = tuple[float, float, float, float]


class Tracker(Protocol):
    def update(self, detections: list[Detection], ts_ms: int | None = None) -> list[Detection]: ...
    @property
    def active_count(self) -> int: ...


@dataclass(frozen=True, slots=True)
class TrackView:
    """Read-only snapshot of one track (see ``IoUKalmanTracker.tracks``)."""
    track_id: int
    label: str
    box: Box                    # last observed box (xyxy, normalised)
    hits: int
    confirmed: bool
    time_since_update: int      # 0 = matched in the latest update
    has_embedding: bool


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
    last_box: Box = (0.0, 0.0, 0.0, 0.0)
    last_ts_ms: int | None = None
    embedding: np.ndarray | None = None


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


def _unit(vector: Any) -> np.ndarray | None:
    """L2-normalised float32 copy of an embedding, or None when unusable."""
    if vector is None:
        return None
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if arr.size == 0 or not np.isfinite(norm) or norm <= 1e-12:
        return None
    return arr / norm


def _centre_size(box: Box) -> tuple[float, float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, max(x2 - x1, y2 - y1, 1e-6)


class IoUKalmanTracker:
    """Per-stream BYTETrack-style tracker.

    Args:
        max_age:        Drop a lost track after this many missed updates
                        (used when ``update`` gets no timestamp or
                        ``max_lost_ms`` is None).
        min_hits:       Promote a track from tentative to confirmed after this
                        many consecutive hits.
        iou_threshold:  Primary (high-conf) association IoU cutoff.
        low_threshold:  Secondary (low-conf) association IoU cutoff (BYTETrack).
        score_high:     Detections with score >= this go to the primary stage.
        score_low:      Detections with score in [low, high) go to the secondary
                        stage; anything below `score_low` is dropped entirely.
        per_class:      If True, only associate detections of the same `label`.
        appearance:     Use ``attrs[embedding_key]`` vectors (see module doc).
        appearance_gate: Minimum cosine similarity for an IoU match when both
                        the track and the detection have a vector.
        reid_threshold: Minimum cosine similarity to revive a lost track.
        reid_max_distance: Spatial gate for revival, in units of the lost
                        track's last box size (max side); <= 0 disables it.
        ema_alpha:      Weight of the old vector in the track's moving average.
        embedding_key:  Detection attribute holding the vector.
        max_lost_ms:    With timestamps: keep a lost track this long.
        exit_lost_ms:   With timestamps: keep a lost track that was last seen
                        at the frame border (leaving the view) this long.
        exit_margin:    Border width (normalised) that counts as "at the edge".
        observation_recovery: Second-chance IoU match of leftover detections
                        against lost tracks' last observed boxes (OC-SORT).
        recovery_distance: Nearby recovery radius in last-box sizes; 0 disables.
        recovery_max_ms: With timestamps: only tracks lost at most this long are
                        recovered by distance (None: any live lost track).
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
        appearance: bool = False,
        appearance_gate: float = 0.2,
        reid_threshold: float = 0.5,
        reid_max_distance: float = 4.0,
        ema_alpha: float = 0.9,
        embedding_key: str = "embedding",
        max_lost_ms: int | None = None,
        exit_lost_ms: int | None = None,
        exit_margin: float = 0.02,
        observation_recovery: bool = False,
        recovery_distance: float = 0.0,
        recovery_max_ms: int | None = None,
    ) -> None:
        if not 0.0 <= ema_alpha < 1.0:
            raise ValueError("ema_alpha must be in [0, 1)")
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.low_threshold = low_threshold
        self.score_high = score_high
        self.score_low = score_low
        self.per_class = per_class
        self.appearance = appearance
        self.appearance_gate = appearance_gate
        self.reid_threshold = reid_threshold
        self.reid_max_distance = reid_max_distance
        self.ema_alpha = ema_alpha
        self.embedding_key = embedding_key
        self.max_lost_ms = max_lost_ms
        self.exit_lost_ms = exit_lost_ms
        self.exit_margin = exit_margin
        self.observation_recovery = observation_recovery
        self.recovery_distance = recovery_distance
        self.recovery_max_ms = recovery_max_ms
        self._tracks: list[_Track] = []
        self._next_id = 1
        self._closed_ids: list[int] = []

    @property
    def active_count(self) -> int:
        return sum(1 for t in self._tracks if t.is_confirmed)

    def pop_closed_ids(self) -> list[int]:
        out, self._closed_ids = self._closed_ids, []
        return out

    def tracks(self) -> list[TrackView]:
        """Snapshot of every live (visible or lost) track."""
        return [TrackView(t.track_id, t.label, t.last_box, t.hits, t.is_confirmed,
                          t.time_since_update, t.embedding is not None) for t in self._tracks]

    def _new_track(self, det: Detection, ts_ms: int | None, embedding: np.ndarray | None) -> _Track:
        tid = self._next_id
        self._next_id += 1
        return _Track(
            track_id=tid,
            label=det.label,
            state=KalmanBoxState(box_to_z(det.box)),
            is_confirmed=self.min_hits <= 1,
            last_box=tuple(det.box),
            last_ts_ms=ts_ms,
            embedding=embedding,
        )

    def _match_subset(
        self,
        track_indices: list[int],
        det_indices: list[int],
        track_boxes: np.ndarray,
        det_boxes: np.ndarray,
        det_labels: list[str],
        threshold: float,
        det_vectors: list[np.ndarray | None] | None = None,
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
        if det_vectors is not None:
            # Appearance gate: a clearly different face/object cannot take over
            # a track just because its box overlaps the prediction.
            for i, ti in enumerate(track_indices):
                track_vector = self._tracks[ti].embedding
                if track_vector is None:
                    continue
                for j, dj in enumerate(det_indices):
                    vector = det_vectors[dj]
                    if vector is not None and float(track_vector @ vector) < self.appearance_gate:
                        iou[i, j] = 0.0
        pairs = _greedy_assign(iou, threshold)
        matched_t = {p[0] for p in pairs}
        matched_d = {p[1] for p in pairs}
        return (
            [(track_indices[i], det_indices[j]) for i, j in pairs],
            [ti for k, ti in enumerate(track_indices) if k not in matched_t],
            [dj for k, dj in enumerate(det_indices) if k not in matched_d],
        )

    def _reidentify(
        self,
        track_indices: list[int],
        det_indices: list[int],
        detections: list[Detection],
        det_vectors: list[np.ndarray | None],
    ) -> tuple[list[tuple[int, int]], list[int]]:
        """Match leftover detections to lost tracks by appearance only."""
        candidates_t = [ti for ti in track_indices if self._tracks[ti].embedding is not None]
        candidates_d = [dj for dj in det_indices if det_vectors[dj] is not None]
        if not candidates_t or not candidates_d:
            return [], list(det_indices)
        track_matrix = np.stack([self._tracks[ti].embedding for ti in candidates_t])
        det_matrix = np.stack([det_vectors[dj] for dj in candidates_d])
        similarity = (track_matrix @ det_matrix.T).astype(np.float32)
        for i, ti in enumerate(candidates_t):
            track = self._tracks[ti]
            tx, ty, size = _centre_size(track.last_box)
            for j, dj in enumerate(candidates_d):
                det = detections[dj]
                if self.per_class and det.label != track.label:
                    similarity[i, j] = -1.0
                    continue
                if self.reid_max_distance > 0:
                    dx, dy, _ = _centre_size(tuple(det.box))
                    if np.hypot(dx - tx, dy - ty) > self.reid_max_distance * size:
                        similarity[i, j] = -1.0
        pairs = _greedy_assign(similarity, self.reid_threshold)
        matched = [(candidates_t[i], candidates_d[j]) for i, j in pairs]
        used = {dj for _, dj in matched}
        return matched, [dj for dj in det_indices if dj not in used]

    def _recover_nearby(
        self,
        track_indices: list[int],
        det_indices: list[int],
        detections: list[Detection],
        det_vectors: list[np.ndarray | None] | None,
        ts_ms: int | None,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Leftover detections vs recently lost tracks by centre distance."""
        score = np.full((len(track_indices), len(det_indices)), -1.0, dtype=np.float32)
        for i, ti in enumerate(track_indices):
            track = self._tracks[ti]
            if (self.recovery_max_ms is not None and ts_ms is not None and track.last_ts_ms is not None
                    and ts_ms - track.last_ts_ms > self.recovery_max_ms):
                continue
            tx, ty, size = _centre_size(track.last_box)
            for j, dj in enumerate(det_indices):
                det = detections[dj]
                if self.per_class and det.label != track.label:
                    continue
                dx, dy, det_size = _centre_size(tuple(det.box))
                if not 0.5 <= det_size / size <= 2.0:
                    continue  # a far, small face is not a near, large one
                vector = det_vectors[dj] if det_vectors is not None else None
                if vector is not None and track.embedding is not None \
                        and float(track.embedding @ vector) < self.appearance_gate:
                    continue
                distance = float(np.hypot(dx - tx, dy - ty)) / size
                if distance <= self.recovery_distance:
                    score[i, j] = 1.0 - distance / (self.recovery_distance + 1e-9)
        pairs = _greedy_assign(score, 0.0)
        matched_t = {i for i, _ in pairs}
        matched_d = {j for _, j in pairs}
        return ([(track_indices[i], det_indices[j]) for i, j in pairs],
                [ti for k, ti in enumerate(track_indices) if k not in matched_t],
                [dj for k, dj in enumerate(det_indices) if k not in matched_d])

    def _update_embedding(self, track: _Track, vector: np.ndarray | None, score: float) -> None:
        if vector is None:
            return
        if track.embedding is None:
            track.embedding = vector
            return
        if score < self.score_high:
            return  # low-confidence boxes are often partial; do not drift the average
        mixed = self.ema_alpha * track.embedding + (1.0 - self.ema_alpha) * vector
        track.embedding = _unit(mixed)

    def _is_exiting(self, track: _Track) -> bool:
        """Last seen at the border, or headed out through a border it was close to.

        A prediction alone is not enough: extrapolating a sudden jolt (camera
        shake, a quick lean) carries it off-screen while the object is still in
        view, and closing that track early would count the object twice. It only
        counts when the last observed box was within one box size of the border
        the prediction crossed (a fast exit between two low-rate frames).
        """
        x1, y1, x2, y2 = track.last_box
        m = self.exit_margin
        if x1 <= m or y1 <= m or x2 >= 1.0 - m or y2 >= 1.0 - m:
            return True
        cx, cy = float(track.state.x[0]), float(track.state.x[1])
        size = max(x2 - x1, y2 - y1)
        gaps = [gap for crossed, gap in ((cx < 0.0, x1), (cx > 1.0, 1.0 - x2), (cy < 0.0, y1), (cy > 1.0, 1.0 - y2))
                if crossed]
        return any(gap <= size for gap in gaps)

    def _expired(self, track: _Track, ts_ms: int | None) -> bool:
        if track.time_since_update == 0:
            return False
        timed = ts_ms is not None and track.last_ts_ms is not None
        if timed and self.exit_lost_ms is not None and self._is_exiting(track) \
                and ts_ms - track.last_ts_ms > self.exit_lost_ms:
            return True
        if timed and self.max_lost_ms is not None:
            return ts_ms - track.last_ts_ms > self.max_lost_ms
        return track.time_since_update > self.max_age

    def update(self, detections: list[Detection], ts_ms: int | None = None) -> list[Detection]:
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
        det_vectors = ([_unit(d.attrs.get(self.embedding_key)) for d in detections]
                       if self.appearance else None)
        high_idx = [i for i, d in enumerate(detections) if d.score >= self.score_high]
        low_idx = [
            i for i, d in enumerate(detections)
            if self.score_low <= d.score < self.score_high
        ]
        all_track_idx = list(range(len(self._tracks)))

        # 3) First-stage: confirmed/unconfirmed tracks vs high-conf detections.
        matches_a, unmatched_tracks, unmatched_high = self._match_subset(
            all_track_idx, high_idx, track_boxes, det_boxes, det_labels,
            self.iou_threshold, det_vectors,
        )

        # 3b) Appearance re-identification of lost tracks (motion failed).
        matches_r: list[tuple[int, int]] = []
        if det_vectors is not None:
            matches_r, unmatched_high = self._reidentify(
                unmatched_tracks, unmatched_high, detections, det_vectors)
            revived = {ti for ti, _ in matches_r}
            unmatched_tracks = [ti for ti in unmatched_tracks if ti not in revived]

        # 3c) Observation-centric recovery: lost tracks vs their last observed box.
        if self.observation_recovery and unmatched_tracks and unmatched_high:
            last_boxes = np.array([t.last_box for t in self._tracks], dtype=np.float32).reshape(-1, 4)
            matches_o, unmatched_tracks, unmatched_high = self._match_subset(
                unmatched_tracks, unmatched_high, last_boxes, det_boxes, det_labels,
                self.iou_threshold, det_vectors,
            )
            matches_r += matches_o

        # 3d) Nearby recovery: small, fast objects that no longer overlap their last box.
        if self.recovery_distance > 0 and unmatched_tracks and unmatched_high:
            lost = [ti for ti in unmatched_tracks if self._tracks[ti].time_since_update > 0]
            matches_n, _, unmatched_high = self._recover_nearby(
                lost, unmatched_high, detections, det_vectors, ts_ms)
            recovered = {ti for ti, _ in matches_n}
            unmatched_tracks = [ti for ti in unmatched_tracks if ti not in recovered]
            matches_r += matches_n

        # 4) Second-stage (BYTETrack): unmatched tracks vs low-conf detections.
        matches_b, unmatched_tracks, _ = self._match_subset(
            unmatched_tracks, low_idx, track_boxes, det_boxes, det_labels,
            self.low_threshold, det_vectors,
        )

        # 5) Apply matches: update Kalman, tag detection.
        for ti, di in matches_a + matches_b + matches_r:
            tr = self._tracks[ti]
            d = detections[di]
            relinked = (ti, di) in matches_r
            if relinked:
                # The motion model is stale after a long gap; restart it here.
                tr.state = KalmanBoxState(box_to_z(d.box))
                d.attrs["track_relinked"] = True
            else:
                tr.state.update(box_to_z(d.box))
            tr.hits += 1
            tr.time_since_update = 0
            tr.label = d.label
            tr.last_box = tuple(d.box)
            tr.last_ts_ms = ts_ms
            if det_vectors is not None:
                self._update_embedding(tr, det_vectors[di], d.score)
            if not tr.is_confirmed and tr.hits >= self.min_hits:
                tr.is_confirmed = True
            d.attrs["track_id"] = tr.track_id

        # 6) Spawn new tracks for unmatched high-conf detections.
        for di in unmatched_high:
            d = detections[di]
            tr = self._new_track(d, ts_ms, det_vectors[di] if det_vectors is not None else None)
            self._tracks.append(tr)
            d.attrs["track_id"] = tr.track_id

        # 7) Prune dead tracks.
        survivors: list[_Track] = []
        for tr in self._tracks:
            if self._expired(tr, ts_ms):
                self._closed_ids.append(tr.track_id)
                continue
            survivors.append(tr)
        self._tracks = survivors
        return detections


def make_tracker(algorithm: str = "iou_kalman", **kwargs) -> Tracker:
    if algorithm in {"iou_kalman", "bytetrack"}:
        return IoUKalmanTracker(**kwargs)
    raise ValueError(f"unknown tracking algorithm: {algorithm!r}")
