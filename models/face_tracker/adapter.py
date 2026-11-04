"""Face counting stage: detect, re-identify, track and count faces per camera.

Per frame and per stream:

1. **Detect** faces with RetinaFace (``detector_model``) or YuNet
   (``light_detector_model``), BYTETrack style: boxes scoring at least
   ``low_confidence`` are kept, but only boxes at ``face_confidence`` or above
   can start (and therefore count) a track; weaker ones only keep existing
   tracks alive through blur and partial occlusion.
2. **Embed lazily** with the AuraFace model (``model``): a face is aligned on
   its five keypoints and embedded only when it is new (it does not overlap a
   face tracked in the previous frame) or its track's vector is older than
   ``embed_refresh_ms``. Tiny or strongly turned faces are not embedded.
3. **Track** with the core ``IoUKalmanTracker`` in appearance mode: the vector
   stops ID swaps when people cross, and re-links a face that drops out for a
   while (head turn, occlusion, missed detection) to its old ID; a face that
   reappears where it was last seen is recovered by box overlap even without a
   vector (observation-centric recovery). A track lost
   at the frame border is closed after ``exit_ms`` (the person left); one lost
   in the middle of the view is kept for ``hold_ms``.
4. **Count** each track once, after ``min_hits`` detections: ``unique_faces``
   (cumulative for this session), ``faces_in_frame`` (counted faces visible
   now) and ``new_faces``; with a configured line or zone also ``line_in``,
   ``line_out``, ``zone_entries`` and ``zone_occupancy`` (see ``counting.py``).
   A person who leaves the view and comes back is a new track and counts
   again; nothing is remembered across cameras or sessions.

Face vectors stay in memory inside the tracker and are never published.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from inferops.adapters.outbound.observability import TRACKS_ACTIVE, TRACKS_CLOSED, TRACKS_STARTED, get_logger
from inferops.adapters.outbound.stages.base import Stage
from inferops.domain.errors import ConfigError
from inferops.domain.tracking import IoUKalmanTracker
from inferops.domain.types import Detection, Result, StageItem

from ..face_detector import yunet
from ..face_detector.yunet import FaceDetection
from ..face_retinaface import retinaface
from .align import align_face, frontal, to_tensor
from .counting import CountingGeometry, LineCounter, ZoneCounter, parse_counting, simple_line

log = get_logger(__name__)

COUNTER_KEYS = ("faces_in_frame", "unique_faces", "new_faces")
NEW_PERSON_ALERT_LEVELS = ("off", "low", "medium", "high", "critical")
LINE_KEYS = ("line_in", "line_out")
ZONE_KEYS = ("zone_entries", "zone_occupancy")


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _overlap(a, b) -> float:
    """Intersection over the smaller box: 1.0 when one box lies inside the other."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / smaller if smaller > 0 else 0.0


def _input_spec(registry, model_name: str, default_name: str, default_size: int) -> tuple[str, int]:
    spec = registry.get_spec(model_name)
    if not spec.inputs:
        return default_name, default_size
    shape = spec.inputs[0].shape
    size = int(shape[-1]) if len(shape) >= 4 and int(shape[-1]) > 0 else default_size
    return spec.inputs[0].name or default_name, size


@dataclass
class _StreamState:
    tracker: IoUKalmanTracker
    geometry: CountingGeometry
    line: LineCounter | None
    zone: ZoneCounter | None
    unique: int = 0
    counted: set[int] = field(default_factory=set)
    counted_ms: dict[int, int] = field(default_factory=dict)        # when each live track was counted
    numbers: dict[int, int] = field(default_factory=dict)           # its running number (#12 = 12th person)
    first_seen_ms: dict[int, int] = field(default_factory=dict)     # when each live track was first seen
    pending: dict[int, list[str]] = field(default_factory=dict)   # events seen before a track is counted
    previous: dict[int, tuple[float, float, float, float]] = field(default_factory=dict)  # visible boxes
    embedded_ms: dict[int, int] = field(default_factory=dict)
    last_ts_ms: int | None = None
    touched: float = field(default_factory=time.monotonic)


class FaceCountingStage(Stage):
    """Pipeline stage that counts and tracks faces.

    YAML usage (generated from ``models/face_tracker/model.yaml``)::

        - name: face_tracker
          stage: .models.face_tracker.adapter:FaceCountingStage
          params:
            model: face_tracker
            detector_model: face_retinaface      # or light_detector_model: face_detector
            counting: {cameras: {cam-1: {line: {points: [[0, 0.55], [1, 0.55]]}}}}
    """

    def __init__(
        self,
        model: str = "face_tracker",
        detector_model: str | None = None,
        light_detector_model: str | None = None,
        face_confidence: float = 0.6,
        low_confidence: float = 0.3,
        nms_threshold: float = 0.4,
        min_face_px: float = 20,
        max_faces: int = 64,
        reid: bool = True,
        min_reid_face_px: float = 40,
        reid_threshold: float = 0.45,
        appearance_gate: float = 0.2,
        reid_max_distance: float = 4.0,
        embed_refresh_ms: int = 1000,
        embed_batch_size: int = 1,
        min_hits: int = 3,
        min_track_ms: int = 300,
        iou_threshold: float = 0.3,
        low_iou_threshold: float = 0.5,
        hold_ms: int = 5000,
        exit_ms: int = 500,
        exit_margin: float = 0.02,
        recovery_distance: float = 1.0,
        recovery_ms: int = 1000,
        counting: dict[str, Any] | None = None,
        line: str = "off",
        line_position: float = 0.5,
        line_in: str = "down",
        new_highlight_ms: int = 3000,
        new_person_alert: str = "off",
        state_ttl_s: float = 300.0,
        fallback_model: str | None = None,
    ) -> None:
        params = dict(locals())
        params.pop("self")
        params.pop("__class__", None)
        params["fallback_model"] = None  # injected by the loader for every model stage; no fallback
        super().__init__(**params)
        if not 0.0 <= low_confidence <= face_confidence <= 1.0:
            raise ValueError("need 0 <= low_confidence <= face_confidence <= 1")
        if min_hits < 1 or max_faces < 1 or embed_batch_size < 1:
            raise ValueError("min_hits, max_faces and embed_batch_size must be at least 1")
        if hold_ms < 0 or exit_ms < 0:
            raise ValueError("hold_ms and exit_ms must not be negative")
        self.model = model
        self.detector_model = detector_model
        self.light_detector_model = light_detector_model
        self.face_confidence = float(face_confidence)
        self.low_confidence = float(low_confidence)
        self.nms_threshold = float(nms_threshold)
        self.min_face_px = float(min_face_px)
        self.max_faces = int(max_faces)
        self.reid = bool(reid)
        self.min_reid_face_px = float(min_reid_face_px)
        self.embed_refresh_ms = int(embed_refresh_ms)
        self._embed_batch = int(embed_batch_size)
        self.tracker_params = dict(
            min_hits=int(min_hits), iou_threshold=float(iou_threshold), low_threshold=float(low_iou_threshold),
            score_high=self.face_confidence, score_low=self.low_confidence, per_class=True,
            appearance=self.reid, appearance_gate=float(appearance_gate), reid_threshold=float(reid_threshold),
            reid_max_distance=float(reid_max_distance), max_lost_ms=int(hold_ms), exit_lost_ms=int(exit_ms),
            exit_margin=float(exit_margin), observation_recovery=True,
            recovery_distance=float(recovery_distance), recovery_max_ms=int(recovery_ms),
        )
        self.min_track_ms = max(0, int(min_track_ms))
        # The simple line settings give every camera a line unless the `counting`
        # default draws its own; a camera entry in `counting.cameras` replaces both.
        counting = dict(counting or {})
        default = dict(counting.get("default") or {})
        if "line" not in default:
            line_config = simple_line(line, line_position, line_in)
            if line_config:
                default["line"] = line_config
        counting["default"] = default
        self.default_geometry, self.camera_geometry = parse_counting(counting)
        self.new_highlight_ms = max(0, int(new_highlight_ms))
        if new_person_alert not in NEW_PERSON_ALERT_LEVELS:
            raise ValueError(f"new_person_alert must be one of {list(NEW_PERSON_ALERT_LEVELS)}")
        self.new_person_alert = new_person_alert
        self.state_ttl_s = state_ttl_s
        self._states: dict[str, _StreamState] = {}

    async def setup(self, ctx) -> None:
        await super().setup(ctx)
        if not (self.detector_model or self.light_detector_model):
            raise ConfigError("face_tracker needs a face detector: select face_retinaface or face_detector")

    @property
    def active_detector(self) -> str:
        return self.detector_model or self.light_detector_model or ""

    # -- per-stream state ---------------------------------------------------

    def _state(self, item: StageItem) -> _StreamState:
        now = time.monotonic()
        key = item.stream_id or "_"
        state = self._states.get(key)
        if state is not None and state.last_ts_ms is not None and item.ts_ms < state.last_ts_ms - 1000:
            state = None  # media time went backwards (looped/restarted source): a new timeline
        if state is None:
            for stale in [k for k, s in self._states.items() if now - s.touched > self.state_ttl_s]:
                del self._states[stale]
            camera = str(item.payload.get("stream_name") or item.stream_id or "")
            geometry = self.camera_geometry.get(camera, self.default_geometry)
            state = self._states[key] = _StreamState(
                tracker=IoUKalmanTracker(**self.tracker_params),
                geometry=geometry,
                line=LineCounter(geometry.line) if geometry.line else None,
                zone=ZoneCounter(geometry.zone) if geometry.zone else None,
            )
        state.touched = now
        return state

    # -- models ---------------------------------------------------------------

    async def _infer(self, model_name: str, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        backend = await self.ctx.registry.get_backend(model_name)
        return await backend.infer(feeds)

    async def _detect(self, image: np.ndarray) -> list[FaceDetection]:
        registry = self.ctx.registry
        if self.detector_model:
            name, size = _input_spec(registry, self.detector_model, "input", retinaface.INPUT_SIZE)
            feeds, scale = await asyncio.to_thread(retinaface.preprocess, image, name, size)
            raw = await self._infer(self.detector_model, feeds)
            return await asyncio.to_thread(retinaface.decode, raw, scale, size=size,
                                           score_threshold=self.low_confidence,
                                           nms_threshold=self.nms_threshold, top_k=self.max_faces)
        name, size = _input_spec(registry, self.light_detector_model, "input", yunet.INPUT_SIZE)
        feeds, scale = await asyncio.to_thread(yunet.preprocess, image, name, size)
        raw = await self._infer(self.light_detector_model, feeds)
        return await asyncio.to_thread(yunet.decode, raw, scale, score_threshold=self.low_confidence,
                                       nms_threshold=self.nms_threshold, size=size, top_k=self.max_faces)

    async def _embed(self, image: np.ndarray, keypoints: list[np.ndarray]) -> list[np.ndarray]:
        input_name, _ = _input_spec(self.ctx.registry, self.model, "data", 112)
        crops = await asyncio.to_thread(lambda: [align_face(image, k) for k in keypoints])
        vectors: list[np.ndarray] = []
        start = 0
        while start < len(crops):
            chunk = crops[start:start + self._embed_batch]
            try:
                raw = await self._infer(self.model, {input_name: to_tensor(chunk)})
            except Exception:
                if len(chunk) == 1:
                    raise
                log.warning("face_tracker: batched embedding failed; embedding one face per call",
                            extra={"batch": len(chunk)})
                self._embed_batch = 1
                continue
            out = np.asarray(next(iter(raw.values())), np.float32).reshape(len(chunk), -1)
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            vectors.extend(out / np.maximum(norms, 1e-12))
            start += len(chunk)
        return vectors

    def _needs_embedding(self, face: FaceDetection, box, state: _StreamState, ts_ms: int) -> bool:
        if face.score < self.face_confidence or min(face.width, face.height) < self.min_reid_face_px:
            return False
        if not frontal(face.keypoints):
            return False
        for track_id, previous in state.previous.items():
            if _iou(box, previous) >= 0.5:
                embedded = state.embedded_ms.get(track_id)
                if embedded is not None and ts_ms - embedded < self.embed_refresh_ms:
                    return False
        return True

    # -- counting ---------------------------------------------------------------

    def _record(self, state: _StreamState, track_id: int, event: tuple[str, str]) -> None:
        if track_id in state.counted:
            self._apply(state, event)
        else:
            state.pending.setdefault(track_id, []).append(event)

    @staticmethod
    def _apply(state: _StreamState, event: tuple[str, str]) -> None:
        kind, value = event
        if kind == "line" and state.line is not None:
            state.line.apply(value)
        elif kind == "zone" and state.zone is not None:
            state.zone.apply()

    # -- stage ----------------------------------------------------------------

    async def process(self, item: StageItem) -> StageItem:
        image = item.payload.get("image_np")
        if image is None:
            return item
        state = self._state(item)
        state.last_ts_ms = item.ts_ms
        h, w = image.shape[:2]
        faces = [f for f in await self._detect(image) if min(f.width, f.height) >= self.min_face_px]
        faces = faces[:self.max_faces]

        detections: list[Detection] = []
        for face in faces:
            x1, y1, x2, y2 = face.box
            box = (max(0.0, x1 / w), max(0.0, y1 / h), min(1.0, x2 / w), min(1.0, y2 / h))
            detections.append(Detection(label="face", score=face.score, box=box, attrs={
                "orig_w": w, "orig_h": h, "xyxy_pixels": tuple(float(v) for v in face.box),
                "keypoints": [[round(float(x) / w, 5), round(float(y) / h, 5)] for x, y in face.keypoints]}))

        embedded_now: list[int] = []
        if self.reid:
            wanted = [i for i, (face, det) in enumerate(zip(faces, detections, strict=True))
                      if self._needs_embedding(face, det.box, state, item.ts_ms)]
            if wanted:
                vectors = await self._embed(image, [faces[i].keypoints for i in wanted])
                for i, vector in zip(wanted, vectors, strict=True):
                    detections[i].attrs["embedding"] = vector
                embedded_now = wanted

        tracker = state.tracker
        tracker.update(detections, ts_ms=item.ts_ms)
        views = {view.track_id: view for view in tracker.tracks()}

        started = 0
        visible: list[Detection] = []
        for index, det in enumerate(detections):
            det.attrs.pop("embedding", None)  # never published
            track_id = det.attrs.get("track_id")
            if track_id is None:
                continue  # weak box that matched no track: noise
            view = views.get(track_id)
            if view is not None and view.hits == 1 and not det.attrs.get("track_relinked"):
                started += 1
            if index in embedded_now:
                state.embedded_ms[track_id] = item.ts_ms
            state.first_seen_ms.setdefault(track_id, item.ts_ms)
            centre =((det.box[0] + det.box[2]) / 2.0, (det.box[1] + det.box[3]) / 2.0)
            if state.line is not None:
                crossing = state.line.update(track_id, centre)
                if crossing:
                    self._record(state, track_id, ("line", crossing))
            if state.zone is not None and state.zone.update(track_id, centre):
                self._record(state, track_id, ("zone", "entry"))
            visible.append(det)

        new_people: list[dict[str, Any]] = []
        counted_boxes = [det.box for det in visible if det.attrs["track_id"] in state.counted]
        for det in visible:
            track_id = det.attrs["track_id"]
            view = views.get(track_id)
            # Count a track once it has min_hits detections over at least
            # min_track_ms (3 frames at 30 fps is only 0.1 s), and not while its box
            # is a second box on a face that is already counted in this frame.
            if (track_id not in state.counted and view is not None and view.confirmed
                    and item.ts_ms - state.first_seen_ms.get(track_id, item.ts_ms) >= self.min_track_ms
                    and not any(_overlap(det.box, box) >= 0.6 for box in counted_boxes)):
                counted_boxes.append(det.box)
                state.counted.add(track_id)
                state.counted_ms[track_id] = item.ts_ms
                state.unique += 1
                state.numbers[track_id] = state.unique
                # Running number: the 12th person counted on this camera is #12.
                new_people.append({"track_id": track_id, "number": state.unique, "box": list(det.box)})
                for event in state.pending.pop(track_id, []):
                    self._apply(state, event)
            det.attrs["counted"] = track_id in state.counted

        closed = tracker.pop_closed_ids()
        for track_id in closed:
            state.counted.discard(track_id)
            state.counted_ms.pop(track_id, None)
            state.first_seen_ms.pop(track_id, None)
            state.numbers.pop(track_id, None)
            state.pending.pop(track_id, None)
            state.embedded_ms.pop(track_id, None)
            if state.line is not None:
                state.line.forget(track_id)
            if state.zone is not None:
                state.zone.forget(track_id)
        state.previous = {det.attrs["track_id"]: det.box for det in visible}

        counted_visible = [det for det in visible if det.attrs["counted"]]
        counters: dict[str, int] = {
            "faces_in_frame": len(counted_visible),
            "unique_faces": state.unique,
            "new_faces": len(new_people),
        }
        if state.line is not None:
            counters.update(line_in=state.line.count_in, line_out=state.line.count_out)
        if state.zone is not None:
            occupancy = sum(1 for det in counted_visible if state.zone.contains(
                ((det.box[0] + det.box[2]) / 2.0, (det.box[1] + det.box[3]) / 2.0)))
            counters.update(zone_entries=state.zone.entries, zone_occupancy=occupancy)

        item.results.append(Result(kind="detections", detections=visible, model=self.active_detector,
                                   raw={"faces": len(visible), "embedded": len(embedded_now)}))
        item.payload.update(counters)

        def is_new(track_id: int) -> bool:
            counted_at = state.counted_ms.get(track_id)
            return counted_at is not None and 0 <= item.ts_ms - counted_at <= self.new_highlight_ms

        item.payload["face_tracks"] = [
            {"track_id": det.attrs["track_id"], "box": list(det.box), "counted": det.attrs["counted"],
             "number": state.numbers.get(det.attrs["track_id"]), "new": is_new(det.attrs["track_id"]),
             "relinked": bool(det.attrs.get("track_relinked"))}
            for det in visible]
        if new_people:
            item.payload["new_people"] = new_people
            if self.new_person_alert != "off":
                # One alert (and so one incident with its evidence clip) per person,
                # even when several are counted on the same frame.
                camera = str(item.payload.get("stream_name") or item.stream_id or "")
                item.payload.setdefault("alert_candidates", []).extend({
                    "type": "new_person", "severity": self.new_person_alert,
                    "dedupe_key": f"new_person:{item.stream_id}:{person['number']}",
                    "payload": {"rule": "new_person", "message": f"New person #{person['number']} counted",
                                "camera": camera, "unique_faces": state.unique, **person},
                } for person in new_people)
        geometry = state.geometry.describe()
        if geometry:
            item.payload["counting_geometry"] = geometry
        item.payload.setdefault("signals", {}).update(counters)

        pipeline = getattr(self.ctx, "pipeline", "face_counting")
        TRACKS_ACTIVE.labels(pipeline=pipeline, label="face").set(tracker.active_count)
        if started:
            TRACKS_STARTED.labels(pipeline=pipeline, label="face").inc(started)
        if closed:
            TRACKS_CLOSED.labels(pipeline=pipeline, label="face").inc(len(closed))
        return item
