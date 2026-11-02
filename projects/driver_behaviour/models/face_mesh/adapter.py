"""Face stage for driver behaviour: head direction, eye state, mouth activity.

Per frame and per stream:

1. **Locate** the driver face. While a face is tracked, the ROI comes from the
   previous frame's landmarks (no detector call). Otherwise, and every
   ``redetect_interval_ms``, the YuNet ``detector_model`` scans the frame. With
   no detector configured the RF-DETR ``DriverBehaviour`` box is used.
2. **Landmarks**: a square, roll-aligned crop (one ``warpAffine``) feeds the
   FaceMeshV2 model; all 478 landmarks (incl. iris) map back to the frame.
3. **Head pose**: weighted Procrustes fit to MediaPipe's canonical face
   (``pose.py``), One-Euro smoothed, relative to a per-stream auto-calibrated
   neutral ("looking at the road") pose, classified with hysteresis into
   ``front | left | right | up | down``; ``away`` when an established face is
   lost (fully turned head).
4. **Eyes**: ``blendshape_model`` eyeBlink/eyeLookDown scores fused with an
   EAR normalised by the driver's own open-eye EAR (``eyes.py``).
5. **Mouth**: cigarette / food detections intersecting the lower face.

Fusion reads ``look_{left,right,up,down}_score``, ``eyes_closed_score``,
``distracted_score``, ``drowsy_ear``, ``smoking_face_score`` and
``eating_face_score``. All model calls go through the registry, so the stage
runs unchanged on ONNX Runtime CPU/CUDA, TensorRT and Triton.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from inferops.adapters.outbound.observability import get_logger
from inferops.adapters.outbound.stages.base import Stage
from inferops.domain.types import StageItem

from ..face_detector.yunet import FaceDetection
from ..face_detector.yunet import decode as decode_faces
from ..face_detector.yunet import preprocess as preprocess_faces
from .eyes import (
    EYE_BLINK_LEFT,
    EYE_BLINK_RIGHT,
    EYE_LOOK_DOWN_LEFT,
    EYE_LOOK_DOWN_RIGHT,
    LEFT_EYE_RING,
    RIGHT_EYE_RING,
    EyeObservation,
    EyeState,
    EyeStateTracker,
    blendshape_points,
    eye_aspect_ratio,
    eye_width,
)
from .pose import (
    DirectionClassifier,
    DirectionThresholds,
    NeutralPoseCalibrator,
    OneEuroFilter,
    Presence,
    estimate_pose,
)

log = get_logger(__name__)

# Key mouth corners + chin for the smoking / eating intersection region
_LOWER_FACE = (61, 291, 0, 17, 152, 148, 176, 140)
# Face class name produced by RF-DETR driver model
_DRIVER_BEHAVIOUR_CLASS = "DriverBehaviour"
# Mouth-region padding (fraction of frame)
_MOUTH_PAD = 0.04
# Eye corners used for the in-plane ROI rotation (image-left eye first)
_ROI_EYES = (33, 263)
_DIRECTION_LABELS = {"left": "Looking Left", "right": "Looking Right", "up": "Looking Up",
                     "down": "Looking Down", "away": "Looking Away"}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Roi:
    """Square, rotated region: centre and side in pixels, angle in radians."""
    cx: float
    cy: float
    size: float
    angle: float = 0.0


def roi_from_eyes(box: tuple[float, float, float, float], image_left_eye, image_right_eye,
                  scale: float) -> Roi:
    x1, y1, x2, y2 = box
    angle = math.atan2(image_right_eye[1] - image_left_eye[1], image_right_eye[0] - image_left_eye[0])
    return Roi((x1 + x2) / 2.0, (y1 + y2) / 2.0, max(x2 - x1, y2 - y1) * scale, angle)


def roi_from_landmarks(points: np.ndarray, scale: float) -> Roi:
    """Next-frame ROI: landmark extent measured in the face's own rotated frame."""
    left, right = points[_ROI_EYES[0], :2], points[_ROI_EYES[1], :2]
    angle = math.atan2(right[1] - left[1], right[0] - left[0])
    c, s = math.cos(angle), math.sin(angle)
    rotated = points[:, :2] @ np.array([[c, -s], [s, c]])
    low, high = rotated.min(axis=0), rotated.max(axis=0)
    centre = ((low + high) / 2.0) @ np.array([[c, s], [-s, c]])
    return Roi(float(centre[0]), float(centre[1]), float(max(high - low)) * scale, angle)


def snap_roi(roi: Roi, step: float = 0.02) -> Roi:
    """Snap the tracking crop to a grid of ``step`` x face size and 1 degree.

    Each crop comes from the previous frame's landmarks, so sub-pixel landmark
    noise would otherwise feed back into every next crop (jitter, and drift
    between runtimes whose float results differ in the last digits). On the
    grid such differences only matter at a cell boundary and re-converge.
    """
    log_step = math.log1p(step)
    size = math.exp(round(math.log(max(roi.size, 1e-3)) / log_step) * log_step)
    cell = step * size
    angle = math.radians(round(math.degrees(roi.angle)))
    return Roi(round(roi.cx / cell) * cell, round(roi.cy / cell) * cell, size, angle)


def crop_roi(image: np.ndarray, roi: Roi, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``size``x``size`` crop and the 2x3 affine crop px -> image px."""
    k = roi.size / size
    c, s = math.cos(roi.angle) * k, math.sin(roi.angle) * k
    half = size / 2.0
    affine = np.array([[c, -s, roi.cx - c * half + s * half],
                       [s, c, roi.cy - s * half - c * half]], dtype=np.float64)
    patch = cv2.warpAffine(image, affine, (size, size), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_CONSTANT)
    return patch, affine


def landmarks_to_image(landmarks: np.ndarray, affine: np.ndarray, roi: Roi, size: int) -> np.ndarray:
    """Map crop-space (x, y, z) landmarks to image pixels (z in the same scale)."""
    xy = landmarks[:, :2] @ affine[:, :2].T + affine[:, 2]
    return np.column_stack([xy, landmarks[:, 2] * (roi.size / size)])


def parse_face_mesh(raw: dict[str, np.ndarray], face_flag_output: str = "Identity_1",
                    face_flag_is_logit: bool = True) -> tuple[np.ndarray | None, float]:
    """Return crop-space landmarks (N, 3) with N = 478 (or 468) and presence."""
    confidence = 0.0
    # FaceMeshV2's second output is the face-presence logit. Other scalar
    # heads are not evidence of face presence; taking their maximum can turn
    # a rejected seat/background crop into a confident "Driver Face".
    flag = raw.get(face_flag_output)
    if flag is not None and flag.size == 1:
        value = float(flag.flat[0])
        if np.isfinite(value):
            if face_flag_is_logit:
                value = float(1 / (1 + np.exp(-np.clip(value, -80, 80))))
            confidence = float(np.clip(value, 0.0, 1.0))
    landmarks = None
    for array in sorted(raw.values(), key=lambda a: a.size, reverse=True):
        if array.size >= 1404:
            count = 478 if array.size >= 1434 else 468
            landmarks = array.reshape(-1)[: count * 3].reshape(count, 3).astype(np.float64)
            break
    if landmarks is None or not np.isfinite(landmarks).all():
        return None, confidence
    return landmarks, confidence


def _box_area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    return inter / max(_box_area(a) + _box_area(b) - inter, 1e-9)


def _driver_boxes(results, width: int, height: int) -> list[tuple[float, float, float, float]]:
    """RF-DETR ``DriverBehaviour`` boxes in pixels (other classes never count)."""
    boxes = []
    for result in results:
        for d in ((result.raw or {}).get("by_class", {}).get(_DRIVER_BEHAVIOUR_CLASS) or []):
            box = d.get("box")
            if box is not None:
                boxes.append((box[0] * width, box[1] * height, box[2] * width, box[3] * height))
    return boxes


def _face_box_from_landmarks(points: np.ndarray, width: int, height: int) -> tuple[float, float, float, float]:
    xs, ys = points[:, 0] / width, points[:, 1] / height
    pad_x = max(0.02, float(xs.max() - xs.min()) * 0.08)
    pad_y = max(0.02, float(ys.max() - ys.min()) * 0.12)
    return (max(0.0, float(xs.min()) - pad_x), max(0.0, float(ys.min()) - pad_y),
            min(1.0, float(xs.max()) + pad_x), min(1.0, float(ys.max()) + pad_y))


# ---------------------------------------------------------------------------
# Mouth activity (cigarette / food x lower face)
# ---------------------------------------------------------------------------

def _mouth_bbox(landmarks: list[dict[str, float]]) -> tuple[float, float, float, float] | None:
    """Return normalised (x1, y1, x2, y2) covering the mouth + lower-face region."""
    if len(landmarks) < 400:
        return None
    xs = [landmarks[i]["x"] for i in _LOWER_FACE]
    ys = [landmarks[i]["y"] for i in _LOWER_FACE]
    return min(xs) - _MOUTH_PAD, min(ys) - _MOUTH_PAD, max(xs) + _MOUTH_PAD, max(ys) + _MOUTH_PAD


def _intersection_score(detections: list[dict[str, Any]], region: tuple[float, float, float, float]) -> float:
    """Best confidence of a detection whose normalised box overlaps ``region``."""
    rx1, ry1, rx2, ry2 = region
    best = 0.0
    for det in detections:
        box = det.get("box")
        if box is None:
            continue
        dx1, dy1, dx2, dy2 = box
        if min(rx2, dx2) > max(rx1, dx1) and min(ry2, dy2) > max(ry1, dy1):
            best = max(best, float(det.get("score", 0.5)))
    return best


def _apply_mouth_scores(item: StageItem, landmarks: list[dict[str, float]]) -> None:
    # Prefer face bbox as the intersection target — it is far more robust than
    # relying on tiny mouth-landmark coordinates alone.  Food / cups are often
    # held in front of the whole lower face, not just the lip region.
    face_box = item.payload.get("face_box")
    mouth_region = _mouth_bbox(landmarks)
    if face_box is not None:
        fx1, fy1, fx2, fy2 = face_box
        face_h = max(fy2 - fy1, 1e-6)
        eating_region = (fx1 - 0.02, fy1 + face_h * 0.40, fx2 + 0.02, fy2 + 0.06)
        smoking_region = (fx1 - 0.01, fy1 + face_h * 0.50, fx2 + 0.01, fy2 + 0.04)
    elif mouth_region is not None:
        eating_region = (mouth_region[0] - 0.05, mouth_region[1] - 0.08,
                         mouth_region[2] + 0.05, mouth_region[3] + 0.04)
        smoking_region = mouth_region
    else:
        return
    for result in item.results:
        by_class = (result.raw or {}).get("by_class", {})
        if by_class.get("cigarette"):
            score = _intersection_score(by_class["cigarette"], smoking_region)
            if score > item.payload.get("smoking_face_score", 0.0):
                item.payload["smoking_face_score"] = score
        if by_class.get("foodItem"):
            score = _intersection_score(by_class["foodItem"], eating_region)
            if score > item.payload.get("eating_face_score", 0.0):
                item.payload["eating_face_score"] = score


def face_status_label(direction: str | None, eyes_closed: bool = False, drowsy_score: float = 0.0,
                      eating_score: float = 0.0, smoking_score: float = 0.0) -> str:
    labels = [name for name, score in (("Eating", eating_score), ("Smoking", smoking_score)) if score >= 0.5]
    if eyes_closed:
        labels.append("Eyes Closed")
    if drowsy_score >= 0.5:
        labels.append("Drowsy")
    if direction in _DIRECTION_LABELS:
        labels.append(_DIRECTION_LABELS[direction])
    return " | ".join(labels) or "Driver Face"


# ---------------------------------------------------------------------------
# Model calls
# ---------------------------------------------------------------------------

def _input_spec(registry, model_name: str, default_name: str, default_size: int) -> tuple[str, int]:
    spec = registry.get_spec(model_name)
    if not spec.inputs:
        return default_name, default_size
    shape = spec.inputs[0].shape
    size = int(shape[-1]) if len(shape) >= 4 and int(shape[-1]) > 0 else default_size
    return spec.inputs[0].name or default_name, size


async def _infer(stage: FaceMeshStage, model_name: str, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    backend = await stage.ctx.registry.get_backend(model_name)
    return await backend.infer(feeds)


@dataclass(slots=True)
class _Mesh:
    points: np.ndarray        # (N, 3) image pixels
    confidence: float
    roi: Roi


@dataclass
class _StreamState:
    calibrator: NeutralPoseCalibrator
    classifier: DirectionClassifier
    eyes: EyeStateTracker
    presence: Presence
    filters: dict[str, OneEuroFilter] = field(default_factory=lambda: {
        axis: OneEuroFilter(min_cutoff=1.0, beta=0.05) for axis in ("yaw", "pitch", "roll")})
    roi: Roi | None = None
    face_box_px: tuple[float, float, float, float] | None = None
    last_detect_ms: int | None = None
    last_infer_ms: int | None = None
    cached: dict[str, Any] = field(default_factory=dict)
    # Where the driver's face was last seen: (cx, cy, size) px and when.
    anchor: tuple[float, float, float] | None = None
    anchor_ms: int | None = None
    # Detector box (px) of the driver face found at ``detected_ms``.
    detected_box: tuple[float, float, float, float] | None = None
    detected_turn: float = 0.0
    detected_ms: int | None = None
    touched: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

class FaceMeshStage(Stage):
    """
    Pipeline stage that enriches each item with face-derived driver signals.
    Plug it in *after* the object-detection ``infer`` stage and *before*
    ``fuse``.

    YAML usage::

        - name: face_mesh
          stage: .models.face_mesh.adapter:FaceMeshStage
          params:
            model: face_mesh
            detector_model: face_detector
            blendshape_model: face_blendshapes
            neutral: {mode: auto}

    Besides the raw ``payload`` keys the fusion DSL reads, the stage publishes
    the dashboard-facing view through ``payload["signals"]`` and
    ``payload["overlay_boxes"]`` (see ``_publish_signals``).
    """

    # Scalar payload keys mirrored into payload["signals"] for the dashboard.
    SIGNAL_KEYS = (
        "face_source", "face_mesh_confidence", "face_status_label", "head_direction",
        "eye_closure", "eye_left_state", "eye_right_state", "eyes_closed_ms", "perclos",
        "drowsy_ear", "distracted_score", "smoking_face_score", "eating_face_score",
    )
    # Decision keys held through a brief face dropout (< ``away_after_ms``).
    HOLD_KEYS = ("head_direction", "look_left_score", "look_right_score", "look_up_score", "look_down_score",
                 "distracted_score", "eyes_closed_score")
    # Keys replayed for frames skipped by ``min_infer_interval_ms``.
    CACHED_KEYS = (
        "face_mesh_confidence", "face_landmarks", "face_box", "face_source", "face_pose", "head_direction",
        "neutral_pose", "look_left_score", "look_right_score", "look_up_score", "look_down_score",
        "distracted_score", "eye_closure", "eye_left_closure", "eye_right_closure", "eye_left_state",
        "eye_right_state", "eye_state_reliable", "eyes_closed", "eyes_closed_score", "eyes_closed_ms",
        "gaze_down", "perclos", "blink_rate", "drowsy_ear",
    )

    def __init__(
        self,
        model: str = "face_mesh_onnx",
        fallback_model: str | None = None,
        detector_model: str | None = None,
        blendshape_model: str | None = None,
        face_confidence_threshold: float = 0.5,
        detector_score_threshold: float = 0.6,
        allow_haar_fallback: bool = False,
        face_flag_output: str = "Identity_1",
        face_flag_is_logit: bool = True,
        crop_padding: float = 0.25,
        roi_scale: float = 1.5,
        min_infer_interval_ms: int = 0,
        redetect_interval_ms: int = 2000,
        driver_selection: str = "largest",
        mirror: bool = False,
        neutral: dict[str, Any] | None = None,
        thresholds: dict[str, float] | None = None,
        eyes: dict[str, Any] | None = None,
        away_after_ms: int = 600,
        away_max_ms: int = 15000,
        profile_turn_ratio: float = 0.6,
        state_ttl_s: float = 300.0,
    ) -> None:
        if driver_selection not in {"largest", "left", "right"}:
            raise ValueError("driver_selection must be 'largest', 'left' or 'right'")
        params = dict(locals())
        params.pop("self")
        params.pop("__class__", None)
        params["fallback_model"] = None
        super().__init__(**params)
        self.model = model
        self.fallback_model = None  # Legacy parameter is accepted; model fallback is disabled.
        self.detector_model = detector_model
        self.blendshape_model = blendshape_model
        self.face_confidence_threshold = face_confidence_threshold
        self.detector_score_threshold = detector_score_threshold
        # Haar cascades were replaced by the YuNet detector; the flag is accepted for old configs.
        self.allow_haar_fallback = False
        self.face_flag_output = face_flag_output
        self.face_flag_is_logit = face_flag_is_logit
        self.crop_padding = crop_padding
        self.roi_scale = roi_scale
        self.min_infer_interval_ms = max(0, int(min_infer_interval_ms))
        self.redetect_interval_ms = max(0, int(redetect_interval_ms))
        self.driver_selection = driver_selection
        self.mirror = bool(mirror)
        self.neutral = dict(neutral or {})
        NeutralPoseCalibrator(**self.neutral)  # validate early
        self.thresholds = DirectionThresholds(**(thresholds or {}))
        self.eye_params = dict(eyes or {})
        EyeStateTracker(**self.eye_params)
        self.away_after_ms = away_after_ms
        self.away_max_ms = away_max_ms
        self.profile_turn_ratio = profile_turn_ratio
        self.state_ttl_s = state_ttl_s
        self._states: dict[str, _StreamState] = {}

    # -- per-stream state ---------------------------------------------------

    def _state(self, stream_id: str) -> _StreamState:
        now = time.monotonic()
        state = self._states.get(stream_id)
        if state is None:
            for key in [k for k, s in self._states.items() if now - s.touched > self.state_ttl_s]:
                del self._states[key]
            state = self._states[stream_id] = _StreamState(
                calibrator=NeutralPoseCalibrator(**self.neutral),
                classifier=DirectionClassifier(self.thresholds, mirror=self.mirror),
                eyes=EyeStateTracker(**self.eye_params),
                presence=Presence(self.away_max_ms),
            )
        state.touched = now
        return state

    # -- localisation + landmarks ------------------------------------------

    async def _mesh(self, image: np.ndarray, roi: Roi) -> _Mesh:
        input_name, size = _input_spec(self.ctx.registry, self.model, "input_12", 256)
        patch, affine = crop_roi(image, roi, size)
        rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        tensor = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis]
        raw = await _infer(self, self.model, {input_name: np.ascontiguousarray(tensor)})
        landmarks, confidence = parse_face_mesh(raw, self.face_flag_output, self.face_flag_is_logit)
        if landmarks is None:
            return _Mesh(np.empty((0, 3)), confidence, roi)
        return _Mesh(landmarks_to_image(landmarks, affine, roi, size), confidence, roi)

    def _choose(self, faces: list[FaceDetection], state: _StreamState, driver_boxes,
                t_ms: int) -> FaceDetection | None:
        if not faces:
            return None
        if state.face_box_px is not None:
            best = max(faces, key=lambda f: _iou(f.box, state.face_box_px))
            if _iou(best.box, state.face_box_px) >= 0.2:
                return best
        if state.anchor is not None and state.anchor_ms is not None and 0 <= t_ms - state.anchor_ms <= self.away_max_ms:
            # Re-acquire only near the driver's last position: a turned-away
            # driver must read as "away", not hand over to a passenger's face.
            cx, cy, size = state.anchor

            def distance(f: FaceDetection) -> float:
                return math.hypot((f.box[0] + f.box[2]) / 2 - cx, (f.box[1] + f.box[3]) / 2 - cy)

            near = [f for f in faces if distance(f) <= size]
            return min(near, key=distance) if near else None
        # Prefer faces centred inside an RF-DETR DriverBehaviour box (not passengers).
        inside = [f for f in faces if any(d[0] <= (f.box[0] + f.box[2]) / 2 <= d[2]
                                          and d[1] <= (f.box[1] + f.box[3]) / 2 <= d[3] for d in driver_boxes)]
        pool = inside or faces
        if self.driver_selection == "left":
            return min(pool, key=lambda f: f.box[0] + f.box[2])
        if self.driver_selection == "right":
            return max(pool, key=lambda f: f.box[0] + f.box[2])
        return max(pool, key=lambda f: f.width * f.height * f.score)

    async def _detect_roi(self, image: np.ndarray, state: _StreamState, item: StageItem, t_ms: int) -> Roi | None:
        h, w = image.shape[:2]
        driver_boxes = _driver_boxes(item.results, w, h)
        if self.detector_model is None:
            # Legacy path: the RF-DETR DriverBehaviour box is the face hint.
            if not driver_boxes:
                return None
            box = max(driver_boxes, key=_box_area)
            pad = 1.0 + 2.0 * self.crop_padding
            return Roi((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, max(box[2] - box[0], box[3] - box[1]) * pad)
        input_name, size = _input_spec(self.ctx.registry, self.detector_model, "input", 640)
        feeds, scale = preprocess_faces(image, input_name, size)
        raw = await _infer(self, self.detector_model, feeds)
        face = self._choose(decode_faces(raw, scale, score_threshold=self.detector_score_threshold, size=size),
                            state, driver_boxes, t_ms)
        if face is None:
            return None
        state.detected_box, state.detected_ms = face.box, t_ms
        # Nose offset from the eye midpoint in eye-distance units: about 0 for a
        # frontal face, beyond ~0.5 once the nose leaves the eye span (profile).
        eye_l, eye_r, nose = face.keypoints[0], face.keypoints[1], face.keypoints[2]
        state.detected_turn = float((nose[0] - (eye_l[0] + eye_r[0]) / 2) / max(abs(eye_r[0] - eye_l[0]), 1e-6))
        # YuNet keypoints: [0] eye on the image left, [1] eye on the image right.
        return roi_from_eyes(face.box, face.keypoints[0], face.keypoints[1], self.roi_scale)

    async def _locate(self, image: np.ndarray, state: _StreamState, item: StageItem, t_ms: int) -> _Mesh | None:
        due = (state.last_detect_ms is None or t_ms < state.last_detect_ms
               or t_ms - state.last_detect_ms >= self.redetect_interval_ms)
        best: _Mesh | None = None
        if state.roi is not None and not due:
            best = await self._mesh(image, state.roi)
            if best.confidence >= self.face_confidence_threshold and len(best.points):
                return best
        roi = await self._detect_roi(image, state, item, t_ms)
        state.last_detect_ms = t_ms
        if roi is None:
            # A periodic re-detection can miss a strongly turned face that the
            # landmark tracker still holds: keep tracking instead of dropping it.
            if best is None and state.roi is not None:
                best = await self._mesh(image, state.roi)
            return best
        mesh = await self._mesh(image, roi)
        if mesh.confidence >= self.face_confidence_threshold and len(mesh.points):
            return mesh
        return mesh if best is None or mesh.confidence >= best.confidence else best

    # -- stage entry ----------------------------------------------------------

    async def process(self, item: StageItem) -> StageItem:  # type: ignore[override]
        started = time.perf_counter()
        try:
            return await self._process(item)
        finally:
            item.payload["face_stage_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
            self._publish_signals(item)

    async def _process(self, item: StageItem) -> StageItem:
        image: np.ndarray | None = item.payload.get("image_np")
        if image is None or image.size == 0:
            return item
        h, w = image.shape[:2]
        t_ms = int(item.ts_ms)
        state = self._state(item.stream_id)

        if (self.min_infer_interval_ms and state.last_infer_ms is not None and state.cached
                and 0 <= t_ms - state.last_infer_ms < self.min_infer_interval_ms):
            item.payload.update(state.cached)
            self._finish_activity(item)
            return item
        state.last_infer_ms = t_ms

        mesh = await self._locate(image, state, item, t_ms)
        if mesh is not None:
            item.payload["face_mesh_confidence"] = mesh.confidence
        if mesh is None or mesh.confidence < self.face_confidence_threshold or len(mesh.points) < 468:
            state.roi = None
            state.face_box_px = None
            if state.detected_ms == t_ms and state.detected_box is not None:
                self._face_unreadable(item, state, t_ms, w, h)
            else:
                self._face_lost(item, state, t_ms)
            state.cached = {k: item.payload[k] for k in self.CACHED_KEYS if k in item.payload}
            return item

        points = mesh.points
        state.presence.seen(t_ms)
        state.roi = snap_roi(roi_from_landmarks(points, self.roi_scale))
        state.face_box_px = (float(points[:, 0].min()), float(points[:, 1].min()),
                             float(points[:, 0].max()), float(points[:, 1].max()))
        x1, y1, x2, y2 = state.face_box_px
        state.anchor, state.anchor_ms = ((x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1)), t_ms
        normalised = np.column_stack([np.clip(points[:, 0] / w, 0.0, 1.0), np.clip(points[:, 1] / h, 0.0, 1.0),
                                      points[:, 2]]).tolist()
        landmarks = [{"x": x, "y": y, "z": z} for x, y, z in normalised]
        item.payload["face_landmarks"] = landmarks
        item.payload["face_box"] = _face_box_from_landmarks(points, w, h)
        item.payload["face_source"] = "face_mesh"

        # ── head pose relative to the calibrated neutral pose ──────────────
        raw_pose = estimate_pose(points)
        t_s = t_ms / 1000.0
        yaw = state.filters["yaw"](raw_pose.yaw, t_s)
        pitch = state.filters["pitch"](raw_pose.pitch, t_s)
        roll = state.filters["roll"](raw_pose.roll, t_s)
        d_yaw, d_pitch = state.calibrator.offset(yaw, pitch)

        # ── eyes ────────────────────────────────────────────────────────────
        observation = EyeObservation(eye_aspect_ratio(points, LEFT_EYE_RING), eye_aspect_ratio(points, RIGHT_EYE_RING),
                                     width_left_px=eye_width(points, LEFT_EYE_RING),
                                     width_right_px=eye_width(points, RIGHT_EYE_RING))
        if self.blendshape_model and len(points) >= 478:
            raw = await _infer(self, self.blendshape_model, {
                _input_spec(self.ctx.registry, self.blendshape_model, "points", 0)[0]: blendshape_points(points)})
            scores = next(iter(raw.values())).reshape(-1)
            observation.blink_left = float(scores[EYE_BLINK_LEFT])
            observation.blink_right = float(scores[EYE_BLINK_RIGHT])
            observation.look_down = float((scores[EYE_LOOK_DOWN_LEFT] + scores[EYE_LOOK_DOWN_RIGHT]) / 2.0)
        eye = state.eyes.update(observation, t_ms, yaw_offset=d_yaw, pitch_offset=d_pitch,
                                pose_calibrated=state.calibrator.calibrated)

        # Only an attentive pose may define "front": eyes open and not looking down.
        attentive = not eye.reliable or (
            eye.closure < state.eyes.open_threshold
            and (observation.look_down is None or observation.look_down < state.eyes.gaze_down_look))
        # Calibrate on the raw pose: attentive samples are often brief road
        # glances that the smoothing filter would never let reach their angle.
        state.calibrator.update(raw_pose.yaw, raw_pose.pitch, t_ms, attentive=attentive)
        d_yaw, d_pitch = state.calibrator.offset(yaw, pitch)
        if state.calibrator.calibrated:
            direction = state.classifier.update(d_yaw, d_pitch, gaze_down=eye.gaze_down)
        else:
            # Until "front" is known for this camera only the eye cue can tell
            # "down"; head-angle directions wait for calibration.
            direction = "down" if eye.gaze_down else "front"
        if observation.look_down is not None:
            item.payload["eye_look_down"] = round(observation.look_down, 3)
        self._publish_face(item, direction, eye, raw_pose.residual,
                           {"yaw": yaw, "pitch": pitch, "roll": roll}, (d_yaw, d_pitch), state)
        state.cached = {k: item.payload[k] for k in self.CACHED_KEYS if k in item.payload}
        _apply_mouth_scores(item, landmarks)
        self._finish_activity(item)
        return item

    # -- outputs --------------------------------------------------------------

    def _face_unreadable(self, item: StageItem, state: _StreamState, t_ms: int, w: int, h: int) -> None:
        # The detector still sees the driver's face where it was, but it is too
        # small or blurred for landmarks: present, direction unknown, no alarm.
        x1, y1, x2, y2 = state.detected_box
        # Only a landmark-confirmed face establishes a driver; a bare detection
        # (e.g. a tiny picture-in-picture face) may only keep one alive.
        established = state.presence.lost_for_ms(t_ms) is not None
        if established:
            state.presence.seen(t_ms)
            state.anchor, state.anchor_ms = ((x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1)), t_ms
        turned = established and abs(state.detected_turn) >= self.profile_turn_ratio
        item.payload.update({
            "face_box": (max(0.0, x1 / w), max(0.0, y1 / h), min(1.0, x2 / w), min(1.0, y2 / h)),
            "face_source": "detector", "head_direction": "away" if turned else "unknown",
            "distracted_score": 1.0 if turned else 0.0,
            "look_left_score": 0.0, "look_right_score": 0.0, "look_up_score": 0.0, "look_down_score": 0.0,
        })

    def _face_lost(self, item: StageItem, state: _StreamState, t_ms: int) -> None:
        lost_ms = state.presence.lost_for_ms(t_ms)
        if lost_ms is None:
            return
        if lost_ms < self.away_after_ms:
            # Brief dropout: hold the last direction so a missed frame neither
            # resets nor fakes a fusion confirmation.
            item.payload.update({k: state.cached[k] for k in self.HOLD_KEYS if k in state.cached})
            return
        # An established driver face vanished: the head is turned beyond what
        # the landmark model can see (or the face is covered).
        state.classifier.state = "front"
        item.payload.update({
            "head_direction": "away", "distracted_score": 1.0, "face_lost_ms": lost_ms,
            "look_left_score": 0.0, "look_right_score": 0.0, "look_up_score": 0.0, "look_down_score": 0.0,
            "face_status_label": face_status_label("away"),
        })

    def _publish_face(self, item: StageItem, direction: str, eye: EyeState, residual: float,
                      pose: dict[str, float], offsets: tuple[float, float], state: _StreamState) -> None:
        calibrator = state.calibrator
        item.payload["face_pose"] = {
            **{k: round(v, 2) for k, v in pose.items()},
            **{f"{k}_deg": round(v, 2) for k, v in pose.items()},
            "yaw_offset": round(offsets[0], 2), "pitch_offset": round(offsets[1], 2),
            "fit_residual": round(residual, 3),
        }
        item.payload["neutral_pose"] = {"yaw": round(calibrator.yaw, 2), "pitch": round(calibrator.pitch, 2),
                                        "calibrated": calibrator.calibrated, "mode": calibrator.mode}
        item.payload["head_direction"] = direction if calibrator.calibrated or direction != "front" else "calibrating"
        for name in ("left", "right", "up", "down"):
            item.payload[f"look_{name}_score"] = 1.0 if direction == name else 0.0
        item.payload["distracted_score"] = 1.0 if direction != "front" else round(
            0.49 * max(state.classifier.scores(*offsets).values()), 3)
        item.payload.update({
            "eye_closure": round(eye.closure, 3),
            "eye_left_closure": round(eye.closure_left, 3),
            "eye_right_closure": round(eye.closure_right, 3),
            "eye_left_state": eye.left_state,
            "eye_right_state": eye.right_state,
            "eye_state_reliable": eye.reliable,
            "eyes_closed": eye.eyes_closed,
            "eyes_closed_score": 1.0 if eye.eyes_closed else 0.0,
            "eyes_closed_ms": eye.closed_ms,
            "gaze_down": eye.gaze_down,
            "perclos": round(eye.perclos, 3),
            "blink_rate": round(eye.blink_rate_per_min, 1),
            "drowsy_ear": round(eye.drowsy_score, 3),
        })

    def _finish_activity(self, item: StageItem) -> None:
        if "face_box" not in item.payload and item.payload.get("head_direction") != "away":
            return
        item.payload["face_status_label"] = face_status_label(
            item.payload.get("head_direction"),
            eyes_closed=bool(item.payload.get("eyes_closed")),
            drowsy_score=float(item.payload.get("drowsy_ear") or 0.0),
            eating_score=float(item.payload.get("eating_face_score", 0.0)),
            smoking_score=float(item.payload.get("smoking_face_score", 0.0)),
        )

    def _publish_signals(self, item: StageItem) -> None:
        if "face_box" not in item.payload and item.payload.get("head_direction") != "away":
            return
        signals = item.payload.setdefault("signals", {})
        for key in self.SIGNAL_KEYS:
            value = item.payload.get(key)
            if isinstance(value, (int, float, str)) and not isinstance(value, bool):
                signals[key] = round(value, 3) if isinstance(value, float) else value
        pose = item.payload.get("face_pose") or {}
        for axis in ("yaw", "pitch", "roll"):
            value = pose.get(f"{axis}_deg", pose.get(axis))
            if isinstance(value, (int, float)):
                signals[f"head_{axis}_deg"] = round(float(value), 1)
        if "face_box" not in item.payload:
            return
        label = str(item.payload.get("face_status_label") or "Face")
        tone = "danger" if any(k in label for k in ("Drowsy", "Eyes Closed")) else ("warn" if any(
            k in label for k in ("Looking", "Eating", "Smoking")) else "ok")
        item.payload["overlay_boxes"] = [{
            "label": label,
            "box": [float(v) for v in item.payload["face_box"]],
            "tone": tone,
        }]
