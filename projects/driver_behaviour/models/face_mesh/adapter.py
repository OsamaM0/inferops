"""Face Mesh inference stage for driver behaviour analysis.

Runs a MediaPipe FaceMesh-compatible ONNX model on the detected face region.
After the object-detection stage this stage:

1. Finds the driver face crop (from the ``DriverBehaviour`` detection class or
   the largest detection).
2. Resizes the crop to the configured model input size and runs the face-mesh model.
3. Maps the 468 landmarks back to normalised full-frame coordinates.
4. Computes:
   - ``drowsy_ear``           - EAR-based drowsiness score  [0, 1]
   - ``distracted_score``     - head-pose distraction score [0, 1]
   - ``smoking_face_score``   - cigarette x mouth-region intersection [0, 1]
   - ``eating_face_score``    - food x mouth-region intersection     [0, 1]
   - ``face_landmarks``       - list[dict] 468 items, keys x/y/z normalised
   - ``face_mesh_confidence`` - model face-flag confidence [0, 1]

All values are written into ``item.payload`` so the fusion engine can read
them via the existing ``drowsy_ear``, ``distracted_score``, ``smoking_face_score``
and ``eating_face_score`` extract expressions.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from inferops.core.types import StageItem
from inferops.obs import get_logger
from inferops.pipeline.stage import Stage

from .features import (
    DISTRACT_PITCH_DEG,
    DISTRACT_YAW_DEG,
    EAR_DROWSY_THRESHOLD,
    LEFT_EYE,
    RIGHT_EYE,
    behavior_scores_from_face,
    eye_aspect_ratio,
    head_pose,
)

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# MediaPipe 468-landmark mouth / lower-face indices
# ---------------------------------------------------------------------------
# Outer lip ring (indices used for mouth bbox)
_MOUTH_OUTER = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                409, 270, 269, 267, 0, 37, 39, 40, 185)
# Key mouth corners + chin for intersection region
_LOWER_FACE = (61, 291, 0, 17, 152, 148, 176, 140)

# Face class name produced by RF-DETR driver model
_DRIVER_BEHAVIOUR_CLASS = "DriverBehaviour"

# Mouth-region padding (fraction of face bbox size)
_MOUTH_PAD = 0.04

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_face_box(results, frame_w: int, frame_h: int) -> tuple[float, float, float, float] | None:
    """
    Return the best (x1, y1, x2, y2) normalised face box from results.

    Only considers detections labelled ``DriverBehaviour``; other classes
    (phone, cigarette, etc.) are deliberately ignored so they cannot be
    mistaken for the driver's face crop.
    """
    best_score = -1.0
    best_box: tuple | None = None
    for result in results:
        by_class = (result.raw or {}).get("by_class", {})
        candidates = by_class.get(_DRIVER_BEHAVIOUR_CLASS, []) or []
        for d in candidates:
            score = d.get("score", 0.0)
            box = d.get("box")
            if box is not None and score > best_score:
                best_score = score
                best_box = tuple(box)
    return best_box  # type: ignore[return-value]


def _crop_face(
    img: np.ndarray,
    box_norm: tuple[float, float, float, float],
    padding: float = 0.25,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop face region with padding; return (crop_bgr, (x1_px, y1_px, x2_px, y2_px))."""
    h, w = img.shape[:2]
    x1n, y1n, x2n, y2n = box_norm
    pw = padding * (x2n - x1n)
    ph = padding * (y2n - y1n)
    x1 = max(0, int((x1n - pw) * w))
    y1 = max(0, int((y1n - ph) * h))
    x2 = min(w, int((x2n + pw) * w))
    y2 = min(h, int((y2n + ph) * h))
    if x2 <= x1 or y2 <= y1:
        return img, (0, 0, w, h)
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)


def _box_area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _box_looks_like_face(box: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = box
    width = max(x2 - x1, 1e-6)
    height = max(y2 - y1, 1e-6)
    aspect = width / height
    area = width * height
    return 0.55 <= aspect <= 1.35 and 0.015 <= area <= 0.22


def _boxes_are_close(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    left_cx, left_cy = _box_center(left)
    right_cx, right_cy = _box_center(right)
    center_shift = max(abs(left_cx - right_cx), abs(left_cy - right_cy))
    left_area = max(_box_area(left), 1e-6)
    right_area = max(_box_area(right), 1e-6)
    area_ratio = max(left_area, right_area) / min(left_area, right_area)
    return center_shift <= 0.08 and area_ratio <= 1.7


def _select_face_box(
    detection_face_box: tuple[float, float, float, float] | None,
    haar_face: dict[str, Any] | None,
) -> tuple[float, float, float, float] | None:
    if detection_face_box is None:
        if haar_face is None:
            return None
        return tuple(haar_face["face_box"])

    if haar_face is None:
        return detection_face_box

    haar_box = tuple(haar_face["face_box"])
    detection_area = _box_area(detection_face_box)
    haar_area = max(_box_area(haar_box), 1e-6)
    if not _box_looks_like_face(detection_face_box):
        return haar_box
    if detection_area / haar_area >= 1.8:
        return haar_box
    return detection_face_box


def _input_spec_for_model(registry, model_name: str) -> tuple[str, int]:
    spec = registry.get_spec(model_name)
    input_name = "input"
    input_size = 192
    if spec.inputs:
        input_name = spec.inputs[0].name or input_name
        if len(spec.inputs[0].shape) >= 4:
            size = int(spec.inputs[0].shape[-1])
            if size > 0:
                input_size = size
    return input_name, input_size


def _preprocess_for_face_mesh(
    crop_bgr: np.ndarray,
    input_name: str,
    input_size: int,
) -> dict[str, np.ndarray]:
    """Resize to the configured model size and convert to NCHW float32 RGB in [0, 1]."""
    resized = cv2.resize(crop_bgr, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis]  # NCHW
    return {input_name: tensor}


def _parse_face_mesh_output(
    raw: dict[str, np.ndarray],
    crop_box: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
    input_size: int,
) -> tuple[list[dict[str, float]], float]:
    """
    Parse raw model outputs → (landmarks, confidence).

    Supports multiple ONNX export conventions:
    - PINTO/ONNX: landmark output shape (..., 1404) or (..., 468, 3)
    - face_flag output shape (..., 1) or scalar
    """
    x1, y1, x2, y2 = crop_box
    crop_w = max(x2 - x1, 1)
    crop_h = max(y2 - y1, 1)

    # ── locate landmark tensor (size ≥ 1404) ──────────────────────────────
    lm_arr: np.ndarray | None = None
    confidence = 0.0

    outputs_by_size: list[tuple[int, np.ndarray]] = sorted(
        [(v.size, v) for v in raw.values()], key=lambda t: t[0], reverse=True
    )

    for size, arr in outputs_by_size:
        if size >= 1404:
            flat = arr.reshape(-1)[:1404]
            lm_arr = flat.reshape(468, 3)
            break

    # ── locate face-flag tensor (small, scalar-like) ──────────────────────
    for _, arr in outputs_by_size:
        if arr.size == 1:
            raw_flag = float(arr.flat[0])
            # sigmoid if logit, identity if already probability
            if abs(raw_flag) > 1.5:
                import math
                raw_flag = 1.0 / (1.0 + math.exp(-raw_flag))
            confidence = max(confidence, float(np.clip(raw_flag, 0.0, 1.0)))

    if lm_arr is None:
        return [], confidence

    scale_x = crop_w / input_size
    scale_y = crop_h / input_size

    landmarks: list[dict[str, float]] = []
    for lm in lm_arr:
        px = (x1 + float(lm[0]) * scale_x) / frame_w
        py = (y1 + float(lm[1]) * scale_y) / frame_h
        pz = float(lm[2])
        landmarks.append({"x": float(np.clip(px, 0.0, 1.0)),
                           "y": float(np.clip(py, 0.0, 1.0)),
                           "z": pz})
    return landmarks, confidence


def _mouth_bbox(
    landmarks: list[dict[str, float]],
) -> tuple[float, float, float, float] | None:
    """Return normalised (x1, y1, x2, y2) covering the mouth + lower-face region."""
    if len(landmarks) < 400:
        return None
    xs = [landmarks[i]["x"] for i in _LOWER_FACE if i < len(landmarks)]
    ys = [landmarks[i]["y"] for i in _LOWER_FACE if i < len(landmarks)]
    if not xs:
        return None
    return (
        min(xs) - _MOUTH_PAD,
        min(ys) - _MOUTH_PAD,
        max(xs) + _MOUTH_PAD,
        max(ys) + _MOUTH_PAD,
    )


def _intersection_score(
    detections: list[dict[str, Any]],
    region: tuple[float, float, float, float],
) -> float:
    """
    Return the best detection confidence whose box overlaps ``region``.
    region and detection boxes are normalised (x1, y1, x2, y2) in [0, 1].
    """
    rx1, ry1, rx2, ry2 = region
    best = 0.0
    for det in detections:
        box = det.get("box")
        if box is None:
            continue
        dx1, dy1, dx2, dy2 = box
        iw = max(0.0, min(rx2, dx2) - max(rx1, dx1))
        ih = max(0.0, min(ry2, dy2) - max(ry1, dy1))
        if iw > 0 and ih > 0:
            best = max(best, float(det.get("score", 0.5)))
    return best


def _face_box_from_landmarks(
    landmarks: list[dict[str, float]],
) -> tuple[float, float, float, float] | None:
    if not landmarks:
        return None
    xs = [float(lm.get("x", 0.0)) for lm in landmarks]
    ys = [float(lm.get("y", 0.0)) for lm in landmarks]
    if not xs or not ys:
        return None
    min_x = max(0.0, min(xs) - max(0.02, (max(xs) - min(xs)) * 0.08))
    min_y = max(0.0, min(ys) - max(0.02, (max(ys) - min(ys)) * 0.12))
    max_x = min(1.0, max(xs) + max(0.02, (max(xs) - min(xs)) * 0.08))
    max_y = min(1.0, max(ys) + max(0.02, (max(ys) - min(ys)) * 0.12))
    return (min_x, min_y, max_x, max_y)


def _face_status_label(drowsy_score: float | None, pose: dict[str, float] | None) -> str:
    labels: list[str] = []
    yaw = None
    pitch = None
    if pose:
        yaw = pose.get("yaw_deg", pose.get("yaw"))
        pitch = pose.get("pitch_deg", pose.get("pitch"))
    if drowsy_score is not None and drowsy_score >= 0.5:
        labels.append("Drowsy")
    if yaw is not None and abs(yaw) >= DISTRACT_YAW_DEG:
        labels.append("Looking Left" if yaw > 0 else "Looking Right")
    elif pitch is not None and abs(pitch) >= DISTRACT_PITCH_DEG:
        labels.append("Looking Down" if pitch > 0 else "Looking Up")
    return " | ".join(labels) or "Driver Face"


def _face_status_label_with_activity(
    drowsy_score: float | None,
    pose: dict[str, float] | None,
    eating_score: float = 0.0,
    smoking_score: float = 0.0,
) -> str:
    base = _face_status_label(drowsy_score, pose)
    activity: list[str] = []
    if eating_score >= 0.5:
        activity.append("Eating")
    if smoking_score >= 0.5:
        activity.append("Smoking")
    if activity:
        return " | ".join(activity) + (" | " + base if base != "Driver Face" else "")
    return base


def _apply_mouth_scores(item: StageItem, landmarks: list[dict[str, float]]) -> None:
    # Prefer face bbox as the intersection target — it is far more robust than
    # relying on tiny mouth-landmark coordinates alone.  Food / cups are often
    # held in front of the whole lower face, not just the lip region.
    face_box = item.payload.get("face_box")
    mouth_region = _mouth_bbox(landmarks)

    if face_box is not None:
        fx1, fy1, fx2, fy2 = face_box
        face_h = max(fy2 - fy1, 1e-6)
        # eating: food overlaps lower 60 % of face bbox + small outer margin
        eating_region: tuple[float, float, float, float] = (
            fx1 - 0.02,
            fy1 + face_h * 0.40,
            fx2 + 0.02,
            fy2 + 0.06,
        )
        # smoking: cigarette overlaps lower 50 % of face bbox
        smoking_region: tuple[float, float, float, float] = (
            fx1 - 0.01,
            fy1 + face_h * 0.50,
            fx2 + 0.01,
            fy2 + 0.04,
        )
    elif mouth_region is not None:
        eating_region = (
            mouth_region[0] - 0.05,
            mouth_region[1] - 0.08,
            mouth_region[2] + 0.05,
            mouth_region[3] + 0.04,
        )
        smoking_region = mouth_region
    else:
        return

    for result in item.results:
        by_class = (result.raw or {}).get("by_class", {})
        cig_dets = by_class.get("cigarette", [])
        food_dets = by_class.get("foodItem", [])
        if cig_dets:
            score = _intersection_score(cig_dets, smoking_region)
            if score > item.payload.get("smoking_face_score", 0.0):
                item.payload["smoking_face_score"] = score
        if food_dets:
            score = _intersection_score(food_dets, eating_region)
            if score > item.payload.get("eating_face_score", 0.0):
                item.payload["eating_face_score"] = score


async def _infer_face_mesh(
    stage: FaceMeshStage,
    crop_bgr: np.ndarray,
) -> tuple[dict[str, np.ndarray], str, int]:
    model_name = stage.model
    input_name, input_size = _input_spec_for_model(stage.ctx.registry, model_name)
    feeds = _preprocess_for_face_mesh(crop_bgr, input_name, input_size)
    backend = await stage.ctx.registry.get_backend(model_name)
    t0 = time.perf_counter()
    raw = await backend.infer(feeds)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    log.debug("face_mesh_infer", model=model_name, latency_ms=round(elapsed_ms, 1))
    return raw, model_name, input_size


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

class FaceMeshStage(Stage):
    """
    Pipeline stage that enriches each item with face-mesh-derived behaviour
    signals.  Plug it in *after* the object-detection ``infer`` stage and
    *before* the ``fuse`` stage.

    YAML usage::

        - name: face_mesh
          stage: .models.face_mesh.adapter:FaceMeshStage
          params:
            model: face_mesh

    Besides the raw ``payload`` keys the fusion DSL reads, the stage publishes
    the dashboard-facing view through the generic ``payload["signals"]`` and
    ``payload["overlay_boxes"]`` channels (see ``_publish_signals``).
    """

    # Scalar payload keys mirrored into payload["signals"] for the dashboard.
    SIGNAL_KEYS = (
        "face_source", "face_mesh_confidence", "face_status_label",
        "drowsy_ear", "distracted_score", "smoking_face_score", "eating_face_score",
    )

    def __init__(
        self,
        model: str = "face_mesh_onnx",
        fallback_model: str | None = None,
        face_confidence_threshold: float = 0.5,
        crop_padding: float = 0.25,
        min_infer_interval_ms: int = 66,
    ) -> None:
        super().__init__(
            model=model,
            fallback_model=None,
            face_confidence_threshold=face_confidence_threshold,
            crop_padding=crop_padding,
            min_infer_interval_ms=min_infer_interval_ms,
        )
        self.model = model
        self.fallback_model = None  # Legacy parameter is accepted; model fallback is disabled.
        self.face_confidence_threshold = face_confidence_threshold
        self.crop_padding = crop_padding
        self.min_infer_interval_ms = max(0, int(min_infer_interval_ms))
        self._active_model: str = model
        self._haar_face_detector = None
        self._haar_eye_detector = None
        self._haar_profile_detector = None
        # Latched once the cascades prove unloadable, so we warn a single time
        # instead of re-probing the filesystem on every frame.
        self._haar_unavailable = False
        self._face_cache: dict[str, dict[str, Any]] = {}

    async def setup(self, ctx) -> None:
        await super().setup(ctx)
        self._ensure_haar_detectors()

    def _restore_cached_face(
        self,
        item: StageItem,
        face_box: tuple[float, float, float, float],
    ) -> bool:
        cached = self._face_cache.get(item.stream_id)
        if not cached:
            return False
        age_ms = int(item.ts_ms) - int(cached.get("ts_ms", 0))
        cached_face_box = cached.get("face_box")
        cached_payload = cached.get("payload") or {}
        if (
            age_ms < 0
            or age_ms > self.min_infer_interval_ms
            or cached_face_box is None
            or not _boxes_are_close(face_box, tuple(cached_face_box))
        ):
            return False
        item.payload.update(cached_payload)
        landmarks = item.payload.get("face_landmarks") or []
        if landmarks:
            _apply_mouth_scores(item, landmarks)
            item.payload["face_status_label"] = _face_status_label_with_activity(
                item.payload.get("drowsy_ear"),
                item.payload.get("face_pose"),
                eating_score=float(item.payload.get("eating_face_score", 0.0)),
                smoking_score=float(item.payload.get("smoking_face_score", 0.0)),
            )
        return True

    def _store_face_cache(self, item: StageItem) -> None:
        payload = {}
        for key in (
            "face_mesh_confidence",
            "face_landmarks",
            "face_box",
            "face_source",
            "drowsy_ear",
            "distracted_score",
            "face_pose",
            "face_status_label",
        ):
            if key in item.payload:
                payload[key] = item.payload[key]
        face_box = payload.get("face_box")
        if face_box is None:
            return
        self._face_cache[item.stream_id] = {
            "ts_ms": int(item.ts_ms),
            "face_box": tuple(face_box),
            "payload": payload,
        }

    def _ensure_haar_detectors(self) -> None:
        if self._haar_unavailable:
            return
        if (
            self._haar_face_detector is not None
            and self._haar_eye_detector is not None
            and self._haar_profile_detector is not None
        ):
            return
        # Haar detection is an optional assist, not a hard requirement: it only
        # kicks in when the detector produced no plausible face box. OpenCV 5
        # dropped `cv2.CascadeClassifier` from the default module and ships no
        # cascade XMLs, and this runs from setup(), so an unguarded failure took
        # down construction of the entire pipeline. Degrade instead.
        try:
            base = Path(cv2.data.haarcascades)
            cascade_cls = cv2.CascadeClassifier
            face = cascade_cls(str(base / "haarcascade_frontalface_default.xml"))
            eye = cascade_cls(str(base / "haarcascade_eye.xml"))
            profile = cascade_cls(str(base / "haarcascade_profileface.xml"))
            if face.empty():
                raise RuntimeError(
                    f"haarcascade_frontalface_default.xml not found under {base}"
                )
        except Exception as exc:
            self._haar_unavailable = True
            log.warning(
                "haar_face_detector_unavailable",
                err=f"{type(exc).__name__}: {exc}",
                cv2_version=getattr(cv2, "__version__", "unknown"),
                impact="face-mesh falls back to detector boxes only",
            )
            return
        self._haar_face_detector = face
        self._haar_eye_detector = eye
        self._haar_profile_detector = profile

    def _detect_face_with_haar(self, img: np.ndarray) -> dict[str, Any] | None:
        self._ensure_haar_detectors()
        if self._haar_face_detector is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_candidates: list[tuple[int, int, int, int]] = []
        frontal_faces = self._haar_face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(40, 40),
        )
        face_candidates.extend(tuple(int(v) for v in face) for face in frontal_faces)
        if self._haar_profile_detector is not None and not self._haar_profile_detector.empty():
            profile_faces = self._haar_profile_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(40, 40),
            )
            face_candidates.extend(tuple(int(v) for v in face) for face in profile_faces)

            flipped = cv2.flip(gray, 1)
            mirrored_faces = self._haar_profile_detector.detectMultiScale(
                flipped,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(40, 40),
            )
            frame_w = gray.shape[1]
            for x, y, width, height in mirrored_faces:
                face_candidates.append((int(frame_w - (x + width)), int(y), int(width), int(height)))

        faces = face_candidates
        if len(faces) == 0:
            return None

        frame_h, frame_w = gray.shape[:2]
        best: dict[str, Any] | None = None
        for (x, y, width, height) in faces:
            roi = gray[y : y + max(1, int(height * 0.65)), x : x + width]
            eyes = self._haar_eye_detector.detectMultiScale(
                roi,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(10, 10),
            )
            eyes = sorted(eyes, key=lambda eye: eye[2] * eye[3], reverse=True)[:2]
            eyes = sorted(eyes, key=lambda eye: eye[0])

            yaw = 0.0
            drowsy_score = 0.0
            if len(eyes) >= 2:
                left_eye, right_eye = eyes[:2]
                left_cx = left_eye[0] + (left_eye[2] / 2.0)
                right_cx = right_eye[0] + (right_eye[2] / 2.0)
                eye_mid_x = ((left_cx + right_cx) / 2.0) / max(width, 1)
                yaw = float(np.clip(((eye_mid_x - 0.5) / 0.12) * 30.0, -45.0, 45.0))
                openness = ((left_eye[3] + right_eye[3]) / 2.0) / max(height, 1)
                drowsy_score = float(np.clip((0.20 - openness) / 0.10, 0.0, 1.0))
                confidence = 0.9
            elif len(eyes) == 1:
                yaw = 0.0
                drowsy_score = 0.0
                confidence = 0.78
            else:
                confidence = 0.62
                drowsy_score = 0.0

            face_box = (
                max(0.0, x / frame_w),
                max(0.0, y / frame_h),
                min(1.0, (x + width) / frame_w),
                min(1.0, (y + height) / frame_h),
            )
            pose = {
                "yaw": yaw,
                "pitch": 0.0,
                "roll": 0.0,
                "yaw_deg": yaw,
                "pitch_deg": 0.0,
            }
            candidate = {
                "score": int(width * height) + (len(eyes) * 10_000),
                "face_box": face_box,
                "confidence": confidence,
                "drowsy_ear": drowsy_score,
                "face_pose": pose,
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate
        return best

    def _apply_face_fallback(self, item: StageItem, face_info: dict[str, Any]) -> None:
        face_box = tuple(face_info["face_box"])
        pose = dict(face_info.get("face_pose") or {})
        drowsy_score = float(face_info.get("drowsy_ear", 0.0))
        item.payload["face_mesh_confidence"] = float(face_info.get("confidence", 0.0))
        item.payload["face_box"] = face_box
        item.payload["drowsy_ear"] = drowsy_score
        item.payload["face_pose"] = pose
        item.payload["face_status_label"] = _face_status_label(drowsy_score, pose)
        item.payload["face_source"] = "haar"
        item.payload.update(
            behavior_scores_from_face(
                None,
                None,
                pose.get("yaw_deg", pose.get("yaw")),
                pose.get("pitch_deg", pose.get("pitch")),
                ear_threshold=EAR_DROWSY_THRESHOLD,
                yaw_threshold=DISTRACT_YAW_DEG,
                pitch_threshold=DISTRACT_PITCH_DEG,
            )
        )
        item.payload["distracted_score"] = max(
            float(item.payload.get("distracted_score", 0.0)),
            float(face_info.get("distracted_score", 0.0)),
        )

    def _publish_signals(self, item: StageItem) -> None:
        if "face_box" not in item.payload:
            return
        signals = item.payload.setdefault("signals", {})
        for key in self.SIGNAL_KEYS:
            value = item.payload.get(key)
            if isinstance(value, (int, float, str)) and not isinstance(value, bool):
                signals[key] = round(value, 3) if isinstance(value, float) else value
        pose = item.payload.get("face_pose") or {}
        for axis in ("yaw", "pitch"):
            value = pose.get(f"{axis}_deg", pose.get(axis))
            if isinstance(value, (int, float)):
                signals[f"head_{axis}_deg"] = round(float(value), 1)
        label = str(item.payload.get("face_status_label") or "Face")
        tone = "danger" if "Drowsy" in label else ("warn" if any(
            k in label for k in ("Looking", "Eating", "Smoking")) else "ok")
        item.payload["overlay_boxes"] = [{
            "label": label,
            "box": [float(v) for v in item.payload["face_box"]],
            "tone": tone,
        }]

    async def process(self, item: StageItem) -> StageItem:  # type: ignore[override]
        try:
            return await self._process(item)
        finally:
            self._publish_signals(item)

    async def _process(self, item: StageItem) -> StageItem:
        img: np.ndarray | None = item.payload.get("image_np")
        if img is None:
            return item

        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return item

        # ── 1. locate face from prior detection results ────────────────────
        detection_face_box = _get_face_box(item.results, w, h)
        haar_face = None
        if detection_face_box is None or not _box_looks_like_face(detection_face_box):
            haar_face = self._detect_face_with_haar(img)
        face_box = _select_face_box(detection_face_box, haar_face)
        if face_box is None:
            return item

        if self._restore_cached_face(item, tuple(face_box)):
            return item

        # ── 2. crop + preprocess ───────────────────────────────────────────
        crop, crop_box = _crop_face(img, face_box, padding=self.crop_padding)

        # ── 3. infer ───────────────────────────────────────
        raw, used_model, used_input_size = await _infer_face_mesh(
            self,
            crop,
        )

        if raw is None:
            if haar_face is not None:
                self._apply_face_fallback(item, haar_face)
                self._store_face_cache(item)
            return item

        if used_model is not None:
            self._active_model = used_model

        # ── 4. parse landmarks ─────────────────────────────────────────────
        landmarks, confidence = _parse_face_mesh_output(
            raw,
            crop_box,
            w,
            h,
            used_input_size,
        )
        item.payload["face_mesh_confidence"] = confidence

        if confidence < self.face_confidence_threshold or len(landmarks) < 468:
            if haar_face is not None:
                self._apply_face_fallback(item, haar_face)
                self._store_face_cache(item)
            return item

        item.payload["face_landmarks"] = landmarks
        item.payload["face_box"] = _face_box_from_landmarks(landmarks) or tuple(face_box)
        item.payload["face_source"] = "face_mesh"

        # ── 5. EAR + head pose ─────────────────────────────────────────────
        pts = np.array([[lm["x"] * w, lm["y"] * h] for lm in landmarks],
                       dtype=np.float64)

        ear_l = eye_aspect_ratio(pts, LEFT_EYE)
        ear_r = eye_aspect_ratio(pts, RIGHT_EYE)
        pose = head_pose(pts, w, h)

        scores = behavior_scores_from_face(
            ear_l, ear_r, pose.get("yaw"), pose.get("pitch"),
            ear_threshold=EAR_DROWSY_THRESHOLD,
            yaw_threshold=DISTRACT_YAW_DEG,
            pitch_threshold=DISTRACT_PITCH_DEG,
        )
        item.payload.update(scores)
        pose["yaw_deg"] = pose.get("yaw", 0.0)
        pose["pitch_deg"] = pose.get("pitch", 0.0)
        item.payload["face_pose"] = pose
        item.payload["face_status_label"] = _face_status_label(
            item.payload.get("drowsy_ear"),
            pose,
        )
        self._store_face_cache(item)

        # ── 6. smoking / eating intersection with mouth region ─────────────
        _apply_mouth_scores(item, landmarks)

        # update face_status_label to include eating/smoking activity
        item.payload["face_status_label"] = _face_status_label_with_activity(
            item.payload.get("drowsy_ear"),
            pose,
            eating_score=float(item.payload.get("eating_face_score", 0.0)),
            smoking_score=float(item.payload.get("smoking_face_score", 0.0)),
        )

        return item

