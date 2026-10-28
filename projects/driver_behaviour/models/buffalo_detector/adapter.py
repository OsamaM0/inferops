"""SCRFD-10G decoding and an independent driver-face detection stage.

Uses only the detector from Buffalo, without running identity recognition.
Output names and anchor conventions match InsightFace's det_10g.onnx export.
"""
from __future__ import annotations

import cv2
import numpy as np

from inferops.adapters.outbound.stages.base import Stage
from inferops.domain.types import Detection, Frame, Result

from ..face_mesh.adapter import (
    _boxes_are_close,
    _crop_face,
    _input_spec_for_model,
    _parse_face_mesh_output,
    _preprocess_for_face_mesh,
)

_SCORES = ("448", "471", "494")
_BOXES = ("451", "474", "497")
_STRIDES = (8, 16, 32)


def _resized_shape(width: int, height: int, size: int) -> tuple[int, int]:
    scale = min(size / width, size / height)
    return max(1, int(width * scale)), max(1, int(height * scale))


class SCRFDPreprocess:
    async def __call__(self, frame: Frame, spec) -> dict[str, np.ndarray]:
        image = frame.image
        height, width = image.shape[:2]
        size = spec.inputs[0].shape[-1]
        resized_w, resized_h = _resized_shape(width, height, size)
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[:resized_h, :resized_w] = cv2.resize(image, (resized_w, resized_h))
        # InsightFace: RGB, mean=127.5, scale=1/128, top-left letterbox.
        tensor = (canvas[:, :, ::-1].astype(np.float32) - 127.5) / 128.0
        return {spec.inputs[0].name: np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])}


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    order = scores.argsort()[::-1]
    area = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0), axis=1)
    keep = []
    while order.size:
        index = int(order[0])
        keep.append(index)
        remaining = order[1:]
        overlap = np.prod(np.maximum(
            np.minimum(boxes[index, 2:], boxes[remaining, 2:])
            - np.maximum(boxes[index, :2], boxes[remaining, :2]), 0), axis=1)
        iou = overlap / np.maximum(area[index] + area[remaining] - overlap, 1e-6)
        order = remaining[iou <= threshold]
    return keep


class SCRFDPostprocess:
    def __init__(self, conf_threshold: float = 0.8, nms_threshold: float = 0.4,
                 candidate_threshold: float | None = None) -> None:
        self.conf_threshold = conf_threshold if candidate_threshold is None else candidate_threshold
        self.nms_threshold = nms_threshold

    async def __call__(self, raw, frame: Frame, spec) -> Result:
        size = spec.inputs[0].shape[-1]
        width, height = int(frame.width), int(frame.height)
        resized_w, resized_h = _resized_shape(width, height, size)
        candidates, confidences = [], []
        for stride, score_name, box_name in zip(_STRIDES, _SCORES, _BOXES, strict=True):
            scores = raw[score_name].reshape(-1)
            distances = raw[box_name].reshape(-1, 4) * stride
            side = size // stride
            yy, xx = np.mgrid[:side, :side]
            anchors = np.repeat(np.stack((xx, yy), axis=-1).reshape(-1, 2), 2, axis=0) * stride
            if len(scores) != len(anchors) or len(distances) != len(anchors):
                raise ValueError(f"SCRFD output shape does not match stride {stride} at {size}px")
            selected = np.flatnonzero(
                np.isfinite(scores) & (scores >= self.conf_threshold)
                & np.isfinite(distances).all(axis=1) & (distances >= 0).all(axis=1))
            centers, offsets = anchors[selected], distances[selected]
            boxes = np.column_stack((centers - offsets[:, :2], centers + offsets[:, 2:]))
            # Discard candidates whose center lies in letterbox padding.
            box_centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            valid = ((box_centers[:, 0] < resized_w) & (box_centers[:, 1] < resized_h)
                     & (box_centers >= 0).all(axis=1))
            candidates.append(boxes[valid])
            confidences.append(scores[selected][valid])
        boxes = np.concatenate(candidates)
        scores = np.concatenate(confidences)
        boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)] / resized_w, 0, 1)
        boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)] / resized_h, 0, 1)
        valid = ((boxes[:, 2] - boxes[:, 0]) * width >= 16) & ((boxes[:, 3] - boxes[:, 1]) * height >= 16)
        boxes, scores = boxes[valid], scores[valid]
        detections = [Detection(label="DriverFace", score=float(scores[i]), box=tuple(boxes[i]))
                      for i in _nms(boxes, scores, self.nms_threshold)]
        return Result(kind="detections", model=spec.name, detections=detections,
                      raw={"face_detector": "scrfd", "by_class": {
                          "DriverFace": [d.model_dump(mode="json") for d in detections]}})


class FaceDetectionStage(Stage):
    """Driver-region search, local reacquisition and independent mesh verification.

    Candidate scores remain detector scores. Verification does not inflate them.
    Recovery views are fresh model observations; cached boxes are never promoted
    after their measurement expires.
    """

    def __init__(self, model: str = "buffalo_detector", min_infer_interval_ms: int = 100,
                 driver_region: list[float] | None = None, fallback_model=None,
                 verification_model: str | None = None, verification_confidence: float = .8,
                 strong_confidence: float = .8,
                 search_tiles: bool = False, enhance_on_miss: bool = False,
                 full_search_interval_ms: int = 1000, max_search_views: int = 7,
                 negative_cache_ms: int = 350) -> None:
        super().__init__(model=model, min_infer_interval_ms=min_infer_interval_ms,
                         driver_region=driver_region, verification_model=verification_model,
                         verification_confidence=verification_confidence, strong_confidence=strong_confidence,
                         search_tiles=search_tiles,
                         enhance_on_miss=enhance_on_miss, full_search_interval_ms=full_search_interval_ms,
                         max_search_views=max_search_views, negative_cache_ms=negative_cache_ms)
        self.model = model
        self.interval = max(0, int(min_infer_interval_ms))
        self._validate_region(driver_region)
        if not 0 < verification_confidence <= 1:
            raise ValueError("verification_confidence must be in (0, 1]")
        if not 0 < strong_confidence <= 1:
            raise ValueError("strong_confidence must be in (0, 1]")
        if not 1 <= max_search_views <= 12 or full_search_interval_ms < 0:
            raise ValueError("search budget must be 1..12 and full search interval nonnegative")
        self.driver_region = driver_region
        self.verification_model = verification_model
        self.verification_confidence = verification_confidence
        self.strong_confidence = strong_confidence
        self.search_tiles = search_tiles
        self.enhance_on_miss = enhance_on_miss
        self.full_interval = full_search_interval_ms
        self.max_views = max_search_views
        self.negative_cache_ms = max(self.interval, int(negative_cache_ms))
        self._cache = {}
        self._tracks = {}
        self._misses = {}

    @staticmethod
    def _validate_region(region):
        if region is not None and (len(region) != 4 or not all(
                np.isfinite(v) and 0 <= v <= 1 for v in region)
                or region[0] >= region[2] or region[1] >= region[3]):
            raise ValueError("driver_region must be normalized [x1, y1, x2, y2]")

    async def _verify(self, image, detection):
        if not self.verification_model:
            return True, None
        h, w = image.shape[:2]
        crop, crop_box = _crop_face(image, detection.box, padding=.25)
        input_name, size = _input_spec_for_model(self.ctx.registry, self.verification_model)
        backend = await self.ctx.registry.get_backend(self.verification_model)
        raw = await backend.infer(_preprocess_for_face_mesh(crop, input_name, size))
        landmarks, confidence = _parse_face_mesh_output(raw, crop_box, w, h, size)
        if confidence < self.verification_confidence or len(landmarks) < 468:
            return False, None
        points = np.array([[p['x'] * w, p['y'] * h] for p in landmarks])
        # Reject degenerate/collapsed geometry even if the scalar head is high.
        if min(np.ptp(points, axis=0)) < 8 or np.linalg.norm(points[33] - points[263]) < 4:
            return False, None
        spec = self.ctx.registry.get_spec(self.verification_model)
        return True, {"raw": raw, "crop_box": crop_box, "input_size": size,
                      "model": spec.name, "frame_size": (w, h), "face_box": detection.box}

    async def _search(self, item, image, region, track):
        h, w = image.shape[:2]
        rx1, ry1, rx2, ry2 = region
        # Crop BEFORE resizing, so the driver's pixels fill the model input.
        roi = (max(0, int(rx1 * w)), max(0, int(ry1 * h)),
               min(w, int(np.ceil(rx2 * w))), min(h, int(np.ceil(ry2 * h))))
        x1, y1, x2, y2 = roi
        if x2 <= x1 or y2 <= y1:
            raise ValueError("driver_region contains no image pixels")
        views = []
        if track and item.ts_ms - track["full_ts"] < self.full_interval:
            a, b, c, d = track["box"]
            px, py = c - a, d - b
            local = (max(x1, int((a - px) * w)), max(y1, int((b - py) * h)),
                     min(x2, int(np.ceil((c + px) * w))), min(y2, int(np.ceil((d + py) * h))))
            if local[2] > local[0] and local[3] > local[1]:
                views.append(("tracked_crop", local, False))
        views.append(("driver_crop", roi, False))
        recovery = []
        if roi != (0, 0, w, h):
            # Context is important for very large/profile faces. Full-frame
            # fallback still excludes detections outside the selected region.
            recovery.append(("full_context", (0, 0, w, h), False))
        if self.search_tiles:
            for yf in (0., .4):
                for xf in (0., .4):
                    tile = (x1 + int(xf * (x2 - x1)), y1 + int(yf * (y2 - y1)),
                            min(x2, x1 + int((xf + .6) * (x2 - x1))),
                            min(y2, y1 + int((yf + .6) * (y2 - y1))))
                    recovery.append(("overlapping_crop", tile, False))
        if self.enhance_on_miss:
            recovery.append(("contrast_recovery", roi, True))
        if recovery:
            # Spread expensive recovery views across misses instead of blocking
            # live capture with an entire pyramid on every obscured frame.
            phase = self._misses.get(item.stream_id, 0) % len(recovery)
            views.extend(recovery[phase:] + recovery[:phase])
        count, verified_count, total_ms = 0, 0, 0.
        for method, (a, b, c, d), enhance in views[:self.max_views]:
            if c <= a or d <= b:
                continue
            crop = image[b:d, a:c]
            if enhance:
                lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8)).apply(lab[:, :, 0])
                crop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            result = await self.ctx.registry.predict(self.model, crop, stream_id=item.stream_id, ts_ms=item.ts_ms)
            count += 1
            total_ms += result.latency_ms
            candidates = []
            for detection in result.detections or []:
                bx1, by1, bx2, by2 = detection.box
                box = ((a + bx1 * (c - a)) / w, (b + by1 * (d - b)) / h,
                       (a + bx2 * (c - a)) / w, (b + by2 * (d - b)) / h)
                if rx1 <= (box[0] + box[2]) / 2 <= rx2 and ry1 <= (box[1] + box[3]) / 2 <= ry2:
                    candidates.append(detection.model_copy(update={"box": box}, deep=True))
            def rank(det):
                area = (det.box[2] - det.box[0]) * (det.box[3] - det.box[1])
                return (int(bool(track and _boxes_are_close(det.box, track['box']))), area * det.score)
            for detection in sorted(candidates, key=rank, reverse=True)[:4]:
                accepted, verification = await self._verify(image, detection)
                verified_count += int(self.verification_model is not None)
                # High-confidence SCRFD observations can still establish head
                # pose when a profile prevents eyelid-mesh verification.
                if not accepted and detection.score < self.strong_confidence:
                    continue
                detection.attrs.update({"search_method": method, "mesh_verified": bool(verification)})
                raw = {"face_detector": "scrfd", "selected_face_box": list(detection.box),
                       "search_method": method, "search_views": count, "verified_candidates": verified_count,
                       "by_class": {"DriverFace": [detection.model_dump(mode="json")]}}
                output = Result(kind="detections", model=result.model, detections=[detection],
                                latency_ms=total_ms, raw=raw)
                return output, verification, method
        return Result(kind="detections", model=self.model, detections=[], raw={
            "face_detector": "scrfd", "search_views": count, "verified_candidates": verified_count,
            "by_class": {"DriverFace": []}}), None, "unavailable"

    async def process(self, item):
        image = item.payload.get("image_np")
        if image is None or image.size == 0:
            return item
        height, width = image.shape[:2]
        region = item.payload.get("driver_roi") or self.driver_region or [0., 0., 1., 1.]
        self._validate_region(region)
        signature = (width, height, *region)
        cached = self._cache.get(item.stream_id)
        cache_interval = self.interval if cached and cached[2].detections else self.negative_cache_ms
        if cached and cached[1] == signature and 0 <= item.ts_ms - cached[0] < cache_interval:
            result = cached[2].model_copy(deep=True)
            verification = cached[3]
            measurement_ts = cached[0]
        else:
            track = self._tracks.get(item.stream_id)
            if track and (track["signature"] != signature or not 0 < item.ts_ms - track["ts"] <= 1000):
                track = None
            if cached and (cached[1] != signature or item.ts_ms < cached[0]):
                self._misses.pop(item.stream_id, None)
            result, verification, method = await self._search(item, image, region, track)
            measurement_ts = item.ts_ms
            if result.detections:
                self._misses.pop(item.stream_id, None)
                self._tracks[item.stream_id] = {
                    "box": result.detections[0].box, "ts": item.ts_ms, "signature": signature,
                    "full_ts": track['full_ts'] if track and method == 'tracked_crop' else item.ts_ms}
            else:
                self._tracks.pop(item.stream_id, None)
                self._misses[item.stream_id] = self._misses.get(item.stream_id, 0) + max(1, self.max_views - 1)
            self._cache[item.stream_id] = (item.ts_ms, signature, result.model_copy(deep=True), verification)
        item.payload['face_detection_age_ms'] = item.ts_ms - measurement_ts
        item.payload['face_search_method'] = result.raw.get('search_method', 'unavailable')
        if result.detections:
            item.payload['face_detection_confidence'] = result.detections[0].score
        if verification:
            item.payload['_verified_face_mesh'] = verification
        item.results.append(result)
        return item

    async def teardown(self) -> None:
        self._cache.clear()
        self._tracks.clear()
        self._misses.clear()
