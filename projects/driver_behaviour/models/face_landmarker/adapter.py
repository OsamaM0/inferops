"""Complete face bundle with camera-relative pose and temporal blink states."""
from __future__ import annotations

import cv2
import numpy as np

from ..face_mesh.adapter import (
    FaceMeshStage,
    _apply_mouth_scores,
    _box_area,
    _boxes_are_close,
    _crop_face,
    _face_box_from_landmarks,
    _face_status_label_with_activity,
)
from ..face_mesh.features import LEFT_EYE, RIGHT_EYE, eye_aspect_ratio, head_pose


def matrix_to_pose(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        return None
    # Remove the uniform scale in a facial transformation matrix.
    scale = np.linalg.norm(matrix[:3, :3], axis=0)
    if (scale <= 1e-8).any():
        return None
    r = matrix[:3, :3] / scale
    if not np.allclose(r.T @ r, np.eye(3), atol=.03) or not np.isclose(np.linalg.det(r), 1, atol=.03):
        return None
    # MediaPipe metric camera: +X right, +Y up, camera looks along -Z.
    # Publish positive yaw=driver left and positive pitch=driver down.
    return {'yaw': float(np.degrees(np.arctan2(-r[2, 0], np.hypot(r[0, 0], r[1, 0])))),
            'pitch': float(np.degrees(np.arctan2(r[2, 1], r[2, 2]))),
            'roll': -float(np.degrees(np.arctan2(r[1, 0], r[0, 0])))}


def validated_region(value):
    if value is None:
        return (0., 0., 1., 1.)
    region = np.asarray(value, dtype=float)
    if (region.shape != (4,) or not np.isfinite(region).all() or (region < 0).any()
            or (region > 1).any() or region[0] >= region[2] or region[1] >= region[3]):
        raise ValueError('driver_region / driver_roi must be normalized [x1, y1, x2, y2]')
    return tuple(float(v) for v in region)


class FaceLandmarkerStage(FaceMeshStage):
    SIGNAL_KEYS = (*FaceMeshStage.SIGNAL_KEYS, 'eye_blink_left', 'eye_blink_right', 'eye_state_source',
                   'face_detected', 'face_analysis_source', 'face_analysis_age_ms')

    def __init__(self, model='face_landmarker', driver_region=None, min_infer_interval_ms=100,
                 blink_closed_threshold=.5, blink_open_threshold=.3, min_eye_width_px=6.,
                 neutral_yaw_deg=0., neutral_pitch_deg=0., direction_yaw_threshold=25.,
                 direction_pitch_threshold=18., require_detector=True, crop_padding=.25,
                 pose_method='transformation', strong_face_confidence=.8, fallback_model=None):
        super().__init__(model=model, allow_haar_fallback=False, min_infer_interval_ms=min_infer_interval_ms,
                         crop_padding=crop_padding,
                         min_eye_width_px=min_eye_width_px, neutral_yaw_deg=neutral_yaw_deg,
                         neutral_pitch_deg=neutral_pitch_deg, direction_yaw_threshold=direction_yaw_threshold,
                         direction_pitch_threshold=direction_pitch_threshold)
        if not 0 <= blink_open_threshold < blink_closed_threshold <= 1:
            raise ValueError('blink thresholds must satisfy 0 <= open < closed <= 1')
        validated_region(driver_region)
        self.driver_region = driver_region
        self.blink_closed_threshold = blink_closed_threshold
        self.blink_open_threshold = blink_open_threshold
        if not np.isfinite(crop_padding) or not 0 <= crop_padding <= 1:
            raise ValueError('crop_padding must be finite and between 0 and 1')
        self.require_detector = bool(require_detector)
        if pose_method not in {'transformation', 'landmark_pnp'}:
            raise ValueError('pose_method must be transformation or landmark_pnp')
        if not np.isfinite(strong_face_confidence) or not 0 < strong_face_confidence <= 1:
            raise ValueError('strong_face_confidence must be in (0, 1]')
        self.pose_method = pose_method
        self.strong_face_confidence = strong_face_confidence
        self._bundle_cache = {}

    def _apply_blinks(self, item, points, raw_pose, scores):
        h, w = item.payload['image_np'].shape[:2]
        eye_points = points[list(LEFT_EYE + RIGHT_EYE)]
        widths = [np.linalg.norm(points[eye[0]] - points[eye[3]]) for eye in (LEFT_EYE, RIGHT_EYE)]
        if (raw_pose is None or abs(raw_pose['yaw']) > 40 or abs(raw_pose['pitch']) > 35
                or min(widths) < self.min_eye_width_px or not np.isfinite(eye_points).all()
                or (eye_points <= 0).any() or (eye_points >= (w - 1, h - 1)).any()
                or not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any()):
            return
        ts = int(item.ts_ms)
        history = self._eye_history.get(item.stream_id, {})
        if (not 0 < ts - history.get('ts_ms', ts - 1) <= 1000
                or (history.get('face_box') is not None and not _boxes_are_close(
                    tuple(item.payload['face_box']), tuple(history['face_box'])))):
            history = {}
        states = ['closed' if score >= self.blink_closed_threshold else
                  'open' if score <= self.blink_open_threshold else history.get(key, 'unknown')
                  for key, score in zip(('left', 'right'), scores, strict=True)]
        state = states[0] if states[0] == states[1] else 'mixed'
        if 'unknown' in states:
            state = 'unknown'
        since = history.get('since_ms', ts) if state == history.get('state') == 'closed' else ts
        item.payload.update(eyes_state=state, left_eye_state=states[0], right_eye_state=states[1],
                            eye_blink_left=float(scores[0]), eye_blink_right=float(scores[1]),
                            eyes_closed_ms=ts - since if state == 'closed' else 0,
                            eye_state_source='mediapipe_blendshapes',
                            ear_left=eye_aspect_ratio(points, RIGHT_EYE),
                            ear_right=eye_aspect_ratio(points, LEFT_EYE))
        if state in {'open', 'closed', 'mixed'}:
            item.payload['drowsy_ear'] = 1. if state == 'closed' else 0.
        self._eye_history[item.stream_id] = {'ts_ms': ts, 'left': states[0], 'right': states[1],
            'state': state, 'since_ms': since, 'face_box': item.payload['face_box']}

    async def _process(self, item):
        verified = item.payload.pop('_verified_face_analysis', None)
        detection_meta = {key: item.payload[key] for key in (
            'face_detection_age_ms', 'face_detection_confidence', 'face_search_method') if key in item.payload}
        for key in (*self.SIGNAL_KEYS, 'face_pose', 'face_pose_raw', 'face_box', 'face_landmarks',
                    'head_yaw_deg', 'head_pitch_deg', 'head_roll_deg'):
            item.payload.pop(key, None)
            item.payload.get('signals', {}).pop(key, None)
        item.payload.update(eyes_state='unknown', left_eye_state='unknown', right_eye_state='unknown',
                            head_direction='unknown', face_direction='unknown', head_pose_source='unknown',
                            face_detected=0)
        item.payload.pop('overlay_boxes', None)
        image = item.payload.get('image_np')
        if image is None or not image.size:
            return item
        h, w = image.shape[:2]
        selected_box = None
        if self.require_detector:
            detector_results = [result for result in item.results if (result.raw or {}).get('face_detector') == 'scrfd']
            result = detector_results[-1] if detector_results else None
            if result is None or not result.detections:
                self._bundle_cache.pop(item.stream_id, None)
                return item
            detection = result.detections[0]
            selected_box = tuple(detection.box)
            if (not np.isfinite(selected_box).all() or min(selected_box) < 0 or max(selected_box) > 1
                    or _box_area(selected_box) <= 0):
                self._bundle_cache.pop(item.stream_id, None)
                return item
            crop, (x1, y1, x2, y2) = _crop_face(image, selected_box, self.crop_padding)
            item.payload.update(detection_meta)
            item.payload.update(face_box=selected_box, face_detected=1, face_source='scrfd',
                                face_detection_confidence=float(detection.score), face_status_label='Driver Face')
        else:
            source_region = item.payload.get('driver_roi', self.driver_region)
            region = validated_region(source_region)
            x1, y1, x2, y2 = [int(value * size) for value, size in zip(region, (w, h, w, h), strict=True)]
            if x2 <= x1 or y2 <= y1:
                return item
            crop = image[y1:y2, x1:x2]
            item.payload['face_search_method'] = 'driver_region' if source_region is not None else 'full_frame'
        signature = (w, h, x1, y1, x2, y2, selected_box)
        cached = self._bundle_cache.get(item.stream_id)
        age = item.ts_ms - cached[0] if cached else -1
        if (verified is not None and verified['model'] == self.model and verified['frame_size'] == (w, h)
                and tuple(verified['crop_box']) == (x1, y1, x2, y2)):
            raw = verified['raw']
            age = max(0, item.ts_ms - verified['ts_ms'])
            self._bundle_cache[item.stream_id] = (verified['ts_ms'], signature, raw)
        elif cached and cached[1] == signature and 0 <= age < self.min_infer_interval_ms:
            raw = cached[2]
        else:
            backend = await self.ctx.registry.get_backend(self.model)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            raw = await backend.infer({'image': rgb})
            self._bundle_cache[item.stream_id] = (item.ts_ms, signature, raw)
            age = 0
        item.payload['face_analysis_age_ms'] = int(age)
        if not self.require_detector:
            item.payload['face_detection_age_ms'] = int(age)
        candidates = []
        for idx, points in enumerate(raw['landmarks']):
            points = points.copy()
            points[:, 0] = (points[:, 0] * (x2 - x1) + x1) / w
            points[:, 1] = (points[:, 1] * (y2 - y1) + y1) / h
            points[:, 2] *= (x2 - x1) / w
            if not np.isfinite(points).all():
                continue
            landmarks = [dict(zip(('x', 'y', 'z'), map(float, p), strict=True)) for p in points]
            box = _face_box_from_landmarks(landmarks)
            if box is not None and _box_area(box) > 0:
                if selected_box is not None:
                    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    if not selected_box[0] <= cx <= selected_box[2] or not selected_box[1] <= cy <= selected_box[3]:
                        continue
                candidates.append((idx, landmarks, box))
        if not candidates:
            if self.require_detector and detection.score < self.strong_face_confidence:
                # A borderline proposal needs independently accepted face landmarks.
                item.payload.pop('face_box', None)
                item.payload.pop('face_status_label', None)
                item.payload.pop('face_source', None)
                item.payload['face_detected'] = 0
            return item
        idx, landmarks, box = max(candidates, key=lambda entry: _box_area(entry[2]))
        item.payload.update(face_landmarks=landmarks, face_box=selected_box or box, face_detected=1,
                            face_source='scrfd' if self.require_detector else 'mediapipe_face_landmarker',
                            face_analysis_source='mediapipe_face_landmarker')
        points = np.array([[p['x'] * w, p['y'] * h] for p in landmarks])
        # Crop coordinates change the task's virtual camera. Fit the accepted
        # landmarks in the original frame when the detector provides a tight crop.
        raw_pose = (head_pose(points, w, h) if self.pose_method == 'landmark_pnp' else
                    matrix_to_pose(raw['matrices'][idx]) if idx < len(raw['matrices']) else None)
        if raw_pose is not None:
            item.payload.update(face_pose_raw=raw_pose, head_pose_source=(
                'mediapipe_landmark_pnp' if self.pose_method == 'landmark_pnp' else 'mediapipe_transformation'))
        pose = self._apply_pose(item, raw_pose)
        if idx < len(raw['blink_scores']):
            self._apply_blinks(item, points, raw_pose, raw['blink_scores'][idx])
        _apply_mouth_scores(item, landmarks)
        item.payload['face_status_label'] = _face_status_label_with_activity(
            item.payload.get('drowsy_ear'), pose, float(item.payload.get('eating_face_score', 0)),
            float(item.payload.get('smoking_face_score', 0)))
        return item

    async def teardown(self):
        self._bundle_cache.clear()
        await super().teardown()
