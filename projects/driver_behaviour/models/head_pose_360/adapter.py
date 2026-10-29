"""Fit head orientation independently of whether the eyes can be resolved."""

from __future__ import annotations

import numpy as np

from ..face_mesh.adapter import FaceMeshStage, _apply_mouth_scores, _crop_face
from ..face_monitoring.geometry import head_tensor, rotation_to_pose


class HeadPose360Stage(FaceMeshStage):
    SIGNAL_KEYS = (*FaceMeshStage.SIGNAL_KEYS, "face_detected", "head_pose_age_ms", "face_analysis_age_ms")

    def __init__(
        self,
        model="head_pose_360",
        fallback_model=None,
        crop_padding=0.25,
        min_infer_interval_ms=100,
        neutral_yaw_deg=0.0,
        neutral_pitch_deg=0.0,
        direction_yaw_threshold=25.0,
        direction_pitch_threshold=18.0,
    ):
        if fallback_model is not None:
            raise ValueError("fallback_model is unsupported; choose one explicit model")
        super().__init__(
            model=model,
            allow_haar_fallback=False,
            crop_padding=crop_padding,
            min_infer_interval_ms=min_infer_interval_ms,
            neutral_yaw_deg=neutral_yaw_deg,
            neutral_pitch_deg=neutral_pitch_deg,
            direction_yaw_threshold=direction_yaw_threshold,
            direction_pitch_threshold=direction_pitch_threshold,
        )
        if not np.isfinite(crop_padding) or not 0 <= crop_padding <= 1:
            raise ValueError("crop_padding must be finite and between 0 and 1")
        self._poses = {}

    async def _process(self, item):
        detection_meta = {
            k: item.payload[k] for k in ("face_detection_age_ms", "face_search_method") if k in item.payload
        }
        for key in (
            *self.SIGNAL_KEYS,
            "face_pose",
            "face_pose_raw",
            "face_box",
            "face_landmarks",
            "face_landmarks_106",
            "eye_landmarks_age_ms",
            "eye_state_age_ms",
            "eye_state_source",
            "eye_closed_score_left",
            "eye_closed_score_right",
            "eye_blink_left",
            "eye_blink_right",
            "face_analysis_source",
            "head_yaw_deg",
            "head_pitch_deg",
            "head_roll_deg",
        ):
            item.payload.pop(key, None)
            item.payload.get("signals", {}).pop(key, None)
        item.payload.pop("_verified_face_analysis", None)
        item.payload.pop("overlay_boxes", None)
        item.payload.update(
            face_detected=0,
            eyes_state="unknown",
            left_eye_state="unknown",
            right_eye_state="unknown",
            head_direction="unknown",
            face_direction="unknown",
            head_pose_source="unknown",
        )
        item.payload.update(detection_meta)
        image = item.payload.get("image_np")
        results = [r for r in item.results if (r.raw or {}).get("face_detector") == "scrfd"]
        result = results[-1] if results else None
        if image is None or not image.size or result is None or not result.detections:
            self._poses.pop(item.stream_id, None)
            return item
        detection = result.detections[0]
        box = tuple(detection.box)
        if not np.isfinite(box).all() or min(box) < 0 or max(box) > 1 or box[0] >= box[2] or box[1] >= box[3]:
            self._poses.pop(item.stream_id, None)
            return item
        item.payload.update(
            face_box=box,
            face_detected=1,
            face_source="scrfd",
            face_detection_confidence=float(detection.score),
            face_status_label="Driver Face",
        )
        height, width = image.shape[:2]
        signature = (width, height, *box)
        cached = self._poses.get(item.stream_id)
        age = item.ts_ms - cached[0] if cached else -1
        if cached and cached[1] == signature and 0 <= age < self.min_infer_interval_ms:
            pose = cached[2]
        else:
            crop, _ = _crop_face(image, box, self.crop_padding)
            spec = self.ctx.registry.get_spec(self.model)
            raw = await self.ctx.registry.infer(self.model, {spec.inputs[0].name: head_tensor(crop)})
            pose = rotation_to_pose(raw[spec.outputs[0].name])
            self._poses[item.stream_id] = (item.ts_ms, signature, pose)
            age = 0
        item.payload.update(head_pose_age_ms=int(age), face_analysis_age_ms=int(age))
        if pose is not None:
            item.payload.update(face_pose_raw=dict(pose), head_pose_source="6drepnet360")
        self._apply_pose(item, pose)
        # Existing behavior objects use the lower face box; no mesh is required.
        _apply_mouth_scores(item, [])
        return item

    async def teardown(self):
        self._poses.clear()
        await super().teardown()
