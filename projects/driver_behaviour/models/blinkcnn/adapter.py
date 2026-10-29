"""Frame-wise eye closure with bilateral confirmation and per-camera history."""

from __future__ import annotations

import numpy as np

from ..face_mesh.adapter import FaceMeshStage, _boxes_are_close, _face_status_label_with_activity
from ..face_monitoring.geometry import closure_scores, eye_inputs


class BlinkCNNStage(FaceMeshStage):
    SIGNAL_KEYS = (
        *FaceMeshStage.SIGNAL_KEYS,
        "face_detected",
        "face_analysis_source",
        "face_analysis_age_ms",
        "head_pose_age_ms",
        "eye_landmarks_age_ms",
        "eye_state_age_ms",
        "eye_state_source",
        "eye_closed_score_left",
        "eye_closed_score_right",
        "eye_blink_left",
        "eye_blink_right",
    )

    def __init__(
        self,
        model="blinkcnn",
        fallback_model=None,
        min_infer_interval_ms=100,
        closed_threshold=0.53,
        open_threshold=0.1325,
        min_eye_width_px=6.0,
    ):
        if fallback_model is not None:
            raise ValueError("fallback_model is unsupported; choose one explicit model")
        super().__init__(
            model=model,
            allow_haar_fallback=False,
            min_infer_interval_ms=min_infer_interval_ms,
            min_eye_width_px=min_eye_width_px,
        )
        if not 0 <= open_threshold < closed_threshold <= 1:
            raise ValueError("thresholds must satisfy 0 <= open < closed <= 1")
        self.closed_threshold, self.open_threshold = closed_threshold, open_threshold
        self._scores = {}

    def _update_eyes(self, item, scores, ears):
        ts = int(item.ts_ms)
        history = self._eye_history.get(item.stream_id, {})
        if not 0 < ts - history.get("ts_ms", ts - 1) <= 1000 or (
            history.get("face_box") is not None
            and not _boxes_are_close(tuple(item.payload["face_box"]), tuple(history["face_box"]))
        ):
            history = {}
        states = [
            "closed"
            if value >= self.closed_threshold
            else "open"
            if value <= self.open_threshold
            else history.get(key, "unknown")
            for key, value in zip(("left", "right"), scores, strict=True)
        ]
        state = "unknown" if "unknown" in states else states[0] if states[0] == states[1] else "mixed"
        since = history.get("since_ms", ts) if state == history.get("state") == "closed" else ts
        item.payload.update(
            left_eye_state=states[0],
            right_eye_state=states[1],
            eyes_state=state,
            eye_closed_score_left=float(scores[0]),
            eye_closed_score_right=float(scores[1]),
            eye_blink_left=float(scores[0]),
            eye_blink_right=float(scores[1]),
            ear_left=float(ears[0]),
            ear_right=float(ears[1]),
            eye_state_source="blinkcnn",
            eyes_closed_ms=ts - since if state == "closed" else 0,
        )
        if state != "unknown":
            item.payload["drowsy_ear"] = float(state == "closed")
        self._eye_history[item.stream_id] = {
            "ts_ms": ts,
            "left": states[0],
            "right": states[1],
            "state": state,
            "since_ms": since,
            "face_box": item.payload["face_box"],
        }

    async def _process(self, item):
        for key in (
            "drowsy_ear",
            "ear_left",
            "ear_right",
            "eyes_closed_ms",
            "eye_state_source",
            "eye_closed_score_left",
            "eye_closed_score_right",
            "eye_blink_left",
            "eye_blink_right",
            "eye_state_age_ms",
        ):
            item.payload.pop(key, None)
            item.payload.get("signals", {}).pop(key, None)
        item.payload.update(eyes_state="unknown", left_eye_state="unknown", right_eye_state="unknown")
        image, box = item.payload.get("image_np"), item.payload.get("face_box")
        landmarks = item.payload.get("face_landmarks_106")
        inputs = None
        if image is not None and image.size and box is not None and landmarks is not None:
            height, width = image.shape[:2]
            points = np.asarray(landmarks, dtype=float) * [width, height]
            inputs = eye_inputs(image, points, item.payload.get("face_pose_raw"), self.min_eye_width_px)
        if inputs is None:
            self._scores.pop(item.stream_id, None)
        else:
            tensor, ears = inputs
            # Changing dimensions, the selected face or eyelid positions invalidates cached scores.
            signature = (image.shape[:2], tuple(box), tuple(np.asarray(landmarks).reshape(-1)))
            cached = self._scores.get(item.stream_id)
            age = item.ts_ms - cached[0] if cached else -1
            if cached and cached[1] == signature and 0 <= age < self.min_infer_interval_ms:
                scores = cached[2]
            else:
                spec = self.ctx.registry.get_spec(self.model)
                raw = await self.ctx.registry.infer(self.model, {spec.inputs[0].name: tensor})
                scores = closure_scores(raw[spec.outputs[0].name])
                self._scores[item.stream_id] = (item.ts_ms, signature, scores)
                age = 0
            item.payload["eye_state_age_ms"] = int(age)
            item.payload["face_analysis_age_ms"] = max(
                int(age), item.payload.get("head_pose_age_ms", 0), item.payload.get("eye_landmarks_age_ms", 0)
            )
            if scores is not None:
                self._update_eyes(item, scores, ears)
        if box is not None:
            item.payload["face_analysis_source"] = "scrfd_6drepnet360_2d106_blinkcnn"
            item.payload["face_status_label"] = _face_status_label_with_activity(
                item.payload.get("drowsy_ear"),
                item.payload.get("face_pose"),
                float(item.payload.get("eating_face_score", 0)),
                float(item.payload.get("smoking_face_score", 0)),
            )
        return item

    async def teardown(self):
        self._scores.clear()
        await super().teardown()
