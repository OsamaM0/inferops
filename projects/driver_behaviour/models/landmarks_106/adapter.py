"""Locate eyelids for BlinkCNN using the detector's selected face."""

from __future__ import annotations

from inferops.adapters.outbound.stages.base import Stage

from ..face_monitoring.geometry import decode_landmarks, landmark_tensor


class Landmarks106Stage(Stage):
    def __init__(self, model="landmarks_106", min_infer_interval_ms=100, fallback_model=None):
        if fallback_model is not None:
            raise ValueError("fallback_model is unsupported; choose one explicit model")
        super().__init__(model=model, min_infer_interval_ms=min_infer_interval_ms)
        self.model, self.interval = model, max(0, int(min_infer_interval_ms))
        self._points = {}

    async def process(self, item):
        item.payload.pop("face_landmarks_106", None)
        box = item.payload.get("face_box")
        image = item.payload.get("image_np")
        if box is None or image is None or not image.size:
            self._points.pop(item.stream_id, None)
            return item
        height, width = image.shape[:2]
        signature = (width, height, *box)
        cached = self._points.get(item.stream_id)
        age = item.ts_ms - cached[0] if cached else -1
        if cached and cached[1] == signature and 0 <= age < self.interval:
            points = cached[2]
        else:
            tensor, transform = landmark_tensor(image, box)
            spec = self.ctx.registry.get_spec(self.model)
            raw = await self.ctx.registry.infer(self.model, {spec.inputs[0].name: tensor})
            points = decode_landmarks(raw[spec.outputs[0].name], transform)
            self._points[item.stream_id] = (item.ts_ms, signature, points)
            age = 0
        item.payload["eye_landmarks_age_ms"] = int(age)
        if points is not None:
            # JSON-safe, normalized full-frame coordinates; not a MediaPipe mesh.
            item.payload["face_landmarks_106"] = (points / [width, height]).tolist()
        return item

    async def teardown(self):
        self._points.clear()
