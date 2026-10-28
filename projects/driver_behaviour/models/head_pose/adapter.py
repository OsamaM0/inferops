"""6DRepNet pose inference on the selected driver crop, independent of FaceMesh."""
from __future__ import annotations

import cv2
import numpy as np

from inferops.adapters.outbound.stages.base import Stage

from ..face_mesh.adapter import _boxes_are_close, _crop_face, _face_status_label, _get_face_box
from ..face_mesh.features import behavior_scores_from_face


def rotation_to_pose(rotation: np.ndarray) -> dict[str, float] | None:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    if (not np.isfinite(matrix).all() or not np.allclose(matrix.T @ matrix, np.eye(3), atol=.02)
            or not np.isclose(np.linalg.det(matrix), 1.0, atol=.02)):
        return None
    sy = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    # Upstream draws its forward axis with y=-cos(yaw)*sin(pitch):
    # positive model pitch points up. Publish positive pitch as driver down.
    # https://github.com/thohemp/6DRepNet/blob/master/sixdrepnet/utils.py
    return {"pitch": -float(np.degrees(np.arctan2(matrix[2, 1], matrix[2, 2]))),
            "yaw": -float(np.degrees(np.arctan2(-matrix[2, 0], sy))),
            "roll": float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))}


class HeadPoseStage(Stage):
    def __init__(self, model: str = "head_pose", crop_padding: float = .2,
                 min_infer_interval_ms: int = 100, fallback_model=None) -> None:
        super().__init__(model=model, crop_padding=crop_padding,
                         min_infer_interval_ms=min_infer_interval_ms)
        self.model = model
        self.padding = crop_padding
        self.interval = max(0, int(min_infer_interval_ms))
        self._cache = {}

    async def process(self, item):
        image = item.payload.get("image_np")
        if image is None or image.size == 0:
            return item
        h, w = image.shape[:2]
        box = _get_face_box(item.results, w, h)
        if box is None:
            self._cache.pop(item.stream_id, None)
            return item
        cached = self._cache.get(item.stream_id)
        if (cached and cached[1] == (w, h) and 0 <= item.ts_ms - cached[0] < self.interval
                and _boxes_are_close(box, cached[2])):
            pose = dict(cached[3])
        else:
            crop, _ = _crop_face(image, box, padding=self.padding)
            spec = self.ctx.registry.get_spec(self.model)
            size = spec.inputs[0].shape[-1]
            rgb = cv2.cvtColor(cv2.resize(crop, (size, size)), cv2.COLOR_BGR2RGB)
            # Qualcomm's export embeds ImageNet normalization inside the graph.
            tensor = np.ascontiguousarray((rgb.astype(np.float32) / 255).transpose(2, 0, 1)[None])
            backend = await self.ctx.registry.get_backend(self.model)
            raw = await backend.infer({spec.inputs[0].name: tensor})
            pose = rotation_to_pose(raw["rotation_matrix"])
            if pose is None:
                self._cache.pop(item.stream_id, None)
                return item
            self._cache[item.stream_id] = (item.ts_ms, (w, h), box, dict(pose))
        item.payload.update({"face_pose_raw": dict(pose), "face_pose": pose, "head_pose_source": "6drepnet",
                             "face_box": box, "face_source": "scrfd"})
        item.payload.update(behavior_scores_from_face(None, None, pose["yaw"], pose["pitch"]))
        item.payload["face_status_label"] = _face_status_label(None, pose)
        return item

    async def teardown(self) -> None:
        self._cache.clear()
