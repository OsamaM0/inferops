"""RetinaFace as a plain ``infer`` stage: preprocess and postprocess plugins.

When ``face_tracker`` is selected it calls RetinaFace itself (``uses``), so
these plugins only run when RetinaFace is selected on its own as a detector.
"""
from __future__ import annotations

import asyncio

from inferops.domain.models import ModelSpec
from inferops.domain.types import Detection, Frame, Result
from inferops.ports.extensions import Postprocessor, Preprocessor

from .retinaface import INPUT_SIZE, decode, preprocess


def input_spec(spec: ModelSpec) -> tuple[str, int]:
    """Input tensor name and square size from the model contract."""
    if not spec.inputs:
        return "input", INPUT_SIZE
    shape = spec.inputs[0].shape
    size = int(shape[-1]) if len(shape) >= 4 and int(shape[-1]) > 0 else INPUT_SIZE
    return spec.inputs[0].name or "input", size


class RetinaFacePreprocess(Preprocessor):
    async def __call__(self, frame: Frame, spec: ModelSpec):
        if frame.image is None:
            raise ValueError("image required")
        name, size = input_spec(spec)
        feeds, scale = await asyncio.to_thread(preprocess, frame.image, name, size)
        frame.meta["retinaface"] = {"scale": scale, "size": size}
        return feeds


class RetinaFacePostprocess(Postprocessor):
    def __init__(self, conf_threshold: float = 0.6, nms_threshold: float = 0.4, min_face_px: float = 0.0) -> None:
        self.conf = conf_threshold
        self.nms = nms_threshold
        self.min_face_px = min_face_px

    async def __call__(self, raw, frame: Frame, spec: ModelSpec) -> Result:
        meta = frame.meta.get("retinaface") or {}
        _, size = input_spec(spec)
        size = int(meta.get("size", size))
        w, h = max(int(frame.width or 1), 1), max(int(frame.height or 1), 1)
        scale = float(meta.get("scale", size / max(w, h)))
        faces = decode(raw, scale, size=size, score_threshold=self.conf, nms_threshold=self.nms)
        detections = []
        for face in faces:
            if min(face.width, face.height) < self.min_face_px:
                continue
            x1, y1, x2, y2 = face.box
            box = (max(0.0, x1 / w), max(0.0, y1 / h), min(1.0, x2 / w), min(1.0, y2 / h))
            detections.append(Detection(label="face", score=face.score, box=box, attrs={
                "orig_w": w, "orig_h": h, "xyxy_pixels": face.box,
                "keypoints": [[float(x) / w, float(y) / h] for x, y in face.keypoints]}))
        return Result(kind="detections", detections=detections, model=spec.name,
                      raw={"faces": len(detections)})
