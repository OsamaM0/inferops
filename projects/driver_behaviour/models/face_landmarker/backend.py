"""Stateless full-bundle inference, safely shared by independent streams."""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np

from inferops.ports.inference import InferenceBackend


class MediaPipeBackend(InferenceBackend):
    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir or '.')
        self._lock = asyncio.Lock()
        self._landmarker = None

    async def load(self, spec):
        self.spec = spec
        await asyncio.to_thread(self._load, spec)
        self.device = 'cpu'
        self.device_info = {'provider': 'MediaPipe Tasks / XNNPACK', 'running_mode': 'IMAGE'}

    def _load(self, spec):
        import mediapipe as mp

        self._mp = mp
        path = Path(spec.artifact.path)
        if not path.is_absolute():
            path = self.base_dir / path
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(path), delegate=mp.tasks.BaseOptions.Delegate.CPU),
            # IMAGE mode prevents one camera's tracking state leaking to another.
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=int(spec.tags.get('num_faces', '2')),
            min_face_detection_confidence=float(spec.tags.get('min_face_detection_confidence', '.5')),
            min_face_presence_confidence=float(spec.tags.get('min_face_presence_confidence', '.8')),
            output_face_blendshapes=True, output_facial_transformation_matrixes=True)
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def _detect(self, image):
        result = self._landmarker.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                                       data=np.ascontiguousarray(image)))
        landmarks = np.asarray([[(p.x, p.y, p.z) for p in face] for face in result.face_landmarks],
                               dtype=np.float32).reshape(-1, 478, 3)
        scores = []
        for categories in result.face_blendshapes:
            by_name = {category.category_name: category.score for category in categories}
            scores.append([by_name.get('eyeBlinkLeft', np.nan), by_name.get('eyeBlinkRight', np.nan)])
        return {'landmarks': landmarks, 'blink_scores': np.asarray(scores, np.float32).reshape(-1, 2),
                'matrices': np.asarray(result.facial_transformation_matrixes, np.float64).reshape(-1, 4, 4)}

    async def infer(self, inputs):
        image = inputs['image']
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
            raise ValueError('MediaPipe expects a nonempty HWC uint8 RGB image')
        async with self._lock:
            if self._landmarker is None:
                raise RuntimeError('MediaPipe backend is not loaded')
            task = asyncio.create_task(asyncio.to_thread(self._detect, image))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                await asyncio.gather(task, return_exceptions=True)
                raise

    async def close(self):
        async with self._lock:
            if self._landmarker is not None:
                await asyncio.to_thread(self._landmarker.close)
                self._landmarker = None
