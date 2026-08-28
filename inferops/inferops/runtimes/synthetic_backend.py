"""Synthetic DETR-format backend for demos and tests without real weights.

Generates per-call (logits, boxes) tensors shaped (1, Q, C) and (1, Q, 4) in
cxcywh-normalised form. A frame counter drives which class is "detected":
phone every Nth call, cigarette occasionally, etc., so the fusion engine sees
realistic time-varying signals when the video pipeline is exercised.

Configure via ModelSpec.tags:
    phone_every: int (default 3)
    cig_every:   int (default 17)
    seed:        int (default 0)
"""
from __future__ import annotations

import numpy as np

from inferops.registry.model_spec import ModelSpec
from inferops.runtimes.base import InferenceBackend


class SyntheticDetectorBackend(InferenceBackend):
    def __init__(self, base_dir=None) -> None:
        self.base_dir = base_dir
        self._n = 0

    async def load(self, spec: ModelSpec) -> None:
        self.spec = spec
        # Pure numpy: always CPU, regardless of what DeviceSpec asks for.
        self.device = "cpu"
        self.device_info = {
            "requested": spec.device.preferred,
            "effective": "cpu",
            "downgraded": spec.device.preferred == "gpu",
            "reason": "synthetic backend runs in numpy on the CPU",
            "active_providers": [],
        }
        tags = spec.tags or {}
        self._phone_every = int(tags.get("phone_every", "3"))
        self._cig_every = int(tags.get("cig_every", "17"))
        self._driver_every = int(tags.get("driver_every", "0"))
        self._classes = int(tags.get("classes", "6"))
        self._queries = int(tags.get("queries", "20"))
        self._rng = np.random.default_rng(int(tags.get("seed", "0")))

    async def infer(self, inputs):
        self._n += 1
        Q, C = self._queries, self._classes
        # base low-confidence logits
        logits = self._rng.normal(loc=-3.0, scale=0.3, size=(1, Q, C)).astype(np.float32)
        boxes = self._rng.uniform(0.05, 0.95, size=(1, Q, 4)).astype(np.float32)

        # Phone class (index 3 in legacy mapping)
        if self._n % self._phone_every == 0:
            logits[0, 0, 3] = 3.5         # sigmoid -> ~0.97
            boxes[0, 0] = [0.55, 0.55, 0.20, 0.18]
        # Cigarette (index 1) occasionally
        if self._n % self._cig_every == 0:
            logits[0, 1, 1] = 3.0
            boxes[0, 1] = [0.30, 0.60, 0.06, 0.06]
        # DriverBehaviour (index 0) – enables face mesh when driver_every > 0
        if self._driver_every > 0 and self._n % self._driver_every == 0:
            logits[0, 2, 0] = 4.0         # strong DriverBehaviour signal
            boxes[0, 2] = [0.50, 0.35, 0.40, 0.50]  # centred face/torso box

        return {"logits": logits, "boxes": boxes}

    async def close(self) -> None:
        pass
