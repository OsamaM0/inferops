"""Synthetic DETR-format detector for demos and tests without real weights.

Emits per-call ``logits`` (1, Q, C) and cxcywh-normalised ``boxes`` (1, Q, 4),
so any DETR-style postprocessor (``inferops.plugins.builtin.detr_decode`` or a
domain's own) decodes it exactly like a real export. Which classes fire, and
how often, comes from the model spec so the backend knows nothing about any
particular domain::

    labels:
      inline: [person, helmet, forklift]
    tags:
      # label=every_n_frames[@cx,cy,w,h]; ';'-separated. Frame 1 counts as 1.
      emit: "helmet=2@0.5,0.3,0.2,0.2; forklift=7"
      queries: "20"
      seed: "0"

A label without a box gets a deterministic box derived from its index. Labels
not listed in ``emit`` never fire; an empty ``emit`` yields background only.
"""
from __future__ import annotations

import re

import numpy as np

from inferops.registry.model_spec import ModelSpec
from inferops.runtimes.base import InferenceBackend

_EMIT_RE = re.compile(r"^\s*([^=@;]+?)\s*=\s*(\d+)\s*(?:@\s*([0-9.,\s]+))?\s*$")
_HOT_LOGIT = 3.5   # sigmoid -> ~0.97, softmax over cold (-3.0) rows -> ~0.99


def _parse_emit(raw: str, labels: list[str]) -> list[tuple[int, int, np.ndarray]]:
    """Return ``[(class_index, every_n, box_cxcywh)]`` from the ``emit`` tag."""
    out: list[tuple[int, int, np.ndarray]] = []
    for chunk in (part for part in raw.split(";") if part.strip()):
        m = _EMIT_RE.match(chunk)
        if not m:
            raise ValueError(f"synthetic backend: cannot parse emit entry {chunk!r}")
        label, every, box = m.group(1), int(m.group(2)), m.group(3)
        if every <= 0:
            continue
        if label not in labels:
            raise ValueError(f"synthetic backend: label {label!r} is not in the model's labels {labels}")
        idx = labels.index(label)
        if box:
            vals = [float(v) for v in box.replace(" ", "").split(",") if v]
            if len(vals) != 4:
                raise ValueError(f"synthetic backend: box for {label!r} needs cx,cy,w,h")
            arr = np.array(vals, dtype=np.float32)
        else:
            # Spread default boxes across the frame so per-class tracks stay distinct.
            arr = np.array([0.2 + 0.15 * (idx % 5), 0.25 + 0.2 * (idx % 3), 0.2, 0.2], dtype=np.float32)
        out.append((idx, every, arr))
    return out


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
        labels = spec.label_list(self.base_dir)
        self._classes = max(int(tags.get("classes", "0") or 0), len(labels), 1)
        self._queries = int(tags.get("queries", "20"))
        self._emit = _parse_emit(tags.get("emit", ""), labels)
        if len(self._emit) > self._queries:
            raise ValueError("synthetic backend: more emit entries than queries")
        self._rng = np.random.default_rng(int(tags.get("seed", "0")))

    async def infer(self, inputs):
        self._n += 1
        Q, C = self._queries, self._classes
        logits = self._rng.normal(loc=-3.0, scale=0.3, size=(1, Q, C)).astype(np.float32)
        boxes = self._rng.uniform(0.05, 0.95, size=(1, Q, 4)).astype(np.float32)
        for slot, (cls_idx, every, box) in enumerate(self._emit):
            if self._n % every == 0:
                logits[0, slot, cls_idx] = _HOT_LOGIT
                boxes[0, slot] = box
        return {"logits": logits, "boxes": boxes}

    async def close(self) -> None:
        pass
