"""Softmax + top-k classification postprocess."""
from __future__ import annotations

import numpy as np

from inferops.domain.models import ModelSpec
from inferops.domain.types import Classification, Frame, Result
from inferops.ports.extensions import Postprocessor


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class SoftmaxTopK(Postprocessor):
    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k

    async def __call__(self, raw, frame: Frame, spec: ModelSpec) -> Result:
        out_name = spec.outputs[0].name if spec.outputs else next(iter(raw))
        logits = raw[out_name]
        if logits.ndim == 2:
            logits = logits[0]
        probs = _softmax(logits)
        idx = np.argsort(-probs)[: self.top_k]
        labels = spec.label_list() or [str(i) for i in range(len(probs))]
        items = [Classification(label=labels[i] if i < len(labels) else str(i),
                                score=float(probs[i]), index=int(i)) for i in idx]
        return Result(kind="classifications", classifications=items, model=spec.name)
