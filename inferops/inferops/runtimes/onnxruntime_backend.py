"""ONNX Runtime backend (CPU + CUDA)."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

from inferops.core.errors import BackendError
from inferops.obs import INFER_BATCH, INFER_LATENCY
from inferops.registry.model_spec import ModelSpec
from inferops.runtimes.base import InferenceBackend


class OnnxRuntimeBackend(InferenceBackend):
    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir
        self._sess = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []

    async def load(self, spec: ModelSpec) -> None:
        import onnxruntime as ort
        self.spec = spec
        if not spec.artifact.path:
            raise BackendError(f"model {spec.name}: onnxruntime backend requires artifact.path")
        p = Path(spec.artifact.path)
        if not p.is_absolute():
            tried = []
            candidates = []
            if self.base_dir:
                candidates += [self.base_dir / p, self.base_dir.parent / p]
            # also try cwd and settings.model_repository sibling
            from inferops.core.config import get_model_repository_root
            from inferops.core.config import settings as _s

            model_repository_root = get_model_repository_root(_s)
            candidates += [
                Path.cwd() / p,
                model_repository_root / p,
                Path(_s.configs_dir).resolve().parent / p,
            ]
            if _s.domain_root:
                candidates.append(Path(_s.domain_root) / p)
            candidates.append(Path(p))
            for c in candidates:
                tried.append(str(c))
                if c.exists():
                    p = c
                    break
            else:
                raise BackendError(f"model file not found, tried: {tried}")
        if not p.exists():
            raise BackendError(f"model file not found: {p}")
        providers = ["CPUExecutionProvider"]
        if spec.device.preferred == "gpu":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._sess = ort.InferenceSession(str(p), providers=providers)
        self._input_names = [i.name for i in self._sess.get_inputs()]
        self._output_names = [o.name for o in self._sess.get_outputs()]

    async def infer(self, inputs):
        if self._sess is None:
            raise BackendError("backend not loaded")
        feeds = {n: inputs[n] for n in self._input_names if n in inputs}
        first = next(iter(feeds.values()))
        INFER_BATCH.labels(model=self.spec.name).observe(first.shape[0] if first.ndim > 0 else 1)
        t0 = time.perf_counter()
        outs = await asyncio.to_thread(self._sess.run, self._output_names, feeds)
        INFER_LATENCY.labels(model=self.spec.name).observe(time.perf_counter() - t0)
        return dict(zip(self._output_names, outs, strict=True))

    async def close(self) -> None:
        self._sess = None
