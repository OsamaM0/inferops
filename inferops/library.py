"""Local, service-free model inference. Images use OpenCV's HWC BGR convention."""
from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from inferops.core.domain import resolve_plugin_ref
from inferops.core.plugins import instantiate
from inferops.core.types import Frame, Result, now_ms
from inferops.registry.loader import load_model_file
from inferops.registry.model_spec import ModelSpec
from inferops.runtimes import build_backend


class Model:
    """An independent model with preprocessing, a lazy backend, and decoding."""

    def __init__(self, spec: ModelSpec, *, base_dir: str | Path | None = None):
        self.spec = spec.model_copy(deep=True)
        self.spec.backend = resolve_plugin_ref(self.spec.backend)
        self.base_dir = Path(base_dir).resolve() if base_dir else None
        if self.spec.artifact.path and self.base_dir and not Path(self.spec.artifact.path).is_absolute():
            from inferops.core.artifacts import resolve_artifact
            artifact, _ = resolve_artifact(self.spec.artifact.path, base_dir=self.base_dir)
            self.spec.artifact.path = str((artifact or self.base_dir / self.spec.artifact.path).resolve())
        self._pre = instantiate(spec.preprocess.plugin, spec.preprocess.params) if spec.preprocess else None
        self._post = instantiate(spec.postprocess.plugin, spec.postprocess.params) if spec.postprocess else None
        self._backend = None
        self._lock = asyncio.Lock()

    @classmethod
    def from_yaml(cls, path: str | Path) -> Model:
        """Load one flat model spec; use from_project for logical runtime variants."""
        path = Path(path).resolve()
        return cls(load_model_file(path), base_dir=path.parent)

    async def load(self) -> Model:
        async with self._lock:
            if self._backend is None:
                backend = build_backend(self.spec, base_dir=self.base_dir)
                try:
                    await backend.load(self.spec)
                    if self.spec.warmup.iterations:
                        await backend.warmup()
                except BaseException:
                    await backend.close()
                    raise
                self._backend = backend
        return self

    async def predict(self, image: np.ndarray, *, stream_id: str = "local") -> Result:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be an HWC BGR array with three channels")
        frame = Frame(stream_id=stream_id, ts_ms=now_ms(), image=image,
                      width=image.shape[1], height=image.shape[0])
        if self._pre is None:
            raise ValueError(f"model {self.spec.name!r} has no preprocessor; use infer(inputs)")
        feeds = await self._pre(frame, self.spec)
        started = time.perf_counter()
        raw = await self.infer(feeds)
        result = await self._post(raw, frame, self.spec) if self._post else Result(
            kind="raw", model=self.spec.name, raw={key: value.tolist() for key, value in raw.items()})
        result.latency_ms = (time.perf_counter() - started) * 1000
        return result

    async def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run named tensors directly, bypassing image preprocessing and decoding."""
        await self.load()
        return await self._backend.infer(inputs)

    async def close(self) -> None:
        async with self._lock:
            if self._backend is not None:
                await self._backend.close()
                self._backend = None

    async def __aenter__(self) -> Model:
        return await self.load()

    async def __aexit__(self, *exc) -> None:
        await self.close()


class InferenceSession:
    """Run named models together. Each model gets its own frame metadata."""

    def __init__(self, models: Mapping[str, Model]):
        if not models:
            raise ValueError("an inference session requires at least one model")
        self.models = dict(models)

    @classmethod
    def from_project(cls, root: str | Path, *, models: list[str] | None = None,
                     runtime: str = "cpu") -> InferenceSession:
        """Load selected model ports from a project folder or installed adapter name."""
        from inferops.core.config import ConfigStack
        from inferops.core.domain import active_package
        from inferops.project_adapters import adapter_root
        from inferops.registry.loader import load_models_with_sources
        from inferops.registry.resolver import ModelResolver

        path = Path(root)
        if not path.is_dir():
            path = adapter_root(str(root))
        previous = active_package()
        try:
            stack = ConfigStack(domain_root=path)
            directory = stack.spec_dir("models")
            resolver = ModelResolver(directory)
            sources = {spec.name: (source, spec) for source, spec in load_models_with_sources(directory)}
            grouped = {variant for variants in resolver.variants.values() for variant in variants}
            names = models if models is not None else [*resolver.ports, *sorted(resolver.names - grouped)]
            selected = {}
            for name in names:
                source, spec = sources[resolver.resolve(name, runtime)]
                if spec.backend == "onnxruntime" and runtime in {"cpu", "cuda"}:
                    spec.device.preferred = "gpu" if runtime == "cuda" else "cpu"
                if spec.artifact.path and not Path(spec.artifact.path).is_absolute():
                    from inferops.core.artifacts import resolve_artifact
                    artifact, _ = resolve_artifact(spec.artifact.path, base_dir=source.parent,
                                                   domain_root=path.resolve())
                    spec.artifact.path = str((artifact or path.resolve() / spec.artifact.path).resolve())
                # Instantiate plugins while this project's relative refs are active.
                selected[name] = Model(spec, base_dir=source.parent)
            return cls(selected)
        finally:
            # Restore relative plugin resolution for callers that already have a project active.
            from inferops.core import domain
            domain._active_package = previous

    async def predict(self, image: np.ndarray, *, stream_id: str = "local",
                      parallel: bool = False) -> dict[str, Result]:
        if parallel:
            # Wait for every task before propagating errors, so close() cannot race a sibling.
            results = await asyncio.gather(
                *(model.predict(image, stream_id=stream_id) for model in self.models.values()),
                return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            return dict(zip(self.models, results, strict=True))
        return {name: await model.predict(image, stream_id=stream_id) for name, model in self.models.items()}

    async def close(self) -> None:
        results = await asyncio.gather(*(model.close() for model in self.models.values()), return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def __aenter__(self) -> InferenceSession:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()
