"""Engine-owned backend pool with explicit project/plugin dependencies."""
from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np

from inferops.domain.errors import RegistryError
from inferops.domain.models import ModelSpec
from inferops.domain.types import Frame, Result
from inferops.ports.inference import InferenceBackend
from inferops.ports.projects import ProjectDefinition


class _BoundedBackend:
    def __init__(self, backend: InferenceBackend, concurrency: int) -> None:
        self._backend = backend
        self._slots = asyncio.Semaphore(concurrency)
        self._active: set[asyncio.Task] = set()
        self._closed = False

    @property
    def spec(self): return self._backend.spec

    @property
    def device(self): return self._backend.device

    @property
    def device_info(self): return self._backend.device_info

    async def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        async with self._slots:
            if self._closed:
                raise RuntimeError("backend pool is closed")
            # Cancelling an await cannot interrupt a thread/CUDA call safely.
            # Wait for it before releasing a slot or allowing backend teardown.
            task = asyncio.create_task(self._backend.infer(inputs))
            self._active.add(task)
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                await asyncio.gather(task, return_exceptions=True)
                raise
            finally:
                self._active.discard(task)

    async def close(self) -> None:
        self._closed = True
        if self._active:
            await asyncio.gather(*self._active, return_exceptions=True)
        await self._backend.close()


class ModelPool:
    def __init__(self, definition: ProjectDefinition, *, max_concurrency: int = 1) -> None:
        if max_concurrency < 1:
            raise ValueError("model concurrency must be positive")
        self.definition = definition
        self.max_concurrency = max_concurrency
        self._backends: dict[str, _BoundedBackend] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._plugins: dict[str, tuple[Any, Any]] = {}
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def _loaded(self) -> dict[str, _BoundedBackend]:
        """Compatibility view for migrated project stages/status adapters."""
        return self._backends

    def get_spec(self, name: str) -> ModelSpec:
        resolved = self.resolve_name(name)
        if resolved not in self.definition.models:
            raise RegistryError(f"unknown model: {name}")
        return self.definition.models[resolved].model_copy(deep=True)

    def resolve_name(self, name: str) -> str:
        if name not in self.definition.models and (self.definition.resolver is None or (name not in self.definition.resolver.names and name not in self.definition.resolver.ports)):
            raise RegistryError(f"unknown model: {name}")
        return self.definition.resolve_model(name)

    def source_dir(self, name: str):
        return self.definition.source_dirs.get(self.definition.resolve_model(name))

    def loaded_backends(self):
        return dict(self._backends)

    def list_models(self):
        return self.list_specs()

    def list_specs(self) -> list[ModelSpec]:
        return [spec.model_copy(deep=True) for spec in self.definition.models.values()]

    def logical_models(self) -> list[dict]:
        result, grouped = [], set()
        for name, variants in (self.definition.resolver.variants if self.definition.resolver else {}).items():
            grouped.update(variants)
            result.append({"name": name, "variants": variants, "selected_variant": self.resolve_name(name),
                           "loaded_variants": [variant for variant in variants if variant in self._backends]})
        for name in self.definition.models.keys() - grouped:
            result.append({"name": name, "variants": [name], "selected_variant": name,
                           "loaded_variants": [name] if name in self._backends else []})
        return sorted(result, key=lambda row: row["name"])

    async def get_backend(self, name: str) -> _BoundedBackend:
        if self._closed:
            raise RuntimeError("model pool is closed")
        name = self.resolve_name(name)
        lock = self._locks.setdefault(name, asyncio.Lock())
        async with lock:
            if self._closed:
                raise RuntimeError("model pool is closed")
            if name not in self._backends:
                spec = self.get_spec(name)
                backend = self.definition.backend_factory(spec, self.definition.source_dirs.get(name))
                try:
                    await backend.load(spec)
                    if spec.warmup.iterations and hasattr(backend, "warmup"):
                        await backend.warmup()
                except BaseException:
                    await backend.close()
                    raise
                self._backends[name] = _BoundedBackend(backend, self.max_concurrency)
            return self._backends[name]

    async def infer(self, name: str, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        backend = await self.get_backend(name)
        return await backend.infer(inputs)

    async def predict(self, name: str, image: np.ndarray, *, stream_id: str = "local",
                      ts_ms: int = 0, seq: int = 0) -> Result:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be an HWC BGR array with three channels")
        spec = self.get_spec(name)
        if spec.name not in self._plugins:
            loader = self.definition.object_loader
            pre = loader(spec.preprocess.plugin)(**spec.preprocess.params) if spec.preprocess else None
            post = loader(spec.postprocess.plugin)(**spec.postprocess.params) if spec.postprocess else None
            self._plugins[spec.name] = pre, post
        pre, post = self._plugins[spec.name]
        if pre is None:
            raise ValueError(f"model {name!r} has no image preprocessor; use get_backend().infer(inputs)")
        frame = Frame(stream_id=stream_id, ts_ms=ts_ms, seq=seq, image=image,
                      width=image.shape[1], height=image.shape[0])
        feeds = await pre(frame, spec)
        backend = await self.get_backend(name)
        started = time.monotonic()
        outputs = await backend.infer(feeds)
        result = await post(outputs, frame, spec) if post else Result(
            kind="raw", model=spec.name,
            raw={key: value.tolist() if hasattr(value, "tolist") else value for key, value in outputs.items()})
        result.model = result.model or spec.name
        result.latency_ms = (time.monotonic() - started) * 1000
        return result

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # A concurrent lazy load must finish/notice closure before teardown.
        acquired = []
        try:
            for lock in self._locks.values():
                await lock.acquire()
                acquired.append(lock)
            outcomes = await asyncio.gather(*(backend.close() for backend in self._backends.values()),
                                            return_exceptions=True)
            self._backends.clear()
            self._plugins.clear()
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    raise outcome
        finally:
            for lock in acquired:
                lock.release()
