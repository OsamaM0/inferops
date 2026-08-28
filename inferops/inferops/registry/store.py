"""In-memory model registry with hot-reload."""
from __future__ import annotations

import asyncio
from contextlib import suppress
from pathlib import Path

from inferops.core.config import get_spec_dir, settings
from inferops.core.errors import RegistryError
from inferops.obs import get_logger
from inferops.registry.loader import load_models_dir
from inferops.registry.model_spec import ModelSpec
from inferops.runtimes.base import InferenceBackend

log = get_logger(__name__)


class ModelRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ModelSpec] = {}
        self._backends: dict[str, InferenceBackend] = {}
        self._lock = asyncio.Lock()
        self._base_dir: Path | None = None

    @property
    def base_dir(self) -> Path | None:
        return self._base_dir

    async def reload_from(self, directory: str | Path) -> int:
        directory = Path(directory)
        async with self._lock:
            await self._teardown_locked()
            specs = load_models_dir(directory)
            for s in specs:
                self._specs[s.name] = s
            self._base_dir = directory
            log.info("registry_loaded", count=len(specs), dir=str(directory))
            return len(specs)

    async def _teardown_locked(self) -> None:
        for b in self._backends.values():
            with suppress(Exception):
                await b.close()
        self._backends.clear()
        self._specs.clear()

    def list_models(self) -> list[ModelSpec]:
        return list(self._specs.values())

    def get_spec(self, name: str) -> ModelSpec:
        if name not in self._specs:
            raise RegistryError(f"unknown model: {name}")
        return self._specs[name]

    def loaded_backends(self) -> dict[str, InferenceBackend]:
        """Backends built so far, for status reporting.

        A copy, so callers cannot mutate the cache, and read without the lock
        because status endpoints must never block the inference path.
        """
        return dict(self._backends)

    async def get_backend(self, name: str) -> InferenceBackend:
        # Fast path first: this runs once per inference call, and taking the
        # registry lock on every frame serialised the whole pipeline behind any
        # concurrent model load.
        cached = self._backends.get(name)
        if cached is not None:
            return cached
        spec = self.get_spec(name)
        async with self._lock:
            cached = self._backends.get(name)
            if cached is not None:
                return cached
            from inferops.runtimes import build_backend
            backend = build_backend(spec, base_dir=self._base_dir)
            try:
                await backend.load(spec)
                if spec.warmup.iterations > 0:
                    await backend.warmup()
            except Exception:
                # A backend that failed partway can still hold a gRPC channel,
                # an ORT session or CUDA memory. Release it, and leave nothing
                # in the cache so the next call retries cleanly instead of
                # handing out a permanently broken backend.
                with suppress(Exception):
                    await backend.close()
                raise
            self._backends[name] = backend
            return backend

    async def close(self) -> None:
        async with self._lock:
            await self._teardown_locked()


_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


async def autoload() -> None:
    """Load configs/models/ at startup."""
    reg = get_registry()
    models_dir = get_spec_dir("models", settings)
    if models_dir.exists():
        await reg.reload_from(models_dir)
