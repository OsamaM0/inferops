"""In-memory model registry with hot-reload."""
from __future__ import annotations

import asyncio
from contextlib import suppress
from pathlib import Path

from inferops.core.config import get_spec_dir, settings
from inferops.core.errors import ConfigError, RegistryError
from inferops.obs import get_logger
from inferops.registry.loader import load_models_with_sources
from inferops.registry.model_spec import ModelSpec
from inferops.runtimes.base import InferenceBackend

log = get_logger(__name__)


class ModelRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ModelSpec] = {}
        self._backends: dict[str, InferenceBackend] = {}
        self._lock = asyncio.Lock()
        self._base_dir: Path | None = None
        self._source_dirs: dict[str, Path] = {}
        self._resolver = None

    def resolve_name(self, name: str) -> str:
        return self._resolver.resolve(name, settings.execution_mode) if self._resolver else name

    @property
    def base_dir(self) -> Path | None:
        return self._base_dir

    def source_dir(self, name: str) -> Path | None:
        """Directory owning a runtime's configuration and relative assets."""
        return self._source_dirs.get(self.resolve_name(name), self._base_dir)

    async def reload_from(self, directory: str | Path) -> int:
        directory = Path(directory)
        sources = load_models_with_sources(directory)
        from inferops.registry.resolver import ModelResolver, canonical_mode

        resolver = ModelResolver(directory)
        names = [spec.name for _, spec in sources]
        if len(set(names)) != len(names):
            raise ConfigError(f"duplicate model names in {directory}")
        async with self._lock:
            await self._teardown_locked()
            self._resolver = resolver
            for path, s in sources:
                mode = canonical_mode(settings.execution_mode)
                if s.backend == "onnxruntime" and mode in {"cpu", "cuda"}:
                    s.device.preferred = "cpu" if mode == "cpu" else "gpu"
                self._specs[s.name] = s
                self._source_dirs[s.name] = path.parent
            self._base_dir = directory
            log.info("registry_loaded", count=len(sources), dir=str(directory))
            return len(sources)

    async def _teardown_locked(self) -> None:
        for b in self._backends.values():
            with suppress(Exception):
                await b.close()
        self._backends.clear()
        self._specs.clear()
        self._source_dirs.clear()

    def list_models(self) -> list[ModelSpec]:
        return list(self._specs.values())

    def logical_models(self) -> list[dict]:
        """Report model ports separately from their configured runtime variants."""
        models = []
        grouped = set()
        if self._resolver:
            for name, variants in self._resolver.variants.items():
                grouped.update(variants)
                try:
                    selected = self.resolve_name(name)
                except ConfigError:
                    selected = None
                models.append({"name": name, "variants": variants, "selected_variant": selected,
                               "loaded_variants": [variant for variant in variants if variant in self._backends]})
        for name in self._specs.keys() - grouped:
            models.append({"name": name, "variants": [name], "selected_variant": name,
                           "loaded_variants": [name] if name in self._backends else []})
        return sorted(models, key=lambda model: model["name"])

    def get_spec(self, name: str) -> ModelSpec:
        name = self.resolve_name(name)
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
        name = self.resolve_name(name)
        # Fast path first: this runs once per inference call, and taking the
        # registry lock on every frame serialised the whole pipeline behind any
        # concurrent model load.
        cached = self._backends.get(name)
        if cached is not None:
            return cached
        async with self._lock:
            cached = self._backends.get(name)
            if cached is not None:
                return cached
            from inferops.runtimes import build_backend
            spec = self.get_spec(name)
            backend = build_backend(spec, base_dir=self.source_dir(name))
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
