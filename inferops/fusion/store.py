"""FusionSpec registry/loader."""
from __future__ import annotations

from pathlib import Path

from inferops.core.config import get_spec_dir, settings
from inferops.core.errors import RegistryError
from inferops.core.yaml_loader import load_dir
from inferops.fusion.engine import FusionEngine
from inferops.fusion.spec import FusionSpec


class FusionStore:
    def __init__(self) -> None:
        self._specs: dict[str, FusionSpec] = {}
        self._engines: dict[str, FusionEngine] = {}

    async def reload_from(self, directory: str | Path) -> int:
        self._specs.clear()
        self._engines.clear()
        for _p, data in load_dir(directory):
            spec = FusionSpec(**data)
            self._specs[spec.name] = spec
        return len(self._specs)

    def list(self): return list(self._specs.values())

    def get(self, name: str) -> FusionEngine:
        if name not in self._specs:
            raise RegistryError(f"unknown fusion: {name}")
        if name not in self._engines:
            self._engines[name] = FusionEngine(self._specs[name])
        return self._engines[name]


_store: FusionStore | None = None


def get_fusion_store() -> FusionStore:
    global _store
    if _store is None:
        _store = FusionStore()
    return _store


async def autoload() -> None:
    s = get_fusion_store()
    d = get_spec_dir("fusion", settings)
    if d.exists():
        await s.reload_from(d)
