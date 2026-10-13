"""AlertSpec registry."""
from __future__ import annotations

from pathlib import Path

from inferops.alerts.engine import AlertEngine
from inferops.alerts.spec import AlertSpec
from inferops.core.config import get_spec_dir, settings
from inferops.core.errors import RegistryError
from inferops.core.yaml_loader import load_dir


class AlertStore:
    def __init__(self) -> None:
        self._specs: dict[str, AlertSpec] = {}
        self._engines: dict[str, AlertEngine] = {}

    async def reload_from(self, directory: str | Path) -> int:
        for e in self._engines.values():
            await e.close()
        self._specs.clear()
        self._engines.clear()
        for _p, data in load_dir(directory):
            spec = AlertSpec(**data)
            self._specs[spec.name] = spec
        return len(self._specs)

    def list(self): return list(self._specs.values())

    def get(self, name: str) -> AlertEngine:
        if name not in self._specs:
            raise RegistryError(f"unknown alerts spec: {name}")
        if name not in self._engines:
            self._engines[name] = AlertEngine(self._specs[name])
        return self._engines[name]


_store: AlertStore | None = None


def get_alert_store() -> AlertStore:
    global _store
    if _store is None:
        _store = AlertStore()
    return _store


async def autoload() -> None:
    s = get_alert_store()
    d = get_spec_dir("alerts", settings)
    if d.exists():
        await s.reload_from(d)
