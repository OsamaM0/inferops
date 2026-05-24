"""Pipeline registry — loads PipelineSpec YAMLs."""
from __future__ import annotations

from pathlib import Path

from inferops.core.config import get_spec_dir, settings
from inferops.core.errors import RegistryError
from inferops.core.yaml_loader import load_dir
from inferops.pipeline.runner import PipelineRunner
from inferops.pipeline.spec import PipelineSpec
from inferops.registry import get_registry


class PipelineStore:
    def __init__(self) -> None:
        self._specs: dict[str, PipelineSpec] = {}
        self._runners: dict[str, PipelineRunner] = {}

    async def reload_from(self, directory: str | Path) -> int:
        await self.close()
        self._specs.clear()
        for _path, data in load_dir(directory):
            spec = PipelineSpec(**data)
            self._specs[spec.name] = spec
        return len(self._specs)

    def list(self) -> list[PipelineSpec]:
        return list(self._specs.values())

    def get(self, name: str) -> PipelineSpec:
        if name not in self._specs:
            raise RegistryError(f"unknown pipeline: {name}")
        return self._specs[name]

    async def get_runner(self, name: str) -> PipelineRunner:
        if name in self._runners:
            return self._runners[name]
        spec = self.get(name)
        from inferops.adapters.transport import get_transport
        runner = PipelineRunner(spec, registry=get_registry(), transport=get_transport())
        await runner.setup()
        self._runners[name] = runner
        return runner

    async def close(self) -> None:
        for r in self._runners.values():
            await r.teardown()
        self._runners.clear()


_store: PipelineStore | None = None


def get_pipeline_store() -> PipelineStore:
    global _store
    if _store is None:
        _store = PipelineStore()
    return _store


async def autoload() -> None:
    s = get_pipeline_store()
    d = get_spec_dir("pipelines", settings)
    if d.exists():
        await s.reload_from(d)
