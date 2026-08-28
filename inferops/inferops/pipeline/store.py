"""Pipeline registry — loads PipelineSpec YAMLs."""
from __future__ import annotations

import asyncio
from contextlib import suppress
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
        # Guards runner construction. Without it two concurrent requests for the
        # same pipeline each build and set up a runner; the loser is overwritten
        # in `_runners` and orphaned with its stages still holding Kafka
        # producers, S3 clients and WebSocket registrations.
        self._lock = asyncio.Lock()

    async def reload_from(self, directory: str | Path) -> int:
        async with self._lock:
            await self._close_locked()
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
        # Fast path: an already-built runner needs no lock, keeping the
        # per-request hot path free of contention.
        cached = self._runners.get(name)
        if cached is not None:
            return cached
        spec = self.get(name)
        async with self._lock:
            # Re-check: another task may have built it while we waited.
            cached = self._runners.get(name)
            if cached is not None:
                return cached
            from inferops.adapters.transport import get_transport
            runner = PipelineRunner(spec, registry=get_registry(), transport=get_transport())
            try:
                await runner.setup()
            except Exception:
                # Never cache a half-built runner, and release whatever its
                # stages already acquired before setup() raised.
                with suppress(Exception):
                    await runner.teardown()
                raise
            self._runners[name] = runner
            return runner

    async def _close_locked(self) -> None:
        for r in self._runners.values():
            # One failing teardown must not strand the remaining runners.
            with suppress(Exception):
                await r.teardown()
        self._runners.clear()

    async def close(self) -> None:
        async with self._lock:
            await self._close_locked()


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
