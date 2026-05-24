"""Async DAG runner.

Supported topologies:
- Linear pipeline (source -> stage1 -> ... -> stageN -> sinks)  — implemented here.
- Item-by-item processing with bounded concurrency.
- Sinks receive a fanout of every emitted item from the last stage.

The runner is reusable in two modes:
- One-shot: `run_one(item)` for /v1/infer/{pipeline}
- Long-running: `start()` then iterate the source until cancelled.
"""
from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from typing import Any

from inferops.core.errors import PipelineError
from inferops.core.types import StageItem
from inferops.obs import ERRORS, PIPELINE_RUNS, get_logger
from inferops.pipeline.graph import linear_order
from inferops.pipeline.registry import resolve_stage
from inferops.pipeline.spec import PipelineSpec
from inferops.pipeline.stage import SinkStage, SourceStage, Stage, StageContext

log = get_logger(__name__)


def _instantiate(decl) -> Stage:
    cls = resolve_stage(decl.stage)
    return cls(**(decl.params or {}))


class PipelineRunner:
    def __init__(self, spec: PipelineSpec, *, registry: Any = None, transport: Any = None) -> None:
        self.spec = spec
        self._registry = registry
        self._transport = transport
        self._source: SourceStage | None = None
        self._stages: list[Stage] = []
        self._sinks: list[SinkStage] = []
        self._setup_done = False

    async def setup(self) -> None:
        if self._setup_done:
            return
        order_names = linear_order(self.spec) if (self.spec.source or self.spec.stages) else []
        decls = {}
        if self.spec.source:
            decls[self.spec.source.name] = self.spec.source
        for s in self.spec.stages:
            decls[s.name] = s
        ordered_decls = [decls[n] for n in order_names]

        for decl in ordered_decls:
            stage = _instantiate(decl)
            ctx = StageContext(pipeline=self.spec.name, name=decl.name, params=decl.params or {},
                               registry=self._registry, transport=self._transport)
            await stage.setup(ctx)
            if isinstance(stage, SourceStage):
                self._source = stage
            else:
                self._stages.append(stage)

        for decl in self.spec.sinks:
            sink = _instantiate(decl)
            ctx = StageContext(pipeline=self.spec.name, name=decl.name, params=decl.params or {},
                               registry=self._registry, transport=self._transport)
            await sink.setup(ctx)
            if not isinstance(sink, SinkStage):
                # Allow non-sink-typed stages declared in sinks list
                pass
            self._sinks.append(sink)  # type: ignore[arg-type]

        self._setup_done = True

    async def teardown(self) -> None:
        for s in [*([self._source] if self._source else []), *self._stages, *self._sinks]:
            with suppress(Exception):
                await s.teardown()

    async def _through_stages(self, item: StageItem) -> list[StageItem]:
        current = [item]
        for stage in self._stages:
            next_items: list[StageItem] = []
            for it in current:
                out = await stage(it)
                if out is None:
                    continue
                if isinstance(out, list):
                    next_items.extend(out)
                else:
                    next_items.append(out)
            current = next_items
            if not current:
                break
        return current

    async def _fan_sinks(self, items: list[StageItem]) -> None:
        if not items or not self._sinks:
            return
        await asyncio.gather(*[s(it) for s in self._sinks for it in items], return_exceptions=False)

    async def _finalize_sinks(self) -> None:
        if self._sinks:
            await asyncio.gather(*[s.finalize() for s in self._sinks], return_exceptions=True)

    async def run_one(self, item: StageItem) -> list[StageItem]:
        if not self._setup_done:
            await self.setup()
        t0 = time.perf_counter()
        try:
            outs = await self._through_stages(item)
            await self._fan_sinks(outs)
            await self._finalize_sinks()
            PIPELINE_RUNS.labels(pipeline=self.spec.name, result="ok").inc()
            return outs
        except Exception as e:
            ERRORS.labels(stage=self.spec.name, kind="runner").inc()
            PIPELINE_RUNS.labels(pipeline=self.spec.name, result="error").inc()
            log.exception("pipeline_run_failed", pipeline=self.spec.name, err=str(e))
            raise PipelineError(str(e)) from e
        finally:
            log.debug("pipeline_run_done", pipeline=self.spec.name,
                      ms=(time.perf_counter() - t0) * 1000)

    async def start(self, max_concurrency: int | None = None) -> None:
        if self._source is None:
            raise PipelineError("pipeline has no source; cannot start()")
        if not self._setup_done:
            await self.setup()
        max_c = max_concurrency or self.spec.limits.max_concurrency
        sem = asyncio.Semaphore(max_c)

        async def _one(it: StageItem) -> None:
            async with sem:
                try:
                    outs = await self._through_stages(it)
                    await self._fan_sinks(outs)
                except Exception:
                    ERRORS.labels(stage=self.spec.name, kind="streaming").inc()
                    log.exception("streaming_item_failed")

        tasks: set[asyncio.Task] = set()
        async for item in self._source.run():
            # Drop incoming frames when all worker slots are occupied.
            # This keeps end-to-end latency bounded for real-time sources
            # (RTSP, MJPEG) instead of building an unbounded in-memory queue.
            if len(tasks) >= max_c:
                log.debug("frame_dropped", stream_id=item.stream_id,
                          seq=item.seq, active_tasks=len(tasks))
                continue
            t = asyncio.create_task(_one(item))
            tasks.add(t)
            t.add_done_callback(tasks.discard)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self._finalize_sinks()
