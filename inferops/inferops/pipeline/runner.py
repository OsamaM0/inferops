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
        # Strong references to in-flight background sink tasks. asyncio keeps
        # only a weak reference, so an unreferenced task can be garbage
        # collected before it finishes writing.
        self._sink_tasks: set[asyncio.Task] = set()

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

        # Auto-inject tracking stage right after the first `infer` stage if
        # the pipeline declares a `tracking:` block with `enabled: true` and
        # the user didn't already place a `track` stage explicitly.
        tracking = self.spec.tracking
        if tracking and tracking.enabled and not any(d.stage == "track" for d in ordered_decls):
            anchor_name = tracking.insert_after or next(
                (d.name for d in ordered_decls if d.stage == "infer"), ""
            )
            if anchor_name:
                from inferops.pipeline.spec import StageDecl
                track_decl = StageDecl(
                    name="auto_track",
                    stage="track",
                    params={
                        "algorithm": tracking.algorithm,
                        "by": tracking.by,
                        "max_age": tracking.max_age,
                        "min_hits": tracking.min_hits,
                        "iou_threshold": tracking.iou_threshold,
                        "low_threshold": tracking.low_threshold,
                        "score_high": tracking.score_high,
                        "score_low": tracking.score_low,
                        "per_class": tracking.per_class,
                        "model_filter": list(tracking.model_filter),
                    },
                )
                idx = next(i for i, d in enumerate(ordered_decls) if d.name == anchor_name)
                ordered_decls.insert(idx + 1, track_decl)
                log.info("tracking_auto_injected", pipeline=self.spec.name,
                         after=anchor_name, algorithm=tracking.algorithm)

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
        # Let background sink writes finish before closing the stages they write
        # through, otherwise shutdown races them and loses the last evidence.
        if self._sink_tasks:
            await asyncio.gather(*list(self._sink_tasks), return_exceptions=True)
            self._sink_tasks.clear()
        # finalize() is the documented "all items are done" hook; this is the
        # only place it belongs for request-driven pipelines, and it is what
        # flushes SinkS3's buffered clip.
        with suppress(Exception):
            await self._finalize_sinks()
        for s in [*([self._source] if self._source else []), *self._stages, *self._sinks]:
            with suppress(Exception):
                await s.teardown()
        # Drop the stage instances and clear the flag so a torn-down runner is
        # rebuilt cleanly instead of silently reusing closed resources or
        # accumulating a second copy of every stage on the next setup().
        self._source = None
        self._stages = []
        self._sinks = []
        self._setup_done = False

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

    async def _run_sinks_bg(self, items: list[StageItem]) -> None:
        """Background wrapper: run sinks without blocking the HTTP response.

        Deliberately does NOT call ``_finalize_sinks()``. ``finalize()`` is
        documented as "called once after all items are processed"; calling it
        per request made SinkS3 flush and clear its pre/post-roll buffer after
        every frame, so evidence clips were never assembled and concurrent
        requests wiped each other's buffers.
        """
        try:
            await self._fan_sinks(items)
        except Exception:
            log.exception("sink_background_failed", pipeline=self.spec.name)

    async def run_one(self, item: StageItem) -> list[StageItem]:
        if not self._setup_done:
            await self.setup()
        t0 = time.perf_counter()
        try:
            outs = await self._through_stages(item)
            # Evidence/sink writes run in the background so the HTTP response is
            # returned immediately. asyncio only holds a weak reference to a
            # task, so without keeping one here the GC could collect it
            # mid-flight and silently drop the evidence and database writes.
            task = asyncio.create_task(self._run_sinks_bg(outs))
            self._sink_tasks.add(task)
            task.add_done_callback(self._sink_tasks.discard)
            PIPELINE_RUNS.labels(pipeline=self.spec.name, result="ok").inc()
            if outs:
                stage_ms = {k: v for k, v in outs[0].trace.items() if k.startswith("stage_ms.")}
                log.info("pipeline_stage_timings", pipeline=self.spec.name, **stage_ms)
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
