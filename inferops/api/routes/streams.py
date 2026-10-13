"""Streams: start/stop long-running pipelines (RTSP, MJPEG, or any source)."""
from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, Depends, HTTPException

from inferops.adapters.transport import get_transport
from inferops.api.deps import pipelines
from inferops.core.errors import RegistryError
from inferops.obs import get_logger
from inferops.pipeline.runner import PipelineRunner
from inferops.pipeline.spec import PipelineSpec, StageDecl
from inferops.registry import get_registry

log = get_logger(__name__)

router = APIRouter()
PipelinesDep = Depends(pipelines)

_RUNNING: dict[str, asyncio.Task] = {}
_META: dict[str, dict] = {}
# The runner is set up in the request handler but torn down inside the task.
# If the task is cancelled before it ever runs, that finally never executes, so
# stop_stream needs its own handle to release the runner. teardown() is
# idempotent, so calling it from both places is safe.
_RUNNERS: dict[str, PipelineRunner] = {}


def _build_live_runner(base_spec: PipelineSpec, source_stage: str | None,
                       source_params: dict | None) -> PipelineRunner:
    """Return a fresh PipelineRunner, optionally with the source swapped.

    Always a *dedicated* runner. A stream owns its runner for its whole life and
    tears it down when it ends, so it must never be handed the shared cached
    instance from the pipeline store — doing that tore down the runner that
    /v1/infer and /v1/live keep using, poisoning them for the rest of the
    process.
    """
    spec = base_spec.model_copy(deep=True)
    if source_stage is not None:
        # decode_any is a pass-through when image_np is already set (live
        # sources produce image_np directly), so no other stage changes are
        # needed.
        spec.source = StageDecl(name="src", stage=source_stage, params=source_params or {})
    return PipelineRunner(spec, registry=get_registry(), transport=get_transport())


def _reap_finished() -> None:
    """Drop completed streams and surface any exception they died with.

    Finished tasks were never evicted, so /v1/streams grew forever and a task
    that crashed had its exception retrieved by nobody.
    """
    for sid in [s for s, t in _RUNNING.items() if t.done()]:
        task = _RUNNING.pop(sid, None)
        _META.pop(sid, None)
        _RUNNERS.pop(sid, None)
        if task is None or task.cancelled():
            continue
        exc = task.exception()
        if exc is not None:
            log.warning("stream_task_failed", stream_id=sid, err=str(exc))


@router.post("/streams")
async def start_stream(
    pipeline: str,
    rtsp_url: str | None = None,
    mjpeg_url: str | None = None,
    stream_id: str | None = None,
    fps: float = 5.0,
    store=PipelinesDep,
):
    """Start a long-running pipeline stream.

    - No extra params → use the pipeline's configured source (e.g. Kafka).
    - ``rtsp_url``  → replace source with SourceRTSP for a live camera feed.
    - ``mjpeg_url`` → replace source with SourceMJPEG for an HTTP MJPEG feed.
    """
    try:
        base_spec = store.get(pipeline)
    except RegistryError as e:
        raise HTTPException(404, str(e)) from e

    _reap_finished()
    sid = stream_id or uuid.uuid4().hex

    # Reusing a live stream_id used to overwrite the entry, orphaning a task
    # that could then never be stopped.
    existing = _RUNNING.get(sid)
    if existing is not None and not existing.done():
        raise HTTPException(409, f"stream {sid!r} is already running")

    if rtsp_url:
        runner = _build_live_runner(
            base_spec, "source_rtsp",
            {"url": rtsp_url, "stream_id": sid, "fps": fps},
        )
        input_desc = rtsp_url
    elif mjpeg_url:
        runner = _build_live_runner(
            base_spec, "source_mjpeg",
            {"url": mjpeg_url, "stream_id": sid, "fps": fps},
        )
        input_desc = mjpeg_url
    else:
        # Dedicated runner using the pipeline's own configured source.
        runner = _build_live_runner(base_spec, None, None)
        input_desc = "pipeline default source"

    await runner.setup()

    async def _run():
        try:
            await runner.start()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("stream_failed", stream_id=sid, pipeline=pipeline)
            raise
        finally:
            await runner.teardown()

    _RUNNING[sid] = asyncio.create_task(_run())
    _RUNNERS[sid] = runner
    _META[sid] = {"pipeline": pipeline, "input": input_desc, "fps": fps}
    return {"stream_id": sid, "pipeline": pipeline, "input": input_desc, "status": "started"}


@router.get("/streams")
async def list_streams():
    _reap_finished()
    return [
        {"stream_id": k, "done": t.done(), **_META.get(k, {})}
        for k, t in _RUNNING.items()
    ]


@router.delete("/streams/{stream_id}")
async def stop_stream(stream_id: str):
    t = _RUNNING.pop(stream_id, None)
    _META.pop(stream_id, None)
    runner = _RUNNERS.pop(stream_id, None)
    if t is None:
        raise HTTPException(404, "stream not found")
    t.cancel()
    # Previously this returned "stopped" immediately, so the caller could
    # observe a stream that was still holding its capture device and sinks.
    # asyncio.wait() waits for the task to finish unwinding — including the
    # teardown in its finally — without re-raising the cancellation the way
    # awaiting the task directly (or through shield) would.
    done, _pending = await asyncio.wait({t}, timeout=10.0)
    if not done:
        log.warning("stream_stop_timeout", stream_id=stream_id)
    elif not t.cancelled() and t.exception() is not None:
        log.warning("stream_stop_error", stream_id=stream_id, err=str(t.exception()))
    # Covers the case where the task was cancelled before it ever ran, so its
    # finally never fired and the runner set up in start_stream would leak.
    if runner is not None:
        try:
            await runner.teardown()
        except Exception as exc:
            log.warning("stream_teardown_failed", stream_id=stream_id, err=str(exc))
    return {"stream_id": stream_id, "status": "stopped", "done": t.done()}

