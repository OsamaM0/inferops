"""Streams: start/stop long-running pipelines (RTSP, MJPEG, or any source)."""
from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, Depends, HTTPException

from inferops.adapters.transport import get_transport
from inferops.api.deps import pipelines
from inferops.core.errors import RegistryError
from inferops.pipeline.runner import PipelineRunner
from inferops.pipeline.spec import PipelineSpec, StageDecl
from inferops.registry import get_registry

router = APIRouter()
PipelinesDep = Depends(pipelines)

_RUNNING: dict[str, asyncio.Task] = {}
_META: dict[str, dict] = {}


def _build_live_runner(base_spec: PipelineSpec, source_stage: str,
                       source_params: dict) -> PipelineRunner:
    """Return a fresh PipelineRunner with the source swapped for a live source."""
    spec = base_spec.model_copy(deep=True)
    spec.source = StageDecl(name="src", stage=source_stage, params=source_params)
    # decode_any is a pass-through when image_np is already set (live sources
    # produce image_np directly), so no other stage changes are needed.
    return PipelineRunner(spec, registry=get_registry(), transport=get_transport())


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

    sid = stream_id or uuid.uuid4().hex

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
        runner = await store.get_runner(pipeline)
        input_desc = "pipeline default source"

    await runner.setup()

    async def _run():
        try:
            await runner.start()
        finally:
            await runner.teardown()

    _RUNNING[sid] = asyncio.create_task(_run())
    _META[sid] = {"pipeline": pipeline, "input": input_desc, "fps": fps}
    return {"stream_id": sid, "pipeline": pipeline, "input": input_desc, "status": "started"}


@router.get("/streams")
async def list_streams():
    return [
        {"stream_id": k, "done": t.done(), **_META.get(k, {})}
        for k, t in _RUNNING.items()
    ]


@router.delete("/streams/{stream_id}")
async def stop_stream(stream_id: str):
    t = _RUNNING.pop(stream_id, None)
    _META.pop(stream_id, None)
    if t is None:
        raise HTTPException(404, "stream not found")
    t.cancel()
    return {"stream_id": stream_id, "status": "stopped"}

