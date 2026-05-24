"""Pipeline routes + /infer."""
from __future__ import annotations

import asyncio
import os
import tempfile
from contextlib import suppress
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from inferops.adapters.postgres_dao import create_job, get_job, list_jobs, update_job
from inferops.api.deps import pipelines
from inferops.core.config import get_spec_dir, settings
from inferops.core.errors import RegistryError
from inferops.core.types import StageItem, new_id, now_ms

router = APIRouter()
_background_tasks: set[asyncio.Task] = set()

PipelinesDep = Depends(pipelines)
OptionalImageFile = File(None)
RequiredVideoFile = File(...)
ApiStreamId = Form("api")
UploadStreamId = Form("frontend-upload")


def _track_background_task(task: asyncio.Task) -> None:
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


def _summarize_run(pipeline: str, runner, outs: list[StageItem]) -> dict:
    alerts, events = [], []
    det_counts: dict[str, int] = {}
    evidence_uris: list[str] = []
    for it in outs:
        alerts.extend(it.payload.get("alerts", []))
        events.extend(it.payload.get("fusion_events", []))
        evidence_uris.extend(it.payload.get("evidence_uris", []))
        for r in it.results:
            for d in (r.detections or []):
                det_counts[d.label] = det_counts.get(d.label, 0) + 1
    spec = runner.spec
    stages = ([spec.source.name + " (" + spec.source.stage + ")"] if spec.source else [])
    stages += [f"{s.name} ({s.stage})" for s in spec.stages]
    stages += [f"{s.name} ({s.stage})" for s in spec.sinks]
    return {
        "pipeline": pipeline,
        "pipeline_shape": " -> ".join(stages),
        "frames": len(outs),
        "detections_by_class": det_counts,
        "fusion_events": events,
        "alerts": alerts,
        "evidence_uris": evidence_uris,
    }


async def _run_job(job_id: str, pipeline: str, payload: dict, store) -> None:
    try:
        await update_job(job_id, status="running")
        runner = await store.get_runner(pipeline)
        item = StageItem(stream_id=payload.pop("stream_id", job_id), ts_ms=now_ms(), payload=payload)
        outs = await runner.run_one(item)
        await update_job(job_id, status="succeeded", result=_summarize_run(pipeline, runner, outs))
    except Exception as exc:
        await update_job(job_id, status="failed", result={"error": str(exc)})


@router.get("/pipelines")
async def list_pipelines(store = PipelinesDep):
    return [p.model_dump(mode="json") for p in store.list()]


@router.get("/pipelines/{name}")
async def get_pipeline(name: str, store = PipelinesDep):
    try:
        return store.get(name).model_dump(mode="json")
    except RegistryError as e:
        raise HTTPException(404, str(e)) from e


@router.post("/pipelines/reload")
async def reload_pipelines(store = PipelinesDep):
    d = get_spec_dir("pipelines", settings)
    n = await store.reload_from(d) if d.exists() else 0
    return {"reloaded": n, "dir": str(d)}


@router.post("/jobs/{pipeline}")
async def start_job(pipeline: str, payload: dict | None = None, store = PipelinesDep):
    try:
        store.get(pipeline)
    except RegistryError as e:
        raise HTTPException(404, str(e)) from e
    job_id = new_id()
    await create_job({"id": job_id, "pipeline": pipeline, "status": "queued", "result": None})
    _track_background_task(asyncio.create_task(_run_job(job_id, pipeline, dict(payload or {}), store)))
    return {"job_id": job_id, "pipeline": pipeline, "status": "queued"}


@router.get("/jobs")
async def get_jobs(limit: int = 100):
    return await list_jobs(limit)


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    row = await get_job(job_id)
    if row is None:
        raise HTTPException(404, "job not found")
    return row


@router.post("/infer/{pipeline}")
async def infer(pipeline: str,
                image: UploadFile | None = OptionalImageFile,
                stream_id: str = ApiStreamId,
                model_override: str | None = Form(None),
                store = PipelinesDep):
    try:
        runner = await store.get_runner(pipeline)
    except RegistryError as e:
        raise HTTPException(404, str(e)) from e
    item = StageItem(stream_id=stream_id, ts_ms=now_ms())
    if image is not None:
        item.payload["image_bytes"] = await image.read()
    if model_override:
        item.payload["model_override"] = model_override
    outs = await runner.run_one(item)
    # Drop binary payload fields so JSON serialisation works.
    def _slim(o):
        d = o.model_dump(mode="json", exclude={"payload"})
        p = {k: v for k, v in o.payload.items()
             if k not in {"image_bytes", "image_np", "feeds", "frame_meta"}}
        d["payload"] = p
        return d
    return [_slim(o) for o in outs]


@router.post("/run/{pipeline}")
async def run_pipeline(pipeline: str, payload: dict | None = None,
                       store = PipelinesDep):
    """Run a pipeline once with a JSON payload (e.g. {'video_path': '/...mp4'})."""
    try:
        runner = await store.get_runner(pipeline)
    except RegistryError as e:
        raise HTTPException(404, str(e)) from e
    payload = payload or {}
    item = StageItem(stream_id=payload.pop("stream_id", "api"), ts_ms=now_ms(),
                     payload=payload)
    outs = await runner.run_one(item)
    return _summarize_run(pipeline, runner, outs)


@router.post("/run/{pipeline}/upload")
async def run_pipeline_upload(
    pipeline: str,
    video: UploadFile = RequiredVideoFile,
    stream_id: str = UploadStreamId,
    store = PipelinesDep,
):
    """Upload a video file, run it through a pipeline once, and return a summary."""
    try:
        runner = await store.get_runner(pipeline)
    except RegistryError as e:
        raise HTTPException(404, str(e)) from e

    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            while chunk := await video.read(1024 * 1024):
                tmp.write(chunk)

        item = StageItem(
            stream_id=stream_id or "frontend-upload",
            ts_ms=now_ms(),
            payload={"video_path": tmp_path},
        )
        outs = await runner.run_one(item)
        summary = _summarize_run(pipeline, runner, outs)
        summary["stream_id"] = item.stream_id
        summary["uploaded_filename"] = video.filename
        return summary
    finally:
        if tmp_path:
            with suppress(OSError):
                os.unlink(tmp_path)
