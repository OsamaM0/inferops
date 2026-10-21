"""Camera/source registry, cached annotated previews, history and capacity API."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response

from inferops.adapters.outbound.observability.capacity_profiles import (
    model_fingerprints,
    service_profile,
    software_fingerprint,
    workload_fingerprint,
)
from inferops.domain.capacity import assess_capacity
from inferops.domain.sources import SourceSpec

router = APIRouter()
Upload = File(...)


def build_source_manager(application, *, store=None, root=None):
    from inferops.adapters.inbound.registered_sources import registered_frames
    from inferops.adapters.outbound.media.mediamtx import configured_relay
    from inferops.adapters.outbound.media.previews import render_preview
    from inferops.adapters.outbound.persistence.source_catalog import MemorySourceStore, PostgresSourceStore
    from inferops.adapters.outbound.persistence.sql import SqlRepository
    from inferops.application.incidents import wire_result
    from inferops.application.sources import SourceManager

    root = Path(root or os.environ.get("INFEROPS_SOURCES_PATH", ".run/sources"))
    scope = getattr(application.repository, "scope", None)
    if scope:
        root = root / scope
    root.mkdir(parents=True, exist_ok=True)

    async def render(*args):
        return await asyncio.to_thread(render_preview, *args)

    manager = SourceManager(
        application,
        store
        or (PostgresSourceStore(
            application.repository, preview_root=root / "previews",
            results_per_camera=int(os.environ.get("INFEROPS_SOURCE_RESULT_HISTORY", "1000")),
        ) if isinstance(application.repository, SqlRepository) else MemorySourceStore()),
        frames=registered_frames,
        render=render,
        serialize=wire_result,
        max_active=int(os.environ.get("INFEROPS_MAX_ACTIVE_SOURCES", "8")),
        relay=configured_relay(scope=scope or "combined"),
    )
    manager.root = root
    return manager


def manager(request):
    return request.app.state.sources


async def forward_owned(request, kind, record_id, *, body=None, worker_id=None):
    if getattr(request.app.state, "role", "combined") != "api":
        return None
    controls = request.app.state.worker_controls
    repository = request.app.state.engine.application.repository
    scope = await repository.record_scope(kind, record_id)
    if scope and scope != "api":
        owner_id = scope.removeprefix("camera-")
        if worker_id and worker_id != owner_id:
            raise HTTPException(409, "camera belongs to another worker; explicit reassignment is required")
        worker_id = owner_id
    elif scope == "api" and worker_id:
        raise HTTPException(409, "camera belongs to the API owner")
    if worker_id:
        control = controls.get(worker_id)
        if control is None:
            raise HTTPException(503, "owning worker is not in the active control manifest")
        params = [(key, value) for key, value in request.query_params.multi_items() if key != "worker_id"]
        return await control.forward(request.method, request.url.path, body=body, params=params)
    return None


@router.get("/workers")
async def workers(request: Request):
    controls = getattr(request.app.state, "worker_controls", {})
    return {"workers": [{"worker_id": worker_id, "source_limit": control.spec.max_active_sources}
                        for worker_id, control in sorted(controls.items())]}


@router.get("/cameras")
async def list_cameras(request: Request):
    selected = manager(request)
    controls = getattr(request.app.state, "worker_controls", {})
    statuses = await asyncio.gather(*(control.status() for control in controls.values()))
    remote = [{**camera, "worker_id": status["worker_id"]}
              for status in statuses for camera in status["cameras"]]
    return {
        "cameras": [selected.snapshot(key) for key in sorted(selected.specs)] + remote,
        "active": len(selected.tasks) + sum(status["active"] for status in statuses),
        "active_limit": selected.max_active + sum(control.spec.max_active_sources for control in controls.values()),
        "workers": [{key: value for key, value in status.items() if key != "cameras"} for status in statuses],
        "ownership_error": getattr(request.app.state.engine.application, "ownership_error", None),
        "media_router": {"enabled": selected.relay is not None, "error": selected.relay_error},
    }


@router.post("/cameras", status_code=201)
async def add_camera(request: Request, spec: SourceSpec, worker_id: str | None = None):
    forwarded = await forward_owned(request, "camera", spec.camera_id, body=spec.model_dump(mode="json"), worker_id=worker_id)
    if forwarded is not None:
        return forwarded
    return await manager(request).put(spec, create=True)


@router.put("/cameras/{camera_id}")
async def edit_camera(request: Request, camera_id: str, payload: dict):
    forwarded = await forward_owned(request, "camera", camera_id, body=payload)
    if forwarded is not None:
        return forwarded
    selected = manager(request)
    existing = selected.specs.get(camera_id)
    if existing is None:
        raise HTTPException(404, "source not found")
    if payload.get("camera_id", camera_id) != camera_id:
        raise HTTPException(400, "camera_id is immutable")
    values = {**existing.model_dump(), **payload, "camera_id": camera_id}
    values["uri"] = payload.get("uri") or existing.uri
    return await selected.put(SourceSpec.model_validate(values))


@router.post("/cameras/{camera_id}/start")
async def start_camera(request: Request, camera_id: str):
    forwarded = await forward_owned(request, "camera", camera_id)
    if forwarded is not None:
        return forwarded
    return await manager(request).run(camera_id)


@router.post("/cameras/{camera_id}/stop")
async def stop_camera(request: Request, camera_id: str):
    forwarded = await forward_owned(request, "camera", camera_id)
    if forwarded is not None:
        return forwarded
    return await manager(request).stop(camera_id)


@router.delete("/cameras/{camera_id}")
async def remove_camera(request: Request, camera_id: str):
    forwarded = await forward_owned(request, "camera", camera_id)
    if forwarded is not None:
        return forwarded
    await manager(request).remove(camera_id)
    return {"camera_id": camera_id, "status": "removed", "history_preserved": True}


@router.get("/cameras/{camera_id}/preview.jpg")
async def preview(request: Request, camera_id: str):
    forwarded = await forward_owned(request, "camera", camera_id)
    if forwarded is not None:
        return forwarded
    jpeg = manager(request).previews.get(camera_id)
    if jpeg is None:
        raise HTTPException(404, "no processed frame yet")
    return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/cameras/{camera_id}/runs")
async def camera_runs(request: Request, camera_id: str, limit: int = Query(50, ge=1, le=500)):
    forwarded = await forward_owned(request, "camera", camera_id)
    if forwarded is not None:
        return forwarded
    return await manager(request).store.list_runs(camera_id, limit)


def live_urls(request, path):
    """Use browser-facing endpoints, including when the owner is a remote worker."""
    from urllib.parse import quote, urlsplit

    result = {"path": path, "annotations": False}
    for protocol, default_port in (("webrtc", 8889), ("hls", 8888)):
        prefix = f"INFEROPS_MEDIAMTX_{protocol.upper()}"
        base = os.getenv(f"{prefix}_PUBLIC_URL", "").strip()
        if base:
            parsed = urlsplit(base)
            if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
                    or parsed.password or parsed.query or parsed.fragment):
                raise HTTPException(503, f"{prefix}_PUBLIC_URL must be an HTTP(S) URL without credentials/query")
        else:
            base = str(request.url.replace(port=int(os.getenv(f"{prefix}_PORT", str(default_port))),
                                           path="/", query=""))
        result[f"{protocol}_url"] = (f"{base.rstrip('/')}/{quote(path, safe='/')}"
                                      "?muted=true&autoplay=true&playsInline=true")
    return result


@router.get("/cameras/{camera_id}/live")
async def live_video(request: Request, camera_id: str):
    forwarded = await forward_owned(request, "camera", camera_id)
    if forwarded is not None:
        if forwarded.status_code != 200:
            return forwarded
        # The worker's HTTP hostname is internal; build public URLs at the API.
        return live_urls(request, json.loads(forwarded.body)["path"])
    selected = manager(request)
    async with selected._lock:
        spec = selected.specs[camera_id]
        if spec.kind != "rtsp" or selected.relay is None:
            raise HTTPException(409, "Live video requires an RTSP source and MediaMTX routing")
        await selected.relay.prepare(spec)
        return live_urls(request, selected.relay.path(camera_id))


def recording_urls(request, path, rows):
    from urllib.parse import urlencode, urlsplit

    base = os.getenv("INFEROPS_MEDIAMTX_PLAYBACK_PUBLIC_URL", "").strip()
    if base:
        parsed = urlsplit(base)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise HTTPException(503, "INFEROPS_MEDIAMTX_PLAYBACK_PUBLIC_URL must be an HTTP(S) URL without credentials/query")
    else:
        base = str(request.url.replace(port=int(os.getenv("INFEROPS_MEDIAMTX_PLAYBACK_PORT", "9996")),
                                       path="/", query=""))
    return {"path": path, "annotations": False, "recordings": [
        {"start": row["start"], "duration": row["duration"],
         "url": f"{base.rstrip('/')}/get?" + urlencode({"path": path, "start": row["start"],
                                                       "duration": row["duration"], "format": "mp4"})}
        for row in rows]}


@router.get("/cameras/{camera_id}/recordings")
async def camera_recordings(request: Request, camera_id: str, start: datetime | None = None,
                            end: datetime | None = None, limit: int = Query(100, ge=1, le=1000)):
    if any(value is not None and value.tzinfo is None for value in (start, end)):
        raise HTTPException(422, "recording times must include a timezone offset")
    if start and end and start >= end:
        raise HTTPException(422, "start must be before end")
    forwarded = await forward_owned(request, "camera", camera_id)
    if forwarded is not None:
        if forwarded.status_code != 200:
            return forwarded
        payload = json.loads(forwarded.body)
        return recording_urls(request, payload["path"], payload["recordings"])
    selected = manager(request)
    spec = selected.specs.get(camera_id)
    if spec is None:
        raise HTTPException(404, "unknown camera")
    if spec.kind != "rtsp" or selected.relay is None:
        raise HTTPException(409, "Recordings require an RTSP source and MediaMTX routing")
    rows = await selected.relay.recordings(camera_id, start=start.isoformat() if start else None,
                                          end=end.isoformat() if end else None)
    return recording_urls(request, selected.relay.path(camera_id), rows[-limit:])


@router.get("/cameras/{camera_id}/results")
async def camera_results(request: Request, camera_id: str, limit: int = Query(50, ge=1, le=500)):
    forwarded = await forward_owned(request, "camera", camera_id)
    if forwarded is not None:
        return forwarded
    return await manager(request).store.list_results(camera_id, limit)


@router.post("/cameras/media", status_code=201)
async def upload_media(request: Request, file: UploadFile = Upload):
    suffix = Path(file.filename or "").suffix.lower()
    images = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if suffix not in images | {".mp4", ".mkv", ".avi", ".mov", ".webm"}:
        raise HTTPException(400, "unsupported image/video extension")
    root = manager(request).root / "uploads"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{uuid.uuid4().hex}{suffix}"
    limit = int(os.environ.get("INFEROPS_SOURCE_UPLOAD_MAX_BYTES", str(512 * 1024**2)))
    size = 0
    try:
        with path.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, "upload exceeds configured limit")
                await asyncio.to_thread(out.write, chunk)

        def verify():
            if suffix in images:
                import cv2

                if cv2.imread(str(path)) is None:
                    raise ValueError("invalid image")
            else:
                import av

                with av.open(str(path)) as container:
                    if next(container.decode(video=0), None) is None:
                        raise ValueError("video contains no decodable frames")

        await asyncio.to_thread(verify)
    except asyncio.CancelledError:
        path.unlink(missing_ok=True)
        raise
    except Exception as error:
        path.unlink(missing_ok=True)
        if isinstance(error, HTTPException):
            raise
        raise HTTPException(400, "uploaded media is not decodable") from error
    return {"uri": str(path.resolve()), "kind": "image" if suffix in images else "video", "bytes": size}


@router.get("/system/hardware")
async def hardware(request: Request):
    from inferops.adapters.outbound.observability.hardware import hardware_snapshot

    selected = manager(request)
    cache = getattr(request.app.state, "hardware_cache", None)
    if cache is None or time.monotonic() - cache[0] > 3:
        value = await asyncio.to_thread(hardware_snapshot, selected.root)
        if request.app.state.engine.application.definition.mode in {"triton"}:
            from urllib.parse import urlsplit, urlunsplit

            from inferops.adapters.outbound.observability.triton_metrics import inference_gpus
            from inferops.adapters.outbound.projects.settings import settings
            target = urlsplit(settings.triton.http_url)
            default_metrics = urlunsplit((target.scheme, f"{target.hostname}:8002", "", "", ""))
            gpus = await inference_gpus(os.getenv("INFEROPS_TRITON_METRICS_URL", default_metrics))
            value.update(gpus=gpus, available=bool(gpus), gpu_scope="inference_server")
        request.app.state.hardware_cache = (time.monotonic(), value)
    return request.app.state.hardware_cache[1]


@router.get("/system/capacity")
async def capacity(
    request: Request,
    pipeline: str,
    width: int = Query(1920, ge=64, le=7680),
    height: int = Query(1080, ge=64, le=4320),
    fps: float = Query(5, gt=0, le=120),
    evidence_fps: float | None = Query(None, gt=0, le=120),
):
    application = request.app.state.engine.application

    def read():
        reports = []
        root = Path(os.environ.get("INFEROPS_CAPACITY_REPORTS", ".run/capacity"))
        for path in sorted(root.glob("*/report.json"))[-50:]:
            try:
                value = json.loads(path.read_text())
                reports.append({**value, "report_id": path.parent.name})
            except (OSError, ValueError):
                continue
        return (
            reports,
            model_fingerprints(application),
            workload_fingerprint(application),
            software_fingerprint(),
        )

    reports, artifacts, workload, software = await asyncio.to_thread(read)
    if getattr(request.app.state, "worker_controls", {}):
        reports = []  # Single-owner capacity profiles cannot qualify a distributed layout.
    info = await hardware(request)
    assessment = assess_capacity(
        reports,
        info,
        pipeline=pipeline,
        runtime=application.definition.mode,
        model_artifacts=artifacts,
        width=width,
        height=height,
        fps=fps,
        evidence_fps=evidence_fps,
        evidence=bool(application.evidence),
        workload_fingerprint=workload,
        software_fingerprint=software,
        service_profile=service_profile(application, media_router="mediamtx" if manager(request).relay else "direct",
                                        media_recording=bool(getattr(manager(request).relay, "record", False))),
        evidence_policy=json.loads(json.dumps(asdict(application.evidence.config)))
        if application.evidence
        else None,
    )
    return {
        **assessment,
        "runtime": application.definition.mode,
        "pipeline": pipeline,
        "width": width,
        "height": height,
        "fps": fps,
        "evidence_fps": evidence_fps or fps,
        "source_sample_fps": max(fps, evidence_fps or fps),
        "registered": len(manager(request).specs),
        "active": len(manager(request).tasks),
        "configured_active_limit": manager(request).max_active,
        "evidence_enabled": bool(application.evidence),
        "hardware": info,
    }
