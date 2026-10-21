"""HTTP transport for the shared pipeline/session application services."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import (
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response

from inferops import __version__
from inferops.domain.errors import BackendError, ConfigError, PipelineError, RegistryError
from inferops.domain.processing import MediaFrame

log = logging.getLogger(__name__)
OptionalImageFile = File(None)
RequiredVideoFile = File(...)


def _byte_range(value: str, size: int) -> tuple[int, int]:
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value.strip())
    if not match or size == 0:
        raise ValueError("unsatisfiable byte range")
    start, end = match.groups()
    if not start:
        if not end or int(end) <= 0:
            raise ValueError("unsatisfiable byte range")
        return max(0, size - int(end)), size - 1
    first = int(start)
    last = min(int(end), size - 1) if end else size - 1
    if first >= size or last < first:
        raise ValueError("unsatisfiable byte range")
    return first, last


def _control_packet(value: str) -> dict:
    packet = json.loads(value)
    if not isinstance(packet, dict):
        raise ValueError("control packet must be a JSON object")
    for name in ("media_time_ns", "client_ts_ms", "seq"):
        item = packet.get(name)
        if item is not None and (type(item) is not int or item < 0):
            raise ValueError(f"{name} must be a nonnegative integer")
    return packet


def slim(item):
    result = item.model_dump(mode="json", exclude={"payload"})
    result["payload"] = {k: v for k, v in item.payload.items()
                         if k not in {"image_np", "image_bytes", "feeds", "frame_meta"} and not k.startswith("_")}
    result["session_id"] = item.payload.get("session_id")
    result["evidence"] = item.payload.get("evidence", [])
    return result


class RuntimeCORS:
    """Resolve server options after explicit startup, without activating globals."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            return await self.app(scope, receive, send)
        engine = getattr(scope["app"].state, "engine", None)
        origins = getattr(engine, "http_options", {}).get("cors_origins", ["*"])
        await CORSMiddleware(self.app, allow_origins=origins, allow_methods=["*"],
                             allow_headers=["*"])(scope, receive, send)


def create_application(engine=None, *, hub=None, source_store=None, source_root=None):
    streams = {}

    @asynccontextmanager
    async def lifespan(app):
        from inferops.application.jobs import JobService
        if engine is None:
            from inferops.bootstrap import build_server_engine
            selected, selected_hub = build_server_engine(hub=hub)
        else:
            from inferops.adapters.outbound.messaging.delivery import DeliveryHub, ServerDelivery
            selected, selected_hub = engine, hub or DeliveryHub()
            if (hasattr(selected.application, "delivery") and selected.application.delivery is None
                    and getattr(selected.application, "role", "combined") != "camera-worker"):
                selected.application.delivery = ServerDelivery(selected_hub, definitions=selected.application.definition,
                                                               broker_enabled=False)
                if hasattr(selected.application.repository, "outbox_transform"):
                    selected.application.repository.outbox_transform = selected.application.delivery.prepare
        app.state.engine, app.state.hub = selected, selected_hub
        jobs = None
        sources = None
        checkpoints = None
        controls = {}
        watchdog = None
        try:
            import cv2
            cv2.setNumThreads(int(os.environ.get("INFEROPS_OPENCV_THREADS", "1")))
            await selected.start()
            role = getattr(selected.application, "role", "combined")
            if role in {"api", "camera-worker"}:
                from inferops.adapters.inbound.workers.camera_worker import start_watchdog
                watchdog = start_watchdog(selected.application,
                    timeout_s=float(os.environ.get("INFEROPS_WORKER_WATCHDOG_SECONDS", "60")))
            app.state.role = role
            app.state.worker_controls = controls
            if role != "camera-worker":
                jobs = JobService(selected)
                app.state.jobs = jobs
                if hasattr(selected.application.repository, "interrupt_jobs"):
                    await jobs.start()
            from inferops.adapters.inbound.http.cameras import build_source_manager
            sources = build_source_manager(selected.application, store=source_store, root=source_root)
            app.state.sources = sources
            if role == "camera-worker" and os.environ.get("INFEROPS_WORKER_MANIFEST"):
                from inferops.adapters.outbound.persistence.worker_manifest import load_worker_manifest
                manifest = load_worker_manifest(os.environ["INFEROPS_WORKER_MANIFEST"])
                sources.max_active = next(w.max_active_sources for w in manifest.workers
                                          if w.worker_id == os.environ["INFEROPS_WORKER_ID"])
            await sources.start()
            manifest_path = os.environ.get("INFEROPS_WORKER_MANIFEST")
            if manifest_path and role != "combined":
                from inferops.adapters.outbound.persistence.worker_manifest import load_worker_manifest
                manifest = load_worker_manifest(manifest_path, control_only=role == "api")
                if role == "api":
                    from inferops.adapters.outbound.messaging.worker_control import HTTPWorkerControl
                    token = os.environ.get("INFEROPS_WORKER_TOKEN")
                    if not token:
                        raise ValueError("API worker routing requires INFEROPS_WORKER_TOKEN")
                    for worker in manifest.workers:
                        controls[worker.worker_id] = HTTPWorkerControl(worker, token)
                else:
                    worker_id = os.environ["INFEROPS_WORKER_ID"]
                    sources.max_active = next(w.max_active_sources for w in manifest.workers if w.worker_id == worker_id)
                    for assignment in manifest.cameras:
                        if assignment.worker_id == worker_id and assignment.source.camera_id not in sources.specs:
                            await sources.put(assignment.source)
                            if assignment.source.autostart:
                                await sources.run(assignment.source.camera_id)
            from inferops.adapters.outbound.inference.checkpoint_jobs import CheckpointJobs
            if role != "camera-worker":
                checkpoints = CheckpointJobs(sources.root / "models")
                app.state.checkpoints = checkpoints
                await checkpoints.start()
            yield
        finally:
            if watchdog:
                watchdog[0].set()
                await asyncio.to_thread(watchdog[1].join, 2)
            for control in controls.values():
                await control.close()
            if checkpoints:
                await checkpoints.close()
            source_error=None
            if sources:
                try:
                    await sources.close()
                except Exception as error:
                    source_error=error
            for task in list(streams.values()):
                task.cancel()
            await asyncio.gather(*list(streams.values()), return_exceptions=True)
            try:
                if jobs:
                    await jobs.close()
            finally:
                await selected.close()
            if source_error:
                raise RuntimeError("source history shutdown failed") from source_error

    app = FastAPI(title="InferOps", version=__version__, lifespan=lifespan)

    @app.middleware("http")
    async def internal_worker_security(request, call_next):
        import secrets
        if (getattr(request.app.state, "role", "combined") == "camera-worker"
                and request.url.path not in {"/healthz", "/readyz"}):
            actual = request.headers.get("x-worker-token", "")
            expected = os.environ.get("INFEROPS_WORKER_TOKEN", "")
            if not expected or not secrets.compare_digest(actual, expected):
                return JSONResponse(status_code=403, content={"detail": "worker control token required"})
            if not request.url.path.startswith(("/v1/cameras", "/v1/evidence/records", "/v1/retention", "/metrics")):
                return JSONResponse(status_code=403, content={"detail": "endpoint belongs to the API role"})
        return await call_next(request)
    from inferops.adapters.inbound.http.cameras import router as camera_router
    from inferops.adapters.inbound.http.configs import router as config_router
    from inferops.adapters.inbound.http.model_preparation import router as preparation_router
    app.include_router(preparation_router, prefix="/v1", tags=["model-preparation"])
    from inferops.adapters.inbound.http.retention import router as retention_router
    app.include_router(retention_router, prefix="/v1", tags=["retention"])
    app.include_router(config_router, prefix="/v1", tags=["configs"])
    app.include_router(camera_router, prefix="/v1", tags=["cameras"])
    app.add_middleware(RuntimeCORS)
    from inferops.ports.ownership import OwnershipLost
    for exception in (BackendError, PipelineError, OwnershipLost):
        async def unavailable(request, exc):
            return JSONResponse(status_code=503, content={"detail": str(exc)})
        app.add_exception_handler(exception, unavailable)
    for exception in (RegistryError, KeyError):
        async def missing(request, exc):
            return JSONResponse(status_code=404, content={"detail": str(exc)})
        app.add_exception_handler(exception, missing)
    for exception in (ValueError, ConfigError):
        async def invalid(request, exc):
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        app.add_exception_handler(exception, invalid)
    async def forbidden(request, exc):
        return JSONResponse(status_code=403, content={"detail": str(exc)})
    app.add_exception_handler(PermissionError, forbidden)

    def active(request):
        return request.app.state.engine

    @app.get("/healthz")
    async def health():
        return {"status": "ok"}

    @app.get("/readyz")
    async def ready(request: Request):
        application = active(request).application
        ready = application.started and not getattr(application, "ownership_error", None)
        return JSONResponse(status_code=200 if ready else 503,
                            content={"status": "ready" if ready else "unavailable",
                                     "ownership_error": getattr(application, "ownership_error", None)})

    @app.get("/metrics")
    async def metrics(request: Request):
        from inferops.adapters.outbound.observability import metrics_text
        data = metrics_text()
        sources = getattr(request.app.state, "sources", None)
        if sources:
            for camera_id, state in sources.states.items():
                for name in ("decoded", "processed", "dropped", "sampling_skips", "capture_overwrites",
                             "admission_expired", "decode_errors", "reconnects", "receive_to_result_ms"):
                    value = state.get(name, 0)
                    data += f'inferops_camera_{name}{{camera_id="{camera_id}"}} {value}\n'.encode()
        for device, gate in getattr(active(request).application.models, "_gpu_slots", {}).items():
            for name, value in gate.diagnostics().items():
                data += f'inferops_admission_{name}{{device="gpu-{device}"}} {value}\n'.encode()
        evidence = active(request).application.evidence
        if evidence:
            data += f"\ninferops_evidence_frames_dropped {evidence.dropped_frames}\n".encode()
            if hasattr(evidence, "diagnostics"):
                for name, value in evidence.diagnostics().items():
                    if name != "frames_dropped":
                        data += f"inferops_evidence_{name} {value}\n".encode()
                data += f"inferops_evidence_spool_bytes {await evidence.spool.usage_bytes()}\n".encode()
                data += f"inferops_evidence_spool_capacity_bytes {evidence.config.max_spool_bytes}\n".encode()
        repository = active(request).application.repository
        if hasattr(repository, "diagnostics"):
            try:
                snapshot = await asyncio.wait_for(repository.diagnostics(), timeout=3)
                for name in ("pending_outbox", "oldest_outbox_age_seconds", "expired_outbox_leases", "outbox_failures", "active_sessions"):
                    data += f"inferops_{name} {snapshot.get(name, 0)}\n".encode()
                for state, count in snapshot["evidence_states"].items():
                    data += f'inferops_evidence_records{{state="{state}"}} {count}\n'.encode()
                data += b"inferops_persistence_metrics_available 1\n"
            except Exception:
                data += b"inferops_persistence_metrics_available 0\n"
        return Response(data, media_type="text/plain; version=0.0.4")

    @app.get("/v1/models")
    async def models(request: Request, view: Literal["variants", "logical"] = "variants"):
        application = active(request).application
        if view == "variants":
            return [s.model_dump(mode="json") for s in application.definition.models.values()]
        return application.models.logical_models()

    @app.get("/v1/models/{name}")
    async def model(request: Request, name: str):
        return active(request).application.models.get_spec(name).model_dump(mode="json")

    @app.get("/v1/pipelines")
    async def pipelines(request: Request):
        return [spec.model_dump(mode="json") for spec in active(request).application.definition.pipelines.values()]

    @app.get("/v1/pipelines/{name}")
    async def pipeline(request: Request, name: str):
        return active(request).application.definition.pipelines[name].model_dump(mode="json")

    @app.post("/v1/sessions")
    async def open_session(request: Request, payload: dict):
        if not isinstance(payload.get("pipeline"), str) or not payload["pipeline"]:
            raise HTTPException(400, "pipeline is required")
        if not isinstance(payload.get("stream_id", ""), str):
            raise HTTPException(400, "stream_id must be a string")
        session = await active(request).open_session(payload["pipeline"], stream_name=payload.get("stream_id", ""),
                                                     owner_kind="http")
        return {"session_id": session.id, "capability": session.capability,
                "stream_id": session.stream_name, "timeline_id": session.timeline_id}

    @app.delete("/v1/sessions/{session_id}")
    async def close_session(request: Request, session_id: str, x_session_capability: str | None = Header(None)):
        session = active(request).application.sessions.get(session_id)
        if session is None:
            raise HTTPException(404, "session not found")
        session.authorize(x_session_capability)
        await session.close("client_closed")
        return {"session_id": session_id, "status": "closed"}

    @app.post("/v1/infer/{pipeline}")
    async def infer(request: Request, pipeline: str, image: UploadFile | None = OptionalImageFile,
                    stream_id: str = Form("api"), session_id: str | None = Form(None),
                    media_time_ns: int | None = Form(None, ge=0), seq: int | None = Form(None, ge=0),
                    x_session_capability: str | None = Header(None), model_override: str | None = Form(None)):
        selected = active(request)
        owned = session_id is None
        if owned:
            session = await selected.open_session(pipeline, stream_name=stream_id, owner_kind="http")
        else:
            session = selected.application.sessions.get(session_id)
            if session is None:
                raise HTTPException(404, "session not found")
            session.authorize(x_session_capability)
            if session.spec.name != pipeline:
                raise HTTPException(409, "session belongs to another pipeline")
        try:
            payload = {"image_bytes": await image.read()} if image else {}
            if model_override:
                payload["model_override"] = model_override
            async for frame in selected.application.source_factory(payload):
                frame = MediaFrame(frame.image, media_time_ns, seq if seq is not None else session.last_sequence + 1,
                                   "client_capture" if media_time_ns is not None else "estimated_capture_interval",
                                   payload=frame.payload)
                return [slim(item) for item in await session.process(frame)]
            return []
        finally:
            if owned:
                await session.close()

    @app.post("/v1/run/{pipeline}")
    async def run(request: Request, pipeline: str, payload: dict | None = None):
        payload = payload or {}
        return await active(request).run(pipeline, payload, stream_name=payload.get("stream_id", "api"), owner_kind="http")

    @app.post("/v1/run/{pipeline}/upload")
    async def upload(request: Request, pipeline: str, video: UploadFile = RequiredVideoFile, stream_id: str = Form("frontend-upload")):
        suffix = Path(video.filename or "input.mp4").suffix
        name = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as output:
                name = output.name
                while chunk := await video.read(1024 * 1024):
                    output.write(chunk)
            result = await active(request).run(pipeline, Path(name), stream_name=stream_id, owner_kind="http")
            return {**result, "stream_id": stream_id, "uploaded_filename": video.filename}
        finally:
            if name:
                Path(name).unlink(missing_ok=True)

    @app.get("/v1/events")
    async def events(request: Request, pipeline: str | None = None, limit: int = 100):
        return await active(request).application.repository.list_events(pipeline, limit)

    @app.get("/v1/alerts")
    async def alerts(request: Request, limit: int = 100):
        return await active(request).application.repository.list_incidents(limit)

    @app.get("/v1/evidence/records")
    async def evidence_records(request: Request, session_id: str | None = None, limit: int = 100):
        return await active(request).application.repository.list_evidence(session_id, limit)

    @app.get("/v1/evidence/records/{evidence_id}")
    async def evidence_record(request: Request, evidence_id: str):
        row = await active(request).application.repository.get_evidence(evidence_id)
        if row is None:
            raise HTTPException(404, "evidence not found")
        return row

    @app.post("/v1/evidence/records/{evidence_id}/retry")
    async def evidence_retry(request: Request, evidence_id: str):
        from inferops.adapters.inbound.http.cameras import forward_owned
        forwarded = await forward_owned(request, "evidence", evidence_id)
        if forwarded is not None:
            return forwarded
        captures = active(request).application.evidence
        if captures is None:
            raise HTTPException(409, "evidence capture disabled")
        return await captures.retry(evidence_id)

    @app.delete("/v1/evidence/records/{evidence_id}")
    async def evidence_delete(request: Request, evidence_id: str):
        from inferops.adapters.inbound.http.cameras import forward_owned
        forwarded = await forward_owned(request, "evidence", evidence_id)
        if forwarded is not None:
            return forwarded
        captures = active(request).application.evidence
        if captures is None:
            raise HTTPException(409, "evidence capture disabled")
        try:
            return await captures.delete(evidence_id)
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        except OSError as error:
            raise HTTPException(503, "artifact deletion pending; retry this request") from error

    @app.get("/v1/evidence/records/{evidence_id}/assets/{kind}")
    async def evidence_asset(request: Request, evidence_id: str, kind: Literal["snapshot", "clip"]):
        row = await evidence_record(request, evidence_id)
        if row.get("deleted"):
            raise HTTPException(410, "evidence was manually deleted")
        asset = row.get("artifacts", {}).get(kind)
        if asset is None:
            raise HTTPException(404, "artifact not saved")
        captures = active(request).application.evidence
        if captures is None:
            raise HTTPException(409, "evidence storage disabled")
        try:
            data = await captures.storage.read(asset["uri"])
        except FileNotFoundError as error:
            raise HTTPException(404, "artifact unavailable") from error
        if data is None:
            raise HTTPException(404, "artifact unavailable")
        content_type = "video/mp4" if kind == "clip" else "image/jpeg"
        # Browser video players need range support for seeking.
        header = request.headers.get("range")
        if header:
            try:
                first, last = _byte_range(header, len(data))
            except ValueError:
                return Response(status_code=416, headers={"Content-Range": f"bytes */{len(data)}"})
            return Response(data[first:last + 1], status_code=206, media_type=content_type,
                            headers={"Content-Range": f"bytes {first}-{last}/{len(data)}", "Accept-Ranges": "bytes"})
        return Response(data, media_type=content_type, headers={"Accept-Ranges": "bytes"})

    @app.post("/v1/jobs/{pipeline}")
    async def submit_job(request: Request, pipeline: str, payload: dict | None = None):
        return await request.app.state.jobs.submit(pipeline, payload or {})

    @app.get("/v1/jobs")
    async def jobs(request: Request, limit: int = 100):
        return await active(request).application.repository.list_jobs(limit)

    @app.get("/v1/jobs/{job_id}")
    async def job(request: Request, job_id: str):
        row = await active(request).application.repository.get_job(job_id)
        if row is None:
            raise HTTPException(404, "job not found")
        return row

    @app.post("/v1/jobs/{job_id}/cancel")
    async def cancel_job(request: Request, job_id: str):
        return await request.app.state.jobs.cancel(job_id)

    @app.post("/v1/jobs/{job_id}/retry")
    async def retry_job(request: Request, job_id: str):
        return await request.app.state.jobs.retry(job_id)

    @app.websocket("/v1/live/{pipeline}")
    async def live(websocket: WebSocket, pipeline: str, stream_id: str = "dashboard-live", model: str | None = None):
        from inferops.adapters.outbound.observability import E2E_LATENCY, FRAMES_DROPPED, WS_CLIENTS
        selected = websocket.app.state.engine
        try:
            session = await selected.open_session(pipeline, stream_name=stream_id, owner_kind="websocket")
        except RegistryError as exc:
            await websocket.close(code=4404, reason=str(exc))
            return
        await websocket.accept()
        WS_CLIENTS.labels(channel=f"live:{pipeline}").inc()
        latest = asyncio.Queue(maxsize=1)
        send_lock = asyncio.Lock()
        dropped = 0

        async def send(payload):
            async with send_lock:
                await websocket.send_json(payload)

        async def processor():
            while True:
                frame, received = await latest.get()
                try:
                    outputs = await session.process(frame, observed=True)
                    if outputs:
                        from inferops.adapters.inbound.http.serialization import live_result
                        response = live_result(outputs[0])
                        response["session_id"] = session.id
                        response["timeline_id"] = session.timeline_id
                        response["evidence"] = outputs[0].payload.get("evidence", [])
                        response["latency_ms"] = round((time.monotonic() - received) * 1000, 2)
                        response["frames_dropped"] = dropped
                        await send(response)
                        E2E_LATENCY.labels(pipeline=pipeline).observe(time.monotonic() - received)
                except (WebSocketDisconnect, RuntimeError):
                    raise
                except Exception as exc:
                    await send({"error": str(exc), "session_id": session.id})
                finally:
                    latest.task_done()

        async def receiver():
            nonlocal dropped
            pending_meta = {}
            sequence = 0
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                if message.get("text") is not None:
                    try:
                        pending_meta = _control_packet(message["text"])
                    except (json.JSONDecodeError, ValueError) as exc:
                        pending_meta = {}
                        await send({"error": "invalid_control", "detail": str(exc), "session_id": session.id})
                    continue
                if message.get("bytes") is not None:
                    received = time.monotonic()
                    meta, pending_meta = pending_meta, {}
                    try:
                        async for decoded in selected.application.source_factory(message["bytes"]):
                            media = meta.get("media_time_ns")
                            payload = dict(decoded.payload)
                            payload["live_mode"] = True
                            if model:
                                payload["model_override"] = model
                            if meta.get("seq") is not None:
                                payload["client_seq"] = meta["seq"]
                            frame = MediaFrame(decoded.image, media if media is not None else time.monotonic_ns() - session.origin,
                                               sequence, "client_capture" if media is not None else "estimated_capture_interval",
                                               presentation_ts_ms=meta.get("client_ts_ms"), payload=payload)
                            sequence += 1
                            await session.observe(frame)
                            if latest.full():
                                latest.get_nowait()
                                latest.task_done()
                                dropped += 1
                                FRAMES_DROPPED.labels(pipeline=pipeline, reason="latest_frame_replaced").inc()
                            latest.put_nowait((frame, received))
                    except (WebSocketDisconnect, RuntimeError):
                        raise
                    except Exception as exc:
                        await send({"error": str(exc), "session_id": session.id})

        tasks = [asyncio.create_task(processor(), name=f"live-process-{session.id}"),
                 asyncio.create_task(receiver(), name=f"live-receive-{session.id}")]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if not task.cancelled() and task.exception() is not None:
                    error = task.exception()
                    if not isinstance(error, (WebSocketDisconnect, RuntimeError)):
                        log.error("live_task_failed", exc_info=error)
        except asyncio.CancelledError:
            # Client/server shutdown can cancel the handler before disconnect is read.
            pass
        finally:
            from anyio import CancelScope
            with CancelScope(shield=True):
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                WS_CLIENTS.labels(channel=f"live:{pipeline}").dec()
                await session.close("disconnected")

    @app.websocket("/v1/ws/{channel}")
    async def subscribe(websocket: WebSocket, channel: str):
        from inferops.adapters.outbound.observability import WS_CLIENTS
        await websocket.accept()
        WS_CLIENTS.labels(channel=channel).inc()
        queue = asyncio.Queue(maxsize=64)
        selected_hub = websocket.app.state.hub
        selected_hub.subscribe(channel, queue)
        async def sender():
            while True:
                await websocket.send_json(await queue.get())

        async def receiver():
            while True:
                if (await websocket.receive())["type"] == "websocket.disconnect":
                    return

        tasks = [asyncio.create_task(sender(), name="subscription-send"),
                 asyncio.create_task(receiver(), name="subscription-receive")]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if not task.cancelled():
                    error = task.exception()
                    if error is not None and not isinstance(error, (WebSocketDisconnect, RuntimeError)):
                        log.error("subscription_task_failed", exc_info=error)
        except asyncio.CancelledError:
            pass
        finally:
            selected_hub.unsubscribe(channel, queue)
            WS_CLIENTS.labels(channel=channel).dec()
            from anyio import CancelScope
            with CancelScope(shield=True):
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    @app.post("/v1/streams")
    async def start_stream(request: Request, pipeline: str, rtsp_url: str | None = None,
                           mjpeg_url: str | None = None, stream_id: str | None = None, fps: float = Query(5, gt=0)):
        from inferops.adapters.inbound.sources import camera_frames, process_live_frames
        selected = active(request)
        if rtsp_url and mjpeg_url:
            raise HTTPException(400, "provide at most one rtsp_url or mjpeg_url")
        declaration = None
        if not (rtsp_url or mjpeg_url):
            definition = selected.application.definition
            if pipeline not in definition.pipelines:
                raise RegistryError(f"unknown pipeline: {pipeline}")
            declaration = definition.pipelines[pipeline].source
            if declaration is None or declaration.stage == "source_rest":
                raise HTTPException(400, "provide a camera URL or configure an autonomous pipeline source")
        session = await selected.open_session(pipeline, stream_name=stream_id or "", owner_kind="camera")
        async def loop():
            try:
                if declaration and declaration.stage == "source_kafka":
                    from inferops.adapters.inbound.workers.topic_worker import run_topic_worker
                    await run_topic_worker(pipeline, input_topic=declaration.params["topic"], output_topic=None,
                                           group=declaration.params.get("group"), engine=selected, manage_engine=False)
                elif declaration and declaration.stage not in {"source_rtsp", "source_mjpeg"}:
                    from inferops.adapters.inbound.workers.execution import (
                        item_frames,
                        require_durable_processing,
                    )
                    from inferops.adapters.outbound.stages.base import StageContext
                    source = selected.application.stage_factory(declaration)
                    await source.setup(StageContext(pipeline=pipeline, name=declaration.name,
                                                    params=declaration.params, registry=session.models,
                                                    transport=selected.application.transport))
                    try:
                        sequence = 0
                        async for item in source.run():
                            async for frame in item_frames(selected, item, sequence=sequence):
                                require_durable_processing(await session.process(frame))
                                sequence = frame.sequence + 1
                    finally:
                        await source.teardown()
                else:
                    from inferops.adapters.outbound.observability import FRAMES_DROPPED
                    url = rtsp_url or mjpeg_url or declaration.params["url"]
                    rate = declaration.params.get("fps", fps) if declaration else fps
                    await process_live_frames(session, camera_frames(url, fps=rate), on_drop=lambda:
                        FRAMES_DROPPED.labels(pipeline=pipeline, reason="latest_frame_replaced").inc())
            finally:
                await session.close("source_closed")
                streams.pop(session.id, None)
        streams[session.id] = asyncio.create_task(loop())
        def completed(task):
            if not task.cancelled() and task.exception() is not None:
                log.error("camera_stream_failed", extra={"session_id": session.id}, exc_info=task.exception())
        streams[session.id].add_done_callback(completed)
        return {"stream_id": session.id, "session_id": session.id, "capability": session.capability,
                "stream_name": stream_id, "pipeline": pipeline, "status": "started"}

    @app.get("/v1/streams")
    async def list_streams():
        return [{"stream_id": name, "done": task.done()} for name, task in streams.items()]

    @app.delete("/v1/streams/{session_id}")
    async def stop_stream(request: Request, session_id: str, x_session_capability: str | None = Header(None)):
        task = streams.get(session_id)
        if task is None:
            raise HTTPException(404, "stream not found")
        session = active(request).application.sessions.get(session_id)
        if session is None:
            raise HTTPException(404, "session not found")
        session.authorize(x_session_capability)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        # A task cancelled before its first instruction never runs its finally.
        await session.close("client_closed")
        streams.pop(session_id, None)
        return {"stream_id": session_id, "status": "stopped"}

    # UI assets and original S3 browsing retain their compatibility addresses.
    from inferops.adapters.inbound.http import evidence as evidence_browser
    from inferops.adapters.inbound.http import system
    from inferops.adapters.inbound.http.ui import mount_ui
    mount_ui(app)
    app.include_router(evidence_browser.router, prefix="/v1", tags=["evidence"])
    app.include_router(system.router, prefix="/v1", tags=["system"])

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return Response(status_code=204)

    @app.get("/")
    async def root(request: Request):
        if "text/html" in request.headers.get("accept", ""):
            return RedirectResponse("/ui/", status_code=307)
        return {"name": "inferops", "version": __version__,
                "docs": "/docs", "frontend": "/frontend/", "ui": "/ui/",
                "interfaces": "/v1/interfaces", "guide": "/ui/guide", "metrics": "/metrics"}

    return app
