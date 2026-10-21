"""Stable camera/source registration and session lifecycle for one control owner."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress

from inferops.domain.sources import FrameCadence, SourceSpec
from inferops.ports.sources import SourceRelay, SourceStore

log = logging.getLogger(__name__)


class SourceManager:
    def __init__(self, application, store: SourceStore, *, frames, render, serialize, max_active=8,
                 relay: SourceRelay | None = None):
        self.application, self.store = application, store
        self.frames, self.render, self.serialize = frames, render, serialize
        self.max_active = max_active
        if max_active < 1:
            raise ValueError("active source limit must be positive")
        self.specs, self.tasks, self.states, self.previews = {}, {}, {}, {}
        self._lock = asyncio.Lock()
        self.relay, self.relay_error, self._relay_task = relay, None, None
        self._autostart_pending = set()

    async def start(self):
        await self.store.start()
        for spec in await self.store.list_sources():
            self.specs[spec.camera_id] = spec
            runs = await self.store.list_runs(spec.camera_id, 1)
            if runs:
                self.states[spec.camera_id] = runs[0]
                jpeg = await self.store.get_preview(spec.camera_id)
                if jpeg and spec.preview_enabled:
                    self.previews[spec.camera_id] = jpeg
        if self.relay:
            self._relay_task = asyncio.create_task(self._maintain_relay(), name="source-relay-reconcile")
        for spec in list(self.specs.values()):
            if spec.autostart:
                try:
                    await self.run(spec.camera_id)
                except Exception as error:
                    if self.relay and spec.kind == "rtsp":
                        self._autostart_pending.add(spec.camera_id)
                    pending = spec.camera_id in self._autostart_pending
                    self.states[spec.camera_id] = {
                        "status": "waiting" if pending else "error",
                        "error": "Autostart is waiting and will retry; check media routing and capacity"
                        if pending else "autostart failed; check capacity/configuration",
                    }
                    log.warning("Source autostart deferred: camera_id=%s error_type=%s",
                                spec.camera_id, type(error).__name__)

    async def _maintain_relay(self):
        # Dynamic MediaMTX paths are volatile. Restore them after a router restart.
        while True:
            try:
                for camera_id in list(self.specs):
                    async with self._lock:
                        spec = self.specs.get(camera_id)
                        if spec and spec.kind == "rtsp":
                            await self.relay.prepare(spec)
                self.relay_error = None
            except Exception:
                if self.relay_error is None:
                    log.warning("Media relay reconciliation failed; it will retry")
                self.relay_error = "Media relay unavailable; reconnect and path restoration will retry"
            else:
                for camera_id in tuple(self._autostart_pending):
                    # Keep startup intent until capacity/routing recovers or
                    # an explicit edit/stop clears it.
                    with suppress(Exception):
                        await self.run(camera_id, retry_autostart=True)
            await asyncio.sleep(10)

    async def put(self, spec: SourceSpec, *, create=False):
        async with self._lock:
            if create and spec.camera_id in self.specs:
                raise ValueError("camera_id is already registered")
            if spec.camera_id in self.tasks:
                raise ValueError("stop the source before editing its configuration")
            if spec.pipeline not in self.application.definition.pipelines:
                raise ValueError("unknown pipeline")
            if hasattr(getattr(self.application, "repository", None), "claim_camera"):
                await self.application.repository.claim_camera(spec.camera_id)
            if self.relay:
                await self.relay.prepare(spec)
                previous = self.specs.get(spec.camera_id)
                if previous and previous.kind == "rtsp" and spec.kind != "rtsp":
                    await self.relay.remove(spec.camera_id)
            await self.store.put_source(spec)
            self.specs[spec.camera_id] = spec
            self._autostart_pending.discard(spec.camera_id)
            self.previews.pop(spec.camera_id, None)
            self.states.pop(spec.camera_id, None)
            return self.snapshot(spec.camera_id)

    def snapshot(self, camera_id):
        spec = self.specs[camera_id]
        # Credentials stay on the server. Editing uses a blank URI to retain it.
        public = spec.model_dump(exclude={"uri"})
        return {
            **public,
            "configured": True,
            **self.states.get(camera_id, {"status": "stopped"}),
            "active": camera_id in self.tasks,
            "live_video_available": self.relay is not None and spec.kind == "rtsp",
        }

    async def run(self, camera_id, *, retry_autostart=False):
        async with self._lock:
            spec = self.specs[camera_id]
            if retry_autostart and camera_id not in self._autostart_pending:
                return self.snapshot(camera_id)
            if hasattr(getattr(self.application, "repository", None), "claim_camera"):
                await self.application.repository.claim_camera(camera_id)
            if camera_id in self.tasks:
                return self.snapshot(camera_id)
            if len(self.tasks) >= self.max_active:
                raise ValueError(
                    f"active source limit ({self.max_active}) reached; stop a source or change configuration"
                )
            capture_spec = await self.relay.prepare(spec) if self.relay else spec
            session = await self.application.open_session(
                spec.pipeline, stream_name=camera_id, owner_kind="camera"
            )
            state = dict(
                camera_id=camera_id,
                session_id=session.id,
                started_at=time.time(),
                status="starting",
                processed=0,
                decoded=0,
                dropped=0,
                sampling_skips=0,
                admission_expired=0,
                decode_errors=0,
                capture_overwrites=0,
                reconnects=0,
                connection="opening",
                fps=0,
                latency_ms=0,
                preview_version=0,
            )
            self.states[camera_id] = state
            self.previews.pop(camera_id, None)
            try:
                await self.store.save_run(state)
            except BaseException:
                await session.close("catalog_failed")
                raise
            self.tasks[camera_id] = asyncio.create_task(
                self._run(capture_spec, session, state), name=f"source-{camera_id}"
            )

            def completed(task):
                if not task.cancelled() and task.exception() is not None:
                    state.update(
                        status="error",
                        error="session finalization failed; inspect evidence and catalog storage",
                    )
                    log.error("Source finalization failed: camera_id=%s", camera_id)

            self.tasks[camera_id].add_done_callback(completed)
            self._autostart_pending.discard(camera_id)
            # Install the task's cleanup before a stop request can cancel it.
            await asyncio.sleep(0)
            return self.snapshot(camera_id)

    async def _run(self, spec, session, state):
        preview_at = 0
        began = time.monotonic()
        queue = asyncio.Queue(maxsize=1)
        sentinel = object()

        async def receive():
            frames = self.frames(spec, state)
            cadence = FrameCadence(spec.inference_fps)
            try:
                async for frame in frames:
                    frame.payload["live_mode"] = True
                    await session.observe(frame)
                    media_ns = frame.media_time_ns or 0
                    if not cadence.due(media_ns, frame.payload.get("connection_generation")):
                        state["sampling_skips"] += 1
                        continue
                    if queue.full():
                        queue.get_nowait()
                        state["dropped"] += 1
                    queue.put_nowait((frame, time.monotonic()))
                await queue.put(sentinel)
            finally:
                if hasattr(frames, "aclose"):
                    await frames.aclose()

        reader = asyncio.create_task(receive(), name=f"read-{spec.camera_id}")

        # Wake a consumer if a reader fails instead of leaving it waiting forever.
        def reader_done(task):
            if not task.cancelled() and task.exception() is not None:
                if queue.full():
                    queue.get_nowait()
                queue.put_nowait(sentinel)

        reader.add_done_callback(reader_done)
        try:
            while True:
                pending = await queue.get()
                if pending is sentinel:
                    await reader
                    break
                frame, received = pending
                captured = frame.payload.get("_captured_monotonic_ns", int(received * 1e9)) / 1e9
                deadline = captured + spec.max_frame_age_ms / 1000
                if spec.kind != "image" and time.monotonic() >= deadline:
                    state["admission_expired"] += 1
                    continue
                if spec.kind != "image":
                    frame.payload["_admission_deadline"] = deadline
                state["status"] = "running"
                started = time.monotonic()
                outputs = await session.process(frame, observed=True)
                if frame.payload.get("_admission_expired"):
                    state["admission_expired"] += 1
                    continue
                state["latency_ms"] = round((time.monotonic() - started) * 1000, 1)
                state["receive_to_result_ms"] = round((time.monotonic() - captured) * 1000, 1)
                state["processed"] += 1
                state["fps"] = round(state["processed"] / max(0.001, time.monotonic() - began), 2)
                state["last_result_at"] = time.time()
                state["processed_width"], state["processed_height"] = (
                    frame.image.shape[1],
                    frame.image.shape[0],
                )
                wire = [self.serialize(item) for item in outputs]
                await self.store.save_result(
                    spec.camera_id,
                    session.id,
                    {
                        "camera_id": spec.camera_id,
                        "session_id": session.id,
                        "at": time.time(),
                        "sequence": frame.sequence,
                        "results": wire,
                    },
                )
                if spec.preview_enabled and outputs and time.monotonic() >= preview_at:
                    # Encode exactly the frame whose annotations were produced;
                    # one cached JPEG per camera, shared by all viewers.
                    self.previews[spec.camera_id] = await self.render(
                        frame.image, outputs[-1], spec, session.annotator
                    )
                    await self.store.save_preview(spec.camera_id, self.previews[spec.camera_id])
                    state["preview_version"] += 1
                    preview_at = time.monotonic() + 1 / spec.preview_fps
            state["status"] = "completed"
        except asyncio.CancelledError:
            state["status"] = "stopped"
            raise
        except Exception as error:
            # Decoder exceptions can contain RTSP credentials; never expose them.
            state.update(status="error", error="processing failed; verify input, pipeline and server logs")
            log.error(
                "Source processing failed: camera_id=%s error_type=%s", spec.camera_id, type(error).__name__
            )
        finally:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
            try:
                await session.close(state["status"])
            finally:
                state["finished_at"] = time.time()
                try:
                    await self.store.save_run(state)
                finally:
                    self.tasks.pop(spec.camera_id, None)

    async def stop(self, camera_id):
        async with self._lock:
            if camera_id in self._autostart_pending:
                self.states[camera_id].update(status="stopped", error=None)
            self._autostart_pending.discard(camera_id)
            task = self.tasks.get(camera_id)
            if task:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            return self.snapshot(camera_id)

    async def remove(self, camera_id):
        await self.stop(camera_id)
        async with self._lock:
            if camera_id in self.tasks:
                raise ValueError("source restarted while removing it; stop it first")
            if self.relay and self.specs[camera_id].kind == "rtsp":
                await self.relay.remove(camera_id)
            await self.store.delete_source(camera_id)
            self.specs.pop(camera_id)
            self.previews.pop(camera_id, None)
            self.states.pop(camera_id, None)

    async def close(self):
        if self._relay_task:
            self._relay_task.cancel()
            await asyncio.gather(self._relay_task, return_exceptions=True)
        for camera_id in list(self.tasks):
            await self.stop(camera_id)
        try:
            await self.store.close()
        finally:
            if self.relay:
                await self.relay.close()
