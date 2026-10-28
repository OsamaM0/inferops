"""Session-owned execution and explicit temporal clocks."""
from __future__ import annotations

import asyncio
import hashlib
import secrets
import time
import uuid
from dataclasses import replace

from inferops.application.execution import build_steps, execute_steps
from inferops.application.incidents import accept_incidents, wire_result
from inferops.domain.errors import PipelineError
from inferops.domain.processing import MediaFrame
from inferops.domain.types import StageItem


class ProcessingSession:
    def __init__(self, application, pipeline: str, *, stream_name: str = "", owner_kind: str = "python"):
        self.application = application
        self.definition = application.definition
        self.models = application.models
        self.stage_factory = application.stage_factory
        self.spec = self.definition.pipelines[pipeline].model_copy(deep=True)
        self.id = uuid.uuid4().hex
        self.capability = secrets.token_urlsafe(32)
        self.stream_name, self.owner_kind = stream_name, owner_kind
        self.timeline_id = uuid.uuid4().hex
        self.generation = 1
        self.ingress_timeline = self.timeline_id
        self.last_observed_ns = None
        self.connection_generation = None
        self.lock = asyncio.Lock()
        self.last_media_ns = None
        self.last_sequence = -1
        self.last_active = time.monotonic()
        self.origin = time.monotonic_ns()
        self.steps = []
        self.fusions = {}
        self.policies = {}
        self.closed = False
        self.replies = {}
        self.sinks = []
        self.annotator = application.annotators.get(pipeline)

    async def start(self):
        self.models.register_session(self.id)
        await self.application.repository.save_session({
            "id": self.id, "pipeline": self.spec.name, "stream_name": self.stream_name,
            "owner": self.application.owner, "owner_kind": self.owner_kind,
            "generation": self.generation, "state": "active", "status": "open", "revision": self.definition.revision,
            "capability_hash": hashlib.sha256(self.capability.encode()).hexdigest(),
        })
        await build_steps(self)
        return self

    def authorize(self, capability: str | None):
        if not capability or not secrets.compare_digest(capability, self.capability):
            raise PermissionError("session capability required")
        if self.closed:
            raise PipelineError("session is closed")


    async def observe(self, frame: MediaFrame):
        media_ns = frame.media_time_ns if frame.media_time_ns is not None else time.monotonic_ns() - self.origin
        generation = frame.payload.get("connection_generation")
        reconnected = (generation is not None and self.connection_generation is not None
                       and generation != self.connection_generation)
        if reconnected or (self.last_observed_ns is not None and media_ns < self.last_observed_ns):
            self.ingress_timeline = uuid.uuid4().hex
        self.connection_generation = generation
        self.last_observed_ns = media_ns
        frame.payload["_ingress_timeline"] = self.ingress_timeline
        if self.application.evidence:
            await self.application.evidence.observe(self.id, self.ingress_timeline, media_ns, frame.sequence,
                                                    frame.image, frame.payload)
        return media_ns

    async def process(self, frame: MediaFrame, *, observed: bool = False):
        from inferops.application.admission import AdmissionExpired, execution_deadline, execution_key
        if self.closed:
            raise PipelineError("session is closed")
        key = execution_key.set(self.id)
        deadline = execution_deadline.set(frame.payload.get("_admission_deadline"))
        outcome = "ok"
        try:
            timeout = self.spec.limits.timeout_ms
            if timeout:
                return await asyncio.wait_for(self._process(frame, observed=observed), timeout / 1000)
            return await self._process(frame, observed=observed)
        except AdmissionExpired:
            outcome = "admission_expired"
            frame.payload["_admission_expired"] = True
            return []
        except asyncio.CancelledError:
            outcome = "cancelled"
            await self.close("cancelled")
            raise
        except BaseException as exc:
            outcome = "error"
            await self.close("processing_failed")
            if isinstance(exc, asyncio.TimeoutError):
                raise PipelineError("processing deadline exceeded") from exc
            raise
        finally:
            if self.application.observer:
                self.application.observer.completed(self.spec.name, outcome)
            execution_deadline.reset(deadline)
            execution_key.reset(key)

    async def _process(self, frame: MediaFrame, *, observed: bool = False):
        async with self.lock:
            self.last_active = time.monotonic()
            media_ns = frame.media_time_ns if frame.media_time_ns is not None else time.monotonic_ns() - self.origin
            if frame.media_time_ns is None:
                frame = replace(frame, media_time_ns=media_ns)
            if hasattr(self.application.repository, "get_session"):
                ownership = await self.application.repository.get_session(self.id)
                if ownership is None or ownership.get("owner") != self.application.owner or ownership.get("generation", 1) != self.generation or ownership.get("status") != "open":
                    raise PipelineError("session ownership was interrupted")
            ingress_timeline = frame.payload.get("_ingress_timeline") if observed else None
            # Equal PTS is valid; sequence is the tie-breaker. A backwards PTS is a new timeline.
            if (ingress_timeline and ingress_timeline != self.timeline_id) or (self.last_media_ns is not None and media_ns < self.last_media_ns):
                if self.application.evidence:
                    await self.application.evidence.finish_session(self.id, "timeline_discontinuity")
                for _, kind, stage in self.steps:
                    if kind == "stage":
                        await stage.teardown()
                for sink in self.sinks:
                    await sink.finalize()
                    await sink.teardown()
                self.timeline_id = ingress_timeline or uuid.uuid4().hex
                self.ingress_timeline = self.timeline_id
                self.last_observed_ns = None
                observed = False
                self.last_sequence = -1
                self.replies.clear()
                await build_steps(self)
            if frame.sequence in self.replies:
                return self.replies[frame.sequence]
            if frame.sequence <= self.last_sequence:
                raise PipelineError("frame sequence must increase within a timeline")
            if not observed:
                await self.observe(frame)
            previous_sequence = self.last_sequence
            previous_media = self.last_media_ns
            self.last_media_ns, self.last_sequence = media_ns, frame.sequence
            checkpoints = {name: policy.checkpoint() for name, policy in self.policies.items()}
            fusion_checkpoints = {name: engine.checkpoint() for name, engine in self.fusions.items()}
            payload = dict(frame.payload)
            payload.update(image_np=frame.image, session_id=self.id, timeline_id=self.timeline_id,
                           media_time_ns=media_ns, timestamp_provenance=frame.timestamp_provenance,
                           stream_name=self.stream_name)
            if frame.image is not None:
                payload.update(width=frame.image.shape[1], height=frame.image.shape[0])
            item = StageItem(stream_id=self.id, ts_ms=media_ns // 1_000_000, seq=frame.sequence, payload=payload)
            current = [item]
            current = await execute_steps(self, current)
            for value in current:
                if self.application.evidence:
                    await self.application.evidence.annotate_frame(self.id, self.timeline_id, media_ns, value,
                                                                   self.annotator)
                value.ts_ms = frame.presentation_ts_ms if frame.presentation_ts_ms is not None else time.time_ns() // 1_000_000
                value.stream_id = self.stream_name or self.id
            await accept_incidents(current, repository=self.application.repository, evidence=self.application.evidence,
                                   definition=self.definition, pipeline=self.spec, session_id=self.id,
                                   timeline_id=self.timeline_id, stream_name=self.stream_name, fusions=self.fusions,
                                   broker_context=frame.payload.get("_broker_event"),
                                   camera_id=self.stream_name if self.owner_kind == "camera" else None)
            if any(value.payload.get("incident_error") for value in current):
                for name, checkpoint in checkpoints.items():
                    self.policies[name].restore(checkpoint)
                for name, checkpoint in fusion_checkpoints.items():
                    self.fusions[name].restore(checkpoint)
                self.last_sequence, self.last_media_ns = previous_sequence, previous_media
                return current
            for value in current:
                for sink in self.sinks:
                    await sink(value)
                for declaration in self.spec.sinks:
                    if declaration.stage == "sink_ws" and self.application.delivery is not None:
                        channel = declaration.params.get("channel", "live.{stream_id}").format(stream_id=self.id)
                        await self.application.delivery.publish_transient(channel, wire_result(value))
            if self.application.observer:
                self.application.observer.accepted(current)
            # Cache only wire-safe fields, with a fixed entry bound, for same-session retry safety.
            self.replies[frame.sequence] = [v.model_copy(update={"payload": {
                k: value for k, value in v.payload.items() if k not in {"image_np", "image_bytes", "feeds", "frame_meta"}
            }}) for v in current]
            if len(self.replies) > 16:
                self.replies.pop(next(iter(self.replies)))
            return current



    async def close(self, reason="source_closed"):
        async with self.lock:
            if self.closed:
                return
            self.closed = True
            errors = []
            if self.application.evidence:
                try:
                    await self.application.evidence.finish_session(self.id, reason)
                except Exception as exc:
                    errors.append(exc)
            for stage in [stage for _, kind, stage in self.steps if kind == "stage"] + self.sinks:
                for close in (stage.finalize, stage.teardown):
                    try:
                        await close()
                    except Exception as exc:
                        errors.append(exc)
            try:
                await self.application.repository.close_session(self.id, "closed")
            except Exception as exc:
                errors.append(exc)
            self.models.unregister_session(self.id)
            self.application.sessions.pop(self.id, None)
            await self.application.collect_retired_models()
            if errors:
                raise PipelineError(f"session shutdown failures: {errors}") from errors[0]


