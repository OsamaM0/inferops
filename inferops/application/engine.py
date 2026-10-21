"""Engine resource ownership and session admission/lifecycle."""
from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import suppress

from inferops.application.sessions import ProcessingSession
from inferops.domain.errors import PipelineError, RegistryError


class PipelineApplication:
    def __init__(self, definition, models, repository, *, stage_factory, source_factory,
                 evidence=None, delivery=None, transport=None, annotators=None,
                 max_sessions=128, idle_timeout_s=300, recover_sessions=False, prewarm_models=()):
        self.definition, self.models, self.repository = definition, models, repository
        self.stage_factory, self.source_factory = stage_factory, source_factory
        self.evidence, self.delivery, self.transport = evidence, delivery, transport
        self.annotators = annotators or {}
        self.owner = uuid.uuid4().hex
        self.sessions = {}
        self.relay = None
        self.started = False
        self._start_lock = asyncio.Lock()
        self._closing = False
        self._sweeper = None
        self.retired_models = []
        self.max_sessions = max_sessions
        if max_sessions < 1 or idle_timeout_s <= 0:
            raise ValueError("session capacity and idle timeout must be positive")
        self.idle_timeout_s = idle_timeout_s
        self.recover_sessions = recover_sessions
        self.prewarm_models = tuple(prewarm_models)
        self._heartbeat = None
        self.ownership_error = None
        self.observer = None
        self.last_ownership_renewal = time.monotonic()

    async def start(self):
        async with self._start_lock:
            try:
                return await self._start()
            except BaseException:
                # Cleanup tries every owned resource; retain the startup cause.
                with suppress(Exception):
                    await self.close()
                raise

    async def _start(self):
        if self._closing:
            raise RuntimeError("engine is closed")
        if self.started:
            return self
        await self.repository.start()
        if hasattr(self.repository, "renew_ownership"):
            async def heartbeat():
                while True:
                    try:
                        await self.repository.renew_ownership()
                        self.last_ownership_renewal = time.monotonic()
                    except Exception:
                        self.ownership_error = "ownership renewal failed; restart this owner"
                        self.started = False
                        # Fence first; then close native work without renewing.
                        for session in list(self.sessions.values()):
                            with suppress(Exception):
                                await session.close("ownership_lost")
                        return
                    await asyncio.sleep(self.repository.lease_s / 4)
            self._heartbeat = asyncio.create_task(heartbeat(), name="owner-heartbeat")
        if self.recover_sessions and hasattr(self.repository, "interrupt_other_sessions"):
            await self.repository.interrupt_other_sessions(self.owner)
        for model in self.prewarm_models:
            await self.models.get_backend(model)
        if self.evidence:
            await self.evidence.start()
            await self.evidence.recover()
        if self.delivery:
            from inferops.application.delivery import OutboxRelay
            self.relay = OutboxRelay(self.repository, self.delivery)
            await self.relay.start()
        self.started = True
        async def expire_sessions():
            while True:
                await asyncio.sleep(15)
                for session in list(self.sessions.values()):
                    if session.owner_kind == "http" and time.monotonic() - session.last_active > self.idle_timeout_s:
                        await session.close("idle_timeout")
        self._sweeper = asyncio.create_task(expire_sessions(), name="session-expiry")
        return self

    async def open_session(self, pipeline, *, stream_name="", owner_kind="python"):
        if self.ownership_error:
            raise PipelineError(self.ownership_error)
        if pipeline not in self.definition.pipelines:
            raise RegistryError(f"unknown pipeline: {pipeline}")
        await self.start()
        if len(self.sessions) >= self.max_sessions:
            raise PipelineError("session capacity exhausted")
        session = ProcessingSession(self, pipeline, stream_name=stream_name, owner_kind=owner_kind)
        # Reserve admission before setup yields, so concurrent clients cannot
        # all pass the limit while their persistence/stage setup is pending.
        self.sessions[session.id] = session
        try:
            await session.start()
        except BaseException:
            await session.close("setup_failed")
            raise
        return session

    async def collect_retired_models(self):
        active = {session.models for session in self.sessions.values()}
        for models in list(self.retired_models):
            if models not in active:
                await models.close()
                self.retired_models.remove(models)

    async def close(self):
        self._closing = True
        if self._sweeper:
            self._sweeper.cancel()
            await asyncio.gather(self._sweeper, return_exceptions=True)
        errors = []
        for session in list(self.sessions.values()):
            try:
                await session.close("shutdown")
            except Exception as exc:
                errors.append(exc)
        storage = self.evidence.storage if self.evidence else None
        for resource in (self.evidence, self.relay, self.models, *self.retired_models, self.transport, storage):
            if resource is not None:
                try:
                    await resource.close()
                except Exception as exc:
                    errors.append(exc)
        if self._heartbeat:
            self._heartbeat.cancel()
            await asyncio.gather(self._heartbeat, return_exceptions=True)
        try:
            await self.repository.close()
        except Exception as exc:
            errors.append(exc)
        self.started = False
        if errors:
            raise RuntimeError(f"application shutdown failures: {errors}") from errors[0]

