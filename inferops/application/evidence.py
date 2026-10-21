"""Session-isolated capture orchestration through explicit persistence/media ports."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from inferops.domain.evidence import EncodedFrame, EvidenceConfig, EvidenceState
from inferops.ports.media import EvidenceSpool, MediaEncoder

log = logging.getLogger(__name__)


@dataclass
class _Capture:
    manifest: dict[str, Any]
    deadline: float


class EvidenceService:
    """Capture decisions and recovery, with no concrete client or filesystem imports.

    Call ``request`` only after incident/evidence acceptance commits. ``observe``
    takes media time; the local capture deadline uses monotonic process time.
    ``spool`` is an explicit adapter instance, not a filesystem path.
    """

    def __init__(self, repository: Any, storage: Any, spool: EvidenceSpool,
                 config: EvidenceConfig | None = None, *, encoder: MediaEncoder,
                 annotate: Any = None):
        self.repository = repository
        self.storage = storage
        self.spool = spool
        self.config = config or EvidenceConfig()
        self.encoder = encoder
        self.annotate = annotate
        self._rings: dict[str, deque[EncodedFrame]] = {}
        self._last_frame_error: dict[str, tuple[str, int, str]] = {}
        self._captures: dict[str, _Capture] = {}
        self._queued: set[str] = set()
        self._queue: asyncio.Queue[str | None] = asyncio.Queue(self.config.max_active_captures)
        self._workers: list[asyncio.Task] = []
        self._retry_tasks: set[asyncio.Task] = set()
        self._attempts: dict[str, int] = {}
        self._watcher: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._started = False
        self._closing = False
        self.dropped_frames = 0

    def diagnostics(self) -> dict[str, int]:
        return {"active_captures": len(self._captures), "queued_captures": len(self._queued),
                "retry_waits": len(self._retry_tasks), "frames_dropped": self.dropped_frames,
                "ring_bytes": sum(len(frame.jpeg) for ring in self._rings.values() for frame in ring)}

    def _trim_rings(self) -> None:
        """Enforce the byte bounds after both raw capture and annotation."""
        for ring in self._rings.values():
            while ring and sum(len(frame.jpeg) for frame in ring) > self.config.per_session_bytes:
                ring.popleft()
                self.dropped_frames += 1
        while sum(len(frame.jpeg) for ring in self._rings.values() for frame in ring) > self.config.global_bytes:
            oldest = min((ring for ring in self._rings.values() if ring),
                         key=lambda ring: ring[0].media_time_ns)
            oldest.popleft()
            self.dropped_frames += 1

    def _capture_fits(self, manifest: dict[str, Any], frame: EncodedFrame) -> bool:
        # Encoding workers read one capture at a time. Bound that material as
        # well as the look-behind ring, rather than allowing disk quota-sized
        # captures to be loaded into memory.
        return sum(entry.get("bytes", 0) for entry in manifest["frames"]
                   if entry["sequence"] != frame.sequence) + len(frame.jpeg) <= self.config.per_session_bytes

    async def start(self) -> None:
        if self._started:
            return
        await self.spool.start()
        self._started = True
        self._workers = [asyncio.create_task(self._worker(), name=f"evidence-{i}")
                         for i in range(self.config.workers)]
        self._watcher = asyncio.create_task(self._deadlines(), name="evidence-deadlines")

    async def observe(self, session_id: str, timeline_id: str, media_time_ns: int,
                      seq: int, image: Any, item: Any = None) -> None:
        if self._closing or image is None:
            return
        await self.start()
        async with self._lock:
            ring = self._rings.setdefault(session_id, deque())
            if ring and ring[-1].timeline_id != timeline_id:
                await self._finish_session_locked(session_id, "timeline_discontinuity")
                ring.clear()
            interval = int(1e9 / self.config.sample_fps)
            if ring and media_time_ns >= ring[-1].media_time_ns and media_time_ns - ring[-1].media_time_ns < interval:
                for evidence_id, capture in list(self._captures.items()):
                    manifest = capture.manifest
                    if manifest["session_id"] == session_id and manifest["timeline_id"] == timeline_id and media_time_ns >= manifest["request_end_ns"]:
                        await self._finalize(evidence_id)
                return
            try:
                jpeg = await asyncio.to_thread(self.encoder.snapshot, image, item, self.annotate)
            except Exception as error:
                await self._encoding_failed_locked(session_id, timeline_id, media_time_ns,
                                                   "frame_encoding_failed", error)
                return
            frame = EncodedFrame(timeline_id, int(media_time_ns), int(seq), jpeg)
            if len(jpeg) > min(self.config.per_session_bytes, self.config.global_bytes):
                self.dropped_frames += 1
                return
            ring.append(frame)
            # An incident's frame reaches rule evaluation after capture has
            # advanced. Retain bounded history beyond the requested pre-window
            # so normal inference delay does not erase its earliest frames.
            cutoff = media_time_ns - int((self.config.pre_seconds + self.config.processing_lag_seconds) * 1e9)
            while ring and ring[0].media_time_ns < cutoff:
                ring.popleft()
            self._trim_rings()
            for evidence_id, capture in list(self._captures.items()):
                manifest = capture.manifest
                if manifest["session_id"] != session_id or manifest["timeline_id"] != timeline_id:
                    continue
                if manifest["request_start_ns"] <= media_time_ns <= manifest["request_end_ns"]:
                    if not self._capture_fits(manifest, frame):
                        await self._finalize(evidence_id, "capture_material_capacity")
                        continue
                    try:
                        capture.manifest = await self.spool.write(manifest, [frame])
                        await self.repository.update_evidence(evidence_id,
                            spool_generation=capture.manifest["generation"],
                            actual_end_ns=media_time_ns)
                    except Exception as error:
                        await self._finalize(evidence_id, getattr(error, "reason_code", "spool_write_failed"))
                        continue
                if media_time_ns >= manifest["request_end_ns"]:
                    await self._finalize(evidence_id)

    async def annotate_frame(self, session_id: str, timeline_id: str,
                             media_time_ns: int, item: Any, annotate: Any = None) -> None:
        """Add inference overlays after processing, before requesting evidence."""
        if not hasattr(self.encoder, "annotate"):
            return
        async with self._lock:
            ring = self._rings.get(session_id)
            if not ring:
                return
            for index in range(len(ring) - 1, -1, -1):
                frame = ring[index]
                if frame.timeline_id != timeline_id or frame.media_time_ns != media_time_ns:
                    continue
                try:
                    jpeg = await asyncio.to_thread(self.encoder.annotate, frame.jpeg, item,
                        self.annotate if annotate is None else annotate)
                except Exception as error:
                    await self._encoding_failed_locked(session_id, timeline_id, media_time_ns,
                                                       "frame_annotation_failed", error)
                    return
                replacement = EncodedFrame(timeline_id, media_time_ns, frame.sequence, jpeg)
                ring[index] = replacement
                self._trim_rings()
                for evidence_id, capture in list(self._captures.items()):
                    manifest = capture.manifest
                    if manifest["session_id"] != session_id or manifest["timeline_id"] != timeline_id:
                        continue
                    if any(entry["sequence"] == frame.sequence for entry in manifest["frames"]):
                        if not self._capture_fits(manifest, replacement):
                            await self._finalize(evidence_id, "capture_material_capacity")
                            continue
                        try:
                            capture.manifest = await self.spool.write(manifest, [replacement])
                            await self.repository.update_evidence(evidence_id,
                                spool_generation=capture.manifest["generation"])
                        except Exception as error:
                            await self._finalize(evidence_id, getattr(error, "reason_code", "spool_write_failed"))
                return

    async def _encoding_failed_locked(self, session_id: str, timeline_id: str,
                                      media_time_ns: int, reason: str, error: Exception) -> None:
        self.dropped_frames += 1
        self._last_frame_error[session_id] = (timeline_id, media_time_ns, reason)
        log.warning(reason, extra={"session_id": session_id, "media_time_ns": media_time_ns},
                    exc_info=(type(error), error, error.__traceback__))
        for evidence_id, capture in list(self._captures.items()):
            if capture.manifest["session_id"] == session_id and capture.manifest["timeline_id"] == timeline_id:
                try:
                    await self._finalize(evidence_id, reason)
                except Exception:
                    # The committed request/spool remains recoverable; evidence
                    # failures must not abort an otherwise usable inference.
                    log.exception("evidence_failure_recording_failed", extra={"evidence_id": evidence_id})

    async def request(self, record: dict[str, Any]) -> None:
        await self.start()
        evidence_id = str(record["id"])
        async with self._lock:
            if evidence_id in self._captures or evidence_id in self._queued:
                return
            persisted = await self.repository.get_evidence(evidence_id)
            if persisted is None:
                raise KeyError(evidence_id)
            # A repeated acceptance callback must not replace completed or
            # retryable material. Recovery/manual retry owns existing spools.
            if persisted.get("state", EvidenceState.PENDING.value) != EvidenceState.PENDING.value or persisted.get("spool_generation"):
                return
            record = persisted
            if len(self._captures) + len(self._queued) >= self.config.max_active_captures or self._closing:
                await self.repository.update_evidence(evidence_id, state=EvidenceState.FAILED.value,
                    errors=["capture_capacity"], retry_available=False)
                return
            trigger = int(record["media_time_ns"])
            start = int(record.get("request_start_ns", trigger - self.config.pre_seconds * 1e9))
            end = int(record.get("request_end_ns", trigger + self.config.post_seconds * 1e9))
            manifest = {
                "evidence_id": evidence_id, "generation": int(record.get("spool_generation") or 0) + 1,
                "session_id": str(record["session_id"]), "timeline_id": str(record["timeline_id"]),
                "media_time_ns": trigger, "request_start_ns": start, "request_end_ns": end,
                "frames": [], "closed": False, "errors": [],
            }
            failure = self._last_frame_error.get(manifest["session_id"])
            if failure and failure[0] == manifest["timeline_id"] and start <= failure[1] <= end:
                manifest["errors"].append(failure[2])
            ring = self._rings.get(manifest["session_id"], ())
            frames = [frame for frame in ring if frame.timeline_id == manifest["timeline_id"]
                      and start <= frame.media_time_ns <= end]
            try:
                manifest = await self.spool.write(manifest, frames)
                await self.repository.update_evidence(evidence_id, state=EvidenceState.PENDING.value,
                    spool_generation=manifest["generation"], retry_available=bool(frames))
            except Exception as error:
                await self.repository.update_evidence(evidence_id, state=EvidenceState.FAILED.value,
                    errors=[getattr(error, "reason_code", "spool_write_failed")], retry_available=False)
                return
            self._captures[evidence_id] = _Capture(manifest, time.monotonic() + self.config.capture_timeout_seconds)
            if self.config.post_seconds == 0 or (ring and ring[-1].timeline_id == manifest["timeline_id"]
                                                and ring[-1].media_time_ns >= end):
                await self._finalize(evidence_id)

    async def _finalize(self, evidence_id: str, reason: str | None = None) -> None:
        capture = self._captures.pop(evidence_id, None)
        if capture is None:
            return
        manifest = {**capture.manifest, "closed": True}
        if reason:
            manifest["errors"] = list(dict.fromkeys([*manifest.get("errors", []), reason]))
        try:
            manifest = await self.spool.write(manifest, [])
        except Exception as error:
            # Already-durable frames remain usable even if a closing generation
            # cannot be committed. Database reasons carry that incompleteness.
            manifest["errors"] = list(dict.fromkeys([*manifest["errors"],
                getattr(error, "reason_code", "spool_write_failed")]))
        try:
            await self.repository.update_evidence(evidence_id, errors=manifest["errors"],
                spool_generation=manifest["generation"], retry_available=bool(manifest["frames"]))
        finally:
            await self._enqueue(evidence_id)

    async def _enqueue(self, evidence_id: str) -> None:
        if evidence_id in self._queued:
            return
        try:
            self._queue.put_nowait(evidence_id)
            self._queued.add(evidence_id)
        except asyncio.QueueFull:
            await self.repository.update_evidence(evidence_id, state=EvidenceState.FAILED.value,
                errors=["encoding_capacity"], retry_available=True)

    async def finish_session(self, session_id: str, reason: str = "source_closed") -> None:
        async with self._lock:
            await self._finish_session_locked(session_id, reason)
            self._rings.pop(session_id, None)
            self._last_frame_error.pop(session_id, None)

    async def _finish_session_locked(self, session_id: str, reason: str) -> None:
        for evidence_id, capture in list(self._captures.items()):
            if capture.manifest["session_id"] == session_id:
                await self._finalize(evidence_id, reason)

    async def _deadlines(self) -> None:
        while True:
            await asyncio.sleep(min(1.0, self.config.capture_timeout_seconds / 4))
            async with self._lock:
                for evidence_id, capture in list(self._captures.items()):
                    if time.monotonic() >= capture.deadline:
                        await self._finalize(evidence_id, "capture_timeout")

    async def _worker(self) -> None:
        while True:
            evidence_id = await self._queue.get()
            try:
                if evidence_id is None:
                    return
                await self._save(evidence_id)
            except Exception as error:
                if evidence_id is not None:
                    # Repository failure must not kill the worker. The accepted
                    # request and durable spool are reconciled after recovery.
                    with suppress(Exception):
                        await self._failure(evidence_id, error)
            finally:
                if evidence_id is not None:
                    self._queued.discard(evidence_id)
                self._queue.task_done()

    @staticmethod
    def _coverage(manifest: dict[str, Any], frames: list[EncodedFrame], interval: int) -> list[list[int]]:
        if not frames:
            return [[manifest["request_start_ns"], manifest["request_end_ns"]]]
        missing = []
        start, end = manifest["request_start_ns"], manifest["request_end_ns"]
        if frames[0].media_time_ns > start + interval:
            missing.append([start, frames[0].media_time_ns])
        if frames[-1].media_time_ns < end - interval:
            missing.append([frames[-1].media_time_ns, end])
        for first, second in pairwise(frames):
            if second.media_time_ns - first.media_time_ns > interval * 2:
                missing.append([max(start, first.media_time_ns), min(end, second.media_time_ns)])
        return [span for span in missing if span[1] > span[0]]

    async def _save(self, evidence_id: str) -> None:
        if hasattr(self.repository, "assert_evidence_owner"):
            await self.repository.assert_evidence_owner(evidence_id)
        record = await self.repository.get_evidence(evidence_id)
        if record is None:
            return  # Unknown spool material is retained for operator reconciliation.
        if record.get("state") == EvidenceState.SAVED.value:
            # A crash may have happened after the DB commit and before cleanup.
            await self.spool.delete(evidence_id)
            return
        manifest, frames = await self.spool.read(evidence_id)
        frames = sorted(frames, key=lambda frame: (frame.media_time_ns, frame.sequence))
        self._attempts[evidence_id] = int(record.get("attempts") or 0) + 1
        await self.repository.update_evidence(evidence_id, attempts=self._attempts[evidence_id])
        if not frames:
            await self.repository.update_evidence(evidence_id, state=EvidenceState.FAILED.value,
                errors=list(dict.fromkeys([*manifest.get("errors", []), "no_capture_material"])),
                retry_available=False)
            return
        artifacts = dict(record.get("artifacts") or {})
        if getattr(self.repository, "scope", None):
            artifacts = {}  # A new owner never references a stale generation's mutable keys.
        for name, content_type in (("snapshot", "image/jpeg"), ("clip", "video/mp4")):
            filename = "snapshot.jpg" if name == "snapshot" else "clip.mp4"
            data = await self.spool.read_artifact(evidence_id, filename)
            if data is None:
                if name == "snapshot":
                    data = min(frames, key=lambda frame: abs(frame.media_time_ns - manifest["media_time_ns"])).jpeg
                else:
                    data = await asyncio.to_thread(self.encoder.clip, frames, self.config.sample_fps)
                await self.spool.write_artifact(evidence_id, filename, data)
            digest = hashlib.sha256(data).hexdigest()
            previous = artifacts.get(name, {})
            uri = previous.get("uri")
            if not uri or previous.get("sha256") != digest or not await self.storage.verify(uri, digest):
                prefix = getattr(self.repository, "artifact_prefix", "evidence")
                uri = await self.storage.put(f"{prefix}/{evidence_id}/{filename}", data, content_type)
                if not await self.storage.verify(uri, digest):
                    raise RuntimeError("Stored artifact checksum verification failed")
            artifacts[name] = {"uri": uri, "sha256": digest, "bytes": len(data), "content_type": content_type}
            # This progress is durable even when the second artifact fails.
            await self.repository.update_evidence(evidence_id, artifacts=artifacts)
        missing = self._coverage(manifest, frames, int(1e9 / self.config.sample_fps))
        errors = list(dict.fromkeys([*manifest.get("errors", []),
            *(code for code in record.get("errors", []) if code != "artifact_delivery_failed")]))
        if missing:
            errors.append("missing_media_context")
        if not manifest.get("closed"):
            errors.append("capture_interrupted")
        errors = list(dict.fromkeys(errors))
        state = EvidenceState.INCOMPLETE.value if errors or missing else EvidenceState.SAVED.value
        completed = {
            "schema_version": 1, "evidence_id": evidence_id,
            "session_id": manifest["session_id"], "timeline_id": manifest["timeline_id"],
            "media_time_ns": manifest["media_time_ns"], "state": state,
            "requested_window": {"start_ns": manifest["request_start_ns"], "end_ns": manifest["request_end_ns"]},
            "actual_window": {"start_ns": frames[0].media_time_ns, "end_ns": frames[-1].media_time_ns},
            "sample_fps": self.config.sample_fps, "missing_intervals": missing,
            "artifacts": artifacts, "errors": errors,
        }
        manifest_data = json.dumps(completed, sort_keys=True, separators=(",", ":")).encode()
        retained = await self.spool.read_artifact(evidence_id, "evidence-manifest.json")
        if retained != manifest_data:
            await self.spool.write_artifact(evidence_id, "evidence-manifest.json", manifest_data)
        manifest_digest = hashlib.sha256(manifest_data).hexdigest()
        manifest_uri = record.get("manifest_uri")
        if (getattr(self.repository, "scope", None) or not manifest_uri or record.get("manifest_sha256") != manifest_digest
                or not await self.storage.verify(manifest_uri, manifest_digest)):
            prefix = getattr(self.repository, "artifact_prefix", "evidence")
            manifest_uri = await self.storage.put(f"{prefix}/{evidence_id}/manifest.json",
                                                  manifest_data, "application/json")
            if not await self.storage.verify(manifest_uri, manifest_digest):
                raise RuntimeError("Stored evidence manifest checksum verification failed")
        await self.repository.update_evidence(evidence_id, state=state, artifacts=artifacts,
            manifest_uri=manifest_uri, manifest_sha256=manifest_digest,
            actual_start_ns=frames[0].media_time_ns, actual_end_ns=frames[-1].media_time_ns,
            missing_intervals=missing, errors=errors, retry_available=False)
        # Incomplete material is retained for inspection; completed object assets
        # are never deleted as a capacity-management policy.
        if state == EvidenceState.SAVED.value:
            await self.spool.delete(evidence_id)

    async def _failure(self, evidence_id: str, error: Exception) -> None:
        record = await self.repository.get_evidence(evidence_id)
        if record is None:
            return
        code = getattr(error, "reason_code", "artifact_delivery_failed")
        errors = list(dict.fromkeys([*record.get("errors", []), code]))
        attempt = self._attempts.get(evidence_id, int(record.get("attempts") or 0))
        permanent = code in {"spool_corrupt", "spool_capacity"} or isinstance(error, ValueError)
        can_retry = not permanent and attempt <= len(self.config.retry_delays) and not self._closing
        state = EvidenceState.PENDING.value if can_retry else (
            EvidenceState.INCOMPLETE.value if record.get("artifacts") else EvidenceState.FAILED.value)
        await self.repository.update_evidence(evidence_id, state=state, errors=errors,
            retry_available=not permanent, last_error=str(error))
        if can_retry:
            task = asyncio.create_task(self._delayed_retry(evidence_id, self.config.retry_delays[max(0, attempt - 1)]))
            self._retry_tasks.add(task)
            task.add_done_callback(self._retry_tasks.discard)

    async def _delayed_retry(self, evidence_id: str, delay: float) -> None:
        await asyncio.sleep(delay)
        if not self._closing:
            await self._enqueue(evidence_id)

    async def retry(self, evidence_id: str) -> dict[str, Any]:
        await self.start()
        record = await self.repository.get_evidence(evidence_id)
        if record is None:
            raise KeyError(evidence_id)
        if record.get("deleted"):
            raise ValueError("Evidence was manually deleted")
        if record.get("state") == EvidenceState.SAVED.value:
            return record
        _, frames = await self.spool.read(evidence_id)
        if not frames:
            raise ValueError("No retained capture material for retry")
        await self.repository.update_evidence(evidence_id, state=EvidenceState.PENDING.value,
            errors=[code for code in record.get("errors", []) if code != "artifact_delivery_failed"],
            retry_available=True)
        await self._enqueue(evidence_id)
        return await self.repository.get_evidence(evidence_id)

    async def delete(self, evidence_id: str, *, retention_before=None) -> dict[str, Any]:
        """Explicit manual deletion, with a durable retryable tombstone.

        The audit record remains queryable. Publishing the tombstone before
        deleting objects prevents readers/recovery from serving or recreating
        assets if deletion is interrupted between external side effects.
        """
        async with self._lock:
            record = await self.repository.get_evidence(evidence_id)
            if record is None:
                raise KeyError(evidence_id)
            if evidence_id in self._captures or evidence_id in self._queued or record.get("state") == EvidenceState.PENDING.value:
                raise ValueError("Evidence capture or delivery is still pending")
            if record.get("deleted") and not record.get("deletion_pending"):
                return record
            artifacts = dict(record.get("deleted_artifacts") or record.get("artifacts") or {})
            manifest_uri = record.get("deleted_manifest_uri") or record.get("manifest_uri")
            await self.repository.update_evidence(evidence_id, deleted=True, deletion_pending=True,
                **({"_retention_before": retention_before} if retention_before is not None else {}),
                deleted_artifacts=artifacts, deleted_manifest_uri=manifest_uri,
                manifest_uri=None, artifacts={}, state=EvidenceState.FAILED.value,
                previous_state=record.get("previous_state", record.get("state")),
                errors=["manually_deleted"], retry_available=False)
            uris = {value["uri"] if isinstance(value, dict) else value for value in artifacts.values()}
            if manifest_uri:
                uris.add(manifest_uri)
            for uri in sorted(uris):
                with suppress(FileNotFoundError, KeyError):
                    await self.storage.delete(uri)
            await self.spool.delete(evidence_id)
            await self.repository.update_evidence(evidence_id, deletion_pending=False)
            return await self.repository.get_evidence(evidence_id)

    async def recover(self) -> None:
        await self.start()
        ids = set(await self.spool.list_ids())
        # Include accepted requests that crashed before their first spool commit.
        records = ([row async for row in self.repository.recovery_evidence()]
                   if hasattr(self.repository, "recovery_evidence")
                   else await self.repository.list_evidence(limit=100_000))
        ids.update(str(record["id"]) for record in records
                   if record.get("state") == EvidenceState.PENDING.value or record.get("deletion_pending"))
        for evidence_id in sorted(ids):
            record = await self.repository.get_evidence(evidence_id)
            if record is None:
                continue
            if getattr(self.repository, "scope", None):
                scope = await self.repository.record_scope("evidence", evidence_id)
                if scope != self.repository.scope:
                    continue  # A foreign spool is never evidence of missing material.
                await self.repository.assert_evidence_owner(evidence_id)
            if record.get("deleted"):
                if record.get("deletion_pending"):
                    try:
                        await self.delete(evidence_id)
                    except Exception:
                        log.exception("evidence_deletion_recovery_failed", extra={"evidence_id": evidence_id})
                continue
            if record.get("state") == EvidenceState.SAVED.value:
                await self.spool.delete(evidence_id)
                continue
            if record.get("state") != EvidenceState.PENDING.value:
                # Exhausted retries and completed-but-incomplete captures are
                # terminal until an explicit retry. Restart is not a new try.
                continue
            try:
                manifest, frames = await self.spool.read(evidence_id)
                if not manifest.get("closed"):
                    manifest = {**manifest, "closed": True,
                                "errors": list(dict.fromkeys([*manifest.get("errors", []), "process_interrupted"]))}
                    manifest = await self.spool.write(manifest, [])
                await self.repository.update_evidence(evidence_id, state=EvidenceState.PENDING.value,
                    spool_generation=manifest["generation"], retry_available=bool(frames))
                await self._enqueue(evidence_id)
            except Exception as error:
                state = EvidenceState.INCOMPLETE.value if record.get("artifacts") else EvidenceState.FAILED.value
                await self.repository.update_evidence(evidence_id, state=state,
                    errors=[getattr(error, "reason_code", "capture_material_missing")], retry_available=False,
                    last_error=str(error))

    async def wait_idle(self) -> None:
        """Wait for currently queued encoding/delivery work, not live captures."""
        await self._queue.join()

    async def close(self) -> None:
        if not self._started:
            return
        self._closing = True
        if self._watcher:
            self._watcher.cancel()
            with suppress(asyncio.CancelledError):
                await self._watcher
        for task in list(self._retry_tasks):
            task.cancel()
        if self._retry_tasks:
            await asyncio.gather(*self._retry_tasks, return_exceptions=True)

        async def drain():
            async with self._lock:
                for evidence_id in list(self._captures):
                    await self._finalize(evidence_id, "shutdown")
                self._rings.clear()
                self._last_frame_error.clear()
            await self.wait_idle()
            for _ in self._workers:
                await self._queue.put(None)
            await asyncio.gather(*self._workers)

        try:
            await asyncio.wait_for(drain(), timeout=self.config.shutdown_timeout_seconds)
        except asyncio.TimeoutError:
            # Cancel async ownership without deleting capture/artifact material.
            # Pending DB requests and committed spool generations reconcile on
            # restart, including writes already dispatched to a worker thread.
            log.warning("evidence_shutdown_timeout", extra={"pending_evidence": sorted(self._queued)})
        finally:
            for worker in self._workers:
                worker.cancel()
            await asyncio.gather(*self._workers, return_exceptions=True)
            while not self._queue.empty():
                self._queue.get_nowait()
                self._queue.task_done()
            self._queued.clear()
            self._captures.clear()
            if hasattr(self.spool, "close"):
                await self.spool.close()
            self._rings.clear()
            self._last_frame_error.clear()
            self._workers.clear()
            self._started = False
