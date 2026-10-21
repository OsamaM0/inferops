"""Inbound worker translation into session-owned application processing."""
from __future__ import annotations

import asyncio
import base64
import time
from dataclasses import replace

from inferops.domain.processing import MediaFrame


async def item_frames(engine, item, *, sequence: int, broker_event=None):
    """Translate a source envelope without using its stream name as identity."""
    payload = dict(item.payload)
    media = payload.get("media_time_ns")
    if media is None and payload.get("timestamp_provenance") in {"source", "presentation_timestamp", "supplied"}:
        media = item.ts_ms * 1_000_000
    encoded = payload.pop("image_base64", None)
    if encoded is not None:
        payload["image_bytes"] = base64.b64decode(encoded, validate=True)
    if broker_event is not None:
        if "video_path" in payload:
            raise ValueError("broker inputs require one frame; submit video files through the job API")
        payload["_broker_event"] = broker_event
    has_media = any(key in payload for key in ("image_np", "image_bytes", "video_path"))
    if has_media:
        count = 0
        async for decoded in engine.application.source_factory(payload):
            merged = {**payload, **decoded.payload}
            yield replace(decoded, media_time_ns=media if media is not None else (
                              decoded.media_time_ns if "video_path" in payload else None),
                          sequence=sequence + count, presentation_ts_ms=item.ts_ms,
                          timestamp_provenance=payload.get("timestamp_provenance", decoded.timestamp_provenance)
                          if media is not None or "video_path" in payload else "estimated_capture_interval",
                          payload=merged)
            count += 1
        if not count:
            raise ValueError("source envelope has no decodable media")
        return
    if item.frame_ref:
        store = getattr(engine.application.evidence, "storage", None)
        if store is None:
            raise ValueError("frame_ref requires an explicitly configured object store")
        data = await store.read(item.frame_ref)
        if data is None:
            raise ValueError("referenced frame is unavailable")
        payload["image_bytes"] = data
        async for decoded in engine.application.source_factory(payload):
            yield replace(decoded, media_time_ns=media, sequence=sequence,
                          presentation_ts_ms=item.ts_ms, payload=payload)
        return
    # Metadata-only pipelines remain usable; image stages reject missing pixels.
    yield MediaFrame(None, media, sequence, payload.get("timestamp_provenance", "estimated_capture_interval"),
                     item.ts_ms, payload)


def require_durable_processing(outputs):
    for item in outputs:
        error = item.payload.get("incident_error")
        if error:
            raise RuntimeError(f"durable processing failed: {error}")


async def process_source(engine, pipeline, source):
    """Offline backpressure or one processing frame plus one live pending frame."""
    live = source.params.get("live", source.__class__.__name__ in {"SourceRTSP", "SourceMJPEG"})
    session = await engine.open_session(pipeline, stream_name=getattr(source, "stream_id", ""), owner_kind="worker")
    origin = time.monotonic_ns()

    def capture_time(frame):
        return (replace(frame, media_time_ns=time.monotonic_ns() - origin,
                        timestamp_provenance="estimated_capture_interval")
                if frame.media_time_ns is None else frame)
    try:
        if not live:
            sequence = 0
            async for item in source.run():
                async for frame in item_frames(engine, item, sequence=sequence):
                    require_durable_processing(await session.process(capture_time(frame)))
                    sequence = frame.sequence + 1
            return
        queue = asyncio.Queue(maxsize=1)
        stop = object()
        dropped = 0

        async def ingest():
            nonlocal dropped
            sequence = 0
            try:
                async for item in source.run():
                    async for frame in item_frames(engine, item, sequence=sequence):
                        frame = capture_time(frame)
                        frame.payload["live_mode"] = True
                        await session.observe(frame)
                        sequence = frame.sequence + 1
                        if queue.full():
                            queue.get_nowait()
                            dropped += 1
                        queue.put_nowait(frame)
            except asyncio.CancelledError:
                raise
            except BaseException:
                await queue.put(stop)
                raise
            else:
                await queue.put(stop)

        task = asyncio.create_task(ingest())
        try:
            while True:
                frame = await queue.get()
                if frame is stop:
                    await task  # Surface acquisition failures after draining pending frames.
                    break
                captured = frame.payload.get("_captured_monotonic_ns", time.monotonic_ns()) / 1e9
                if time.monotonic() - captured >= .5:
                    dropped += 1
                    continue
                frame.payload["_admission_deadline"] = captured + .5
                frame.payload["source_frames_dropped"] = dropped
                require_durable_processing(await session.process(frame, observed=True))
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    finally:
        await session.close("source_closed")
