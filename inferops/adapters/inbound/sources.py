"""Bounded frame sources; media time is independent of processing speed."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

from inferops.domain.processing import MediaFrame
from inferops.domain.sources import FrameCadence


def _video_frames(path, every_n=1):
    import av

    from inferops.adapters.outbound.media.camera import bgr_array
    with av.open(str(path)) as container:
        container.streams.video[0].thread_count = 1
        for sequence, frame in enumerate(container.decode(video=0)):
            if sequence % every_n:
                continue
            if frame.pts is None or frame.time_base is None:
                rate = container.streams.video[0].average_rate
                if not rate:
                    raise ValueError("video frames require PTS or an advertised frame rate")
                media = int(sequence / float(rate) * 1e9)
                provenance = "estimated_frame_rate"
            else:
                media = int(frame.pts * frame.time_base * 1_000_000_000)
                provenance = "presentation_timestamp"
            yield MediaFrame(bgr_array(frame), media, sequence, provenance)


def _next(iterator):
    return next(iterator, None)


async def source_frames(source, *, every_n=1):
    if hasattr(source, "__aiter__"):
        async for frame in source:
            yield frame
        return
    if isinstance(source, MediaFrame):
        yield source
        return
    payload = dict(source) if isinstance(source, dict) else {}
    if "video_path" in payload:
        source = Path(payload["video_path"])
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            source = await asyncio.to_thread(path.read_bytes)
        else:
            frames = _video_frames(path, every_n)
            try:
                while True:
                    # Closing a generator while its native decoder is still
                    # running raises "generator already executing". Drain that
                    # native read before cancellation reaches the close path.
                    reading = asyncio.create_task(asyncio.to_thread(_next, frames))
                    try:
                        frame = await asyncio.shield(reading)
                    except asyncio.CancelledError:
                        await asyncio.gather(reading, return_exceptions=True)
                        raise
                    if frame is None:
                        break
                    yield frame
            finally:
                await asyncio.to_thread(frames.close)
            return
    if "image_base64" in payload:
        import base64
        payload["image_bytes"] = base64.b64decode(payload.pop("image_base64"), validate=True)
    if "image_np" in payload:
        source = payload.pop("image_np")
    elif "image_bytes" in payload:
        source = payload.pop("image_bytes")
    if isinstance(source, (bytes, bytearray)):
        import cv2
        import numpy as np
        source = cv2.imdecode(np.frombuffer(source, np.uint8), cv2.IMREAD_COLOR)
        if source is None:
            raise ValueError("invalid image")
    if hasattr(source, "shape"):
        provenance = "supplied" if "media_time_ns" in payload else "single_image"
        yield MediaFrame(source, payload.pop("media_time_ns", 0), payload.pop("seq", 0), provenance, payload=payload)
    elif isinstance(source, dict):
        provenance = "supplied" if "media_time_ns" in payload else "estimated_capture_interval"
        yield MediaFrame(None, payload.pop("media_time_ns", None), payload.pop("seq", 0), provenance, payload=payload)


async def camera_frames(url, *, fps=5, kind="rtsp", transport="tcp", timeout_s=5,
                        max_width=7680, max_height=4320):
    """Compatibility entry point using the same bounded reader as the registry."""
    from types import SimpleNamespace

    from inferops.adapters.outbound.media.camera import capture_frames
    spec = SimpleNamespace(uri=url, camera_id="legacy", kind=kind, inference_fps=fps or 120,
                           evidence_fps=None, rtsp_transport=transport, timeout_s=timeout_s,
                           max_width=max_width, max_height=max_height)
    frames = capture_frames(spec, {"decoded": 0, "dropped": 0})
    try:
        async for frame in frames:
            yield frame
    finally:
        await frames.aclose()


async def process_live_frames(session, frames, *, on_drop=None, on_result=None, inference_fps=None):
    """Observe capture independently of bounded latest-frame inference admission."""
    queue = asyncio.Queue(maxsize=1)
    stop = object()
    dropped = 0
    cadence = FrameCadence(inference_fps) if inference_fps is not None else None

    async def receive():
        nonlocal dropped
        try:
            async for frame in frames:
                received = time.monotonic()
                frame.payload["live_mode"] = True
                await session.observe(frame)
                if cadence and not cadence.due(frame.media_time_ns or 0, frame.payload.get("connection_generation")):
                    continue
                if queue.full():
                    queue.get_nowait()
                    dropped += 1
                    if on_drop:
                        on_drop()
                queue.put_nowait((frame, received))
        except asyncio.CancelledError:
            raise
        except BaseException:
            # A cancelled consumer cannot drain a full queue. Error signaling
            # must not block reader cleanup on that same consumer.
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(stop)
            raise
        else:
            await queue.put(stop)

    reader = asyncio.create_task(receive(), name=f"capture-{session.id}")
    try:
        while True:
            pending = await queue.get()
            if pending is stop:
                await reader
                return dropped
            frame, received = pending
            captured = frame.payload.get("_captured_monotonic_ns", int(received * 1e9)) / 1e9
            if time.monotonic() - captured >= .5:
                dropped += 1
                if on_drop:
                    on_drop()
                continue
            frame.payload["_admission_deadline"] = captured + .5
            frame.payload["source_frames_dropped"] = dropped
            outputs = await session.process(frame, observed=True)
            if frame.payload.get("_admission_expired"):
                dropped += 1
                if on_drop:
                    on_drop()
                continue
            if on_result:
                await on_result(outputs, time.monotonic() - received)
    finally:
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
        if hasattr(frames, "aclose"):
            await frames.aclose()
