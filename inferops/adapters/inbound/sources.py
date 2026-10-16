"""Bounded frame sources; media time is independent of processing speed."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

from inferops.domain.processing import MediaFrame


def _video_frames(path, every_n=1):
    import av
    with av.open(str(path)) as container:
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
            yield MediaFrame(frame.to_ndarray(format="bgr24"), media, sequence, provenance)


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
                    frame = await asyncio.to_thread(_next, frames)
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


async def camera_frames(url, *, fps=5, kind="rtsp"):
    import cv2
    capture = await asyncio.to_thread(cv2.VideoCapture, url)
    origin = time.monotonic_ns()
    sequence = 0
    try:
        while True:
            ok, image = await asyncio.to_thread(capture.read)
            if not ok:
                raise ValueError("camera source disconnected")
            yield MediaFrame(image, time.monotonic_ns() - origin, sequence, "estimated_capture_interval")
            sequence += 1
            if fps:
                await asyncio.sleep(1 / fps)
    finally:
        await asyncio.to_thread(capture.release)


async def process_live_frames(session, frames, *, on_drop=None, on_result=None):
    """Observe capture independently of bounded latest-frame inference admission."""
    queue = asyncio.Queue(maxsize=1)
    stop = object()
    dropped = 0

    async def receive():
        nonlocal dropped
        try:
            async for frame in frames:
                received = time.monotonic()
                frame.payload["live_mode"] = True
                await session.observe(frame)
                if queue.full():
                    queue.get_nowait()
                    dropped += 1
                    if on_drop:
                        on_drop()
                queue.put_nowait((frame, received))
        except asyncio.CancelledError:
            raise
        except BaseException:
            await queue.put(stop)
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
            frame.payload["source_frames_dropped"] = dropped
            outputs = await session.process(frame, observed=True)
            if on_result:
                await on_result(outputs, time.monotonic() - received)
    finally:
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
