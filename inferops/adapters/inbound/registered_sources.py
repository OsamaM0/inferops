"""Per-source decoders: paced files and latest-frame, reconnecting RTSP readers."""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace

from inferops.adapters.inbound.sources import source_frames
from inferops.adapters.outbound.media.camera import _resize, capture_frames


async def registered_frames(spec, stats):
    if spec.kind == "rtsp":
        frames = capture_frames(spec, stats)
        try:
            async for frame in frames:
                yield frame
        finally:
            await frames.aclose()
        return
    sequence = 0
    timeline_offset = 0
    while True:
        origin = time.monotonic()
        first_pts = None
        last_media = 0
        next_sample = 0
        count = 0
        frames = source_frames(spec.uri)
        try:
            async for frame in frames:
                stats["decoded"] += 1
                stats["source_width"], stats["source_height"] = frame.image.shape[1], frame.image.shape[0]
                pts = frame.media_time_ns or 0
                if first_pts is None:
                    first_pts = pts
                relative = pts - first_pts
                if relative < next_sample:
                    continue
                next_sample += int(1e9 / max(spec.inference_fps, spec.evidence_fps or 0))
                # Pace file replay while preserving PTS for temporal rules.
                await asyncio.sleep(max(0, origin + relative / 1e9 - time.monotonic()))
                image = await asyncio.to_thread(_resize, frame.image, spec)
                last_media = timeline_offset + relative
                yield replace(
                    frame,
                    image=image,
                    sequence=sequence,
                    media_time_ns=last_media,
                    payload={**frame.payload, "camera_id": spec.camera_id},
                )
                sequence += 1
                count += 1
        finally:
            await frames.aclose()
        if not spec.loop or not count:
            return
        timeline_offset = last_media + int(1e9 / spec.inference_fps)


