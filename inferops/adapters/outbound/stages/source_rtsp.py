"""Compatibility stage over the shared bounded PyAV camera reader."""
from __future__ import annotations

from inferops.adapters.inbound.sources import camera_frames
from inferops.adapters.outbound.stages.base import SourceStage
from inferops.domain.types import StageItem, now_ms


class SourceRTSP(SourceStage):
    kind = "rtsp"

    def __init__(self, url: str, stream_id: str = "rtsp", fps: float = 0.0,
                 reconnect_s: float = 0.5, transport: str = "tcp",
                 latency_budget_ms: int = 100, timeout_s: float = 5, loop: bool = False):
        if fps < 0 or timeout_s <= 0 or reconnect_s <= 0 or transport not in {"tcp", "udp"}:
            raise ValueError("invalid camera rate, transport or timeout")
        super().__init__(url=url, stream_id=stream_id, fps=fps, reconnect_s=reconnect_s,
                         transport=transport, latency_budget_ms=latency_budget_ms,
                         timeout_s=timeout_s, loop=loop, live=True)
        self.url, self.stream_id, self.fps = url, stream_id, fps
        self.timeout_s, self.transport, self.reconnect_s = timeout_s, transport, reconnect_s
        self.loop = loop
        self.stats = {"decoded": 0, "dropped": 0}

    async def run(self):
        frames = camera_frames(self.url, fps=self.fps, kind=self.kind,
                               transport=self.transport, timeout_s=self.timeout_s,
                               loop=self.loop, stats=self.stats, stream_id=self.stream_id,
                               reconnect_s=self.reconnect_s)
        try:
            async for frame in frames:
                yield StageItem(stream_id=self.stream_id, ts_ms=now_ms(), seq=frame.sequence,
                                payload={**frame.payload, "image_np": frame.image,
                                         "width": frame.image.shape[1], "height": frame.image.shape[0],
                                         "media_time_ns": frame.media_time_ns,
                                         "timestamp_provenance": frame.timestamp_provenance})
        finally:
            await frames.aclose()
