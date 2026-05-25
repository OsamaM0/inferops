"""WebSocket fan-out sink (low-latency, orjson, binary frames)."""
from __future__ import annotations

import numpy as np
import orjson

from inferops.pipeline.stage import SinkStage

# Types that cannot be JSON-serialised and must be stripped from the payload
# before sending over WebSocket (numpy arrays serialised to lists can be
# hundreds of MB for a single HD frame).
_STRIP_TYPES = (np.ndarray, bytes, bytearray)


def _slim_result(r) -> dict:
    """Hand-built dict — avoids pydantic.model_dump() overhead per result."""
    out = {"kind": getattr(r, "kind", None), "model": getattr(r, "model", None)}
    dets = getattr(r, "detections", None)
    if dets:
        out["detections"] = [
            {
                "label": d.label,
                "score": float(d.score),
                "box": list(d.box) if d.box is not None else None,
                "attrs": getattr(d, "attrs", None) or {},
            }
            for d in dets
        ]
    cls = getattr(r, "classifications", None)
    if cls:
        out["classifications"] = [
            {"label": c.label, "score": float(c.score)} for c in cls
        ]
    return out


class _Hub:
    def __init__(self) -> None:
        self.clients: dict[str, set] = {}   # channel -> set[WebSocket]

    def subscribe(self, channel: str, ws) -> None:
        self.clients.setdefault(channel, set()).add(ws)

    def unsubscribe(self, channel: str, ws) -> None:
        if channel in self.clients:
            self.clients[channel].discard(ws)

    async def publish_bytes(self, channel: str, data: bytes) -> None:
        dead = []
        for ws in list(self.clients.get(channel, set())):
            try:
                await ws.send_bytes(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.unsubscribe(channel, ws)


HUB = _Hub()


class SinkWS(SinkStage):
    def __init__(self, channel: str = "default") -> None:
        super().__init__(channel=channel)
        self.channel = channel

    async def process(self, item):
        ch = self.channel.format(stream_id=item.stream_id) if "{" in self.channel else self.channel
        if not self.clients_for(ch):
            return None
        msg = {
            "id": item.id,
            "stream_id": item.stream_id,
            "seq": item.seq,
            "ts_ms": item.ts_ms,
            "frame_ref": item.frame_ref,
            "results": [_slim_result(r) for r in item.results],
        }
        # keep only json-safe payload entries (typically alert/fusion/track meta)
        payload = {k: v for k, v in item.payload.items()
                   if not isinstance(v, _STRIP_TYPES) and k != "live_mode"}
        if payload:
            msg["payload"] = payload
        data = orjson.dumps(msg)
        await HUB.publish_bytes(ch, data)
        return None

    def clients_for(self, channel: str) -> int:
        return len(HUB.clients.get(channel, ()))
