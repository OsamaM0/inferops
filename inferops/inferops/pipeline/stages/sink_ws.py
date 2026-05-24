"""WebSocket fan-out sink."""
from __future__ import annotations

import numpy as np

from inferops.pipeline.stage import SinkStage

# Types that cannot be JSON-serialised and must be stripped from the payload
# before sending over WebSocket (numpy arrays serialised to lists can be
# hundreds of MB for a single HD frame).
_STRIP_TYPES = (np.ndarray, bytes, bytearray)


class _Hub:
    def __init__(self) -> None:
        self.clients: dict[str, set] = {}   # channel -> set[WebSocket]

    def subscribe(self, channel: str, ws) -> None:
        self.clients.setdefault(channel, set()).add(ws)

    def unsubscribe(self, channel: str, ws) -> None:
        if channel in self.clients:
            self.clients[channel].discard(ws)

    async def publish(self, channel: str, payload) -> None:
        dead = []
        for ws in list(self.clients.get(channel, set())):
            try:
                await ws.send_json(payload)
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
        # Serialise everything except raw image arrays / bytes so the JSON
        # payload stays small (results, metadata, stream_id, ts_ms, …).
        data = item.model_dump(mode="json", exclude={"payload"})
        data["payload"] = {k: v for k, v in item.payload.items()
                           if not isinstance(v, _STRIP_TYPES)}
        await HUB.publish(ch, data)
        return None
