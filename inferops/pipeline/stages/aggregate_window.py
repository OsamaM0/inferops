"""Rolling-window aggregator (groups by stream_id)."""
from __future__ import annotations

import time
from collections import defaultdict, deque

from inferops.pipeline.stage import Stage


class AggregateWindow(Stage):
    def __init__(self, window_ms: int = 5000, key: str = "stream_id") -> None:
        super().__init__(window_ms=window_ms, key=key)
        self.window_ms = window_ms
        self.key = key
        self._buf: dict[str, deque] = defaultdict(deque)

    async def process(self, item):
        k = getattr(item, self.key, "") or "default"
        now = item.ts_ms or int(time.time() * 1000)
        q = self._buf[k]
        q.append((now, item))
        cutoff = now - self.window_ms
        while q and q[0][0] < cutoff:
            q.popleft()
        item.payload["window_size"] = len(q)
        return item
