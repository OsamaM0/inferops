"""Conditional routing stage (filters items)."""
from __future__ import annotations

from inferops.adapters.outbound.stages.base import Stage


class RouteBy(Stage):
    def __init__(self, when: str = "True") -> None:
        super().__init__(when=when)
        self.when = when
        self._code = compile(when, "<route_by>", "eval")

    async def process(self, item):
        env = {"item": item, "stream_id": item.stream_id, "results": item.results,
               "payload": item.payload, "seq": item.seq}
        try:
            ok = bool(eval(self._code, {"__builtins__": {}}, env))
        except Exception:
            ok = False
        return item if ok else None
