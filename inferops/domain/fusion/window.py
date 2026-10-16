"""Rolling windows + reducers (in-memory)."""
from __future__ import annotations

from collections import deque
from typing import Any

from inferops.domain.fusion.spec import WindowSpec


class Window:
    """A rolling window for a single signal+entity, with named reducers."""

    def __init__(self, spec: WindowSpec) -> None:
        self.spec = spec
        self._buf: deque[tuple[int, float]] = deque()
        self._ewma: float | None = None

    def add(self, ts_ms: int, value: float) -> None:
        if self.spec.kind == "ewma":
            a = float(self.spec.alpha or 0.3)
            self._ewma = value if self._ewma is None else (a * value + (1 - a) * self._ewma)
            return
        self._buf.append((ts_ms, float(value)))
        self._prune(ts_ms)

    def _prune(self, ts_ms: int) -> None:
        if self.spec.kind == "rolling_time" and self.spec.duration_ms:
            cutoff = ts_ms - self.spec.duration_ms
            while self._buf and self._buf[0][0] < cutoff:
                self._buf.popleft()
        elif self.spec.kind == "rolling_count" and self.spec.count:
            while len(self._buf) > self.spec.count:
                self._buf.popleft()

    def _values(self) -> list[float]:
        return [v for _, v in self._buf]

    # exposed methods (called by expression DSL: e.g. phone_score.fraction_above(0.4))
    def latest(self) -> float | None:
        if self.spec.kind == "ewma":
            return self._ewma
        return self._buf[-1][1] if self._buf else None

    def mean(self) -> float | None:
        if self.spec.kind == "ewma":
            return self._ewma
        vs = self._values()
        return sum(vs) / len(vs) if vs else None

    def max(self) -> float | None:
        vs = self._values()
        return max(vs) if vs else None

    def min(self) -> float | None:
        vs = self._values()
        return min(vs) if vs else None

    def count(self) -> int:
        return len(self._buf)

    def fraction_above(self, threshold: float) -> float:
        vs = self._values()
        if not vs:
            return 0.0
        return sum(1 for v in vs if v >= threshold) / len(vs)

    def fraction_below(self, threshold: float) -> float:
        vs = self._values()
        if not vs:
            return 0.0
        return sum(1 for v in vs if v < threshold) / len(vs)

    def slope(self) -> float:
        if len(self._buf) < 2:
            return 0.0
        (t0, v0), (t1, v1) = self._buf[0], self._buf[-1]
        dt = (t1 - t0) / 1000.0
        return (v1 - v0) / dt if dt > 0 else 0.0

    def derivative(self) -> float:
        return self.slope()

    def asdict(self) -> dict[str, Any]:
        return {
            "latest": self.latest(), "mean": self.mean(),
            "max": self.max(), "min": self.min(), "count": self.count(),
        }
