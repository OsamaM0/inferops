"""Line-crossing and zone counters for tracked objects (normalised coordinates).

Configuration (stage parameter ``counting``), per camera id with a default::

    counting:
      default: {}
      cameras:
        cam-1:
          line: {points: [[0.0, 0.55], [1.0, 0.55]], invert: false, margin: 0.01}
          zone: {points: [[0.1, 0.2], [0.9, 0.2], [0.9, 0.9], [0.1, 0.9]]}

**Line.** IN means crossing from the left-hand to the right-hand side of the
arrow A -> B as seen on the image (y grows downwards): for a horizontal line
drawn left to right, moving *down* the image is IN. ``invert: true`` swaps IN
and OUT. A track changes side only after its centre is further than
``margin`` from the line (hysteresis), and only crossings within the segment
count. Sides alternate, so jitter cannot produce repeated INs, and a track
first seen beyond the line is not counted until it crosses.

**Zone.** A polygon. Each track counts as one entry the first time its centre
is inside; occupancy is the number of tracks inside right now.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

Point = tuple[float, float]


def _points(value: Any, minimum: int, what: str) -> tuple[Point, ...]:
    points = value.get("points") if isinstance(value, Mapping) else value
    if not isinstance(points, (list, tuple)) or len(points) < minimum:
        raise ValueError(f"counting {what} needs at least {minimum} [x, y] points")
    out = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"counting {what} point {point!r} is not [x, y]")
        x, y = (float(v) for v in point)
        if not (-0.5 <= x <= 1.5 and -0.5 <= y <= 1.5):
            raise ValueError(f"counting {what} point {point!r} is not normalised (0..1)")
        out.append((x, y))
    return tuple(out)


def _cross(a: Point, b: Point, p: Point) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


@dataclass(frozen=True, slots=True)
class LineSpec:
    a: Point
    b: Point
    invert: bool = False
    margin: float = 0.01

    @classmethod
    def from_config(cls, value: Mapping[str, Any]) -> LineSpec:
        a, b = _points(value, 2, "line")[:2]
        if a == b:
            raise ValueError("counting line endpoints must differ")
        margin = float(value.get("margin", 0.01)) if isinstance(value, Mapping) else 0.01
        invert = bool(value.get("invert", False)) if isinstance(value, Mapping) else False
        return cls(a, b, invert, max(0.0, margin))

    @property
    def length(self) -> float:
        return ((self.b[0] - self.a[0]) ** 2 + (self.b[1] - self.a[1]) ** 2) ** 0.5

    def signed_distance(self, p: Point) -> float:
        """> 0 on the right-hand side of A -> B (image coordinates)."""
        return _cross(self.a, self.b, p) / self.length

    def crossed(self, p: Point, q: Point) -> bool:
        """Does the movement p -> q intersect the segment A-B?"""
        d1, d2 = _cross(self.a, self.b, p), _cross(self.a, self.b, q)
        d3, d4 = _cross(p, q, self.a), _cross(p, q, self.b)
        return d1 * d2 <= 0 and d3 * d4 <= 0


class LineCounter:
    def __init__(self, spec: LineSpec) -> None:
        self.spec = spec
        self.count_in = 0
        self.count_out = 0
        self._side: dict[int, tuple[int, Point]] = {}   # track -> (side, last point on that side)

    def update(self, track_id: int, point: Point) -> str | None:
        """Record a track position; returns "in"/"out" when it crossed, else None."""
        distance = self.spec.signed_distance(point)
        if abs(distance) <= self.spec.margin:
            return None
        side = 1 if distance > 0 else -1
        previous = self._side.get(track_id)
        self._side[track_id] = (side, point)
        if previous is None or previous[0] == side or not self.spec.crossed(previous[1], point):
            return None
        entered = side > 0
        if self.spec.invert:
            entered = not entered
        return "in" if entered else "out"

    def apply(self, event: str) -> None:
        if event == "in":
            self.count_in += 1
        elif event == "out":
            self.count_out += 1

    def forget(self, track_id: int) -> None:
        self._side.pop(track_id, None)


class ZoneCounter:
    def __init__(self, points: tuple[Point, ...]) -> None:
        self.points = points
        self.entries = 0
        self._seen: set[int] = set()

    def contains(self, p: Point) -> bool:
        """Ray casting point-in-polygon."""
        inside = False
        x, y = p
        n = len(self.points)
        for i in range(n):
            (x1, y1), (x2, y2) = self.points[i], self.points[(i + 1) % n]
            if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
        return inside

    def update(self, track_id: int, point: Point) -> bool:
        """True the first time this track is inside the zone (one entry per track)."""
        if track_id in self._seen or not self.contains(point):
            return False
        self._seen.add(track_id)
        return True

    def apply(self) -> None:
        self.entries += 1

    def forget(self, track_id: int) -> None:
        self._seen.discard(track_id)


@dataclass(frozen=True, slots=True)
class CountingGeometry:
    line: LineSpec | None = None
    zone: tuple[Point, ...] | None = None

    @classmethod
    def from_config(cls, value: Mapping[str, Any] | None) -> CountingGeometry:
        value = value or {}
        if not isinstance(value, Mapping):
            raise ValueError("counting geometry must be a mapping with optional line/zone")
        unknown = set(value) - {"line", "zone"}
        if unknown:
            raise ValueError(f"unknown counting keys {sorted(unknown)}; use line and zone")
        line = LineSpec.from_config(value["line"]) if value.get("line") else None
        zone = _points(value["zone"], 3, "zone") if value.get("zone") else None
        return cls(line, zone)

    def describe(self) -> dict[str, Any]:
        """JSON-friendly geometry for overlays."""
        out: dict[str, Any] = {}
        if self.line:
            out["line"] = {"points": [list(self.line.a), list(self.line.b)], "invert": self.line.invert}
        if self.zone:
            out["zone"] = {"points": [list(p) for p in self.zone]}
        return out


LINE_DIRECTIONS = {"horizontal": ("down", "up"), "vertical": ("left", "right")}


def simple_line(orientation: str, position: float, direction: str) -> dict[str, Any] | None:
    """A full-width (or full-height) counting line from three simple settings.

    ``orientation`` is horizontal, vertical or off; ``position`` is the line's
    y (horizontal) or x (vertical) in 0..1; ``direction`` is the IN movement:
    down/up for a horizontal line, left/right for a vertical one.
    """
    if orientation == "off":
        return None
    if orientation not in LINE_DIRECTIONS:
        raise ValueError(f"counting line orientation must be horizontal, vertical or off, not {orientation!r}")
    position = float(position)
    if not 0.0 <= position <= 1.0:
        raise ValueError("counting line position must be between 0 and 1")
    natural, inverted = LINE_DIRECTIONS[orientation]
    if direction not in (natural, inverted):
        raise ValueError(f"a {orientation} counting line needs IN direction {natural} or {inverted}, not {direction!r}")
    # Left-to-right horizontal: its right-hand side is down. Top-to-bottom vertical: left.
    points = [[0.0, position], [1.0, position]] if orientation == "horizontal" else [[position, 0.0], [position, 1.0]]
    return {"points": points, "invert": direction == inverted}


def parse_counting(config: Mapping[str, Any] | None) -> tuple[CountingGeometry, dict[str, CountingGeometry]]:
    """(default geometry, per-camera geometry) from the ``counting`` parameter."""
    config = config or {}
    if not isinstance(config, Mapping):
        raise ValueError("counting must be a mapping with default and cameras")
    unknown = set(config) - {"default", "cameras"}
    if unknown:
        raise ValueError(f"unknown counting keys {sorted(unknown)}; use default and cameras")
    cameras = config.get("cameras") or {}
    if not isinstance(cameras, Mapping):
        raise ValueError("counting.cameras must map camera ids to line/zone settings")
    return (CountingGeometry.from_config(config.get("default")),
            {str(camera): CountingGeometry.from_config(value) for camera, value in cameras.items()})
