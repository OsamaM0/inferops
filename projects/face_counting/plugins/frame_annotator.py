"""Face counting overlay for evidence clips and previews.

Draws each tracked face (green with its running number once counted, amber
while still tentative, magenta "NEW #n" for a few seconds after it is counted),
the counting line with its IN arrow, the zone polygon, a "NEW PERSON" banner
and a counter panel. The camera preview and incident evidence both use it; it
is wired into the evidence sink in ``project.yaml``::

    - {name: evidence, stage: evidence, params: {annotator: .plugins.frame_annotator:FaceCountAnnotator}}
"""
from __future__ import annotations

import cv2
import numpy as np

from inferops.domain.types import StageItem
from inferops.ports.extensions import FrameAnnotator

_COUNTED = (60, 200, 60)      # BGR green
_TENTATIVE = (0, 190, 255)    # amber
_NEW = (255, 0, 255)          # magenta: just counted
_LINE = (255, 200, 0)         # cyan
_ZONE = (255, 120, 200)       # violet
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _px(point, w: int, h: int) -> tuple[int, int]:
    return round(float(point[0]) * w), round(float(point[1]) * h)


def _label(frame: np.ndarray, text: str, origin: tuple[int, int], color) -> None:
    (tw, th), _ = cv2.getTextSize(text, _FONT, 0.5, 1)
    x, y = origin
    y = max(y, th + 6)
    cv2.rectangle(frame, (x, y - th - 6), (x + tw + 6, y), color, cv2.FILLED)
    cv2.putText(frame, text, (x + 3, y - 4), _FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


class FaceCountAnnotator(FrameAnnotator):
    def __call__(self, frame: np.ndarray, item: StageItem) -> np.ndarray:
        h, w = frame.shape[:2]
        payload = item.payload or {}
        geometry = payload.get("counting_geometry") or {}

        zone = (geometry.get("zone") or {}).get("points")
        if zone:
            polygon = np.array([_px(p, w, h) for p in zone], np.int32)
            shade = frame.copy()
            cv2.fillPoly(shade, [polygon], _ZONE)
            cv2.addWeighted(shade, 0.15, frame, 0.85, 0, dst=frame)
            cv2.polylines(frame, [polygon], True, _ZONE, 2, cv2.LINE_AA)

        line = geometry.get("line") or {}
        if line.get("points"):
            a, b = (_px(p, w, h) for p in line["points"][:2])
            cv2.line(frame, a, b, _LINE, 2, cv2.LINE_AA)
            # IN arrow: from the line's middle towards its right-hand side (flipped by invert).
            dx, dy = b[0] - a[0], b[1] - a[1]
            length = max((dx * dx + dy * dy) ** 0.5, 1.0)
            sign = -1.0 if line.get("invert") else 1.0
            mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
            tip = (int(mid[0] - sign * dy / length * 40), int(mid[1] + sign * dx / length * 40))
            cv2.arrowedLine(frame, mid, tip, _LINE, 2, cv2.LINE_AA, tipLength=0.35)
            _label(frame, "IN", (tip[0] + 4, tip[1] + 8), _LINE)

        fresh = []
        for track in payload.get("face_tracks") or []:
            x1, y1 = _px(track["box"][:2], w, h)
            x2, y2 = _px(track["box"][2:], w, h)
            number = track.get("number")
            if track.get("new"):
                color, text, thickness = _NEW, f"NEW #{number}", 3
                fresh.append(number)
            elif track.get("counted"):
                color, text, thickness = _COUNTED, f"#{number}" if number else f"ID {track['track_id']}", 2
            else:
                color, text, thickness = _TENTATIVE, f"ID {track['track_id']}", 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            _label(frame, text, (x1, y1 - 2), color)

        if fresh:
            banner = "NEW PERSON " + ", ".join(f"#{n}" for n in sorted(n for n in fresh if n))
            (tw, th), _ = cv2.getTextSize(banner, _FONT, 0.7, 2)
            x = max(0, (w - tw) // 2)
            cv2.rectangle(frame, (x - 8, h - th - 22), (x + tw + 8, h - 6), _NEW, cv2.FILLED)
            cv2.putText(frame, banner, (x, h - 14), _FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        rows = [f"In view: {payload.get('faces_in_frame', 0)}", f"Unique: {payload.get('unique_faces', 0)}"]
        if "line_in" in payload:
            rows.append(f"IN: {payload.get('line_in', 0)}  OUT: {payload.get('line_out', 0)}")
        if "zone_entries" in payload:
            rows.append(f"Zone: {payload.get('zone_occupancy', 0)} now, {payload.get('zone_entries', 0)} entered")
        width = max(cv2.getTextSize(row, _FONT, 0.6, 2)[0][0] for row in rows) + 20
        panel = frame[0:22 * len(rows) + 14, 0:min(width, w)]
        panel[:] = (panel * 0.35).astype(frame.dtype)
        for i, row in enumerate(rows):
            cv2.putText(frame, row, (10, 26 + 22 * i), _FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return frame
