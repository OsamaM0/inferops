"""Driver-behaviour frame annotator.

Draws detection class colours and face-mesh overlays on BGR frames.
Registered as a ``FrameAnnotator`` plugin and wired into ``sink_s3``
via the pipeline YAML:

    - name: evidence
      stage: sink_s3
      params:
        evidence_mode: both
        annotator: examples.driver_behaviour.plugins.frame_annotator:DriverAnnotator
"""
from __future__ import annotations

import cv2
import numpy as np

from inferops.core.types import StageItem
from inferops.plugins.base import FrameAnnotator

# ── Detection class colours (BGR) ─────────────────────────────────────────
_CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "seatbelt":  ( 50, 200,  50),   # green
    "phone":     ( 40,  40, 220),   # red
    "cigarette": (  0, 130, 255),   # orange
    "foodItem":  (  0, 210, 210),   # yellow
    "person":    (220, 150,  40),   # blue
    "wheel":     (180, 180,  60),   # olive
}
_DEFAULT_COLOR: tuple[int, int, int] = (200, 80, 200)   # purple


def _face_color(label: str) -> tuple[int, int, int]:
    if "Drowsy" in label:
        return (60, 60, 220)        # red
    if "Looking" in label:
        return (0, 180, 255)        # amber
    if "Eating" in label or "Smoking" in label:
        return (0, 130, 255)        # orange
    return (212, 244, 125)          # lime (normal)


class DriverAnnotator(FrameAnnotator):
    """Draws per-class coloured detection boxes and the face-mesh overlay."""

    def __call__(self, frame: np.ndarray, item: StageItem) -> np.ndarray:
        h, w = frame.shape[:2]

        # ── re-colour detection boxes with domain-specific palette ────────
        for result in (item.results or []):
            for det in (result.detections or []):
                color = _CLASS_COLORS.get(det.label, _DEFAULT_COLOR)
                px = (det.attrs or {}).get("xyxy_pixels")
                if px and len(px) == 4:
                    x1, y1, x2, y2 = (int(v) for v in px)
                else:
                    x1 = int(det.box[0] * w)
                    y1 = int(det.box[1] * h)
                    x2 = int(det.box[2] * w)
                    y2 = int(det.box[3] * h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{det.label} {det.score:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                by = max(y1 - 4, th + 6)
                cv2.rectangle(frame, (x1, by - th - 4), (x1 + tw + 4, by), color, cv2.FILLED)
                cv2.putText(frame, label, (x1 + 2, by - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # ── face-mesh overlay ─────────────────────────────────────────────
        face_box = item.payload.get("face_box")
        if isinstance(face_box, (list, tuple)) and len(face_box) == 4:
            fx1 = int(float(face_box[0]) * w)
            fy1 = int(float(face_box[1]) * h)
            fx2 = int(float(face_box[2]) * w)
            fy2 = int(float(face_box[3]) * h)
            face_label = str(item.payload.get("face_status_label") or "Driver Face")
            color = _face_color(face_label)
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), color, 2)
            (tw, th), _ = cv2.getTextSize(face_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            by = max(fy1 - 4, th + 8)
            cv2.rectangle(frame, (fx1, by - th - 6), (fx1 + tw + 8, by), color, cv2.FILLED)
            cv2.putText(frame, face_label, (fx1 + 4, by - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

        return frame
