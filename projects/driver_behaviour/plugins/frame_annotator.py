"""Driver-behaviour frame annotator.

Draws detection class colours and face-mesh overlays on BGR frames.
Registered as a ``FrameAnnotator`` plugin and wired into ``sink_s3``
via the pipeline YAML:

    - name: evidence
      stage: sink_s3
      params:
        evidence_mode: both
        annotator: .plugins.frame_annotator:DriverAnnotator
"""

from __future__ import annotations

import cv2
import numpy as np

from inferops.core.types import StageItem
from inferops.plugins.base import FrameAnnotator

# ── Base detection class colours (BGR) ────────────────────────────────────
# Orange  → suspicious / dangerous items that need monitoring
# Green   → compliance / safety items
_CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "seatbelt": (40, 200, 50),  # green  - worn seatbelt is safe
    "phone": (0, 140, 255),  # orange - phone present (possibly alerting)
    "cigarette": (0, 140, 255),  # orange - cigarette present
    "foodItem": (0, 140, 255),  # orange - food / drink present
    "person": (80, 200, 200),  # teal
    "wheel": (40, 180, 40),  # green  - hands on wheel is safe
}
_COLOR_ALERT: tuple[int, int, int] = (0, 0, 220)  # red   - actively alerting
_DEFAULT_COLOR: tuple[int, int, int] = (200, 80, 200)  # purple

# Human-readable banners for each fusion / alert type
_ALERT_LABELS: dict[str, str] = {
    "phone_use": "MOBILE PHONE USE",
    "no_seatbelt": "NO SEATBELT",
    "smoking": "SMOKING",
    "eating": "EATING / DRINKING",
    "distraction": "DRIVER DISTRACTED",
    "drowsiness": "DROWSINESS",
    "hands_off_wheel": "HANDS OFF WHEEL",
}


def _danger_classes(item: StageItem) -> set[str]:
    """
    Return detection-class names that are ACTIVELY in an alert state and
    should be drawn with a red bounding box.

    All thresholds are evaluated directly on per-frame signals — they are
    not throttled by alert / fusion cooldowns, so the red box appears
    instantly when the condition becomes true.
    """
    danger: set[str] = set()

    # ── phone: any detection above confidence threshold ────────────────
    for r in item.results or []:
        if any(d.get("score", 0.0) >= 0.45 for d in (r.raw or {}).get("by_class", {}).get("phone", [])):
            danger.add("phone")
            break

    # ── cigarette: face-mesh confirms mouth intersection ───────────────
    if (item.payload.get("smoking_face_score") or 0.0) >= 0.15:
        danger.add("cigarette")
    else:
        # Fallback: very high-confidence raw detection with no face mesh
        for r in item.results or []:
            if any(
                d.get("score", 0.0) >= 0.60 for d in (r.raw or {}).get("by_class", {}).get("cigarette", [])
            ):
                danger.add("cigarette")
                break

    # ── foodItem: face-mesh confirms mouth region intersection ─────────
    if (item.payload.get("eating_face_score") or 0.0) >= 0.40:
        danger.add("foodItem")

    # ── wheel: hands detected off the steering wheel ───────────────────
    hand_on_wheel = item.payload.get("hand_on_wheel")
    if hand_on_wheel is not None and hand_on_wheel < 0.5:
        danger.add("wheel")

    return danger


def _active_alert_lines(item: StageItem) -> list[str]:
    """
    Build a deduplicated list of human-readable alert descriptions to show
    in the on-screen banner. Sources: face_status_label + fusion_events +
    stored alerts.
    """
    seen: set[str] = set()
    lines: list[str] = []

    def _add(text: str) -> None:
        if text not in seen:
            seen.add(text)
            lines.append(text)

    face_label = str(item.payload.get("face_status_label") or "")
    if "Drowsy" in face_label:
        _add("DROWSINESS DETECTED")
    if "Looking" in face_label:
        _add(face_label.upper())
    if "Eating" in face_label:
        _add("EATING / DRINKING")
    if "Smoking" in face_label:
        _add("SMOKING")

    for ev in item.payload.get("fusion_events") or []:
        label = _ALERT_LABELS.get(ev.get("type", ""), ev.get("type", "").upper().replace("_", " "))
        _add(label)
    for a in item.payload.get("alerts") or []:
        label = _ALERT_LABELS.get(a.get("type", ""), a.get("type", "").upper().replace("_", " "))
        _add(label)

    return lines


def _face_color(label: str) -> tuple[int, int, int]:
    if "Drowsy" in label:
        return (0, 0, 220)  # red
    if "Looking" in label:
        return (0, 180, 255)  # amber
    if "Eating" in label or "Smoking" in label:
        return (0, 130, 255)  # orange
    return (212, 244, 125)  # lime - normal / safe


class DriverAnnotator(FrameAnnotator):
    """Draws per-class coloured detection boxes and the face-mesh overlay."""

    def __call__(self, frame: np.ndarray, item: StageItem) -> np.ndarray:
        h, w = frame.shape[:2]
        danger = _danger_classes(item)

        # ── per-class detection boxes ──────────────────────────────────────
        for result in item.results or []:
            for det in result.detections or []:
                # RED when the class is actively alerting, otherwise class default
                if det.label in danger:
                    color = _COLOR_ALERT
                elif det.label == "seatbelt":
                    # Low-confidence seatbelt → orange warning; high → green safe
                    color = _CLASS_COLORS["seatbelt"] if det.score >= 0.4 else (0, 140, 255)
                else:
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
                label_text = f"{det.label} {det.score:.2f}"
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                by = max(y1 - 4, th + 6)
                cv2.rectangle(frame, (x1, by - th - 4), (x1 + tw + 4, by), color, cv2.FILLED)
                cv2.putText(
                    frame,
                    label_text,
                    (x1 + 2, by - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

        # ── face-mesh overlay ──────────────────────────────────────────────
        face_box = item.payload.get("face_box")
        if isinstance(face_box, (list, tuple)) and len(face_box) == 4:
            fx1 = int(float(face_box[0]) * w)
            fy1 = int(float(face_box[1]) * h)
            fx2 = int(float(face_box[2]) * w)
            fy2 = int(float(face_box[3]) * h)
            face_label = str(item.payload.get("face_status_label") or "Driver Face")
            fcolor = _face_color(face_label)
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), fcolor, 2)
            (tw, th), _ = cv2.getTextSize(face_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            by = max(fy1 - 4, th + 8)
            cv2.rectangle(frame, (fx1, by - th - 6), (fx1 + tw + 8, by), fcolor, cv2.FILLED)
            cv2.putText(
                frame,
                face_label,
                (fx1 + 4, by - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # ── alert overlay: red border + top banner ─────────────────────────
        alert_lines = _active_alert_lines(item)
        face_label_active = str(item.payload.get("face_status_label") or "")
        is_alerting = (
            bool(danger)
            or any(kw in face_label_active for kw in ("Drowsy", "Looking", "Eating", "Smoking"))
            or bool(alert_lines)
        )

        if is_alerting:
            # Thick red border around the entire frame
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), _COLOR_ALERT, 4)

        if alert_lines:
            # Semi-transparent dark-red banner at the top
            line_h = 26
            banner_h = len(alert_lines) * line_h + 10
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 160), cv2.FILLED)
            cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)
            for i, line in enumerate(alert_lines):
                cv2.putText(
                    frame,
                    line,
                    (8, (i + 1) * line_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        return frame
