"""Eye open/closed state, gaze-down disambiguation and PERCLOS.

Closure per eye fuses two independent cues:

* the MediaPipe blendshape ``eyeBlink*`` score (learned, robust to pose), and
* a geometric eye-aspect ratio normalised by the driver's own open-eye EAR,
  learned online, so narrow and wide eyes are judged against themselves.

Seen from a cab camera, eyes looking down at a phone *look* closed. The
blendshape model separates the two: ``eyeLookDown*`` is high for lowered gaze
(0.6-0.8 on the sample phone video) and low when looking ahead (0.1-0.3).
Lowered lids with a down gaze are reported as gaze-down (distraction), not as
closed eyes (drowsiness); both alarm. The sample videos contain no genuine
microsleep, so ``gaze_down_look_strong`` is the knob to revisit with such data.

"Left"/"right" are anatomical: the driver's left eye is on the image right.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

# MediaPipe FaceBlendshapesGraph landmark subset (kLandmarksSubsetIdxs), fed in
# image-pixel coordinates; verified bit-exact against MediaPipe's FaceLandmarker.
BLENDSHAPE_LANDMARKS = (
    0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67,
    70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148,
    149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191,
    195, 197, 234, 246, 249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297,
    300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374,
    375, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415,
    454, 466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477,
)
EYE_BLINK_LEFT, EYE_BLINK_RIGHT = 9, 10
EYE_LOOK_DOWN_LEFT, EYE_LOOK_DOWN_RIGHT = 11, 12

# Six-point EAR rings: corner, two upper lid points, corner, two lower lid points.
LEFT_EYE_RING = (263, 387, 385, 362, 380, 373)   # driver's left eye (image right)
RIGHT_EYE_RING = (33, 160, 158, 133, 153, 144)   # driver's right eye (image left)


def eye_aspect_ratio(points: np.ndarray, ring: tuple[int, ...]) -> float:
    p1, p2, p3, p4, p5, p6 = points[list(ring), :2]
    width = float(np.linalg.norm(p1 - p4))
    return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * width + 1e-6))


def blendshape_points(points: np.ndarray) -> np.ndarray:
    """(1, 146, 2) float32 input for the blendshape model, in image pixels."""
    return points[list(BLENDSHAPE_LANDMARKS), :2][np.newaxis].astype(np.float32)


@dataclass(slots=True)
class EyeObservation:
    ear_left: float
    ear_right: float
    blink_left: float | None = None
    blink_right: float | None = None
    look_down: float | None = None
    width_left_px: float | None = None    # eye corner-to-corner width in frame pixels
    width_right_px: float | None = None


def eye_width(points: np.ndarray, ring: tuple[int, ...]) -> float:
    return float(np.linalg.norm(points[ring[0], :2] - points[ring[3], :2]))


@dataclass(slots=True)
class EyeState:
    closure_left: float
    closure_right: float
    closure: float
    left_state: str        # open | closed | unknown
    right_state: str
    eyes_closed: bool
    gaze_down: bool
    closed_ms: int
    perclos: float
    blink_rate_per_min: float
    drowsy_score: float
    reliable: bool = True  # False when the eyes are too small in the frame to judge


class EyeStateTracker:
    """Per-stream eye state with hysteresis, microsleep duration and PERCLOS."""

    def __init__(self, *, close_threshold: float = 0.55, open_threshold: float = 0.40,
                 blendshape_weight: float = 0.6, gaze_down_look: float = 0.5,
                 gaze_down_look_strong: float = 0.6, gaze_down_pitch_deg: float = 6.0,
                 occluded_eye_yaw_deg: float = 35.0,
                 perclos_window_ms: int = 30000, perclos_alarm: float = 0.25,
                 microsleep_ms: int = 1500, baseline_ear: float = 0.30, min_eye_px: float = 10.0):
        self.close_threshold = close_threshold
        self.open_threshold = open_threshold
        self.blendshape_weight = blendshape_weight
        self.gaze_down_look = gaze_down_look
        self.gaze_down_look_strong = gaze_down_look_strong
        self.gaze_down_pitch = gaze_down_pitch_deg
        self.occluded_eye_yaw = occluded_eye_yaw_deg
        # Below ~10 px corner-to-corner both cues saturate near 0.5 (measured on
        # the sample videos), i.e. lid state is not resolvable: report unknown.
        self.min_eye_px = min_eye_px
        self.perclos_window_ms = perclos_window_ms
        self.perclos_alarm = perclos_alarm
        self.microsleep_ms = microsleep_ms
        self.baseline = {"left": baseline_ear, "right": baseline_ear}
        self._open_ears: dict[str, deque[float]] = {"left": deque(maxlen=300), "right": deque(maxlen=300)}
        self.closed = False
        self._closed_since: int | None = None
        self._history: deque[tuple[int, bool]] = deque()
        self._blinks: deque[int] = deque()

    def _geometric(self, side: str, ear: float) -> float:
        return float(np.clip(1.0 - ear / max(self.baseline[side], 1e-3), 0.0, 1.0))

    def _learn_baseline(self, side: str, ear: float, blink: float | None) -> None:
        if blink is not None and blink > 0.3:
            return
        samples = self._open_ears[side]
        samples.append(ear)
        if len(samples) >= 15:
            # Open-eye reference: a high percentile, robust to blinks that slip through.
            self.baseline[side] = float(np.clip(np.percentile(samples, 80), 0.15, 0.45))

    def update(self, obs: EyeObservation, t_ms: int, *, yaw_offset: float = 0.0,
               pitch_offset: float = 0.0, pose_calibrated: bool = True) -> EyeState:
        closures, visible = {}, {}
        for side, ear, blink, width in (("left", obs.ear_left, obs.blink_left, obs.width_left_px),
                                        ("right", obs.ear_right, obs.blink_right, obs.width_right_px)):
            geometric = self._geometric(side, ear)
            closures[side] = (geometric if blink is None
                              else self.blendshape_weight * blink + (1 - self.blendshape_weight) * geometric)
            visible[side] = width is None or width >= self.min_eye_px
            if visible[side]:
                self._learn_baseline(side, ear, blink)
        # A strongly turned head hides the far eye; trust the visible one.
        if yaw_offset > self.occluded_eye_yaw:
            sides = ["right"]
        elif yaw_offset < -self.occluded_eye_yaw:
            sides = ["left"]
        else:
            sides = ["left", "right"]
        closure = sum(closures[side] for side in sides) / len(sides)
        if not all(visible[side] for side in sides):
            return self._unknown(closures, closure, t_ms)

        lowered = closure >= self.close_threshold
        # Lowered lids + eyes looking down = reading a phone / the dash. Drivers
        # do this mostly with the eyes (head pitch moved only ~4 deg on the phone
        # video), so strong eye evidence suffices; moderate evidence also needs
        # the head pitched down from the calibrated road pose.
        look = obs.look_down
        gaze_down = bool(lowered and look is not None and (
            look >= self.gaze_down_look_strong
            or (look >= self.gaze_down_look and (pitch_offset >= self.gaze_down_pitch or not pose_calibrated))))
        if self.closed:
            self.closed = closure > self.open_threshold and not gaze_down
        else:
            self.closed = lowered and not gaze_down
        if self.closed:
            self._closed_since = t_ms if self._closed_since is None else self._closed_since
        else:
            if self._closed_since is not None and t_ms - self._closed_since < 500:
                self._blinks.append(t_ms)
            self._closed_since = None
        closed_ms = 0 if self._closed_since is None else max(0, t_ms - self._closed_since)

        self._history.append((t_ms, self.closed))
        perclos, blink_rate, drowsy = self._statistics(t_ms, closed_ms)

        def state(side: str) -> str:
            if side not in sides:
                return "unknown"
            return "closed" if closures[side] >= self.close_threshold else "open"

        return EyeState(closures["left"], closures["right"], closure, state("left"), state("right"),
                        self.closed, gaze_down, closed_ms, perclos, blink_rate, drowsy)

    def _statistics(self, t_ms: int, closed_ms: int) -> tuple[float, float, float]:
        while self._history and t_ms - self._history[0][0] > self.perclos_window_ms:
            self._history.popleft()
        while self._blinks and t_ms - self._blinks[0] > 60000:
            self._blinks.popleft()
        if not self._history:
            return 0.0, 0.0, 0.0
        perclos = sum(closed for _, closed in self._history) / len(self._history)
        span_ms = t_ms - self._history[0][0]
        span_min = max(span_ms / 60000.0, 1 / 60.0)
        # Long closures dominate; PERCLOS counts once a third of its window is observed.
        drowsy = float(np.clip(closed_ms / (2.0 * self.microsleep_ms), 0.0, 1.0))
        if span_ms >= self.perclos_window_ms / 3:
            drowsy = max(drowsy, float(np.clip(perclos / (2.0 * self.perclos_alarm), 0.0, 1.0)))
        return perclos, len(self._blinks) / min(span_min, 1.0), drowsy

    def _unknown(self, closures: dict[str, float], closure: float, t_ms: int) -> EyeState:
        # Unresolvable eyes neither start nor extend a closure, and stay out of PERCLOS.
        self.closed, self._closed_since = False, None
        perclos, blink_rate, drowsy = self._statistics(t_ms, 0)
        return EyeState(closures["left"], closures["right"], closure, "unknown", "unknown", False, False,
                        0, perclos, blink_rate, drowsy, reliable=False)
