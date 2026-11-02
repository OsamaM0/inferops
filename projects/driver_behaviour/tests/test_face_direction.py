"""Head direction and eye state: geometry, temporal logic and the full stage.

No weights are needed: fake model backends return a canonical face rotated to
a known pose, so expected directions are exact.
"""
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from inferops.adapters.outbound.projects.loader import ProjectPluginResolver
from inferops.domain.models import ModelSpec
from inferops.domain.types import StageItem

DOMAIN_ROOT = Path(__file__).resolve().parents[1]
load_object = ProjectPluginResolver(DOMAIN_ROOT, "face-direction-tests").load_object
pose = load_object(".models.face_mesh.pose:estimate_pose").__globals__
eyes = load_object(".models.face_mesh.eyes:eye_aspect_ratio").__globals__
adapter = load_object(".models.face_mesh.adapter:crop_roi").__globals__
yunet = load_object(".models.face_detector.yunet:decode").__globals__

estimate_pose = pose["estimate_pose"]
canonical = np.load(pose["CANONICAL_FACE"])["vertices"].astype(np.float64)


def rotation(yaw: float, pitch: float, roll: float = 0.0) -> np.ndarray:
    """Head rotation in the canonical frame (x: image right, y: up, z: towards camera).

    +yaw turns the nose to image right (the driver's left), +pitch lowers it.
    """
    a, b, c = (math.radians(v) for v in (yaw, pitch, roll))
    ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    rz = np.array([[math.cos(c), math.sin(c), 0], [-math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return ry @ rx @ rz


EYELIDS = ((263, 362, (387, 385, 380, 373)), (33, 133, (160, 158, 153, 144)))


def face_points(yaw=0.0, pitch=0.0, roll=0.0, *, scale=7.5, centre=(128.0, 128.0), noise=0.0, seed=0,
                eyes_closed=False):
    """478 image-space landmarks (x right, y down, z away) of a posed canonical face."""
    face = canonical.copy()
    if eyes_closed:  # lids meet on the line between the eye corners
        for corner_a, corner_b, lids in EYELIDS:
            face[list(lids), 1] = (face[corner_a, 1] + face[corner_b, 1]) / 2
    rotated = face @ rotation(yaw, pitch, roll).T * scale
    iris = np.repeat(rotated[[159, 386]], 5, axis=0)  # 10 iris points, placed on the eyes
    points = np.vstack([rotated, iris])
    image = np.column_stack([centre[0] + points[:, 0], centre[1] - points[:, 1], -points[:, 2]])
    if noise:
        image[:, :2] += np.random.default_rng(seed).normal(0, noise, (len(image), 2))
    return image


# ── pose ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("yaw", [-60, -30, -10, 0, 15, 35, 60])
@pytest.mark.parametrize("pitch", [-35, -10, 0, 20, 40])
def test_pose_recovers_known_rotation_and_signs(yaw, pitch):
    estimated = estimate_pose(face_points(yaw, pitch, roll=5, noise=0.3))
    assert estimated.yaw == pytest.approx(yaw, abs=2.0)
    assert estimated.pitch == pytest.approx(pitch, abs=2.0)
    assert estimated.residual < 0.2


def test_pose_is_scale_and_translation_invariant():
    near = estimate_pose(face_points(20, -10, scale=20, centre=(900, 500)))
    far = estimate_pose(face_points(20, -10, scale=2, centre=(40, 30)))
    assert (near.yaw, near.pitch) == pytest.approx((far.yaw, far.pitch), abs=0.5)


# ── neutral calibration and direction ───────────────────────────────────────

def test_calibrator_learns_off_axis_front_and_ignores_glances():
    calibrator = pose["NeutralPoseCalibrator"](warmup_ms=4000, min_samples=20)
    for t in range(0, 4200, 66):
        calibrator.update(22.0, 6.0, t, attentive=True)
        calibrator.update(-40.0, 30.0, t + 1, attentive=False)  # mirror / phone glances
    assert calibrator.calibrated
    assert (calibrator.yaw, calibrator.pitch) == pytest.approx((22.0, 6.0), abs=0.01)
    for t in range(4200, 60000, 66):  # a long look away never becomes "front"
        calibrator.update(-35.0, 25.0, t, attentive=True)
    assert (calibrator.yaw, calibrator.pitch) == pytest.approx((22.0, 6.0), abs=0.01)


def test_calibrator_never_falls_back_to_the_camera_axis():
    calibrator = pose["NeutralPoseCalibrator"](warmup_ms=4000, min_samples=20, timeout_ms=12000)
    for t in range(0, 20000, 66):
        calibrator.update(25.0, 15.0, t, attentive=False)
    assert not calibrator.calibrated
    for t in range(20000, 20400, 66):  # past the timeout a few attentive frames suffice
        calibrator.update(20.0, 4.0, t, attentive=True)
    assert calibrator.calibrated and calibrator.yaw == pytest.approx(20.0)


def test_fixed_neutral_uses_configured_offsets():
    calibrator = pose["NeutralPoseCalibrator"](mode="fixed", yaw=15.0, pitch=-5.0)
    assert calibrator.calibrated and calibrator.offset(40.0, -5.0) == (25.0, 0.0)


def test_direction_hysteresis_and_mirror():
    classify = pose["DirectionClassifier"]()
    assert [classify.update(y, 0.0) for y in (0, 20, 26, 20, 19, 17)] == [
        "front", "front", "left", "left", "left", "front"]
    assert classify.update(0.0, 19.0) == "down"
    assert classify.update(0.0, 13.0) == "down" and classify.update(0.0, 11.0) == "front"
    assert classify.update(0.0, -21.0) == "up"
    assert classify.update(-30.0, 0.0) == "right"
    assert classify.update(0.0, 2.0, gaze_down=True) == "down"
    mirrored = pose["DirectionClassifier"](mirror=True)
    assert mirrored.update(30.0, 0.0) == "right"


def test_one_euro_filter_smooths_jitter_but_follows_turns():
    filt = pose["OneEuroFilter"](min_cutoff=1.0, beta=0.05)
    jitter = [filt(v, i / 15) for i, v in enumerate(np.random.default_rng(1).normal(0, 2, 60))]
    assert np.std(jitter[10:]) < 1.0
    turned = [filt(40.0, (60 + i) / 15) for i in range(15)]
    assert turned[-1] > 35.0


# ── eyes ────────────────────────────────────────────────────────────────────

def observation(blink=0.05, ear=0.30, look_down=0.1, width=20.0):
    return eyes["EyeObservation"](ear, ear, blink, blink, look_down, width, width)


def test_eyes_closed_hysteresis_duration_and_drowsiness():
    tracker = eyes["EyeStateTracker"]()
    for t in range(0, 1000, 66):
        state = tracker.update(observation(), t)
    assert not state.eyes_closed and state.left_state == "open"
    for t in range(1000, 4000, 66):
        state = tracker.update(observation(blink=0.9, ear=0.05), t)
    assert state.eyes_closed and state.closed_ms >= 2900 and state.drowsy_score >= 0.9
    state = tracker.update(observation(blink=0.5, ear=0.15), 4066)  # between thresholds: stays closed
    assert state.eyes_closed
    state = tracker.update(observation(), 4132)
    assert not state.eyes_closed and state.closed_ms == 0


def test_lowered_gaze_is_looking_down_not_closed_eyes():
    tracker = eyes["EyeStateTracker"]()
    state = tracker.update(observation(blink=0.75, ear=0.08, look_down=0.75), 0, pitch_offset=2.0)
    assert state.gaze_down and not state.eyes_closed
    state = tracker.update(observation(blink=0.75, ear=0.08, look_down=0.55), 66, pitch_offset=2.0)
    assert not state.gaze_down and state.eyes_closed  # moderate cue, head level: closed
    state = tracker.update(observation(blink=0.75, ear=0.08, look_down=0.55), 132, pitch_offset=9.0)
    assert state.gaze_down and not state.eyes_closed


def test_tiny_eyes_are_unknown_never_closed():
    tracker = eyes["EyeStateTracker"]()
    for t in range(0, 3000, 66):
        state = tracker.update(observation(blink=0.9, ear=0.05, width=6.0), t)
    assert not state.reliable and not state.eyes_closed and state.left_state == "unknown"


def test_far_eye_is_ignored_when_the_head_is_turned():
    tracker = eyes["EyeStateTracker"]()
    obs = eyes["EyeObservation"](0.05, 0.30, 0.9, 0.05, 0.1, 4.0, 20.0)  # left eye hidden and tiny
    state = tracker.update(obs, 0, yaw_offset=45.0)
    assert state.reliable and not state.eyes_closed and state.left_state == "unknown"


# ── YuNet decode and crop geometry ──────────────────────────────────────────

def yunet_outputs(cells: list[tuple[int, int, int, float]]) -> dict[str, np.ndarray]:
    """Raw YuNet tensors with a face at each (stride, row, col, score)."""
    out = {}
    for stride in (8, 16, 32):
        n = (640 // stride) ** 2
        out[f"cls_{stride}"] = np.zeros((1, n, 1), np.float32)
        out[f"obj_{stride}"] = np.zeros((1, n, 1), np.float32)
        out[f"bbox_{stride}"] = np.zeros((1, n, 4), np.float32)
        out[f"kps_{stride}"] = np.zeros((1, n, 10), np.float32)
    for stride, row, col, score in cells:
        index = row * (640 // stride) + col
        out[f"cls_{stride}"][0, index, 0] = out[f"obj_{stride}"][0, index, 0] = score
        out[f"bbox_{stride}"][0, index] = (0.0, 0.0, math.log(5.0), math.log(5.0))
        out[f"kps_{stride}"][0, index] = (-1.2, -0.6, 1.2, -0.6, 0.0, 0.0, -0.8, 0.8, 0.8, 0.8)
    return out


def test_yunet_decode_places_box_keypoints_and_cell_score():
    faces = yunet["decode"](yunet_outputs([(32, 10, 10, 0.9), (16, 2, 3, 0.7)]), scale=0.5)
    assert [round(f.score, 2) for f in faces] == [0.9, 0.7]
    # Cell (10, 10) at stride 32 -> centre 320, size exp(log 5) * 32 = 160; / scale 0.5.
    assert faces[0].box == pytest.approx((480.0, 480.0, 800.0, 800.0))
    assert faces[0].keypoints[0] == pytest.approx((563.2, 601.6))
    assert yunet["decode"](yunet_outputs([(8, 1, 1, 0.5)]), scale=1.0, score_threshold=0.6) == []


def test_crop_roi_maps_landmarks_back_exactly():
    roi = adapter["Roi"](cx=300.0, cy=220.0, size=180.0, angle=math.radians(17))
    image = np.zeros((480, 640, 3), np.uint8)
    image[250, 330] = 255
    patch, affine = adapter["crop_roi"](image, roi, 256)
    y, x = np.unravel_index(patch[..., 0].argmax(), patch.shape[:2])
    back = adapter["landmarks_to_image"](np.array([[x, y, 0.0]]), affine, roi, 256)
    assert back[0, :2] == pytest.approx((330, 250), abs=1.0)
    centre = adapter["landmarks_to_image"](np.array([[128.0, 128.0, 0.0]]), affine, roi, 256)
    assert centre[0, :2] == pytest.approx((300.0, 220.0))


def test_roi_snaps_to_a_grid_so_float_noise_cannot_feed_back():
    snap, Roi = adapter["snap_roi"], adapter["Roi"]
    base = snap(Roi(100.0, 100.0, 120.0, 0.0))
    assert snap(Roi(100.0 + 1e-4, 100.0 - 1e-4, 120.0 + 1e-4, 1e-5)) == base
    moved = snap(Roi(110.0, 100.0, 120.0, 0.0))
    assert moved.cx != base.cx and abs(moved.cx - 110.0) <= 0.02 * moved.size
    assert abs(base.size - 120.0) <= 0.02 * 120.0


# ── full stage with fake models ─────────────────────────────────────────────

class FakeModels:
    """Detector + FaceMesh + blendshapes; mesh output follows ``self.pose``."""

    def __init__(self):
        self.pose = (0.0, 0.0)
        # True/False, or "crop": present only if the crop shows a bright face patch.
        self.present = True
        self.faces = [(32, 10, 10, 0.9)]
        self.blink, self.look_down = 0.05, 0.1
        self.calls = {"face_detector": 0, "face_mesh": 0, "face_blendshapes": 0}

    def get_spec(self, name):
        shapes = {"face_detector": ("input", [1, 3, 640, 640]), "face_mesh": ("input_12", [1, 3, 256, 256]),
                  "face_blendshapes": ("points", [1, 146, 2])}
        input_name, shape = shapes[name]
        return ModelSpec(name=name, inputs=[{"name": input_name, "shape": shape}])

    async def get_backend(self, name):
        models = self

        class Backend:
            async def infer(self, feeds):
                models.calls[name] += 1
                if name == "face_detector":
                    return yunet_outputs(models.faces)
                if name == "face_mesh":
                    # Like the real model, put the landmarks on the face visible
                    # in the crop: the bright patch (centre and extent).
                    ys, xs = np.nonzero(feeds["input_12"][0].mean(axis=0) > 0.5)
                    visible = xs.size > 100
                    centre = (xs.mean(), ys.mean()) if visible else (128.0, 128.0)
                    extent = max(np.ptp(xs), np.ptp(ys)) if visible else 120.0
                    points = face_points(*models.pose, scale=extent / 18.0, centre=centre,
                                         eyes_closed=models.blink > 0.5)
                    present = visible if models.present == "crop" else models.present
                    return {"Identity": points.astype(np.float32).reshape(1, 1, 1, 1434),
                            "Identity_1": np.array([[[[8.0 if present else -8.0]]]], np.float32),
                            "Identity_2": np.array([[0.999]], np.float32)}
                scores = np.zeros((1, 52), np.float32)
                scores[0, [9, 10]] = models.blink
                scores[0, [11, 12]] = models.look_down
                return {"blendshapes": scores}

        return Backend()


def make_stage(models, **params):
    stage = adapter["FaceMeshStage"](model="face_mesh", detector_model="face_detector",
                                     blendshape_model="face_blendshapes", **params)
    stage.ctx = SimpleNamespace(registry=models)
    return stage


def frame(*patches: tuple[int, int]) -> np.ndarray:
    """Black 640x640 frame with a bright 120 px "face" centred at each (x, y)."""
    image = np.zeros((640, 640, 3), np.uint8)
    for x, y in patches:
        image[y - 60:y + 60, x - 60:x + 60] = 255
    return image


DRIVER = frame((320, 320))


async def run(stage, start_ms, stop_ms, step=66, stream="cab", image=DRIVER):
    item = None
    for t in range(start_ms, stop_ms, step):
        item = await stage.process(StageItem(stream_id=stream, ts_ms=t, payload={"image_np": image}))
    return item


@pytest.mark.asyncio
async def test_stage_calibrates_then_reports_each_direction():
    models = FakeModels()
    stage = make_stage(models)
    models.pose = (20.0, 5.0)  # off-axis camera: the driver's road pose is not 0/0
    item = await run(stage, 0, 4500)
    assert item.payload["neutral_pose"]["calibrated"]
    assert item.payload["head_direction"] == "front" and item.payload["distracted_score"] < 0.5
    assert item.payload["face_status_label"] == "Driver Face"
    t = 4500
    for (yaw, pitch), direction, label in (((50.0, 5.0), "left", "Looking Left"),
                                           ((-10.0, 5.0), "right", "Looking Right"),
                                           ((20.0, 30.0), "down", "Looking Down"),
                                           ((20.0, -22.0), "up", "Looking Up")):
        models.pose = (yaw, pitch)
        item = await run(stage, t, t + 1500)
        t += 1500
        assert item.payload["head_direction"] == direction
        assert item.payload[f"look_{direction}_score"] == 1.0 and item.payload["distracted_score"] == 1.0
        assert label in item.payload["face_status_label"]
        assert item.payload["signals"]["head_direction"] == direction


@pytest.mark.asyncio
async def test_stage_tracks_without_rerunning_the_detector():
    models = FakeModels()
    stage = make_stage(models, redetect_interval_ms=2000)
    await run(stage, 0, 1980)
    assert models.calls["face_detector"] == 1 and models.calls["face_mesh"] == 30


@pytest.mark.asyncio
async def test_stage_eyes_closed_versus_phone_gaze():
    models = FakeModels()
    stage = make_stage(models)
    await run(stage, 0, 4500)
    models.blink, models.look_down = 0.9, 0.1
    item = await run(stage, 4500, 6000)
    assert item.payload["eyes_closed"] and item.payload["eyes_closed_score"] == 1.0
    assert "Eyes Closed" in item.payload["face_status_label"]
    models.look_down = 0.8
    item = await run(stage, 6000, 7000)
    assert not item.payload["eyes_closed"] and item.payload["head_direction"] == "down"


@pytest.mark.asyncio
async def test_lost_established_face_is_away_but_brief_dropouts_hold():
    models = FakeModels()
    stage = make_stage(models, away_after_ms=600)
    await run(stage, 0, 4500)
    models.pose = (50.0, 0.0)
    await run(stage, 4500, 5500)
    models.present, models.faces = False, []
    item = await run(stage, 5500, 5800)
    assert item.payload["head_direction"] == "left"  # held through the dropout
    item = await run(stage, 5800, 6800)
    assert item.payload["head_direction"] == "away" and item.payload["distracted_score"] == 1.0
    assert item.payload["face_status_label"] == "Looking Away"


@pytest.mark.asyncio
async def test_never_seen_face_is_never_away_and_draws_nothing():
    models = FakeModels()
    models.present, models.faces = False, []
    item = await run(make_stage(models), 0, 3000)
    assert "head_direction" not in item.payload
    assert "face_box" not in item.payload and "overlay_boxes" not in item.payload


@pytest.mark.asyncio
async def test_visible_but_unreadable_face_is_unknown_not_away():
    models = FakeModels()
    stage = make_stage(models)
    await run(stage, 0, 4500)
    models.present = False  # detector still finds the face; landmarks rejected
    item = await run(stage, 4500, 6000)
    assert item.payload["head_direction"] == "unknown" and item.payload["distracted_score"] == 0.0
    assert item.payload["face_source"] == "detector"


@pytest.mark.asyncio
async def test_reacquires_the_driver_not_a_passenger():
    models = FakeModels()
    models.present = "crop"
    stage = make_stage(models)
    item = await run(stage, 0, 4500)
    assert item.payload["head_direction"] == "front"
    # The driver turns out of view; only the passenger (left, readable) remains.
    models.faces = [(32, 10, 3, 0.95)]
    item = await run(stage, 4500, 5500, image=frame((96, 320)))
    assert item.payload["head_direction"] == "away"


@pytest.mark.asyncio
async def test_streams_are_isolated():
    models = FakeModels()
    stage = make_stage(models)
    models.pose = (60.0, 0.0)
    await run(stage, 0, 4500, stream="a")
    item = await run(stage, 0, 300, stream="b")
    assert item.payload["head_direction"] == "calibrating"


@pytest.mark.asyncio
async def test_missed_periodic_redetection_keeps_the_tracked_face():
    models = FakeModels()
    stage = make_stage(models, redetect_interval_ms=1000)
    await run(stage, 0, 4500)
    models.pose, models.faces = (50.0, 0.0), []  # turned: the detector no longer sees it
    item = await run(stage, 4500, 7000)
    assert models.calls["face_detector"] > 3
    assert item.payload["head_direction"] == "left" and item.payload["face_source"] == "face_mesh"
