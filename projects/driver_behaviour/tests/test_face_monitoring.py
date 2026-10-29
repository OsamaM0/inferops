"""Legacy landmark math and temporal eye-state regression checks."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from inferops.adapters.outbound.projects.loader import ProjectPluginResolver
from inferops.adapters.outbound.projects.yaml import load_yaml
from inferops.domain.fusion import FusionEngine
from inferops.domain.fusion.spec import FusionSpec
from inferops.domain.types import StageItem

ROOT = Path(__file__).resolve().parents[1]
load = ProjectPluginResolver(ROOT, "face-monitoring-tests").load_object
FaceMeshStage = load(".models.face_mesh.adapter:FaceMeshStage")
crop_face = load(".models.face_mesh.adapter:_crop_face")
head_pose = load(".models.face_mesh.features:head_pose")
POSE_MODEL = head_pose.__globals__["POSE_MODEL"]
POSE_INDICES = head_pose.__globals__["POSE_INDICES"]
scores = load(".models.face_mesh.features:behavior_scores_from_face")


@pytest.mark.parametrize("box", [(.2, .2, .5, .7), (0, 0, .2, .4), (.8, .7, 1, 1)])
def test_face_crop_stays_square_and_maps_frame_edges(box):
    image = np.full((240, 320, 3), 255, np.uint8)
    crop, (x1, y1, x2, y2) = crop_face(image, box)
    assert crop.shape[0] == crop.shape[1] == x2 - x1 == y2 - y1
    assert crop.max() == 255



@pytest.mark.parametrize("pitch,yaw,roll", [(0, 0, 0), (35, 0, 0), (-35, 0, 0),
                                          (0, 35, 0), (0, -35, 0), (15, -25, 12)])
def test_pose_recovers_known_3d_rotations_and_driver_directions(pitch, yaw, roll):
    x, y, z = np.radians([pitch, -yaw, roll])
    rx = np.array(((1, 0, 0), (0, np.cos(x), -np.sin(x)), (0, np.sin(x), np.cos(x))))
    ry = np.array(((np.cos(y), 0, np.sin(y)), (0, 1, 0), (-np.sin(y), 0, np.cos(y))))
    rz = np.array(((np.cos(z), -np.sin(z), 0), (np.sin(z), np.cos(z), 0), (0, 0, 1)))
    rotation, _ = cv2.Rodrigues(rz @ ry @ rx)
    camera = np.array(((640, 0, 320), (0, 640, 240), (0, 0, 1)), dtype=np.float64)
    projected, _ = cv2.projectPoints(POSE_MODEL, rotation, np.array((0., 0., 60.)), camera, np.zeros(4))
    points = np.zeros((478, 2))
    points[list(POSE_INDICES)] = projected.reshape(-1, 2)
    pose = head_pose(points, 640, 480)
    assert pose is not None
    assert (pose["pitch"], pose["yaw"], pose["roll"]) == pytest.approx((pitch, yaw, roll), abs=.01)



def test_degenerate_landmarks_do_not_imply_forward_facing_or_closed_eyes():
    assert head_pose(np.zeros((478, 2)), 640, 480) is None
    assert head_pose(np.full((478, 2), np.nan), 640, 480) is None



def test_configured_thresholds_reach_the_fusion_decision_boundary():
    assert scores(.21, .21, 0, 28)["drowsy_ear"] == pytest.approx(.5)
    assert scores(.3, .3, 0, 28)["distracted_score"] == pytest.approx(.5)
    assert scores(.3, .3, 0, 35)["distracted_score"] > .5
    assert scores(.05, .3, 0, 0)["drowsy_ear"] == 0  # wink



def _eye_item(stage, ts, left=.3, right=.3, stream="driver", pose=None):
    points = np.zeros((478, 2), dtype=np.float64)
    feature_globals = head_pose.__globals__
    # Published left/right are anatomical; legacy constants use image sides.
    for indices, ear, center in ((feature_globals["RIGHT_EYE"], left, 180),
                                  (feature_globals["LEFT_EYE"], right, 100)):
        points[list(indices)] = [(center, 100), (center + 6, 100 - 10 * ear),
                                 (center + 14, 100 - 10 * ear), (center + 20, 100),
                                 (center + 14, 100 + 10 * ear), (center + 6, 100 + 10 * ear)]
    item = StageItem(stream_id=stream, ts_ms=ts, payload={"image_np": np.zeros((240, 320, 3), np.uint8),
                     "eyes_state": "unknown", "left_eye_state": "unknown", "right_eye_state": "unknown"})
    stage._apply_eye_state(item, points, {"yaw": 0, "pitch": 0} if pose is None else pose)
    return item



def test_eye_hysteresis_duration_winks_and_stream_isolation():
    stage = FaceMeshStage(allow_haar_fallback=False)
    assert _eye_item(stage, 0).payload["eyes_state"] == "open"
    assert _eye_item(stage, 100, .1, .1).payload["eyes_closed_ms"] == 0
    assert _eye_item(stage, 500, .22, .22).payload["eyes_closed_ms"] == 400
    assert _eye_item(stage, 500, .22, .22, stream="other").payload["eyes_state"] == "unknown"
    assert _eye_item(stage, 600, .1, .3).payload["eyes_state"] == "mixed"
    assert _eye_item(stage, 700).payload["drowsy_ear"] == 0
    assert _eye_item(stage, 800, .1, .1).payload["eyes_closed_ms"] == 0
    assert _eye_item(stage, 2000, .1, .1).payload["eyes_closed_ms"] == 0  # gap
    assert _eye_item(stage, 0, .1, .1).payload["eyes_closed_ms"] == 0  # rewind



async def test_missing_face_and_extreme_pose_reset_eye_evidence():
    stage = FaceMeshStage(allow_haar_fallback=False)
    _eye_item(stage, 0, .1, .1)
    lost = await stage.process(StageItem(stream_id="driver", ts_ms=100))
    assert lost.payload["signals"]["eyes_state"] == "unknown"
    assert "driver" not in stage._eye_history
    turned = _eye_item(stage, 200, .1, .1, pose={"yaw": 60, "pitch": 0})
    assert turned.payload["eyes_state"] == "unknown" and "drowsy_ear" not in turned.payload



async def test_blinks_never_confirm_but_sustained_bilateral_closure_does():
    stage = FaceMeshStage(allow_haar_fallback=False)
    spec = FusionSpec(**load_yaml(ROOT / "configs/fusion/driver.yaml"))
    spec.outputs.events_topic = None
    engine = FusionEngine(spec)
    events = []
    for ts in range(0, 4500, 100):
        ear = .1 if ts % 1000 < 200 else .3
        events.extend(await engine.ingest_item(_eye_item(stage, ts, ear, ear)))
    assert "drowsiness" not in {e.type for e in events}
    for ts in range(4500, 8000, 100):
        events.extend(await engine.ingest_item(_eye_item(stage, ts, .1, .1)))
    assert "drowsiness" in {e.type for e in events}
