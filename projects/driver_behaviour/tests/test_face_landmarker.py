"""Complete-bundle pose conventions and blink evidence across independent streams."""
import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from inferops.adapters.outbound.projects.loader import ProjectPluginResolver
from inferops.domain.types import Detection, Result, StageItem

load = ProjectPluginResolver(Path(__file__).resolve().parents[1], 'landmarker-tests').load_object
FaceLandmarkerStage = load('.models.face_landmarker.adapter:FaceLandmarkerStage')
matrix_to_pose = load('.models.face_landmarker.adapter:matrix_to_pose')
MediaPipeBackend = load('.models.face_landmarker.backend:MediaPipeBackend')


@pytest.mark.parametrize('pitch,yaw,roll', [(0, 0, 0), (30, 0, 0), (-30, 0, 0),
                                          (0, 35, 0), (0, -35, 0), (15, -25, 12)])
def test_transformation_angles_have_driver_coordinate_signs(pitch, yaw, roll):
    x, y, z = np.radians([pitch, yaw, -roll])
    rx = np.array(((1, 0, 0), (0, np.cos(x), -np.sin(x)), (0, np.sin(x), np.cos(x))))
    ry = np.array(((np.cos(y), 0, np.sin(y)), (0, 1, 0), (-np.sin(y), 0, np.cos(y))))
    rz = np.array(((np.cos(z), -np.sin(z), 0), (np.sin(z), np.cos(z), 0), (0, 0, 1)))
    matrix = np.eye(4)
    matrix[:3, :3] = 1.1 * (rz @ ry @ rx)
    pose = matrix_to_pose(matrix)
    assert [pose[k] for k in ('pitch', 'yaw', 'roll')] == pytest.approx([pitch, yaw, roll])


@pytest.mark.parametrize('bad', [np.zeros((4, 4)), np.full((4, 4), np.nan), -np.eye(4), np.ones((4, 4))])
def test_degenerate_transforms_cannot_imply_forward_facing(bad):
    assert matrix_to_pose(bad) is None


def bundle(scores=(.05, .05), yaw=0):
    points = np.full((478, 3), (.5, .5, 0), dtype=np.float32)
    points[0, :2], points[1, :2] = (.25, .2), (.75, .8)
    for indices, center in (((33, 160, 158, 133, 153, 144), .35),
                             ((362, 385, 387, 263, 373, 380), .6)):
        points[list(indices), :2] = [(center, .4), (center + .02, .38), (center + .06, .38),
                                     (center + .08, .4), (center + .06, .42), (center + .02, .42)]
    matrix = np.eye(4)
    theta = np.radians(yaw)
    matrix[:3, :3] = ((np.cos(theta), 0, np.sin(theta)), (0, 1, 0), (-np.sin(theta), 0, np.cos(theta)))
    return {'landmarks': points[None], 'blink_scores': np.asarray([scores]), 'matrices': matrix[None]}


def stage_with_output(output, **kwargs):
    stage = FaceLandmarkerStage(min_infer_interval_ms=0, require_detector=False, **kwargs)

    async def infer(feeds):
        assert feeds['image'].dtype == np.uint8
        return output[0]

    async def get_backend(name):
        assert name == 'face_landmarker'
        return SimpleNamespace(infer=infer)

    stage.ctx = SimpleNamespace(registry=SimpleNamespace(get_backend=get_backend))
    return stage


def item(ts, stream='driver'):
    return StageItem(ts_ms=ts, stream_id=stream, payload={'image_np': np.zeros((240, 320, 3), np.uint8)})


async def test_blink_coefficients_drive_closure_hysteresis_duration_and_stream_isolation():
    output = [bundle()]
    stage = stage_with_output(output)
    assert (await stage.process(item(0))).payload['eyes_state'] == 'open'
    output[0] = bundle((.8, .8))
    assert (await stage.process(item(100))).payload['eyes_closed_ms'] == 0
    output[0] = bundle((.4, .4))
    closed = (await stage.process(item(500))).payload
    assert closed['eyes_closed_ms'] == 400
    assert closed['drowsy_ear'] == 1  # EAR geometry is open; blendshapes are authoritative.
    assert closed['eye_state_source'] == 'mediapipe_blendshapes'
    assert (await stage.process(item(500, 'passenger'))).payload['eyes_state'] == 'unknown'
    output[0] = bundle((.8, .1))
    wink = (await stage.process(item(600))).payload
    assert wink['eyes_state'] == 'mixed' and wink['drowsy_ear'] == 0
    output[0] = bundle((.8, .8))
    assert (await stage.process(item(2000))).payload['eyes_closed_ms'] == 0
    assert (await stage.process(item(100))).payload['eyes_closed_ms'] == 0


async def test_missing_face_and_extreme_pose_reset_closure_evidence():
    output = [bundle((.8, .8))]
    stage = stage_with_output(output)
    await stage.process(item(0))
    output[0] = {'landmarks': np.empty((0, 478, 3)), 'blink_scores': np.empty((0, 2)),
                 'matrices': np.empty((0, 4, 4))}
    lost = (await stage.process(item(100))).payload
    assert lost['eyes_state'] == 'unknown' and 'drowsy_ear' not in lost
    assert 'driver' not in stage._eye_history
    output[0] = bundle((.9, .9), yaw=60)
    turned = (await stage.process(item(200))).payload
    assert turned['head_direction'] == 'left' and turned['eyes_state'] == 'unknown'
    assert 'drowsy_ear' not in turned
    assert turned['signals']['head_pose_source'] == 'mediapipe_transformation'


async def test_driver_region_maps_landmarks_back_and_calibration_is_applied_once():
    stage = stage_with_output([bundle(yaw=30)], driver_region=(.2, .1, .8, .9), neutral_yaw_deg=10)
    payload = (await stage.process(item(0))).payload
    assert payload['face_landmarks'][0]['x'] == pytest.approx(.35)
    assert payload['face_pose_raw']['yaw'] == pytest.approx(30)
    assert payload['face_pose']['yaw'] == pytest.approx(20)
    assert payload['head_direction'] == 'forward'


async def test_camera_roi_overrides_pipeline_default_and_invalidates_cached_coordinates():
    stage = stage_with_output([bundle()], driver_region=(0, 0, .5, 1))
    stage.min_infer_interval_ms = 100
    first = item(0)
    first.payload['driver_roi'] = [.5, 0, 1, 1]
    p1 = (await stage.process(first)).payload
    assert p1['face_landmarks'][0]['x'] == pytest.approx(.625)
    second = item(50)
    second.payload['driver_roi'] = [0, 0, .5, 1]
    p2 = (await stage.process(second)).payload
    assert p2['face_landmarks'][0]['x'] == pytest.approx(.125)
    assert p2['face_detection_age_ms'] == 0


async def test_task_errors_propagate_instead_of_silently_using_an_old_model():
    stage = stage_with_output([bundle()])

    async def unavailable(name):
        raise RuntimeError('task unavailable')

    stage.ctx.registry.get_backend = unavailable
    with pytest.raises(RuntimeError, match='task unavailable'):
        await stage.process(item(0))


@pytest.mark.parametrize('region', [(0, 0, 0, 1), (-.1, 0, 1, 1), (0, 0, np.nan, 1)])
def test_invalid_driver_regions_are_rejected(region):
    with pytest.raises(ValueError, match='driver_region'):
        FaceLandmarkerStage(driver_region=region)


def detected_item(ts=0, box=(.3, .2, .7, .8)):
    frame = item(ts)
    detection = Detection(label='DriverFace', score=.95, box=box)
    frame.results.append(Result(kind='detections', model='scrfd_onnx', detections=[detection],
                                raw={'face_detector': 'scrfd', 'selected_face_box': list(box)}))
    return frame


async def test_scrfd_crop_is_the_analysis_input_and_coordinates_return_to_original_frame():
    seen = []

    async def infer(feeds):
        seen.append(feeds['image'].shape)
        return bundle()

    async def get_backend(name):
        assert name == 'face_landmarker'
        return SimpleNamespace(infer=infer)

    stage = FaceLandmarkerStage()
    stage.ctx = SimpleNamespace(registry=SimpleNamespace(get_backend=get_backend))
    output = (await stage.process(detected_item(box=(.25, .25, .75, .75)))).payload
    assert seen == [(240, 240, 3)]  # Square face crop, not the 240x320 full frame.
    assert output['face_landmarks'][0]['x'] == pytest.approx((40 + .25 * 240) / 320)
    assert output['face_source'] == 'scrfd' and output['face_detected'] == 1
    assert output['face_analysis_source'] == 'mediapipe_face_landmarker'
    assert output['face_box'] == (.25, .25, .75, .75)
    assert output['head_direction'] == 'forward' and output['eyes_state'] == 'open'


async def test_detected_face_is_retained_when_eye_and_direction_analysis_fails():
    async def get_backend(name):
        async def infer(feeds):
            return {'landmarks': np.empty((0, 478, 3)), 'matrices': np.empty((0, 4, 4)),
                    'blink_scores': np.empty((0, 2))}
        return SimpleNamespace(infer=infer)

    stage = FaceLandmarkerStage()
    stage.ctx = SimpleNamespace(registry=SimpleNamespace(get_backend=get_backend))
    payload = (await stage.process(detected_item())).payload
    assert payload['face_detected'] == 1 and payload['face_box'] == (.3, .2, .7, .8)
    assert payload['face_detection_confidence'] == .95
    assert payload['eyes_state'] == payload['head_direction'] == 'unknown'
    assert 'drowsy_ear' not in payload
    assert payload['overlay_boxes'] and payload['signals']['face_source'] == 'scrfd'


async def test_crop_pose_fits_original_frame_pixels_and_publishes_down(monkeypatch):
    observed = []

    def fit(points, width, height):
        observed.append((points.copy(), width, height))
        return {'yaw': 0., 'pitch': 25., 'roll': 0.}

    stage = stage_with_output([bundle()], pose_method='landmark_pnp')
    monkeypatch.setitem(stage._process.__globals__, 'head_pose', fit)
    payload = (await stage.process(item(0))).payload
    assert observed[0][1:] == (320, 240)
    assert observed[0][0][0] == pytest.approx((80, 48))
    assert payload['head_direction'] == 'down'
    assert payload['head_pose_source'] == 'mediapipe_landmark_pnp'


async def test_no_scrfd_face_cannot_be_replaced_by_a_generic_behavior_detection():
    stage = FaceLandmarkerStage()

    async def forbidden(name):
        raise AssertionError('A missing SCRFD face must not invoke the face analyzer')

    stage.ctx = SimpleNamespace(registry=SimpleNamespace(get_backend=forbidden))
    frame = item(0)
    frame.results.append(Result(kind='detections', raw={'by_class': {'DriverBehaviour': [
        {'score': .99, 'box': [.1, .1, .9, .9]}]}}))
    payload = (await stage.process(frame)).payload
    assert payload['face_detected'] == 0 and 'face_box' not in payload
    assert payload['eyes_state'] == payload['head_direction'] == 'unknown'


async def test_cancelled_inference_drains_native_work_before_backend_close():
    entered, release = threading.Event(), threading.Event()
    closed = []
    backend = MediaPipeBackend()
    backend._landmarker = SimpleNamespace(close=lambda: closed.append(True))

    def detect(image):
        entered.set()
        if not release.wait(5):
            raise RuntimeError('test did not release native inference')
        return {}

    backend._detect = detect
    pending = asyncio.create_task(backend.infer({'image': np.zeros((10, 10, 3), np.uint8)}))
    await asyncio.wait_for(asyncio.to_thread(entered.wait), timeout=2)
    pending.cancel()
    closing = asyncio.create_task(backend.close())
    await asyncio.sleep(.01)
    assert not closed and not pending.done() and not closing.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await closing
    assert closed and backend._landmarker is None
