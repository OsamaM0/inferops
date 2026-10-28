"""Face detection, physical pose conventions, and temporal eye-state regressions."""
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from inferops.adapters.outbound.projects.loader import ProjectPluginResolver
from inferops.adapters.outbound.projects.models import load_model_variants
from inferops.adapters.outbound.projects.yaml import load_yaml
from inferops.domain.fusion import FusionEngine
from inferops.domain.fusion.spec import FusionSpec
from inferops.domain.types import Detection, Frame, Result, StageItem

ROOT = Path(__file__).resolve().parents[1]
load = ProjectPluginResolver(ROOT, "face-monitoring-tests").load_object
SCRFDPreprocess = load(".models.buffalo_detector.adapter:SCRFDPreprocess")
SCRFDPostprocess = load(".models.buffalo_detector.adapter:SCRFDPostprocess")
FaceDetectionStage = load(".models.buffalo_detector.adapter:FaceDetectionStage")
FaceMeshStage = load(".models.face_mesh.adapter:FaceMeshStage")
get_face_box = load(".models.face_mesh.adapter:_get_face_box")
crop_face = load(".models.face_mesh.adapter:_crop_face")
head_pose = load(".models.face_mesh.features:head_pose")
POSE_MODEL = head_pose.__globals__["POSE_MODEL"]
POSE_INDICES = head_pose.__globals__["POSE_INDICES"]
scores = load(".models.face_mesh.features:behavior_scores_from_face")
HeadPoseStage = load(".models.head_pose.adapter:HeadPoseStage")
rotation_to_pose = load(".models.head_pose.adapter:rotation_to_pose")


@pytest.fixture
def detector_spec():
    return load_model_variants(ROOT / "models/buffalo_detector/model.yaml")[0]


async def test_scrfd_preprocessing_matches_rgb_normalization_and_letterbox(detector_spec):
    image = np.zeros((320, 640, 3), np.uint8)
    image[:] = (0, 128, 255)
    frame = Frame(stream_id="s", ts_ms=0, image=image, width=640, height=320)
    feeds = await SCRFDPreprocess()(frame, detector_spec)
    tensor = feeds["input.1"]
    assert tensor.shape == (1, 3, 640, 640) and tensor.dtype == np.float32
    assert tensor[0, :, 0, 0] == pytest.approx(np.array((127.5, .5, -127.5)) / 128)
    assert tensor[0, :, 500, 0] == pytest.approx(np.full(3, -127.5 / 128))


async def test_scrfd_decodes_letterbox_suppresses_duplicates_and_rejects_padding(detector_spec):
    raw = {s.name: np.zeros(s.shape, np.float32) for s in detector_spec.outputs}
    index = (30 * 80 + 40) * 2  # stride-8 center (320, 240)
    raw["448"][index:index + 2] = .95
    raw["451"][index:index + 2] = 5  # 40px distances
    padding_index = (70 * 80 + 40) * 2
    raw["448"][padding_index] = .99
    raw["451"][padding_index] = 5
    frame = Frame(stream_id="s", ts_ms=0, width=1280, height=720)
    # Dict insertion order must not change how output heads are decoded.
    result = await SCRFDPostprocess()(dict(reversed(list(raw.items()))), frame, detector_spec)
    assert len(result.detections) == 1
    assert result.detections[0].box == pytest.approx((280 / 640, 200 / 360, 360 / 640, 280 / 360))
    assert result.raw["face_detector"] == "scrfd"
    raw["451"][index:index + 2] = np.nan
    assert not (await SCRFDPostprocess()(raw, frame, detector_spec)).detections


def test_dedicated_detector_is_authoritative_even_when_it_finds_no_faces():
    generic = Result(kind="detections", raw={"by_class": {
        "DriverBehaviour": [{"score": .99, "box": [.1, .1, .5, .5]}]}})
    dedicated = Result(kind="detections", raw={"face_detector": "scrfd", "by_class": {"DriverFace": []}})
    assert get_face_box([generic, dedicated], 640, 480) is None
    dedicated.raw["by_class"]["DriverFace"] = [{"score": .9, "box": [.6, .2, .9, .7]}]
    assert get_face_box([generic, dedicated], 640, 480) == (.6, .2, .9, .7)


async def test_face_detection_cache_isolated_by_stream_time_and_dimensions():
    calls = []

    async def predict(model, image, **kwargs):
        calls.append(kwargs)
        detections = [Detection(label="DriverFace", score=.95, box=(.1, .1, .3, .4)),
                      Detection(label="DriverFace", score=.99, box=(.6, .1, .9, .5))]
        return Result(kind="detections", model=model, detections=detections,
                      raw={"face_detector": "scrfd", "by_class": {
                          "DriverFace": [d.model_dump(mode="json") for d in detections]}})

    stage = FaceDetectionStage(driver_region=[0, 0, .5, 1])
    stage.ctx = SimpleNamespace(registry=SimpleNamespace(predict=predict))

    async def run(stream, ts, shape=(80, 80, 3)):
        return await stage.process(StageItem(stream_id=stream, ts_ms=ts,
                                             payload={"image_np": np.zeros(shape, np.uint8)}))

    first = await run("driver", 100)
    assert len(first.results[0].detections) == 1
    first.results[0].raw.clear()  # Consumers cannot mutate the cached result.
    assert (await run("driver", 150)).results[0].raw["face_detector"] == "scrfd"
    assert len(calls) == 1
    await run("other", 150)
    await run("driver", 50)  # rewind
    await run("driver", 60, (100, 100, 3))
    assert len(calls) == 4
    await stage.teardown()
    assert not stage._cache


async def test_driver_region_is_cropped_before_inference_and_boxes_map_back():
    seen = []

    async def predict(model, image, **kwargs):
        seen.append(image.shape)
        detection = Detection(label="DriverFace", score=.9, box=(.2, .2, .6, .6))
        return Result(kind="detections", model=model, detections=[detection], raw={})

    stage = FaceDetectionStage()
    stage.ctx = SimpleNamespace(registry=SimpleNamespace(predict=predict))
    item = StageItem(stream_id="s", ts_ms=0, payload={
        "image_np": np.zeros((200, 400, 3), np.uint8), "driver_roi": [.5, .5, 1., 1.]})
    output = await stage.process(item)
    assert seen == [(100, 200, 3)]
    assert output.results[0].detections[0].box == pytest.approx((.6, .6, .8, .8))
    assert output.results[0].raw["selected_face_box"] == pytest.approx((.6, .6, .8, .8))
    # Changing the region invalidates the detection cache immediately.
    item = StageItem(stream_id="s", ts_ms=10, payload={
        "image_np": np.zeros((200, 400, 3), np.uint8), "driver_roi": [0., 0., .5, .5]})
    assert (await stage.process(item)).results[0].detections[0].box == pytest.approx((.1, .1, .3, .3))
    assert len(seen) == 2


async def test_recovery_search_has_a_fixed_budget_and_never_reuses_a_lost_face():
    calls = []
    found = True

    async def predict(model, image, **kwargs):
        calls.append(image.shape)
        detections = [Detection(label="DriverFace", score=.95, box=(.2, .2, .6, .6))] if found else []
        return Result(kind="detections", model=model, detections=detections, raw={})

    stage = FaceDetectionStage(search_tiles=True, enhance_on_miss=True, max_search_views=3)
    stage.ctx = SimpleNamespace(registry=SimpleNamespace(predict=predict))

    def item(ts):
        return StageItem(stream_id="s", ts_ms=ts, payload={"image_np": np.zeros((100, 100, 3), np.uint8)})

    assert (await stage.process(item(0))).results[0].detections
    found = False
    output = await stage.process(item(150))
    assert not output.results[0].detections and output.results[0].raw['search_views'] == 3
    assert "s" not in stage._tracks and len(calls) == 4
    assert not (await stage.process(item(160))).results[0].detections
    assert len(calls) == 4  # A cached negative is not an old positive.


async def test_negative_search_backoff_and_recovery_views_advance_between_misses():
    crops = []
    async def predict(model, image, **kwargs):
        crops.append(float(image.mean()))
        return Result(kind="detections", model=model, raw={}, detections=[])

    image = np.zeros((100, 100, 3), np.uint8)
    image[:, 50:] = 255
    stage = FaceDetectionStage(search_tiles=True, max_search_views=2, negative_cache_ms=350)
    stage.ctx = SimpleNamespace(registry=SimpleNamespace(predict=predict))
    async def run(ts):
        return await stage.process(StageItem(stream_id='s', ts_ms=ts, payload={'image_np': image}))

    await run(0)
    assert len(crops) == 2
    await run(100)
    assert len(crops) == 2
    await run(400)
    assert len(crops) == 4 and crops[1] != crops[3]


async def test_candidate_requires_mesh_presence_and_real_geometry(monkeypatch):
    from inferops.domain.models import ModelSpec

    raw = {"Identity": np.tile([128., 128., 0.], (1, 478)).astype(np.float32),
           "Identity_1": np.array([-8.], np.float32)}

    class Registry:
        def get_spec(self, name):
            return ModelSpec(name="face_mesh_onnx", inputs=[{"name": "input", "shape": [1, 3, 256, 256]}])

        async def get_backend(self, name):
            async def infer(feeds):
                return raw
            return SimpleNamespace(infer=infer)

    stage = FaceDetectionStage(verification_model="face_mesh")
    stage.ctx = SimpleNamespace(registry=Registry())
    detection = Detection(label="DriverFace", score=.7, box=(.2, .2, .8, .8))
    assert not (await stage._verify(np.zeros((200, 200, 3), np.uint8), detection))[0]
    raw["Identity_1"][:] = 10  # A high scalar cannot rescue collapsed landmarks.
    assert not (await stage._verify(np.zeros((200, 200, 3), np.uint8), detection))[0]


@pytest.mark.parametrize("score,expected", [(.95, True), (.65, False)])
async def test_strong_face_retains_pose_path_while_borderline_face_requires_verification(score, expected):
    async def predict(model, image, **kwargs):
        return Result(kind="detections", model=model, raw={}, detections=[
            Detection(label="DriverFace", score=score, box=(.2, .2, .8, .8))])

    async def unavailable(*args):
        return False, None

    stage = FaceDetectionStage(verification_model="face_mesh")
    stage.ctx = SimpleNamespace(registry=SimpleNamespace(predict=predict))
    stage._verify = unavailable
    output = await stage.process(StageItem(payload={"image_np": np.zeros((200, 200, 3), np.uint8)}))
    assert bool(output.results[0].detections) is expected


async def test_verified_mesh_is_reused_without_an_extra_inference(monkeypatch):
    stage = FaceMeshStage(model="face_mesh_onnx", allow_haar_fallback=False)
    image = np.zeros((200, 200, 3), np.uint8)
    box = (.2, .2, .8, .8)
    _, crop_box = crop_face(image, box)

    async def forbidden(*args):
        raise AssertionError("Already verified landmarks must not run twice")

    monkeypatch.setitem(stage._process.__globals__, "_infer_face_mesh", forbidden)
    raw = {"Identity": np.tile([128., 128., 0.], (1, 478)).astype(np.float32),
           "Identity_1": np.array([10.], np.float32)}
    item = StageItem(stream_id="s", ts_ms=0, payload={"image_np": image,
        "_verified_face_mesh": {"raw": raw, "crop_box": crop_box, "frame_size": (200, 200),
                                "input_size": 256, "model": "face_mesh_onnx"}},
        results=[Result(kind="detections", raw={"face_detector": "scrfd", "selected_face_box": box})])
    output = await stage.process(item)
    assert output.payload["face_mesh_confidence"] > .99
    assert "_verified_face_mesh" not in output.payload


@pytest.mark.parametrize("region", [[.5, 0, .2, 1], [0, 0, 2, 1], [0, 0, 1], [0, 0, np.nan, 1]])
def test_invalid_driver_region_rejected(region):
    with pytest.raises(ValueError, match="driver_region"):
        FaceDetectionStage(driver_region=region)


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


@pytest.mark.parametrize("bad", [np.zeros((1, 3, 3)), np.full((1, 3, 3), np.nan),
                                  -np.eye(3)[None], np.ones((1, 3, 3))])
def test_learned_pose_rejects_invalid_rotation_matrices(bad):
    assert rotation_to_pose(bad) is None


def test_learned_pitch_is_translated_from_upstream_up_positive_to_driver_down_positive():
    theta = np.radians(30)
    rotation = np.array(((1, 0, 0), (0, np.cos(theta), -np.sin(theta)),
                         (0, np.sin(theta), np.cos(theta))))
    assert rotation_to_pose(rotation)['pitch'] == pytest.approx(-30)
    assert rotation_to_pose(rotation.T)['pitch'] == pytest.approx(30)
    stage = FaceMeshStage()
    item = StageItem()
    stage._apply_pose(item, {'yaw': 0., 'pitch': 21., 'roll': 0.})
    assert item.payload['head_direction'] == 'down'
    assert item.payload['face_status_label'] == 'Looking Down'
    assert item.payload['distracted_score'] >= .5


async def test_learned_pose_survives_missing_eyelid_landmarks_and_calibrates_once(monkeypatch):
    stage = FaceMeshStage(allow_haar_fallback=False, neutral_yaw_deg=10)

    async def rejected(*args):
        return {"Identity": np.zeros((1, 1434), np.float32),
                "Identity_1": np.array([-10.], np.float32)}, "face_mesh", 256

    monkeypatch.setitem(stage._process.__globals__, "_infer_face_mesh", rejected)
    item = StageItem(stream_id="s", ts_ms=0, payload={
        "image_np": np.zeros((240, 320, 3), np.uint8),
        "face_pose_raw": {"yaw": 45., "pitch": 0., "roll": 0.},
        "face_box": (.1, .1, .5, .8), "head_pose_source": "6drepnet",
    }, results=[Result(kind="detections", raw={"face_detector": "scrfd", "by_class": {
        "DriverFace": [{"score": .95, "box": [.1, .1, .5, .8]}]}})])
    output = await stage.process(item)
    assert output.payload["head_direction"] == "left"
    assert output.payload["signals"]["head_yaw_deg"] == 35
    assert output.payload["eyes_state"] == "unknown"
    assert "drowsy_ear" not in output.payload


async def test_learned_pose_uses_export_normalization_caches_and_does_not_hide_backend_errors():
    from inferops.domain.models import ModelSpec

    feeds = []
    fail = False

    async def infer(tensor):
        if fail:
            raise RuntimeError("pose server unavailable")
        feeds.append(tensor)
        return {"rotation_matrix": np.eye(3, dtype=np.float32)[None]}

    class Registry:
        def get_spec(self, name):
            return ModelSpec(name=name, inputs=[{"name": "image", "shape": [1, 3, 224, 224]}])

        async def get_backend(self, name):
            return SimpleNamespace(infer=infer)

    stage = HeadPoseStage()
    stage.ctx = SimpleNamespace(registry=Registry())

    def item(ts):
        return StageItem(stream_id="s", ts_ms=ts,
                         payload={"image_np": np.full((100, 100, 3), (0, 128, 255), np.uint8)},
                         results=[Result(kind="detections", raw={"face_detector": "scrfd", "by_class": {
                             "DriverFace": [{"score": .95, "box": [.2, .2, .8, .8]}]}})])

    output = await stage.process(item(0))
    assert output.payload["head_pose_source"] == "6drepnet"
    assert feeds[0]["image"][0, :, 100, 100] == pytest.approx((1., 128 / 255, 0.))
    await stage.process(item(50))
    assert len(feeds) == 1
    fail = True
    with pytest.raises(RuntimeError, match="pose server unavailable"):
        await stage.process(item(200))
    await stage.teardown()
    assert not stage._cache
