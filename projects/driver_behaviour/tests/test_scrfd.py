"""SCRFD decoding, independent detection, recovery budgets and stream isolation."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from inferops.adapters.outbound.projects.loader import ProjectPluginResolver
from inferops.adapters.outbound.projects.models import load_model_variants
from inferops.domain.types import Detection, Frame, Result, StageItem

ROOT = Path(__file__).resolve().parents[1]
load = ProjectPluginResolver(ROOT, "scrfd-tests").load_object
SCRFDPreprocess = load(".models.scrfd.adapter:SCRFDPreprocess")
SCRFDPostprocess = load(".models.scrfd.adapter:SCRFDPostprocess")
FaceDetectionStage = load(".models.scrfd.adapter:FaceDetectionStage")


@pytest.fixture
def detector_spec():
    return load_model_variants(ROOT / "models/scrfd/model.yaml")[0]



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


@pytest.mark.parametrize('score,expected', [(.6, False), (.95, True)])
async def test_weak_proposals_need_analysis_but_strong_faces_survive_missing_analysis(score, expected):
    async def predict(model, image, **kwargs):
        return Result(kind='detections', model=model, raw={}, detections=[
            Detection(label='DriverFace', score=score, box=(.2, .2, .8, .8))])

    async def get_backend(name):
        async def infer(feeds):
            return {'landmarks': np.empty((0, 478, 3)), 'matrices': np.empty((0, 4, 4)),
                    'blink_scores': np.empty((0, 2))}
        return SimpleNamespace(infer=infer)

    stage = FaceDetectionStage(verification_model='face_landmarker')
    stage.ctx = SimpleNamespace(registry=SimpleNamespace(predict=predict, get_backend=get_backend))
    out = await stage.process(StageItem(payload={'image_np': np.zeros((200, 200, 3), np.uint8)}))
    assert bool(out.results[0].detections) == expected


async def test_face_analysis_is_reused_after_candidate_verification():
    calls = []
    points = np.zeros((1, 478, 3), np.float32)
    points[0, :, 0] = np.linspace(.3, .7, 478)
    points[0, :, 1] = np.linspace(.25, .75, 478)

    async def predict(model, image, **kwargs):
        return Result(kind='detections', model=model, raw={}, detections=[
            Detection(label='DriverFace', score=.65, box=(.2, .2, .8, .8))])

    async def get_backend(name):
        async def infer(feeds):
            calls.append(feeds['image'].shape)
            return {'landmarks': points, 'matrices': np.eye(4)[None], 'blink_scores': np.zeros((1, 2))}
        return SimpleNamespace(infer=infer)

    ctx = SimpleNamespace(registry=SimpleNamespace(predict=predict, get_backend=get_backend))
    detector = FaceDetectionStage(verification_model='face_landmarker')
    detector.ctx = ctx
    analyzer = load('.models.face_landmarker.adapter:FaceLandmarkerStage')()
    analyzer.ctx = ctx
    frame = StageItem(ts_ms=100, payload={'image_np': np.zeros((200, 200, 3), np.uint8)})
    frame = await detector.process(frame)
    assert frame.results[0].detections and '_verified_face_analysis' in frame.payload
    frame = await analyzer.process(frame)
    assert len(calls) == 1 and frame.payload['face_analysis_source'] == 'mediapipe_face_landmarker'
    assert '_verified_face_analysis' not in frame.payload



@pytest.mark.parametrize("region", [[.5, 0, .2, 1], [0, 0, 2, 1], [0, 0, 1], [0, 0, np.nan, 1]])
def test_invalid_driver_region_rejected(region):
    with pytest.raises(ValueError, match="driver_region"):
        FaceDetectionStage(driver_region=region)


async def test_weak_proposal_is_not_accepted_without_an_independent_verifier():
    async def predict(model, image, **kwargs):
        return Result(kind='detections', model=model, raw={}, detections=[
            Detection(label='DriverFace', score=.65, box=(.2, .2, .8, .8))])

    stage = FaceDetectionStage(verification_model=None, strong_confidence=.8)
    stage.ctx = SimpleNamespace(registry=SimpleNamespace(predict=predict))
    out = await stage.process(StageItem(payload={'image_np': np.zeros((200, 200, 3), np.uint8)}))
    assert not out.results[0].detections
