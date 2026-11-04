"""RetinaFace priors, preprocessing and decoding (no weights needed)."""
import asyncio

import numpy as np
import pytest
from synthetic import Face, face_at, keypoints, load_object, retinaface, retinaface_outputs

from inferops.domain.models import ModelSpec
from inferops.domain.types import Frame

priors, decode, preprocess = retinaface["priors"], retinaface["decode"], retinaface["preprocess"]


def test_priors_follow_the_reference_layout():
    p = priors(640, 640)
    assert p.shape == (16800, 4)                      # 80*80*2 + 40*40*2 + 20*20*2
    # First cell of stride 8: both anchor sizes before the next cell.
    assert p[0] == pytest.approx((4 / 640, 4 / 640, 16 / 640, 16 / 640))
    assert p[1] == pytest.approx((4 / 640, 4 / 640, 32 / 640, 32 / 640))
    assert p[2] == pytest.approx((12 / 640, 4 / 640, 16 / 640, 16 / 640))
    assert p[12800] == pytest.approx((8 / 640, 8 / 640, 64 / 640, 64 / 640))   # stride 16 starts
    assert priors(480, 640).shape == (60 * 80 * 2 + 30 * 40 * 2 + 15 * 20 * 2, 4)


def test_decode_recovers_box_landmarks_and_scale():
    face = face_at(1, 300.0, 200.0, size=100.0, score=0.9)
    faces = decode(retinaface_outputs([face]), scale=0.5, score_threshold=0.5)
    assert len(faces) == 1 and faces[0].score == pytest.approx(0.9)
    # Input pixels / scale 0.5 -> source pixels doubled.
    assert faces[0].box == pytest.approx(tuple(v * 2 for v in face.box), abs=1e-3)
    assert faces[0].keypoints == pytest.approx(keypoints(face.box) * 2, abs=1e-3)
    assert decode(retinaface_outputs([face]), scale=1.0, score_threshold=0.95) == []


def test_nms_keeps_the_best_of_overlapping_boxes_and_sorts_by_score():
    a = face_at(1, 300.0, 300.0, size=64.0, score=0.8)
    b = Face(1, tuple(v + 8 for v in a.box), score=0.95)       # IoU ~0.6 with a
    c = face_at(2, 100.0, 100.0, size=64.0, score=0.7)
    faces = decode(retinaface_outputs([a, b, c]), scale=1.0, score_threshold=0.5, nms_threshold=0.4)
    assert [round(f.score, 2) for f in faces] == [0.95, 0.7]


def test_decode_rejects_outputs_for_a_different_input_size():
    with pytest.raises(ValueError, match="priors"):
        decode(retinaface_outputs([face_at(1, 100, 100)], size=640), scale=1.0, size=320)


def test_preprocess_letterboxes_top_left_with_zero_padding():
    image = np.full((360, 640, 3), 200, np.uint8)
    feeds, scale = preprocess(image, "input", 640)
    tensor = feeds["input"]
    assert tensor.shape == (1, 3, 640, 640) and scale == 1.0
    assert tensor[0, :, :360].mean(axis=(1, 2)) == pytest.approx((96.0, 83.0, 77.0))   # 200 - mean (BGR)
    assert np.all(tensor[0, :, 360:] == 0.0)


def test_standalone_postprocess_returns_normalised_face_detections():
    pre = load_object(".models.face_retinaface.adapter:RetinaFacePreprocess")()
    post = load_object(".models.face_retinaface.adapter:RetinaFacePostprocess")(conf_threshold=0.5, min_face_px=30)
    spec = ModelSpec(name="face_retinaface", inputs=[{"name": "input", "shape": [1, 3, 640, 640]}])
    frame = Frame(stream_id="s", ts_ms=0, image=np.zeros((480, 640, 3), np.uint8), width=640, height=480)
    asyncio.run(pre(frame, spec))
    big, small = face_at(1, 320, 240, size=100), face_at(2, 100, 100, size=20)
    result = asyncio.run(post(retinaface_outputs([big, small]), frame, spec))
    assert [d.label for d in result.detections] == ["face"]      # the 20 px face is filtered
    det = result.detections[0]
    assert det.box == pytest.approx((270 / 640, 190 / 480, 370 / 640, 290 / 480), abs=1e-4)
    assert len(det.attrs["keypoints"]) == 5
