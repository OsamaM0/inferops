"""Five-point alignment to the ArcFace template."""
import math

import numpy as np
import pytest
from synthetic import align, keypoints

TEMPLATE = align["ARCFACE_TEMPLATE"]
similarity_transform, align_face, to_tensor, frontal = (
    align["similarity_transform"], align["align_face"], align["to_tensor"], align["frontal"])


def test_template_maps_to_identity():
    matrix = similarity_transform(TEMPLATE, TEMPLATE)
    assert matrix == pytest.approx(np.hstack([np.eye(2), np.zeros((2, 1))]), abs=1e-9)


def test_recovers_rotation_scale_and_translation():
    angle, scale, shift = math.radians(20), 2.5, np.array([300.0, 150.0])
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    source = TEMPLATE @ (scale * rotation).T + shift
    matrix = similarity_transform(source, TEMPLATE)
    mapped = source @ matrix[:, :2].T + matrix[:, 2]
    assert mapped == pytest.approx(TEMPLATE, abs=1e-6)


def test_align_face_puts_the_keypoints_on_the_template():
    image = np.zeros((480, 640, 3), np.uint8)
    box = (300.0, 200.0, 412.0, 312.0)
    points = keypoints(box)
    nose = tuple(round(v) for v in points[2])
    image[nose[1] - 1:nose[1] + 2, nose[0] - 1:nose[0] + 2] = 255
    crop = align_face(image, points)
    assert crop.shape == (112, 112, 3)
    y, x = np.unravel_index(crop[..., 0].argmax(), crop.shape[:2])
    assert (x, y) == pytest.approx(TEMPLATE[2], abs=1.5)


def test_tensor_is_rgb_nchw_in_minus_one_to_one():
    crop = np.zeros((112, 112, 3), np.uint8)
    crop[..., 2] = 255                                  # red in BGR
    tensor = to_tensor([crop, crop])
    assert tensor.shape == (2, 3, 112, 112) and tensor.dtype == np.float32
    assert tensor[0, :, 0, 0] == pytest.approx((1.0, -1.0, -1.0))


def test_frontal_rejects_profiles():
    points = keypoints((0, 0, 112, 112))
    assert frontal(points)
    profile = points.copy()
    profile[2, 0] = profile[1, 0] + 5                   # nose beyond the image-right eye
    assert not frontal(profile)
    assert not frontal(np.zeros((5, 2)))
