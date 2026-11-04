"""Synthetic scenes for face counting tests: no weights needed.

Each person is a coloured square "face" on a black 640x640 frame. The fake
detector backends return raw RetinaFace (or YuNet) tensors that decode exactly
to the scene's boxes and keypoints, and the fake embedder derives an identity
vector from the colour it sees in the aligned crop, so identity comes from the
image just as with a real model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from inferops.adapters.outbound.projects.loader import ProjectPluginResolver
from inferops.domain.models import ModelSpec

DOMAIN_ROOT = Path(__file__).resolve().parents[1]
load_object = ProjectPluginResolver(DOMAIN_ROOT, "face-counting-tests").load_object
retinaface = load_object(".models.face_retinaface.retinaface:decode").__globals__
align = load_object(".models.face_tracker.align:align_face").__globals__
counting = load_object(".models.face_tracker.counting:parse_counting").__globals__
adapter = load_object(".models.face_tracker.adapter:_iou").__globals__
FaceCountingStage = adapter["FaceCountingStage"]

SIZE = 640
COLORS = {1: (40, 40, 220), 2: (40, 200, 40), 3: (220, 60, 40), 4: (30, 200, 230)}


@dataclass
class Face:
    person: int
    box: tuple[float, float, float, float]   # source pixels
    score: float = 0.95


def face_at(person: int, cx: float, cy: float, size: float = 80.0, score: float = 0.95) -> Face:
    return Face(person, (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2), score)


def keypoints(box) -> np.ndarray:
    """Frontal five-point layout inside the box (ArcFace template proportions)."""
    x1, y1, x2, y2 = box
    template = align["ARCFACE_TEMPLATE"] / 112.0
    return np.column_stack([x1 + template[:, 0] * (x2 - x1), y1 + template[:, 1] * (y2 - y1)])


def render(faces: list[Face]) -> np.ndarray:
    image = np.zeros((SIZE, SIZE, 3), np.uint8)
    for face in faces:
        x1, y1, x2, y2 = (round(v) for v in face.box)
        image[max(0, y1):y2, max(0, x1):x2] = COLORS[face.person]
    return image


def retinaface_outputs(faces: list[Face], size: int = SIZE) -> dict[str, np.ndarray]:
    """Raw RetinaFace tensors that decode to ``faces`` (image == input, scale 1)."""
    priors = retinaface["priors"](size, size)
    n = len(priors)
    loc = np.zeros((1, n, 4), np.float32)
    conf = np.zeros((1, n, 2), np.float32)
    conf[0, :, 0] = 1.0
    marks = np.zeros((1, n, 10), np.float32)
    used: set[int] = set()
    for face in faces:
        x1, y1, x2, y2 = (v / size for v in face.box)
        cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        cost = np.hypot(priors[:, 0] - cx, priors[:, 1] - cy) + np.abs(np.log(priors[:, 2] / bw))
        cost[list(used)] = np.inf
        i = int(np.argmin(cost))
        used.add(i)
        p = priors[i]
        loc[0, i] = ((cx - p[0]) / (0.1 * p[2]), (cy - p[1]) / (0.1 * p[3]),
                     math.log(bw / p[2]) / 0.2, math.log(bh / p[3]) / 0.2)
        conf[0, i] = (1.0 - face.score, face.score)
        k = keypoints(face.box) / size
        marks[0, i] = ((k - p[:2]) / (0.1 * p[2:])).reshape(-1)
    return {"loc": loc, "conf": conf, "landmarks": marks}


def yunet_outputs(faces: list[Face], size: int = SIZE) -> dict[str, np.ndarray]:
    """Raw YuNet tensors that decode to ``faces`` (one stride-16 cell per face)."""
    out = {}
    for stride in (8, 16, 32):
        cells = (size // stride) ** 2
        out[f"cls_{stride}"] = np.zeros((1, cells, 1), np.float32)
        out[f"obj_{stride}"] = np.zeros((1, cells, 1), np.float32)
        out[f"bbox_{stride}"] = np.zeros((1, cells, 4), np.float32)
        out[f"kps_{stride}"] = np.zeros((1, cells, 10), np.float32)
    stride, per_row = 16, size // 16
    for face in faces:
        x1, y1, x2, y2 = face.box
        cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        col, row = int(cx // stride), int(cy // stride)
        index = row * per_row + col
        out[f"cls_{stride}"][0, index, 0] = out[f"obj_{stride}"][0, index, 0] = face.score
        out[f"bbox_{stride}"][0, index] = (cx / stride - col, cy / stride - row,
                                           math.log(bw / stride), math.log(bh / stride))
        kps = keypoints(face.box) / stride - np.array([col, row])
        out[f"kps_{stride}"][0, index] = kps.reshape(-1)
    return out


def identity_vector(rgb_mean: np.ndarray) -> np.ndarray:
    """Deterministic unit vector for the colour an embedder 'sees'."""
    key = tuple(round(v * 4) for v in rgb_mean)     # coarse colour bins in [-1, 1] space
    seed = abs(hash(key)) % (2 ** 32)
    v = np.random.default_rng(seed).normal(size=512)
    return (v / np.linalg.norm(v)).astype(np.float32)


class FakeModels:
    """Registry with a fake detector (RetinaFace or YuNet) and embedder."""

    def __init__(self, detector: str = "face_retinaface"):
        self.detector = detector
        self.faces: list[Face] = []
        self.calls = {"face_retinaface": 0, "face_detector": 0, "face_tracker": 0}
        self.embedded = 0
        self.batch_limit: int | None = None

    def get_spec(self, name):
        shapes = {"face_retinaface": ("input", [1, 3, SIZE, SIZE]), "face_detector": ("input", [1, 3, SIZE, SIZE]),
                  "face_tracker": ("data", [1, 3, 112, 112])}
        input_name, shape = shapes[name]
        return ModelSpec(name=name, inputs=[{"name": input_name, "shape": shape}])

    async def get_backend(self, name):
        models = self

        class Backend:
            async def infer(self, feeds):
                models.calls[name] += 1
                if name == "face_retinaface":
                    return retinaface_outputs(models.faces)
                if name == "face_detector":
                    return yunet_outputs(models.faces)
                batch = feeds["data"]
                if models.batch_limit is not None and batch.shape[0] > models.batch_limit:
                    raise RuntimeError("batch too large")
                models.embedded += batch.shape[0]
                centre = batch[:, :, 36:76, 36:76].mean(axis=(2, 3))     # (N, 3) RGB in [-1, 1]
                return {"1333": np.stack([identity_vector(c) for c in centre])}

        return Backend()


def make_stage(models: FakeModels, **params) -> FaceCountingStage:
    detector = {"detector_model": "face_retinaface"} if models.detector == "face_retinaface" \
        else {"light_detector_model": "face_detector"}
    stage = FaceCountingStage(model="face_tracker", **{**detector, **params})
    stage.ctx = SimpleNamespace(registry=models, pipeline="face_counting")
    return stage
