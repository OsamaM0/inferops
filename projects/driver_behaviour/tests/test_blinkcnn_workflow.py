"""Production head direction, independent eye state, and stream lifecycle contracts."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from inferops.adapters.outbound.projects.loader import ProjectLoader, ProjectPluginResolver
from inferops.domain.types import Detection, Result, StageItem

load = ProjectPluginResolver(Path(__file__).resolve().parents[1], "blinkcnn-tests").load_object
Head = load(".models.head_pose_360.adapter:HeadPose360Stage")
Landmarks = load(".models.landmarks_106.adapter:Landmarks106Stage")
Eyes = load(".models.blinkcnn.adapter:BlinkCNNStage")
pose_from_rotation = load(".models.face_monitoring.geometry:rotation_to_pose")
eye_inputs = load(".models.face_monitoring.geometry:eye_inputs")


def rotation(yaw=0, down=0):
    y, x = np.radians([-yaw, -down])
    ry = np.array(((np.cos(y), 0, np.sin(y)), (0, 1, 0), (-np.sin(y), 0, np.cos(y))))
    rx = np.array(((1, 0, 0), (0, np.cos(x), -np.sin(x)), (0, np.sin(x), np.cos(x))))
    return (ry @ rx)[None].astype(np.float32)


def points106():
    points = np.full((106, 2), (96.0, 96.0), np.float32)
    for indices, center in (((35, 41, 42, 39, 37, 36), 52), ((89, 95, 96, 93, 91, 90), 116)):
        points[list(indices)] = [
            (center, 80),
            (center + 6, 76),
            (center + 18, 76),
            (center + 24, 80),
            (center + 18, 84),
            (center + 6, 84),
        ]
    return (points / 96 - 1).reshape(1, 212)


class Registry:
    def __init__(self):
        self.calls = []
        self.rotation = rotation()
        self.points = points106()
        self.scores = [0.02, 0.03]

    def get_spec(self, name):
        names = {
            "head_pose_360": ("image", "rotation"),
            "landmarks_106": ("data", "fc1"),
            "blinkcnn": ("crops", "sequence"),
        }
        source, target = names[name]
        return SimpleNamespace(inputs=[SimpleNamespace(name=source)], outputs=[SimpleNamespace(name=target)])

    async def infer(self, name, feeds):
        self.calls.append((name, feeds))
        if name == "head_pose_360":
            assert feeds["image"].shape == (1, 3, 224, 224)
            return {"rotation": self.rotation}
        if name == "landmarks_106":
            assert feeds["data"].shape == (1, 3, 192, 192)
            return {"fc1": self.points}
        assert feeds["crops"].shape == (2, 1, 3, 64, 64)
        assert feeds["crops"].dtype == np.float32
        assert feeds["crops"].min() >= 0 and feeds["crops"].max() <= 1
        scores = np.asarray(self.scores, np.float32)
        return {"sequence": np.log(scores / (1 - scores)).reshape(2, 1, 1)}


def frame(ts=0, stream="driver", face=True, box=(0.2, 0.2, 0.8, 0.8), size=400):
    item = StageItem(
        stream_id=stream,
        ts_ms=ts,
        payload={
            "image_np": np.full((size, size, 3), (0, 128, 255), np.uint8),
            "face_detection_age_ms": 7,
            "face_search_method": "driver_crop",
        },
    )
    if face:
        item.results.append(
            Result(
                kind="detections",
                detections=[Detection(label="DriverFace", score=0.95, box=box)],
                raw={"face_detector": "scrfd"},
            )
        )
    return item


def workflow(interval=0):
    registry = Registry()
    stages = [
        Head(min_infer_interval_ms=interval),
        Landmarks(min_infer_interval_ms=interval),
        Eyes(min_infer_interval_ms=interval),
    ]
    for stage in stages:
        stage.ctx = SimpleNamespace(registry=registry)
    return registry, stages


async def run(stages, item):
    for stage in stages:
        item = await stage.process(item)
    return item.payload


@pytest.mark.parametrize(
    "yaw,down,label",
    [
        (0, 0, "forward"),
        (35, 0, "left"),
        (-35, 0, "right"),
        (0, 30, "down"),
        (0, -30, "up"),
        (120, 0, "left"),
        (-120, 0, "right"),
        (170, 0, "left"),
        (-170, 0, "right"),
    ],
)
async def test_all_driver_head_directions_and_full_yaw_range(yaw, down, label):
    registry, stages = workflow()
    registry.rotation = rotation(yaw, down)
    output = await run(stages, frame())
    assert output["head_direction"] == label
    assert output["face_pose_raw"]["yaw"] == pytest.approx(yaw, abs=1e-4)
    assert output["face_pose_raw"]["pitch"] == pytest.approx(down, abs=1e-4)
    assert output["signals"]["head_pose_source"] == "6drepnet360"
    if abs(yaw) > 40:
        assert output["eyes_state"] == "unknown" and "drowsy_ear" not in output
        assert [name for name, _ in registry.calls] == ["head_pose_360", "landmarks_106"]


@pytest.mark.parametrize("bad", [np.zeros((1, 3, 3)), np.full((1, 3, 3), np.nan), -np.eye(3)])
async def test_invalid_head_matrix_keeps_face_and_never_claims_closed_eyes(bad):
    registry, stages = workflow()
    registry.rotation = bad
    registry.scores = [0.99, 0.99]
    output = await run(stages, frame())
    assert output["face_detected"] == 1 and output["face_source"] == "scrfd"
    assert output["head_direction"] == output["eyes_state"] == "unknown"
    assert "drowsy_ear" not in output


async def test_blinkcnn_is_authoritative_and_input_is_rgb_without_double_normalization():
    registry, stages = workflow()
    output = await run(stages, frame())
    assert output["eyes_state"] == "open"
    assert output["eye_state_source"] == output["signals"]["eye_state_source"] == "blinkcnn"
    assert output["head_pose_source"] == "6drepnet360"
    assert output["face_detection_age_ms"] == 7 and output["face_search_method"] == "driver_crop"
    pixels = registry.calls[-1][1]["crops"]
    assert pixels[0, 0, :, 32, 32] == pytest.approx([1, 128 / 255, 0])
    registry.scores = [0.98, 0.97]
    closed = await run(stages, frame(100))
    assert closed["eyes_state"] == "closed" and closed["drowsy_ear"] == 1
    assert closed["ear_left"] > 0.1  # EAR is diagnostic; it cannot override BlinkCNN.
    assert closed["signals"]["eye_closed_score_left"] == pytest.approx(0.98)


async def test_bilateral_closure_hysteresis_duration_winks_and_stream_isolation():
    registry, stages = workflow()
    await run(stages, frame(0))
    registry.scores = [0.9, 0.9]
    assert (await run(stages, frame(100)))["eyes_closed_ms"] == 0
    registry.scores = [0.3, 0.3]
    held = await run(stages, frame(500))
    assert held["eyes_state"] == "closed" and held["eyes_closed_ms"] == 400
    assert (await run(stages, frame(500, "passenger")))["eyes_state"] == "unknown"
    registry.scores = [0.99, 0.01]
    wink = await run(stages, frame(600))
    assert wink["eyes_state"] == "mixed" and wink["drowsy_ear"] == 0
    registry.scores = [0.9, 0.9]
    assert (await run(stages, frame(2000)))["eyes_closed_ms"] == 0
    assert (await run(stages, frame(50)))["eyes_closed_ms"] == 0


async def test_missing_face_clears_previous_outputs_and_never_runs_models():
    registry, stages = workflow()
    registry.scores = [0.99, 0.99]
    item = frame()
    await run(stages, item)
    registry.calls.clear()
    item.results.clear()
    item.ts_ms = 200
    output = await run(stages, item)
    assert not registry.calls
    assert output["face_detected"] == 0
    assert "face_box" not in output and "drowsy_ear" not in output
    assert output["head_direction"] == output["eyes_state"] == "unknown"
    assert "head_yaw_deg" not in output["signals"]


async def test_cache_isolated_by_source_size_time_and_selected_face():
    registry, stages = workflow(interval=100)
    await run(stages, frame(100))
    assert len(registry.calls) == 3
    output = await run(stages, frame(150))
    assert len(registry.calls) == 3
    assert output["head_pose_age_ms"] == output["eye_state_age_ms"] == 50
    await run(stages, frame(150, "other"))
    await run(stages, frame(20))
    await run(stages, frame(30, size=500))
    await run(stages, frame(40, size=500, box=(0.1, 0.1, 0.6, 0.6)))
    assert len(registry.calls) == 15
    for stage in stages:
        await stage.teardown()


async def test_landmark_or_logit_corruption_resets_eye_evidence():
    registry, stages = workflow()
    registry.scores = [0.99, 0.99]
    await run(stages, frame(0))
    registry.points = np.full((1, 212), np.nan)
    bad = await run(stages, frame(100))
    assert bad["head_direction"] == "forward" and bad["eyes_state"] == "unknown"
    assert "drowsy_ear" not in bad
    registry.points = points106()
    registry.scores = [np.nan, 0.9]
    bad = await run(stages, frame(200))
    assert bad["eyes_state"] == "unknown" and "drowsy_ear" not in bad
    assert not stages[-1]._eye_history


async def test_behavior_object_intersections_survive_without_mediapipe():
    _registry, stages = workflow()
    item = frame()
    item.results.insert(
        0,
        Result(
            kind="detections",
            raw={
                "by_class": {
                    "foodItem": [{"score": 0.91, "box": [0.4, 0.6, 0.5, 0.7]}],
                    "cigarette": [{"score": 0.92, "box": [0.4, 0.6, 0.5, 0.7]}],
                }
            },
        ),
    )
    output = await run(stages, item)
    assert output["eating_face_score"] == 0.91 and output["smoking_face_score"] == 0.92


def test_tiny_or_offscreen_eyes_are_not_a_closed_eye_prediction():
    image = np.zeros((200, 200, 3), np.uint8)
    for points in (np.zeros((106, 2)), np.full((106, 2), np.nan)):
        assert eye_inputs(image, points, {"yaw": 0, "pitch": 0}) is None


@pytest.mark.parametrize("values", [(0.1, 0.2), (0.5, 0.5), (1.1, 0.2), (0.5, -0.1)])
def test_invalid_eye_thresholds_are_rejected(values):
    with pytest.raises(ValueError, match="thresholds"):
        Eyes(closed_threshold=values[0], open_threshold=values[1])


@pytest.mark.parametrize("mode", ["cpu", "cuda", "triton", "tensorrt"])
def test_resolved_pipeline_can_construct_all_custom_stages(mode):
    definition = ProjectLoader(Path(__file__).resolve().parents[1], mode=mode).load()
    for name in ("driver_face", "driver_behaviour"):
        for declaration in definition.pipelines[name].stages:
            if ":" in declaration.stage:
                definition.object_loader(declaration.stage)(**declaration.params)
