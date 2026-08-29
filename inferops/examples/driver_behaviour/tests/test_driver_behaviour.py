"""Driver-behaviour domain integration: fusion + alerts end-to-end.

Feeds synthetic detection-laden StageItems mirroring what the RFDetr
postprocess would produce, and asserts that the driver_fusion + driver_alerts
configs emit the expected events and alerts.
"""
from pathlib import Path

import numpy as np
import pytest

from examples.driver_behaviour.plugins.postprocess_rfdetr import RFDetrPostprocess
from inferops.alerts.engine import AlertEngine
from inferops.alerts.spec import AlertSpec
from inferops.core.types import Frame, Result, StageItem
from inferops.core.yaml_loader import load_yaml
from inferops.fusion.engine import FusionEngine
from inferops.fusion.spec import FusionSpec
from inferops.registry.model_spec import ModelSpec

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "examples" / "driver_behaviour" / "configs"


def _item(stream_id: str, ts_ms: int, by_class: dict[str, list[dict]]) -> StageItem:
    it = StageItem(stream_id=stream_id, ts_ms=ts_ms)
    it.results.append(Result(kind="detections", detections=[],
                             raw={"by_class": by_class}, model="rfdetr_driver"))
    return it


@pytest.mark.asyncio
async def test_driver_fusion_phone_use_fires():
    spec = FusionSpec(**load_yaml(DB / "fusion" / "driver.yaml"))
    eng = FusionEngine(spec)
    # turn off Kafka publish for the test
    eng.spec.outputs.events_topic = None

    events: list = []
    ts = 1_000_000
    # 6 high-phone frames within a 5s window -> legacy phone score crosses 0.4.
    for i in range(6):
        events.extend(await eng.ingest_item(
            _item("cam1", ts + i * 200, {"phone": [{"label": "phone", "score": 0.9, "box": [0,0,0.1,0.1]}]})))

    types = [e.type for e in events]
    assert "phone_use" in types


@pytest.mark.asyncio
async def test_driver_alerts_dedupe_and_severity():
    fspec = FusionSpec(**load_yaml(DB / "fusion" / "driver.yaml"))
    aspec_dict = load_yaml(DB / "alerts" / "driver.yaml")
    # silence non-test sinks
    aspec_dict["sinks"] = [{"name": "stdout", "kind": "stdout", "params": {}}]
    aspec_dict["persistence_enabled"] = False
    aspec = AlertSpec(**aspec_dict)

    fusion = FusionEngine(fspec)
    fusion.spec.outputs.events_topic = None
    alerts = AlertEngine(aspec)

    ts = 2_000_000
    item_payload = {"phone": [{"label": "phone", "score": 0.95, "box": [0,0,1,1]}]}
    # drive enough samples to trigger PHONE_USE
    first_alerts: list = []
    for i in range(8):
        it = _item("cam7", ts + i * 200, item_payload)
        evs = await fusion.ingest_item(it)
        if evs:
            it.payload["fusion_events"] = [e.model_dump(mode="json") for e in evs]
            first_alerts.extend(await alerts.ingest_item(it))

    # at least one alert, severity follows the legacy UNSAFE mapping: medium.
    assert first_alerts
    assert any(a.type == "phone_use" and a.severity == "medium" for a in first_alerts)

    # immediate retry should be deduped (no new alert)
    it2 = _item("cam7", ts + 5_000, item_payload)
    evs2 = await fusion.ingest_item(it2)
    if evs2:
        it2.payload["fusion_events"] = [e.model_dump(mode="json") for e in evs2]
        dup = await alerts.ingest_item(it2)
        # Either no fresh event after cooldown, or it gets deduped at the alert layer.
        assert dup == []


@pytest.mark.asyncio
async def test_driver_fusion_face_and_hand_signals():
    spec = FusionSpec(**load_yaml(DB / "fusion" / "driver.yaml"))
    eng = FusionEngine(spec)
    eng.spec.outputs.events_topic = None

    item = StageItem(stream_id="cam-face", ts_ms=3_000_000)
    item.payload.update({
        "drowsy_ear": 0.8,
        "distracted_score": 0.7,
        "hand_on_wheel": 0.0,
    })
    events = await eng.ingest_item(item)
    types = {e.type for e in events}
    assert {"drowsiness", "distraction", "hands_off_wheel"} <= types


@pytest.mark.asyncio
async def test_rfdetr_postprocess_preserves_legacy_pixel_boxes():
    post = RFDetrPostprocess(conf_threshold=0.5)
    logits = np.array([[[0.0, 0.0, 0.0, 4.0, 0.0, 0.0]]], dtype=np.float32)
    boxes = np.array([[[0.5, 0.5, 0.25, 0.5]]], dtype=np.float32)
    frame = Frame(stream_id="cam1", ts_ms=1, width=200, height=100)
    spec = ModelSpec(name="rfdetr_driver", inputs=[])

    result = await post({"logits": logits, "boxes": boxes}, frame, spec)

    det = result.detections[0]
    assert det.label == "phone"
    assert det.box == (0.375, 0.25, 0.625, 0.75)
    assert det.attrs["xyxy_pixels"] == (75.0, 25.0, 125.0, 75.0)


@pytest.mark.asyncio
async def test_rfdetr_drops_the_background_class_column():
    """The Triton export has 7 logit columns for 6 classes.

    The 7th is DETR's no-object slot; it used to be emitted as a detection
    labelled "6", which the fusion rules then reacted to on every frame.
    """
    post = RFDetrPostprocess(conf_threshold=0.5)
    logits = np.full((1, 1, 7), -5.0, dtype=np.float32)
    logits[0, 0, 6] = 4.0                     # confident background query
    boxes = np.array([[[0.5, 0.5, 0.2, 0.2]]], dtype=np.float32)
    frame = Frame(stream_id="cam1", ts_ms=1, width=100, height=100)
    spec = ModelSpec(name="rfdetr_driver", inputs=[])

    result = await post({"logits": logits, "boxes": boxes}, frame, spec)
    assert result.detections == []
    assert result.raw["by_class"] == {}


@pytest.mark.asyncio
async def test_rfdetr_uses_labels_from_the_model_spec():
    """`labels.inline` in the model YAML was dead config: the class list was hardcoded."""
    post = RFDetrPostprocess(conf_threshold=0.5)
    logits = np.array([[[0.0, 4.0, 0.0]]], dtype=np.float32)
    boxes = np.array([[[0.5, 0.5, 0.2, 0.2]]], dtype=np.float32)
    frame = Frame(stream_id="cam1", ts_ms=1, width=100, height=100)
    spec = ModelSpec(name="custom", inputs=[],
                     labels={"inline": ["alpha", "beta", "gamma"]})

    result = await post({"logits": logits, "boxes": boxes}, frame, spec)
    assert [d.label for d in result.detections] == ["beta"]


@pytest.mark.asyncio
async def test_explicit_class_names_still_win_over_the_spec():
    post = RFDetrPostprocess(conf_threshold=0.5, class_names=["x", "y", "z"])
    logits = np.array([[[0.0, 4.0, 0.0]]], dtype=np.float32)
    boxes = np.array([[[0.5, 0.5, 0.2, 0.2]]], dtype=np.float32)
    frame = Frame(stream_id="cam1", ts_ms=1, width=100, height=100)
    spec = ModelSpec(name="custom", inputs=[], labels={"inline": ["alpha", "beta", "gamma"]})

    result = await post({"logits": logits, "boxes": boxes}, frame, spec)
    assert [d.label for d in result.detections] == ["y"]
