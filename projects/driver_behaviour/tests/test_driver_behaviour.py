"""Driver-behaviour domain integration: fusion + alerts end-to-end.

Feeds synthetic detection-laden StageItems mirroring what the RFDetr
postprocess would produce, and asserts that the driver_fusion + driver_alerts
configs emit the expected events and alerts.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from inferops.adapters.outbound.projects.loader import ProjectPluginResolver
from inferops.adapters.outbound.projects.settings import get_spec_dir
from inferops.adapters.outbound.projects.yaml import load_yaml
from inferops.domain.fusion import FusionEngine
from inferops.domain.fusion.spec import FusionSpec
from inferops.domain.incident_spec import AlertSpec
from inferops.domain.incidents import IncidentPolicy as AlertEngine
from inferops.domain.models import ModelSpec
from inferops.domain.types import Frame, Result, StageItem

DOMAIN_ROOT = Path(__file__).resolve().parents[1]
load_object = ProjectPluginResolver(DOMAIN_ROOT, "driver-tests").load_object
DB = get_spec_dir("fusion").parent
RFDetrPostprocess = load_object(".models.rfdetr_driver.adapter:RFDetrPostprocess")


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
    first_alerts = []
    for i in range(8):
        evs = await fusion.ingest_item(_item("cam9", ts + i * 250, item_payload))
        for ev in evs:
            from inferops.domain.incident_spec import AlertCandidate
            a = await alerts.submit(AlertCandidate(
                type=ev.type, entity_id=ev.entity_id, ts_ms=ev.ts_ms, payload=ev.payload))
            if a:
                first_alerts.append(a)

    assert first_alerts, "expected at least one alert"
    assert any(a.type == "phone_use" and a.severity == "medium" for a in first_alerts)

    # Immediately re-submitting the same candidate is deduped (same key, within window).
    from inferops.domain.incident_spec import AlertCandidate
    dup = await alerts.submit(AlertCandidate(type="phone_use", entity_id="cam9",
                                             ts_ms=ts + 8 * 250, payload={}))
    assert dup is None


@pytest.mark.asyncio
async def test_driver_fusion_face_and_hand_signals():
    spec = FusionSpec(**load_yaml(DB / "fusion" / "driver.yaml"))
    eng = FusionEngine(spec)
    eng.spec.outputs.events_topic = None
    ts = 3_000_000
    events: list = []
    for i in range(4):
        it = _item("cam2", ts + i * 300, {})
        it.payload.update({"drowsy_ear": 0.9, "distracted_score": 0.8, "hand_on_wheel": 0.0})
        events.extend(await eng.ingest_item(it))
    types = {e.type for e in events}
    assert {"drowsiness", "distraction", "hands_off_wheel"} <= types


@pytest.mark.asyncio
async def test_rfdetr_postprocess_preserves_legacy_pixel_boxes():
    post = RFDetrPostprocess(conf_threshold=0.5)
    logits = np.full((1, 3, 6), -6.0, dtype=np.float32)
    logits[0, 0, 3] = 6.0  # phone
    boxes = np.array([[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.05, 0.05], [0.9, 0.9, 0.1, 0.1]]], dtype=np.float32)
    spec = ModelSpec(name="rfdetr_driver", inputs=[])
    frame = Frame(stream_id="s", ts_ms=0, width=640, height=480)
    res = await post({"logits": logits, "boxes": boxes}, frame, spec)
    assert len(res.detections) == 1
    det = res.detections[0]
    assert det.label == "phone"
    assert det.attrs["xyxy_pixels"] == pytest.approx((256.0, 192.0, 384.0, 288.0))
    assert res.raw["by_class"]["phone"][0]["label"] == "phone"


@pytest.mark.asyncio
async def test_rfdetr_drops_the_background_class_column():
    """The Triton export has 7 logit columns for 6 labels; column 6 is no-object."""
    logits = np.full((1, 2, 7), -6.0, dtype=np.float32)
    logits[0, 0, 6] = 6.0   # background slot must not become a detection
    logits[0, 1, 3] = 6.0   # phone
    boxes = np.array([[[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]]], dtype=np.float32)
    post = RFDetrPostprocess(conf_threshold=0.5)
    frame = Frame(stream_id="s", ts_ms=0, width=100, height=100)
    spec = ModelSpec(name="rfdetr_driver", inputs=[])
    res = await post({"labels": logits, "dets": boxes}, frame, spec)
    assert [d.label for d in res.detections] == ["phone"]


@pytest.mark.asyncio
async def test_rfdetr_uses_labels_from_the_model_spec():
    logits = np.full((1, 1, 3), -6.0, dtype=np.float32)
    logits[0, 0, 2] = 6.0
    boxes = np.array([[[0.5, 0.5, 0.2, 0.2]]], dtype=np.float32)
    frame = Frame(stream_id="s", ts_ms=0, width=100, height=100)
    post = RFDetrPostprocess(conf_threshold=0.5)
    spec = ModelSpec(name="m", inputs=[], labels=["one", "two", "three"])
    res = await post({"logits": logits, "boxes": boxes}, frame, spec)
    assert res.detections[0].label == "three"

    post = RFDetrPostprocess(conf_threshold=0.5, class_names=["x", "y", "z"])
    res = await post({"logits": logits, "boxes": boxes}, frame, spec)
    assert res.detections[0].label == "z"


@pytest.mark.asyncio
async def test_face_mesh_propagates_model_failures_without_fallback():
    FaceMeshStage = load_object(".models.face_mesh.adapter:FaceMeshStage")
    infer_face_mesh = load_object(".models.face_mesh.adapter:_infer_face_mesh")

    class Registry:
        def get_spec(self, name):
            return ModelSpec(name=name, inputs=[{"name": "input", "shape": [1, 3, 256, 256]}])

        async def get_backend(self, name):
            raise RuntimeError("required weights unavailable")

    stage = FaceMeshStage()
    assert stage.fallback_model is None
    stage.ctx = SimpleNamespace(registry=Registry())
    with pytest.raises(RuntimeError, match="required weights unavailable"):
        await infer_face_mesh(stage, np.zeros((64, 64, 3), dtype=np.uint8))


