"""Every newly counted person becomes its own alert (stage -> the project's alert policy)."""
from pathlib import Path

import pytest
from synthetic import FakeModels, face_at, make_stage, render

from inferops.adapters.outbound.projects.loader import ProjectLoader
from inferops.domain.incidents import IncidentPolicy
from inferops.domain.types import StageItem

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def alert_spec(monkeypatch):
    monkeypatch.setenv("INFEROPS_MODELS_DIR", str(ROOT.parents[1] / "models"))
    definition = ProjectLoader(ROOT, configs_dir=ROOT.parents[1] / "tests").load({})
    stage = definition.pipelines["face_counting"].stages[1]
    assert stage.params["new_person_alert"] == "low"           # the project default
    return definition.alerts["face_counting"]


async def run(stage, models, policy, frames):
    alerts, t = [], 0
    for faces in frames:
        models.faces = faces
        item = await stage.process(StageItem(stream_id="cam-1", ts_ms=t, payload={"image_np": render(faces)}))
        alerts += await policy.ingest_item(item)
        t += 100
    return alerts


@pytest.mark.asyncio
async def test_each_new_person_raises_its_own_alert(alert_spec):
    models = FakeModels()
    stage = make_stage(models, new_person_alert="low")
    two = [face_at(1, 150, 300), face_at(2, 450, 300)]
    three = [*two, face_at(3, 300, 150)]
    alerts = await run(stage, models, IncidentPolicy(alert_spec), [two] * 4 + [three] * 4)
    # Persons 1 and 2 are counted on the same frame, person 3 later: three alerts.
    assert [a.type for a in alerts] == ["new_person"] * 3
    assert [a.payload["number"] for a in alerts] == [1, 2, 3]
    assert {a.severity for a in alerts} == {"low"}
    assert len({a.dedupe_key for a in alerts}) == 3
    assert alerts[2].payload["message"] == "New person #3 counted" and len(alerts[2].payload["box"]) == 4


@pytest.mark.asyncio
async def test_the_alert_severity_follows_the_setting_and_off_disables_it(alert_spec):
    for level, expected in (("high", ["high"]), ("off", [])):
        models = FakeModels()
        stage = make_stage(models, new_person_alert=level)
        alerts = await run(stage, models, IncidentPolicy(alert_spec), [[face_at(1, 300, 300)]] * 4)
        assert [a.severity for a in alerts] == expected
    with pytest.raises(ValueError):
        make_stage(FakeModels(), new_person_alert="loud")
