"""A project's one pipeline, fusion and alerts follow from its selected library models."""
from __future__ import annotations

import pytest

from inferops.adapters.outbound.projects.composition import compose
from inferops.domain.errors import ConfigError

MODELS = {
    "detector": {"name": "detector", "signals": {"phone_score": "detected('phone')"}},
    "finder": {"name": "finder"},
    "face": {"name": "face", "signals": ["look_score"],
             "stage": {"plugin": ".models.face.adapter:FaceStage", "uses": {"finder_model": "finder"},
                       "after": ["detector"], "params": {"threshold": 0.5}}},
}
RULES = {
    "signals": {"wheel": "payload.get('wheel', 1.0)"},
    "derived": {"phone": "phone_score.latest() or 0.0", "looking": "look_score.latest() or 0.0",
                "busy": "phone > 0.5 or looking > 0.5"},
    "rules": [{"name": "phone_use", "when": "phone > 0.5"}, {"name": "looking", "when": "looking > 0.5"},
              {"name": "busy", "when": "busy"}, {"name": "wheel", "when": "wheel < 0.5"}],
    "alerts": {"rated_only": True},
}


def test_all_models_run_in_dependency_order_with_helpers_wired_in():
    result = compose("monitor", {"models": ["face", "finder", "detector"], "tracking": {"enabled": True},
                                 "sinks": [{"name": "out", "stage": "sink_ws"}]}, MODELS, RULES)
    stages = result["pipeline"]["stages"]
    assert [stage["name"] for stage in stages] == ["decode", "detector", "face", "fuse", "alert"]
    assert stages[2]["params"] == {"model": "face", "finder_model": "finder", "threshold": 0.5}
    assert result["pipeline"]["tracking"] == {"model_filter": ["detector"], "enabled": True}
    assert result["pipeline"]["sinks"] == [{"name": "out", "stage": "sink_ws"}]
    assert result["fusion"]["extract"] == {"phone_score": "detected('phone')", "look_score": "signal('look_score')",
                                           "wheel": "payload.get('wheel', 1.0)"}
    assert result["fusion"]["inactive_rules"] == {}
    assert result["alerts"] == {"rated_only": True, "name": "monitor"}


def test_rules_needing_an_unselected_model_switch_off_but_its_signals_stay_declared():
    result = compose("monitor", {"models": ["detector"]}, MODELS, RULES)
    assert [stage["name"] for stage in result["pipeline"]["stages"]] == ["decode", "detector", "fuse", "alert"]
    fusion = result["fusion"]
    # Through derived values too: `busy` needs `looking`, which needs the face model.
    assert fusion["inactive_rules"] == {"looking": ["face"], "busy": ["face"]}
    assert [rule["name"] for rule in fusion["rules"]] == ["phone_use", "wheel"]
    assert "look_score" in fusion["signals"] and "look_score" not in fusion["extract"]


def test_a_helper_without_its_owner_runs_as_a_plain_model_and_unknown_models_fail():
    stages = compose("monitor", {"models": ["finder"]}, MODELS, RULES)["pipeline"]["stages"]
    assert stages[1] == {"name": "finder", "stage": "infer", "params": {"model": "finder"}}
    face_only = compose("monitor", {"models": ["face"]}, MODELS, RULES)["pipeline"]["stages"]
    assert face_only[1]["params"] == {"model": "face", "threshold": 0.5}
    with pytest.raises(ConfigError, match="unknown models"):
        compose("monitor", {"models": ["missing"]}, MODELS, RULES)
