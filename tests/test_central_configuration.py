"""One editable document drives native settings, project loading and Docker."""
from __future__ import annotations

import json
import os

import pytest
import yaml
from dotenv import dotenv_values

from inferops.adapters.outbound.projects import central
from inferops.adapters.outbound.projects.configuration import validate_document
from inferops.adapters.outbound.projects.loader import ProjectLoader
from inferops.adapters.outbound.projects.preparation import selected_surface
from inferops.adapters.outbound.projects.settings import ConfigStack, load_settings
from inferops.domain.errors import ConfigError
from inferops.testing import PROJECT_ROOT
from scripts.configure import render


@pytest.fixture
def central_config(tmp_path, monkeypatch):
    monkeypatch.setenv("INFEROPS_MODELS_DIR", str(PROJECT_ROOT / "models"))
    directory = tmp_path / "configs"
    directory.mkdir()
    data = yaml.safe_load((PROJECT_ROOT / "configs/inferops.yaml").read_text())
    data["runtime"].update(mode="cpu", api_port=8093)
    data["projects"]["driver_behaviour"]["root"] = str(PROJECT_ROOT / "projects/driver_behaviour")
    path = directory / "inferops.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path, data


def test_loader_and_preparation_use_central_settings_in_every_mode(central_config):
    path, data = central_config
    data["projects"]["driver_behaviour"]["settings"] = {"detection_confidence": 0.85, "confirm_ms": 4000}
    path.write_text(yaml.safe_dump(data))
    root = central.project_root(path, data)
    for mode in ("cpu", "cuda", "triton", "tensorrt"):
        definition = ProjectLoader(root, configs_dir=path.parent, mode=mode).load()
        params = definition.models["rfdetr_driver_onnx"].postprocess.params
        assert params["conf_threshold"] == 0.85
        rules = central.documents(path, data, root, path.parent)
        assert any("confirm_ms: 4000" in yaml.safe_dump(value) for value in rules.values())
        _, selected = selected_surface(ConfigStack(path.parent, domain_root=root), mode)
        assert next(spec for _, spec in selected if spec.name.startswith("rfdetr_driver")).postprocess.params["conf_threshold"] == 0.85


def test_native_settings_read_central_yaml_and_keep_explicit_environment(central_config, monkeypatch):
    path, data = central_config
    monkeypatch.setattr(central, "_exported", {})
    for key in {*central.environment(path, data), "INFEROPS_CONFIG_VALUES"}:
        existed = key in os.environ
        monkeypatch.setenv(key, os.getenv(key, ""))
        if not existed:
            os.environ.pop(key)
    monkeypatch.setenv("INFEROPS_CONFIGS_DIR", str(path.parent))
    monkeypatch.delenv("INFEROPS_EXECUTION_MODE", raising=False)
    monkeypatch.delenv("INFEROPS_API_PORT", raising=False)
    monkeypatch.delenv("INFEROPS_PROJECT_ROOT", raising=False)
    monkeypatch.delenv("INFEROPS_DOMAIN_ROOT", raising=False)
    settings = load_settings()
    assert settings.api.port == 8093 and settings.execution_mode == "cpu"
    assert settings.domain_root == str(central.project_root(path, data))
    # A second read after editing must not be shadowed by earlier generated env.
    data["runtime"]["api_port"] = 8094
    path.write_text(yaml.safe_dump(data))
    assert load_settings().api.port == 8094
    monkeypatch.setenv("INFEROPS_API_PORT", "8095")
    assert load_settings().api.port == 8095


def test_renderer_keeps_secrets_private_and_does_not_modify_inputs(central_config, tmp_path):
    path, _ = central_config
    env_file = path.parent.parent / ".env"
    env_file.write_text("INFEROPS_S3_SECRET_KEY='a-secret-with-$-and-backticks'\nHF_TOKEN='hf_test'\n")
    before = (path.read_bytes(), env_file.read_bytes())
    values = render(config_file=path, output=tmp_path / "rendered", environ={})
    container = dotenv_values(tmp_path / "rendered/container.env")
    assert values["INFEROPS_API_PORT"] == "8093"
    assert container["INFEROPS_API_PORT"] == "8080"
    assert container["INFEROPS_S3_SECRET_KEY"] == "a-secret-with-$-and-backticks"
    assert container["HF_TOKEN"] == "hf_test"
    assert container["INFEROPS_DOMAIN_ROOT"] == "/project"
    assert container["INFEROPS_DATABASE_URL"].split("@")[1] == "postgres:5432/inferops"
    assert "S3_SECRET_KEY" not in container["INFEROPS_CONFIG_VALUES"]
    assert json.loads(container["INFEROPS_SOURCE_DEFAULTS"])["inference_fps"] == 30
    assert all(file.stat().st_mode & 0o077 == 0 for file in (tmp_path / "rendered").iterdir())
    assert (path.read_bytes(), env_file.read_bytes()) == before


@pytest.mark.parametrize("document", [
    {"runtime": {"mode": "bad"}}, {"runtime": {"api_port": 0}},
    {"runtime": {"services": ["mystery"]}}, {"sources": {"inference_fps": 0}},
    {"sources": {"not_a_setting": 1}}, {"evidence": {"sample_fps": -1}},
    {"projects": {"escape": {"root": ".", "overrides": {"pipelines": {"../escape": {}}}}}},
    {"projects": {"old": {"root": ".", "parameters": {}}}},
])
def test_central_document_rejects_invalid_values(document):
    with pytest.raises((ValueError, ConfigError)):
        validate_document("settings", document)


def test_worker_manifest_can_live_in_the_same_yaml(central_config):
    from inferops.adapters.outbound.persistence.worker_manifest import load_worker_manifest
    path, data = central_config
    data["workers"] = {"version": 1, "workers": [{"worker_id": "a", "endpoint": "http://camera-a:8080"}], "cameras": []}
    path.write_text(yaml.safe_dump(data))
    assert load_worker_manifest(path).workers[0].worker_id == "a"
    assert central.environment(path, data)["INFEROPS_WORKER_MANIFEST"] == str(path)


def test_native_yaml_defaults_do_not_turn_into_docker_host_overrides(central_config, tmp_path, monkeypatch):
    path, data = central_config
    monkeypatch.setattr(central, "_exported", {})
    generated = central.environment(path, data, context="native")
    process = {**generated, "INFEROPS_CONFIG_VALUES": json.dumps(generated)}
    render(config_file=path, output=tmp_path / "generated", environ=process)
    container = dotenv_values(tmp_path / "generated/container.env")
    assert container["INFEROPS_TRITON_URL"] == "triton:8001"
    assert container["INFEROPS_MEDIAMTX_API_URL"] == "http://mediamtx:9997"
    assert container["INFEROPS_DATABASE_URL"].split("@")[1] == "postgres:5432/inferops"


def test_editor_saves_store_only_changes_and_round_trip(central_config):
    path, data = central_config
    root = central.project_root(path, data)
    fusion = root / "rules.yaml"
    effective = central.documents(path, data, root, path.parent)[fusion]
    assert central.store_document(path, data, root, path.parent, fusion, effective) == data
    edited = yaml.safe_load(yaml.safe_dump(effective))
    edited["rules"][0]["when"] = "phone >= 0.6"
    edited["rules"] = [rule for rule in edited["rules"] if rule["name"] != "eating"]
    edited["rules"].append({"name": "yawning", "when": "eating > 2", "severity": "low"})
    saved = central.store_document(path, data, root, path.parent, fusion, edited)
    patch = saved["projects"]["driver_behaviour"]["overrides"]["rules"]
    assert patch == {"rules": [{"name": "phone_use", "when": "phone >= 0.6"},
                               {"name": "yawning", "when": "eating > 2", "severity": "low"},
                               {"name": "eating", "$remove": True}]}
    assert central.documents(path, saved, root, path.parent)[fusion] == edited
    edited["rules"].reverse()
    reordered = central.store_document(path, data, root, path.parent, fusion, edited)
    assert central.documents(path, reordered, root, path.parent)[fusion] == edited


@pytest.mark.parametrize("settings", [{"unknown": 1}, {"detection_confidence": 1.5}, {"confirm_ms": "slow"}])
def test_project_settings_are_declared_typed_and_bounded(central_config, settings):
    path, data = central_config
    data["projects"]["driver_behaviour"]["settings"] = settings
    with pytest.raises(ConfigError, match="setting"):
        central.documents(path, data, central.project_root(path, data), path.parent)


def test_fusion_defaults_expand_rules_windows_and_transitions():
    from inferops.domain.fusion.spec import FusionSpec
    spec = FusionSpec(name="f", defaults={"rule": {"confirm_ms": 100, "cooldown_ms": 5},
                                          "transition": {"min_samples": 2},
                                          "window": {"duration_ms": 900}, "extract": "signal('{name}')"},
                      extract={"a": "detected('a')"}, signals=["a", "b"],
                      rules=[{"name": "r", "when": "a", "cooldown_ms": 7}],
                      state_machine={"initial": "X", "states": ["X", "Y", "Z"],
                                     "transitions": [{"from_state": ["X", "Y"], "to_state": "Z", "when": "a"}]})
    assert (spec.rules[0].confirm_ms, spec.rules[0].cooldown_ms) == (100, 7)
    assert {name: window.duration_ms for name, window in spec.windows.items()} == {"a": 900, "b": 900}
    assert spec.extract == {"a": "detected('a')", "b": "signal('b')"}
    assert [(t.from_state, t.min_samples) for t in spec.state_machine.transitions] == [("X", 2), ("Y", 2)]
