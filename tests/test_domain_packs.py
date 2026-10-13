"""Domain-pack contract: manifest, pack-relative plugin refs, scaffolding, evidence triggers."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from inferops.core.config import ConfigStack
from inferops.core.domain import (
    DomainManifest,
    activate_domain_imports,
    discover_domains,
    load_domain_manifest,
    manifest_env,
    resolve_plugin_ref,
    scaffold_domain,
)
from inferops.core.plugins import load_object
from inferops.core.types import Result, StageItem
from inferops.pipeline.stages.sink_s3 import SinkS3
from inferops.testing import PROJECT_ROOT


def test_workspace_projects_take_precedence_over_legacy_sibling_projects(tmp_path):
    engine = tmp_path / "engine"
    (engine / "configs").mkdir(parents=True)
    (engine / "pyproject.toml").touch()
    (tmp_path / "compose.yaml").touch()
    current = engine / "projects/local_project"
    legacy = tmp_path / "projects/local_project"
    for root in (current, legacy):
        root.mkdir(parents=True)
        (root / "project.yaml").write_text("name: local_project\npipelines: []\n")
    try:
        assert discover_domains(engine / "configs") == [("local_project", current)]
        assert ConfigStack(engine / "configs", active_domain="local_project").domain_root == current
    finally:
        ConfigStack(PROJECT_ROOT / "configs")


def test_manifest_defaults_and_validation(tmp_path: Path):
    root = tmp_path / "my_pack"
    (root / "configs").mkdir(parents=True)
    manifest = load_domain_manifest(root)
    assert manifest is not None and manifest.name == "my_pack" and manifest.triton is False

    (root / "domain.yaml").write_text("label: My Pack\ntriton: true\nprewarm_models: [a, b]\n", encoding="utf-8")
    manifest = load_domain_manifest(root)
    assert manifest.display_label == "My Pack" and manifest.triton and manifest.prewarm_models == ["a", "b"]

    env = manifest_env(manifest, root, root / "model_repository")
    assert env["INFEROPS_PREWARM_MODELS"] == "a,b" and env["INFEROPS_DOMAIN_TRITON"] == "1"

    with pytest.raises(ValueError):
        DomainManifest(name="not-a-package")

    assert load_domain_manifest(tmp_path / "nope") is None


def test_pack_relative_plugin_refs_resolve_against_the_active_pack(tmp_path: Path, monkeypatch):
    root = tmp_path / "packs" / "acme_vision"
    (root / "plugins").mkdir(parents=True)
    (root / "__init__.py").write_text("", encoding="utf-8")
    (root / "plugins" / "__init__.py").write_text("", encoding="utf-8")
    (root / "plugins" / "hello.py").write_text("class Hello:\n    pass\n", encoding="utf-8")

    package = activate_domain_imports(root)
    assert package == "acme_vision"
    assert resolve_plugin_ref(".plugins.hello:Hello") == "acme_vision.plugins.hello:Hello"
    assert load_object(".plugins.hello:Hello").__name__ == "Hello"

    # ConfigStack activates the pack from INFEROPS_DOMAIN_ROOT-style selection too.
    stack = ConfigStack(tmp_path / "configs", domain_root=root)
    assert stack.domain_package == "acme_vision"
    monkeypatch.delattr(sys.modules["inferops.core.domain"], "_active_package", raising=False)


def test_scaffolded_domain_validates_and_runs(tmp_path: Path):
    """`inferops domain init` output must pass validate + preflight untouched."""
    root = tmp_path / "examples" / "ppe_demo"
    scaffold_domain(root, "ppe_demo", ["person", "helmet"])
    assert (root / "project.yaml").exists() and not list((root / "configs/pipelines").glob("*.yaml"))

    env = {"INFEROPS_ACTIVE_DOMAIN": "ppe_demo", "INFEROPS_DOMAIN_ROOT": str(root),
           "INFEROPS_TRANSPORT_KIND": "inproc", "PYTHONPATH": str(PROJECT_ROOT)}
    for command in (["validate"], ["preflight", "--no-require-artifacts"]):
        proc = subprocess.run(
            [sys.executable, "-m", "inferops.cli", *command, "--domain-root", str(root)],
            cwd=PROJECT_ROOT, capture_output=True, text=True, env={**_base_env(), **env},
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr


def _base_env() -> dict[str, str]:
    import os

    return {k: v for k, v in os.environ.items() if not k.startswith("INFEROPS_")}


def _det_item(label: str, score: float, **payload) -> StageItem:
    item = StageItem(stream_id="s", ts_ms=1_000)
    item.results.append(Result(kind="detections", detections=[],
                               raw={"by_class": {label: [{"label": label, "score": score}]}}))
    item.payload.update(payload)
    return item


def test_sink_s3_trigger_rules_are_yaml_driven():
    sink = SinkS3(trigger_rules=[
        {"flag": "phone_use", "class": "phone", "min_score": 0.45},
        {"flag": "no_seatbelt", "class": "seatbelt", "max_score": 0.4},
        {"flag": "drowsiness", "signal": "drowsy_ear", "min": 0.5},
        {"flag": "hands_off", "signal": "hand_on_wheel", "max": 0.5, "default": 1.0},
    ])
    assert sink.current_flags(_det_item("phone", 0.9)) == {"phone_use"}
    assert sink.current_flags(_det_item("phone", 0.2)) == set()
    assert sink.current_flags(_det_item("seatbelt", 0.3)) == {"no_seatbelt"}
    assert sink.current_flags(_det_item("seatbelt", 0.9)) == set()
    assert sink.current_flags(_det_item("x", 0.0, drowsy_ear=0.7)) == {"drowsiness"}
    assert sink.current_flags(_det_item("x", 0.0, hand_on_wheel=0.1)) == {"hands_off"}
    assert sink.current_flags(_det_item("x", 0.0)) == set()          # default 1.0 keeps hands on

    # Signals published through the generic channel count too.
    assert sink.current_flags(_det_item("x", 0.0, signals={"drowsy_ear": 0.9})) == {"drowsiness"}

    with pytest.raises(ValueError, match="flag"):
        SinkS3(trigger_rules=[{"class": "phone"}])


def test_sink_s3_defaults_to_alert_and_fusion_events():
    sink = SinkS3()
    item = _det_item("x", 0.0, alerts=[{"type": "intrusion"}], fusion_events=[{"type": "loitering"}])
    assert sink.current_flags(item) == {"intrusion", "loitering"}
