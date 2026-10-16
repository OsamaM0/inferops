"""Domain-pack contract: manifest, pack-relative plugin refs, scaffolding, evidence triggers."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from inferops.adapters.outbound.projects.manifest import (
    DomainManifest,
    discover_domains,
    load_domain_manifest,
    manifest_env,
    scaffold_domain,
)
from inferops.adapters.outbound.projects.settings import ConfigStack
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

    from inferops.adapters.outbound.projects.loader import ProjectPluginResolver
    original_path = list(sys.path)
    resolver = ProjectPluginResolver(root, "test-revision")
    reference = resolver.resolve(".plugins.hello:Hello")
    assert reference.startswith(resolver.package)
    assert resolver.load_object(".plugins.hello:Hello").__name__ == "Hello"
    stack = ConfigStack(tmp_path / "configs", domain_root=root)
    assert stack.domain_package == "acme_vision"
    assert sys.path == original_path


def test_scaffolded_domain_validates_and_runs(tmp_path: Path):
    """`inferops domain init` output must pass validate + preflight untouched."""
    root = tmp_path / "examples" / "ppe_demo"
    scaffold_domain(root, "ppe_demo", ["person", "helmet"])
    assert (root / "project.yaml").exists() and not list((root / "configs/pipelines").glob("*.yaml"))

    env = {"INFEROPS_ACTIVE_DOMAIN": "ppe_demo", "INFEROPS_DOMAIN_ROOT": str(root),
           "INFEROPS_TRANSPORT_KIND": "inproc", "PYTHONPATH": str(PROJECT_ROOT)}
    for command in (["validate"], ["preflight", "--no-require-artifacts"]):
        proc = subprocess.run(
            [sys.executable, "-m", "inferops.adapters.inbound.cli", *command, "--domain-root", str(root)],
            cwd=PROJECT_ROOT, capture_output=True, text=True, env={**_base_env(), **env},
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr


def _base_env() -> dict[str, str]:
    import os

    return {k: v for k, v in os.environ.items() if not k.startswith("INFEROPS_")}



def test_evidence_is_an_application_service_in_project_configuration():
    from inferops.adapters.outbound.stages.registry import APPLICATION_STAGES, BUILTINS
    assert {"evidence", "sink_s3", "fuse", "alert"} <= APPLICATION_STAGES
    assert "sink_s3" not in BUILTINS


def test_driver_annotation_remains_bound_to_explicit_project_context():
    from inferops.adapters.outbound.projects.loader import ProjectLoader
    definition = ProjectLoader(PROJECT_ROOT / "projects/driver_behaviour").load()
    pipeline = definition.pipelines["driver_behaviour"]
    annotator = next(sink.params["annotator"] for sink in pipeline.sinks if sink.params.get("annotator"))
    assert definition.object_loader(annotator).__name__ == "DriverAnnotator"
