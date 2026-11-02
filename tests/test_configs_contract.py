"""Every shipped model parses, and every project composes with any one of its models."""
from pathlib import Path

import pytest
import yaml

from inferops.adapters.outbound.projects.central import apply_model_settings
from inferops.adapters.outbound.projects.loader import ProjectLoader
from inferops.adapters.outbound.projects.models import expand_model_config, model_config_files
from inferops.adapters.outbound.projects.yaml import load_yaml

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = sorted(path.parent for path in (ROOT / "projects").glob("*/project.yaml"))


def _selections():
    for project in PROJECTS:
        models = yaml.safe_load((project / "project.yaml").read_text()).get("models")
        for selection in [None, *([model] for model in models or [])]:
            yield pytest.param(project, selection, id=f"{project.name}-{'+'.join(selection or ['all'])}")


@pytest.mark.parametrize("path", [*model_config_files(ROOT / "models"),
                                  *(path for project in PROJECTS for path in model_config_files(project / "models"))])
def test_model_specs_load(path):
    model = apply_model_settings(load_yaml(path), {})
    assert all(spec.name for spec in expand_model_config(model))


@pytest.mark.parametrize(("project", "selection"), list(_selections()))
def test_project_composes_with_each_model_alone_and_all_together(project, selection, monkeypatch):
    monkeypatch.setenv("INFEROPS_MODELS_DIR", str(ROOT / "models"))
    manifest = yaml.safe_load((project / "project.yaml").read_text())
    overrides = {project / "project.yaml": {**manifest, "models": selection}} if selection else {}
    definition = ProjectLoader(project, configs_dir=ROOT / "tests").load(overrides)
    assert list(definition.pipelines) == [definition.name]
    assert set(definition.fusion) == set(definition.alerts) == {definition.name}
