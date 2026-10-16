"""Every shipped project's configuration parses to its spec."""
from pathlib import Path

import pytest

from inferops.adapters.outbound.projects.models import load_model_variants, model_config_files
from inferops.adapters.outbound.projects.yaml import load_yaml
from inferops.domain.fusion.spec import FusionSpec
from inferops.domain.incident_spec import AlertSpec
from inferops.domain.pipeline import PipelineSpec

ROOT = Path(__file__).resolve().parents[1]


def _all(subdir: str):
    out = []
    for base in [ROOT / "configs", *sorted((ROOT / "projects").glob("*/configs"))]:
        out.extend(sorted((base / subdir).glob("*.yaml")))
    if subdir == "models":
        for project in sorted((ROOT / "projects").iterdir()):
            out.extend(model_config_files(project / "models"))
    return out


@pytest.mark.parametrize("path", _all("models"))
def test_model_specs_load(path):
    assert all(spec.name for spec in load_model_variants(path))


@pytest.mark.parametrize("path", _all("pipelines"))
def test_pipeline_specs_load(path):
    spec = PipelineSpec(**load_yaml(path))
    assert spec.name


@pytest.mark.parametrize("path", _all("fusion"))
def test_fusion_specs_load(path):
    spec = FusionSpec(**load_yaml(path))
    assert spec.name


@pytest.mark.parametrize("path", _all("alerts"))
def test_alert_specs_load(path):
    spec = AlertSpec(**load_yaml(path))
    assert spec.name
