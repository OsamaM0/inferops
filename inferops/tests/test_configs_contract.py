"""Contract test: every YAML in configs/ and examples/*/configs/ parses to its spec."""
from pathlib import Path

import pytest

from inferops.alerts.spec import AlertSpec
from inferops.core.yaml_loader import load_yaml
from inferops.fusion.spec import FusionSpec
from inferops.pipeline.spec import PipelineSpec
from inferops.registry.model_spec import ModelSpec

ROOT = Path(__file__).resolve().parents[1]


def _all(subdir: str):
    out = []
    for base in [ROOT / "configs", *sorted((ROOT / "examples").glob("*/configs"))]:
        out.extend(sorted((base / subdir).glob("*.yaml")))
    return out


@pytest.mark.parametrize("path", _all("models"))
def test_model_specs_load(path):
    spec = ModelSpec(**load_yaml(path))
    assert spec.name


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
