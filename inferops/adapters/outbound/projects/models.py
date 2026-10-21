"""Load flat model specs or model ports with shared runtime variants."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from inferops.adapters.outbound.projects.yaml import deep_merge, load_yaml
from inferops.domain.errors import ConfigError
from inferops.domain.models import LabelsSpec, ModelSpec


class ModelPortSpec(BaseModel):
    """One model's common contract and its named runtime implementations."""

    model_config = ConfigDict(extra="forbid")
    name: str
    defaults: dict[str, Any] = Field(default_factory=dict)
    variants: dict[str, dict[str, Any]] = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        if not value.isidentifier():
            raise ValueError("model port name must be a Python identifier")
        return value

    @field_validator("variants")
    @classmethod
    def valid_variants(cls, value: dict) -> dict:
        if any(not name.isidentifier() for name in value):
            raise ValueError("variant names must be Python identifiers")
        return value


def expand_model_config(data: dict[str, Any]) -> list[ModelSpec]:
    """Lists replace lists; dictionaries merge recursively. Inputs stay untouched."""
    if "variants" not in data:
        return [ModelSpec(**data)]
    try:
        port = ModelPortSpec(**data)
    except ValueError as exc:
        raise ConfigError(f"invalid model port: {exc}") from exc
    specs = []
    for variant, overrides in port.variants.items():
        merged = deep_merge(port.defaults, overrides)
        unknown = set(merged) - ModelSpec.model_fields.keys()
        if unknown:
            raise ConfigError(f"model port {port.name!r}, variant {variant!r}: unknown fields {sorted(unknown)}")
        merged.setdefault("name", f"{port.name}_{variant}")
        try:
            specs.append(ModelSpec(**merged))
        except Exception as exc:
            raise ConfigError(f"model port {port.name!r}, variant {variant!r}: {exc}") from exc
    return specs


def model_config_files(directory: str | Path) -> list[Path]:
    """Discover only model YAMLs, keeping legacy configs/models/*.yaml usable."""
    root = Path(directory)
    return sorted([*root.glob("*.yaml"), *root.glob("*/model.yaml")])


def load_model_variants(path: str | Path) -> list[ModelSpec]:
    path = Path(path)
    try:
        specs = expand_model_config(load_yaml(path))
        for spec in specs:
            if isinstance(spec.labels, LabelsSpec) and spec.labels.from_file:
                labels = Path(spec.labels.from_file)
                if not labels.is_absolute():
                    spec.labels.from_file = str((path.parent / labels).resolve())
        return specs
    except Exception as exc:
        raise ConfigError(f"invalid model configuration {path}: {exc}") from exc


def load_model_file(path: str | Path) -> ModelSpec:
    """Compatibility entry point for callers expecting exactly one flat spec."""
    specs = load_model_variants(path)
    if len(specs) != 1:
        raise ConfigError(f"{path}: contains several variants; use load_model_variants")
    return specs[0]


def load_models_with_sources(directory: str | Path) -> list[tuple[Path, ModelSpec]]:
    return [(path, spec) for path in model_config_files(directory) for spec in load_model_variants(path)]


def load_models_dir(directory: str | Path) -> list[ModelSpec]:
    return [spec for _, spec in load_models_with_sources(directory)]
