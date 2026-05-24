"""ModelSpec loader."""
from __future__ import annotations

from pathlib import Path

from inferops.core.errors import ConfigError
from inferops.core.yaml_loader import load_dir, load_yaml
from inferops.registry.model_spec import ModelSpec


def load_model_file(path: str | Path) -> ModelSpec:
    data = load_yaml(path)
    try:
        return ModelSpec(**data)
    except Exception as e:
        raise ConfigError(f"invalid model spec {path}: {e}") from e


def load_models_dir(path: str | Path) -> list[ModelSpec]:
    out = []
    for p, data in load_dir(path):
        try:
            out.append(ModelSpec(**data))
        except Exception as e:
            raise ConfigError(f"invalid model spec {p}: {e}") from e
    return out
