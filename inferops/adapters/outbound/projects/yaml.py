"""YAML loader with `${ENV}` interpolation."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from inferops.domain.errors import ConfigError

_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


def _interp(value: Any) -> Any:
    if isinstance(value, str):
        def sub(m: re.Match[str]) -> str:
            name, default = m.group(1), m.group(2) or ""
            return os.environ.get(name, default)
        return _ENV_RE.sub(sub, value)
    if isinstance(value, dict):
        return {k: _interp(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interp(v) for v in value]
    return value


def load_yaml(path: str | Path) -> dict:
    return parse_yaml(Path(path).read_text(encoding="utf-8"))


def parse_yaml(text: str) -> dict:
    """Parse file/editor content using the same environment interpolation."""
    try:
        raw = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML: {exc}") from exc
    return _interp(raw)


def load_dir(path: str | Path) -> list[tuple[Path, dict]]:
    out = []
    base = Path(path)
    if not base.exists():
        return out
    for p in sorted(base.glob("*.yaml")):
        out.append((p, load_yaml(p)))
    return out
