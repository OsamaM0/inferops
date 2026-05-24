"""Resolve `"module:Class"` strings into callables."""
from __future__ import annotations

import importlib
from typing import Any

from inferops.core.errors import ConfigError


def load_object(path: str) -> Any:
    """Load a `module:Attr` or `module.Attr` reference."""
    if not path:
        raise ConfigError("empty plugin reference")
    if ":" in path:
        mod_name, attr = path.split(":", 1)
    else:
        mod_name, _, attr = path.rpartition(".")
        if not mod_name:
            raise ConfigError(f"invalid plugin path: {path!r}")
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as e:
        raise ConfigError(f"cannot import {mod_name!r}: {e}") from e
    try:
        return getattr(mod, attr)
    except AttributeError as e:
        raise ConfigError(f"{mod_name!r} has no attribute {attr!r}") from e


def instantiate(path: str, params: dict | None = None) -> Any:
    cls = load_object(path)
    return cls(**(params or {}))
