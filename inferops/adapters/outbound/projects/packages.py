"""Discover independently installed project adapters through Python entry points."""
from __future__ import annotations

import importlib.metadata
import shutil
from pathlib import Path
from types import ModuleType

ENTRY_POINT_GROUP = "inferops.projects"


def installed_adapters() -> dict[str, importlib.metadata.EntryPoint]:
    found = {}
    for entry in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
        if entry.name in found:
            raise ValueError(f"multiple installed adapters are named {entry.name!r}")
        found[entry.name] = entry
    return found


def adapter_root(name: str) -> Path:
    entries = installed_adapters()
    if name not in entries:
        raise ValueError(f"adapter {name!r} is not installed; available: {', '.join(sorted(entries)) or 'none'}")
    module = entries[name].load()
    if not isinstance(module, ModuleType) or not module.__file__:
        raise ValueError(f"adapter {name!r} entry point must reference a Python package")
    root = Path(module.__file__).resolve().parent
    if not (root / "project.yaml").is_file():
        raise ValueError(f"adapter {name!r} is missing project.yaml")
    return root


def create_project(name: str, destination: str | Path, *, configs_dir: str | Path = "configs") -> Path:
    """Copy an installed adapter to a writable workspace, without weights or tests.

    Models the adapter selects from the shared library are copied into the new
    project's own models/ folder, so the workspace is self-contained.
    """
    import yaml

    from inferops.adapters.outbound.projects.central import models_dir

    root = adapter_root(name)
    destination = Path(destination).resolve()
    if not destination.name.isidentifier():
        raise ValueError("project directory name must be a Python identifier (use underscores, not hyphens)")
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "tests", "*.egg-info", "model_repository", "build", "dist")
    shutil.copytree(root, destination, ignore=ignore)
    library = models_dir(root, configs_dir)
    if library.is_dir() and not library.is_relative_to(root):
        selected = (yaml.safe_load((root / "project.yaml").read_text()) or {}).get("models")
        for model in sorted(library.glob("*/model.yaml")):
            if selected is None or (yaml.safe_load(model.read_text()) or {}).get("name") in selected:
                shutil.copytree(model.parent, destination / "models" / model.parent.name, ignore=ignore)
        if (library / "__init__.py").is_file() and (destination / "models").is_dir():
            shutil.copy2(library / "__init__.py", destination / "models" / "__init__.py")
    return destination
