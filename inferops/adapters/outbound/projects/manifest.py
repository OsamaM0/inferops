"""Domain packs: the unit that turns InferOps into a concrete CV product.

A domain pack is a directory with this layout::

    <root>/
      project.yaml           # entry point (legacy domain.yaml also supported)
      __init__.py            # makes the pack importable as a package
      configs/
        pipelines/*.yaml
        fusion/*.yaml
        alerts/*.yaml
        platform.yaml        # optional overrides merged over core platform.yaml
      models/<name>/         # model.yaml + adapter.py; runtime variants together
      plugins/               # project stages, annotators, actions
      model_repository/      # optional; replaces the core repository when present
      scripts/               # optional custom artifact installers
      tests/                 # optional; run with INFEROPS_ACTIVE_DOMAIN=<name>

Nothing in the core package knows a domain's models, classes, or signals: the
manifest declares what the tooling (``make``, ``setup``, ``serve``, the e2e
scenario) needs, and YAML references plugins through ``<package>.plugins...``
or the pack-relative form ``.plugins.<module>:<Class>``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

MANIFEST_FILENAME = "project.yaml"
LEGACY_MANIFEST_FILENAME = "domain.yaml"
_PACKAGE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

class DomainAsset(BaseModel):
    """A model artifact the domain needs on disk before it can serve."""

    path: str
    installer: str | None = None
    description: str = ""
    required: bool = False


class DomainScenario(BaseModel):
    """What the browser end-to-end scenario should drive for this domain."""

    pipeline: str | None = None
    sample_video: str | None = None
    expect_alert: bool = False


class DomainManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    label: str | None = None
    description: str = ""
    version: int = 1
    # None preserves discovery for legacy projects; [] explicitly enables none.
    pipelines: list[str] | None = None
    platform: dict[str, Any] = Field(default_factory=dict)
    python_extras: list[str] = Field(default_factory=list)
    triton: bool = False
    prewarm_pipelines: list[str] = Field(default_factory=list)
    prewarm_models: list[str] = Field(default_factory=list)
    assets: list[DomainAsset] = Field(default_factory=list)
    scenario: DomainScenario = Field(default_factory=DomainScenario)
    tags: dict[str, str] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not _PACKAGE_RE.fullmatch(value):
            raise ValueError("domain name must be a valid Python identifier (letters, digits, underscore)")
        return value

    @property
    def display_label(self) -> str:
        return self.label or self.name.replace("_", " ").title()


def manifest_path(domain_root: str | Path) -> Path:
    root = Path(domain_root)
    current = root / MANIFEST_FILENAME
    legacy = root / LEGACY_MANIFEST_FILENAME
    return current if current.exists() or not legacy.exists() else legacy


def load_domain_manifest(domain_root: str | Path | None) -> DomainManifest | None:
    """Return the manifest for ``domain_root``.

    A pack without ``domain.yaml`` still works: it gets a manifest with only
    its directory name, so pre-manifest packs keep loading.
    """
    if domain_root is None:
        return None
    root = Path(domain_root)
    if not root.is_dir():
        return None
    path = manifest_path(root)
    if not path.exists():
        return DomainManifest(name=root.name)
    from inferops.adapters.outbound.projects.yaml import load_yaml

    data = load_yaml(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: manifest must be a YAML mapping")
    data.setdefault("name", root.name)
    return DomainManifest(**data)


def project_directories(configs_dir: str | Path = "configs") -> list[Path]:
    """Use workspace projects first; read older layouts when they exist."""
    engine_root = Path(configs_dir).resolve().parent
    roots = [engine_root / parent for parent in ("projects", "examples", "domains")]
    if (engine_root / "pyproject.toml").is_file() and (engine_root.parent / "compose.yaml").is_file():
        roots.append(engine_root.parent / "projects")
    return roots


def discover_domains(configs_dir: str | Path = "configs") -> list[tuple[str, Path]]:
    """List project blocks, preferring projects/ over legacy directories."""
    found: dict[str, Path] = {}
    for base in project_directories(configs_dir):
        if not base.is_dir():
            continue
        for child in sorted(base.iterdir()):
            if child.is_dir() and ((child / "configs").is_dir() or (child / MANIFEST_FILENAME).is_file()
                                   or (child / LEGACY_MANIFEST_FILENAME).is_file()) and child.name not in found:
                found[child.name] = child
    return list(found.items())


def domain_package_name(domain_root: str | Path) -> str:
    name = Path(domain_root).resolve().name
    if not _PACKAGE_RE.fullmatch(name):
        raise ValueError(
            f"domain directory {name!r} is not importable as a Python package; "
            "use letters, digits and underscores"
        )
    return name


def resolve_plugin_ref(ref: str, *, package: str | None = None) -> str:
    """Resolve a relative plugin only with an explicit project namespace."""
    if not ref.startswith("."):
        return ref
    if package is None:
        raise ValueError(f"plugin reference {ref!r} requires an explicit project context; no domain is active")
    return f"{package}{ref}"


def scaffold_domain(root: Path, name: str, labels: list[str], *, label: str | None = None) -> list[Path]:
    """Create an editable project block with models, pipelines, rules and actions."""
    from inferops.adapters.outbound.projects.scaffold import scaffold_domain as create

    return create(root, name, labels, label=label)


def manifest_env(manifest: DomainManifest, domain_root: Path, model_repository: Path) -> dict[str, Any]:
    """Shell-friendly view of the manifest consumed by the Makefile."""
    return {
        "INFEROPS_DOMAIN_NAME": manifest.name,
        "INFEROPS_DOMAIN_ROOT": str(domain_root.resolve()),
        "INFEROPS_DOMAIN_PACKAGE": domain_package_name(domain_root),
        "INFEROPS_DOMAIN_TRITON": "1" if manifest.triton else "0",
        "INFEROPS_DOMAIN_MODEL_REPOSITORY": str(model_repository.resolve()),
        "INFEROPS_PREWARM_PIPELINES": ",".join(manifest.prewarm_pipelines),
        "INFEROPS_PREWARM_MODELS": ",".join(manifest.prewarm_models),
        "INFEROPS_DOMAIN_EXTRAS": ",".join(manifest.python_extras),
        "INFEROPS_DOMAIN_HAS_TESTS": "1" if (domain_root / "tests").is_dir() else "0",
    }
