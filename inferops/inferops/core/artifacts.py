"""Single source of truth for resolving relative model-artifact paths.

This logic used to exist in three places that had drifted apart:

* ``inferops/runtimes/onnxruntime_backend.py`` — the runtime loader
* ``examples/driver_behaviour/plugins/face_mesh_stage.py`` — the "is the real
  model present?" probe
* ``inferops/cli/__main__.py`` — ``inferops preflight``

The CLI copy omitted ``settings.domain_root`` and the configs-dir sibling, so
``preflight`` could report a model as missing that the runtime resolved fine
(and vice versa). Everything now goes through :func:`artifact_candidates`.
"""
from __future__ import annotations

from pathlib import Path

__all__ = ["artifact_candidates", "resolve_artifact"]


def artifact_candidates(
    path: str | Path,
    *,
    base_dir: Path | None = None,
    model_repository: Path | None = None,
    domain_root: str | Path | None = None,
    configs_dir: str | Path | None = None,
) -> list[Path]:
    """Return the ordered locations an artifact may live in.

    An absolute path resolves to itself only. Relative paths are tried against
    the spec directory first (most specific) and the bare path last.
    """
    artifact = Path(path)
    if artifact.is_absolute():
        return [artifact]

    candidates: list[Path] = []
    if base_dir is not None:
        candidates.append(Path(base_dir) / artifact)
        candidates.append(Path(base_dir).parent / artifact)
    if model_repository is not None:
        candidates.append(Path(model_repository) / artifact)
    if domain_root:
        candidates.append(Path(domain_root) / artifact)
    if configs_dir:
        candidates.append(Path(configs_dir).resolve().parent / artifact)
    candidates.append(Path.cwd() / artifact)
    candidates.append(artifact)

    # Preserve order while dropping duplicates, so error messages stay short
    # and callers do not stat the same location twice.
    seen: set[str] = set()
    unique: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def resolve_artifact(
    path: str | Path,
    *,
    base_dir: Path | None = None,
    model_repository: Path | None = None,
    domain_root: str | Path | None = None,
    configs_dir: str | Path | None = None,
) -> tuple[Path | None, list[Path]]:
    """Return ``(first_existing_or_None, candidates_tried)``."""
    candidates = artifact_candidates(
        path,
        base_dir=base_dir,
        model_repository=model_repository,
        domain_root=domain_root,
        configs_dir=configs_dir,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate, candidates
    return None, candidates
