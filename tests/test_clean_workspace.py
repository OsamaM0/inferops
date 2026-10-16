"""Cleanup must distinguish packaging output from source and persistent data."""
from __future__ import annotations

import pytest

from scripts.clean_workspace import clean


@pytest.mark.parametrize("dry_run", [True, False])
def test_cleanup_preserves_source_environment_and_runtime_assets(tmp_path, dry_run):
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname = 'example'\n")
    generated = [
        tmp_path / ".ruff_cache",
        project / ".run" / "cache",
        project / "build" / "lib" / "example",
        project / "dist",
        project / "example.egg-info",
        project / "src" / "__pycache__",
    ]
    protected = [
        project / "src" / "build",
        project / ".venv" / "example.egg-info",
        project / ".run" / "__pycache__",
        project / "model_repository" / "__pycache__",
        tmp_path / ".git" / "__pycache__",
    ]
    for directory in generated + protected:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "sentinel").write_text("keep or remove with its directory")
    (project / ".env").write_text("LOCAL_SETTING=value\n")

    assert clean(tmp_path, dry_run=dry_run) == len(generated)
    for directory in generated:
        assert directory.exists() is dry_run
    for directory in protected:
        assert (directory / "sentinel").is_file()
    assert (project / ".env").read_text() == "LOCAL_SETTING=value\n"


def test_cleanup_does_not_follow_or_delete_symlinks(tmp_path):
    outside = tmp_path / "outside"
    (outside / "__pycache__").mkdir(parents=True)
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "pyproject.toml").touch()
    (root / "build").symlink_to(outside, target_is_directory=True)

    assert clean(root) == 0
    assert (root / "build").is_symlink()
    assert (outside / "__pycache__").is_dir()
