"""Remove caches and packaging output without touching project assets or runtime data."""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO
CACHES = {"__pycache__", ".pytest_cache", ".ruff_cache", ".playwright-cli"}
PRESERVE = {".git", ".venv", ".data", ".run", ".tools", "logs", "model_repository"}


def clean(root: Path, *, dry_run: bool = False) -> int:
    count = 0
    for current, directories, _ in os.walk(root, followlinks=False):
        package_root = (Path(current) / "pyproject.toml").is_file()
        for name in list(directories):
            path = Path(current) / name
            if path.is_symlink() or name in PRESERVE:
                directories.remove(name)
            elif name in CACHES or (
                package_root and (name in {"build", "dist"} or name.endswith(".egg-info"))
            ):
                directories.remove(name)
                print(f"{'Would remove' if dry_run else 'Removed'} {path.relative_to(root)}")
                if not dry_run:
                    shutil.rmtree(path)
                count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(f"{clean(WORKSPACE, dry_run=args.dry_run)} generated directories")
