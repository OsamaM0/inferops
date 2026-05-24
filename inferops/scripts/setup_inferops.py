"""Install InferOps core and optionally one active domain pack."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DOMAIN_EXTRAS = {
    "driver_behaviour": ["triton"],
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)


def _pip_available(repo: Path) -> bool:
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "--version"],
            cwd=repo,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return True


def _ensure_pip(repo: Path) -> None:
    if _pip_available(repo):
        return
    try:
        _run([sys.executable, "-m", "ensurepip", "--upgrade"], cwd=repo)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "The virtual environment exists but does not contain pip. "
            "Install the system venv package for your Python version "
            "(for example: apt install python3.12-venv), then rerun make setup. "
            f"If this virtualenv was created before that package was installed, remove {repo / '.venv'} and retry."
        ) from exc
    if not _pip_available(repo):
        raise SystemExit(
            f"pip is still unavailable in {repo / '.venv'} after running ensurepip. "
            "Remove the virtualenv and rerun make setup."
        )


def _install(repo: Path, *, domain: str | None, include_dev: bool) -> None:
    extras = []
    if include_dev:
        extras.append("dev")
    if domain:
        extras.extend(DOMAIN_EXTRAS.get(domain, []))
    seen = []
    for extra in extras:
        if extra not in seen:
            seen.append(extra)
    spec = ".[{}]".format(",".join(seen)) if seen else "."
    _run([sys.executable, "-m", "pip", "install", "-e", spec], cwd=repo)


def _setup_driver_models(repo: Path, *, force: bool) -> None:
    target = repo / "model_repository" / "rfdetr_driver_behaviour_onnx" / "1" / "model.onnx"
    if target.exists() and not force:
        print(f"driver model already exists: {target}")
        return
    _run([sys.executable, "scripts/download_rfdetr_driver_model.py"], cwd=repo)


def _setup_domain_assets(repo: Path, *, domain: str | None, force: bool) -> None:
    if domain == "driver_behaviour":
        _setup_driver_models(repo, force=force)
    elif domain:
        print(f"no asset installer registered for domain: {domain}")


def _validate(repo: Path, *, domain: str | None) -> None:
    validate_cmd = [sys.executable, "-m", "inferops.cli", "validate"]
    preflight_cmd = [sys.executable, "-m", "inferops.cli", "preflight"]
    if domain:
        validate_cmd.extend(["--domain", domain])
        preflight_cmd.extend(["--domain", domain])
    _run(validate_cmd, cwd=repo)
    _run(preflight_cmd, cwd=repo)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", help="Optional active domain to install and validate")
    parser.add_argument("--skip-install", action="store_true", help="Do not run pip install")
    parser.add_argument("--skip-models", action="store_true", help="Do not download/build domain model assets")
    parser.add_argument("--force-models", action="store_true", help="Download domain model assets even if present")
    parser.add_argument("--no-dev", action="store_true", help="Install runtime dependencies only")
    args = parser.parse_args()

    repo = _repo_root()
    if not args.skip_install:
        _ensure_pip(repo)
        _install(repo, domain=args.domain, include_dev=not args.no_dev)
    if not args.skip_models:
        _setup_domain_assets(repo, domain=args.domain, force=args.force_models)
    _validate(repo, domain=args.domain)
    print("InferOps setup complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
