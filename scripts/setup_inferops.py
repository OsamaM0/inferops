"""Install InferOps core and optionally one active domain pack.

Everything domain-specific (pip extras, model assets) comes from the pack's
``project.yaml`` manifest (or legacy ``domain.yaml``), so this script never changes when a pack is added.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


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


def _domain_extras(repo: Path, domain: str | None, domain_root: str | None = None) -> list[str]:
    """Read python_extras from the manifest without importing the package.

    Runs before ``pip install``, so only PyYAML (a build-time dependency of
    nothing else here) may be relied on; fall back to no extras if missing.
    """
    if not domain and not domain_root:
        return []
    try:
        import yaml
    except ImportError:
        return []
    roots = [Path(domain_root)] if domain_root else [repo / parent / domain for parent in ("projects", "examples", "domains")]
    for root in roots:
        path = root / "project.yaml" if (root / "project.yaml").exists() else root / "domain.yaml"
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return [str(x) for x in data.get("python_extras", [])]
    return []


def _install(repo: Path, *, domain: str | None, include_dev: bool, gpu: bool,
             domain_root: str | None = None) -> None:
    extras = ["server", "cuda" if gpu else "cpu"]
    if include_dev:
        extras.append("dev")
    if gpu:
        extras.append("gpu")
    extras.extend(_domain_extras(repo, domain, domain_root))
    seen = []
    for extra in extras:
        if extra not in seen:
            seen.append(extra)
    spec = ".[{}]".format(",".join(seen)) if seen else "."
    _run([sys.executable, "-m", "pip", "install", "-e", spec], cwd=repo)
    # Fresh virtualenvs may not have PyYAML until the first core install.
    # Re-read now so manifest extras are also installed on the first setup.
    missing_extras = [extra for extra in _domain_extras(repo, domain, domain_root) if extra not in seen]
    if missing_extras:
        _run([sys.executable, "-m", "pip", "install", "-e", f".[{','.join(missing_extras)}]"], cwd=repo)
    if gpu:
        # Clean up a CPU wheel left by an earlier CPU installation.
        _run([sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime"], cwd=repo)
        _run([sys.executable, "-m", "pip", "install", "--force-reinstall", "onnxruntime-gpu[cuda,cudnn]>=1.21,<1.27"], cwd=repo)


def _setup_domain_assets(repo: Path, *, domain: str | None, force: bool, domain_root: str | None = None) -> None:
    if not domain and not domain_root:
        return
    selection = ["--domain-root", domain_root] if domain_root else ["--domain", domain]
    cmd = [sys.executable, "-m", "inferops.cli", "domain", "setup-assets", *selection]
    if force:
        cmd.append("--force")
    _run(cmd, cwd=repo)


def _validate(repo: Path, *, domain: str | None, domain_root: str | None = None) -> None:
    validate_cmd = [sys.executable, "-m", "inferops.cli", "validate"]
    preflight_cmd = [sys.executable, "-m", "inferops.cli", "preflight"]
    if domain or domain_root:
        selection = ["--domain-root", domain_root] if domain_root else ["--domain", domain]
        validate_cmd.extend(selection)
        preflight_cmd.extend(selection)
    _run(validate_cmd, cwd=repo)
    _run(preflight_cmd, cwd=repo)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", help="Optional active domain to install and validate")
    parser.add_argument("--domain-root", help="Project block folder outside the repository")
    parser.add_argument("--skip-install", action="store_true", help="Do not run pip install")
    parser.add_argument("--skip-models", action="store_true", help="Do not download/build domain model assets")
    parser.add_argument("--force-models", action="store_true", help="Download domain model assets even if present")
    parser.add_argument("--no-dev", action="store_true", help="Install runtime dependencies only")
    parser.add_argument("--gpu", action="store_true", help="Install onnxruntime-gpu instead of the CPU wheel")
    args = parser.parse_args()

    repo = _repo_root()
    if not args.skip_install:
        _ensure_pip(repo)
        _install(repo, domain=args.domain, domain_root=args.domain_root, include_dev=not args.no_dev, gpu=args.gpu)
    if not args.skip_models:
        _setup_domain_assets(repo, domain=args.domain, domain_root=args.domain_root, force=args.force_models)
    _validate(repo, domain=args.domain, domain_root=args.domain_root)
    print("InferOps setup complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
