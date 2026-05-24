"""Install native runtime prerequisites for broker-backed local serve."""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

MINIO_RELEASES = {
    ("Linux", "x86_64"): "linux-amd64",
    ("Linux", "aarch64"): "linux-arm64",
}
MINIO_SERVER_BASE = "https://dl.min.io/server/minio/release"
MINIO_CLIENT_BASE = "https://dl.min.io/client/mc/release"
REDPANDA_DEB_SETUP_URL = "https://dl.redpanda.com/nzc4ZYQK3WRGd9sy/redpanda/cfg/setup/bash.deb.sh"
ARCH_ALIASES = {
    "amd64": "x86_64",
    "x86-64": "x86_64",
    "arm64": "aarch64",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _workspace_root() -> Path:
    return _repo_root().parent


def _tools_bin_dir() -> Path:
    return _workspace_root() / ".tools" / "bin"


def _run(cmd: list[str], *, env: dict[str, str] | None = None, dry_run: bool) -> None:
    print("+", " ".join(cmd))
    if dry_run:
        return
    subprocess.check_call(cmd, cwd=_repo_root(), env=env)


def _download(url: str, target: Path, *, executable: bool, dry_run: bool) -> None:
    print(f"+ download {url} -> {target}")
    if dry_run:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "inferops-setup"})
    with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    if executable:
        target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _platform_key() -> tuple[str, str]:
    system = platform.system()
    machine = ARCH_ALIASES.get(platform.machine().lower(), platform.machine().lower())
    return system, machine


def _manual_redpanda_instructions() -> str:
    return (
        "curl -1sLf 'https://dl.redpanda.com/nzc4ZYQK3WRGd9sy/redpanda/cfg/setup/bash.deb.sh' | "
        "sudo -E bash && sudo apt install redpanda -y"
    )


def _ensure_minio_binary(name: str, *, force: bool, dry_run: bool) -> None:
    existing = shutil.which(name)
    if existing:
        print(f"{name} already available: {existing}")
        return
    platform_dir = MINIO_RELEASES.get(_platform_key())
    if platform_dir is None:
        raise SystemExit(
            "Automatic MinIO installation is only supported on Linux x86_64/aarch64. "
            f"Download {name} manually for {_platform_key()[0]} {_platform_key()[1]}."
        )
    target = _tools_bin_dir() / name
    if target.exists() and os.access(target, os.X_OK) and not force:
        print(f"{name} already bootstrapped: {target}")
        return
    base_url = MINIO_SERVER_BASE if name == "minio" else MINIO_CLIENT_BASE
    _download(f"{base_url}/{platform_dir}/{name}", target, executable=True, dry_run=dry_run)


def _privilege_prefix() -> list[str] | None:
    if os.name != "posix":
        return None
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return []
    if shutil.which("sudo"):
        return ["sudo", "-E"]
    return None


def _ensure_redpanda(*, dry_run: bool) -> None:
    rpk = shutil.which("rpk")
    redpanda = shutil.which("redpanda")
    if rpk and redpanda:
        print(f"Redpanda already available: redpanda={redpanda}, rpk={rpk}")
        return
    system, _machine = _platform_key()
    if system != "Linux":
        raise SystemExit(
            "Automatic Redpanda installation is only supported on Debian/Ubuntu Linux. "
            f"Install it manually with: {_manual_redpanda_instructions()}"
        )
    apt_get = shutil.which("apt-get") or shutil.which("apt")
    if apt_get is None:
        raise SystemExit(
            "Automatic Redpanda installation requires apt or apt-get. "
            f"Install it manually with: {_manual_redpanda_instructions()}"
        )
    privilege_prefix = _privilege_prefix()
    if privilege_prefix is None:
        raise SystemExit(
            "Automatic Redpanda installation requires root or sudo access. "
            f"Install it manually with: {_manual_redpanda_instructions()}"
        )
    with tempfile.TemporaryDirectory(prefix="inferops-redpanda-") as tmp_dir:
        setup_script = Path(tmp_dir) / "redpanda-apt-setup.sh"
        _download(REDPANDA_DEB_SETUP_URL, setup_script, executable=True, dry_run=dry_run)
        _run(privilege_prefix + ["bash", str(setup_script)], dry_run=dry_run)
        env = os.environ.copy()
        env.setdefault("DEBIAN_FRONTEND", "noninteractive")
        _run(privilege_prefix + [apt_get, "install", "-y", "redpanda"], env=env, dry_run=dry_run)
    if dry_run:
        return
    rpk = shutil.which("rpk")
    redpanda = shutil.which("redpanda")
    if not (rpk and redpanda):
        raise SystemExit(
            "Redpanda installation completed but `rpk` and `redpanda` are still not on PATH. "
            f"Repair it manually with: {_manual_redpanda_instructions()}"
        )
    print(f"Redpanda ready: redpanda={redpanda}, rpk={rpk}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print the runtime bootstrap steps without executing them")
    parser.add_argument("--force-downloads", action="store_true", help="Re-download MinIO and mc even if the local copies already exist")
    args = parser.parse_args()

    _ensure_minio_binary("minio", force=args.force_downloads, dry_run=args.dry_run)
    _ensure_minio_binary("mc", force=args.force_downloads, dry_run=args.dry_run)
    _ensure_redpanda(dry_run=args.dry_run)
    print("Native runtime setup complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())