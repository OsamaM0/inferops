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

# Pinned versions for observability stack — last LTS-ish releases that
# work well together. Bump deliberately, not automatically.
PROMETHEUS_VERSION = "2.54.1"
GRAFANA_VERSION = "11.2.0"
DCGM_EXPORTER_VERSION = "3.3.7-3.5.0"

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


def _extract_tarball(archive: Path, dest: Path, *, dry_run: bool) -> None:
    print(f"+ extract {archive} -> {dest}")
    if dry_run:
        return
    import tarfile
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tf:
        tf.extractall(dest)


def _ensure_prometheus(*, force: bool, dry_run: bool) -> None:
    """Download Prometheus static binary into .tools/bin."""
    target = _tools_bin_dir() / "prometheus"
    if target.exists() and os.access(target, os.X_OK) and not force:
        print(f"prometheus already bootstrapped: {target}")
        return
    if shutil.which("prometheus") and not force:
        print(f"prometheus on PATH: {shutil.which('prometheus')}")
        return
    system, machine = _platform_key()
    if system != "Linux":
        raise SystemExit("Automatic Prometheus install only supported on Linux.")
    arch = "amd64" if machine == "x86_64" else "arm64"
    name = f"prometheus-{PROMETHEUS_VERSION}.linux-{arch}"
    url = (f"https://github.com/prometheus/prometheus/releases/download/"
           f"v{PROMETHEUS_VERSION}/{name}.tar.gz")
    with tempfile.TemporaryDirectory(prefix="inferops-prom-") as tmp_dir:
        tmp = Path(tmp_dir)
        tar = tmp / "prom.tar.gz"
        _download(url, tar, executable=False, dry_run=dry_run)
        _extract_tarball(tar, tmp, dry_run=dry_run)
        if dry_run:
            return
        src = tmp / name / "prometheus"
        promtool = tmp / name / "promtool"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        shutil.copy2(promtool, _tools_bin_dir() / "promtool")
        target.chmod(0o755)
        (_tools_bin_dir() / "promtool").chmod(0o755)
    print(f"prometheus ready: {target}")


def _ensure_grafana(*, force: bool, dry_run: bool) -> None:
    """Download Grafana OSS into .tools/grafana (bin/grafana-server)."""
    workspace = _workspace_root()
    install_dir = workspace / ".tools" / "grafana"
    server = install_dir / "bin" / "grafana"
    if not server.exists():
        server = install_dir / "bin" / "grafana-server"
    if server.exists() and os.access(server, os.X_OK) and not force:
        print(f"grafana already bootstrapped: {server}")
        return
    if shutil.which("grafana-server") and not force:
        print(f"grafana on PATH: {shutil.which('grafana-server')}")
        return
    system, machine = _platform_key()
    if system != "Linux":
        raise SystemExit("Automatic Grafana install only supported on Linux.")
    arch = "amd64" if machine == "x86_64" else "arm64"
    name = f"grafana-{GRAFANA_VERSION}.linux-{arch}"
    url = (f"https://dl.grafana.com/oss/release/{name}.tar.gz")
    with tempfile.TemporaryDirectory(prefix="inferops-grafana-") as tmp_dir:
        tmp = Path(tmp_dir)
        tar = tmp / "grafana.tar.gz"
        _download(url, tar, executable=False, dry_run=dry_run)
        _extract_tarball(tar, tmp, dry_run=dry_run)
        if dry_run:
            return
        extracted = tmp / f"grafana-v{GRAFANA_VERSION}"
        if not extracted.exists():
            # Some releases extract under a `grafana-<version>` folder.
            candidates = [p for p in tmp.iterdir() if p.is_dir() and p.name.startswith("grafana")]
            if not candidates:
                raise SystemExit("grafana extract: archive layout unexpected")
            extracted = candidates[0]
        if install_dir.exists():
            shutil.rmtree(install_dir)
        shutil.copytree(extracted, install_dir)
    print(f"grafana ready: {install_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print the runtime bootstrap steps without executing them")
    parser.add_argument("--force-downloads", action="store_true", help="Re-download MinIO and mc even if the local copies already exist")
    parser.add_argument("--skip-obs", action="store_true", help="Skip Prometheus/Grafana bootstrap (CI / fast loops)")
    args = parser.parse_args()

    _ensure_minio_binary("minio", force=args.force_downloads, dry_run=args.dry_run)
    _ensure_minio_binary("mc", force=args.force_downloads, dry_run=args.dry_run)
    _ensure_redpanda(dry_run=args.dry_run)
    if not args.skip_obs:
        _ensure_prometheus(force=args.force_downloads, dry_run=args.dry_run)
        _ensure_grafana(force=args.force_downloads, dry_run=args.dry_run)
    print("Native runtime setup complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())