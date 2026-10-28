"""Supervise a native Triton server and InferOps API inside one RunPod Pod."""
from __future__ import annotations

import argparse
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlsplit

APP = Path(__file__).resolve().parents[1]
LOCAL_HOSTS = {"localhost", "127.0.0.1"}


def endpoint(url: str, *, grpc: bool = False) -> tuple[str, int]:
    parsed = urlsplit(f"//{url}" if grpc else url)
    if not parsed.hostname or parsed.port is None or parsed.username or parsed.password:
        raise ValueError(f"Expected {'host:port' if grpc else 'http://host:port'} endpoint: {url}")
    if grpc and parsed.path:
        raise ValueError(f"Triton gRPC URL must be host:port: {url}")
    if not grpc and (parsed.scheme != "http" or parsed.path not in {"", "/"}):
        raise ValueError(f"Native Triton requires a plain HTTP endpoint: {url}")
    return parsed.hostname, parsed.port


def native_command(settings, repository: Path, binary: str) -> list[str]:
    grpc_host, grpc_port = endpoint(settings.triton.url, grpc=True)
    http_host, http_port = endpoint(settings.triton.http_url)
    metrics_host, metrics_port = endpoint(os.environ.get("INFEROPS_TRITON_METRICS_URL", "http://127.0.0.1:8002"))
    if any(host not in LOCAL_HOSTS for host in (grpc_host, http_host, metrics_host)):
        raise ValueError("Native Triton autostart requires local gRPC, HTTP and metrics endpoints")
    if len({grpc_port, http_port, metrics_port}) != 3:
        raise ValueError("Triton gRPC, HTTP and metrics ports must be distinct")
    repository = repository.resolve()
    # The artifact repository may also contain training checkpoints and other
    # non-serving directories. Explicit mode must not load those via '*'.
    models = sorted(path.parent.name for path in repository.glob("*/config.pbtxt") if path.is_file())
    if not repository.is_dir() or not models:
        raise ValueError(f"No prepared Triton model configurations found in {repository}")
    if "*" in models:
        raise ValueError("Triton model directories cannot be named '*'")
    return [
        binary, f"--model-repository={repository}", "--strict-model-config=true",
        "--model-control-mode=explicit", *[f"--load-model={name}" for name in models], "--exit-on-error=true",
        "--rate-limit=execution_count", "--pinned-memory-pool-byte-size=134217728",
        "--cuda-memory-pool-byte-size=0:134217728",
        "--http-address=127.0.0.1", f"--http-port={http_port}",
        "--grpc-address=127.0.0.1", f"--grpc-port={grpc_port}",
        "--metrics-address=127.0.0.1", f"--metrics-port={metrics_port}",
    ]


def listening(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


def stop_owned(process) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def print_startup_log(path: Path, start: int, *, max_bytes: int = 262144) -> None:
    """Include early backend errors, excluding output from previous attempts."""
    with suppress(OSError), path.open("rb") as output:
        end = output.seek(0, os.SEEK_END)
        output.seek(start if start <= end else 0)
        content = output.read(max_bytes + 1)
        print(f"[inferops/runpod] Triton startup output ({path}):", file=sys.stderr, flush=True)
        print(content[:max_bytes].decode("utf-8", errors="replace"), file=sys.stderr, flush=True)
        if len(content) > max_bytes:
            print(f"[inferops/runpod] Startup output truncated at {max_bytes} bytes; full log: {path}", file=sys.stderr, flush=True)


def run_runtime(settings, *, repository: Path, data: Path, triton_log: Path,
                timeout_s: float = 60, autostart: str = "auto") -> int:
    from scripts.runpod_preflight import wait_for_triton

    if autostart not in {"auto", "true", "false"}:
        raise ValueError("INFEROPS_TRITON_AUTOSTART must be auto, true or false")
    triton = api = None
    log_start = 0
    try:
        if settings.execution_mode == "triton":
            if not math.isfinite(timeout_s) or timeout_s <= 0:
                raise ValueError("Triton startup timeout must be finite and positive")
            host, port = endpoint(settings.triton.url, grpc=True)
            if autostart == "true" and host not in LOCAL_HOSTS:
                raise ValueError("INFEROPS_TRITON_AUTOSTART=true requires a local gRPC endpoint")
            binary = os.environ.get("INFEROPS_TRITON_BINARY") or shutil.which("tritonserver")
            if not binary and Path("/opt/tritonserver/bin/tritonserver").is_file():
                binary = "/opt/tritonserver/bin/tritonserver"
            if autostart != "false" and host in LOCAL_HOSTS and not listening(host, port):
                if binary:
                    command = native_command(settings, repository, binary)
                    triton_log.parent.mkdir(parents=True, exist_ok=True)
                    print(f"[inferops/runpod] Starting native Triton; log: {triton_log}", flush=True)
                    with triton_log.open("ab") as output:
                        log_start = output.tell()
                        triton = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
                elif autostart == "true":
                    raise RuntimeError("tritonserver is missing; use a compatible NVIDIA Triton image or configure an external server")

            def check_running():
                if triton is not None and (code := triton.poll()) is not None:
                    raise RuntimeError(f"Native Triton exited with status {code}; see {triton_log}")

            print(f"[inferops/runpod] Checking Triton gRPC readiness at {settings.triton.url}...", flush=True)
            wait_for_triton(settings.triton.url, timeout_s=timeout_s, check_running=check_running)
            check_running()

        print("[inferops/runpod] Starting InferOps API...", flush=True)
        api = subprocess.Popen([sys.executable, str(APP / "scripts/serve_local.py"), "--data", str(data)])
        while (code := api.poll()) is None:
            if triton is not None and (triton_code := triton.poll()) is not None:
                raise RuntimeError(f"Native Triton exited with status {triton_code}; see {triton_log}")
            time.sleep(0.5)
        return code
    except BaseException:
        if triton is not None and triton_log.is_file():
            print_startup_log(triton_log, log_start)
        raise
    finally:
        stop_owned(api)
        stop_owned(triton)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--triton-log", type=Path, required=True)
    args = parser.parse_args()
    from inferops.adapters.outbound.projects.settings import get_model_repository_root, load_settings

    settings = load_settings()
    repository = Path(os.environ.get("INFEROPS_TRITON_REPOSITORY") or get_model_repository_root(settings))

    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        timeout_s = float(os.environ.get("INFEROPS_TRITON_STARTUP_TIMEOUT_SECONDS", "60")) if settings.execution_mode == "triton" else 60
        code = run_runtime(settings, repository=repository, data=args.data, triton_log=args.triton_log,
                           timeout_s=timeout_s, autostart=os.environ.get("INFEROPS_TRITON_AUTOSTART", "auto"))
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"[inferops/runpod] ERROR: {exc}") from exc
    raise SystemExit(code)


if __name__ == "__main__":
    sys.path.insert(0, str(APP))
    main()
