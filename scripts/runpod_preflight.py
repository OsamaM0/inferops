"""Check the configured Triton server before the native RunPod API starts."""
from __future__ import annotations

import argparse
import importlib
import math
import os
import sys
import time
from pathlib import Path


def wait_for_triton(url: str, *, timeout_s: float) -> None:
    """Wait for gRPC readiness using the same endpoint as the inference backend."""
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("Triton startup timeout must be finite and positive")
    try:
        # Import the API actually used by TritonBackend as well as the sync probe.
        importlib.import_module("tritonclient.grpc.aio")
        grpc = importlib.import_module("tritonclient.grpc")
    except ImportError as exc:
        raise RuntimeError(
            "Triton client dependencies are unavailable. Reinstall with "
            "the InferOps virtual environment's python -m pip install -e '.[triton]'."
        ) from exc

    deadline = time.monotonic() + timeout_s
    reason = "server reported not ready"
    with grpc.InferenceServerClient(url=url) as client:
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                if client.is_server_ready(client_timeout=min(5.0, remaining)):
                    return
                reason = "server reported not ready"
            except Exception as exc:
                reason = str(exc)
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(1.0, remaining))

    raise RuntimeError(
        f"Triton gRPC server at {url} was not ready within {timeout_s:g}s: {reason}. "
        "runpod-start.sh starts native PostgreSQL, MediaMTX and InferOps; it does not "
        "launch Triton Server or Docker Compose. Start a compatible Triton server "
        "with the project's models, or set INFEROPS_TRITON_URL to an existing "
        "server's reachable host:port. See docs/runpod.md."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeout", type=float,
        help="Maximum seconds to wait for Triton readiness (default: 60)",
    )
    args = parser.parse_args()

    from inferops.adapters.outbound.projects.settings import load_settings

    settings = load_settings()
    if settings.execution_mode != "triton":
        return
    print(f"[inferops/runpod] Checking Triton gRPC readiness at {settings.triton.url}...", flush=True)
    try:
        timeout_s = args.timeout if args.timeout is not None else float(
            os.environ.get("INFEROPS_TRITON_STARTUP_TIMEOUT_SECONDS", "60")
        )
        wait_for_triton(settings.triton.url, timeout_s=timeout_s)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"[inferops/runpod] ERROR: {exc}") from exc
    print("[inferops/runpod] Triton server is ready.", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
