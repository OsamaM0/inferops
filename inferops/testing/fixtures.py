"""Session fixtures shared by the core and domain test suites.

Import with ``from inferops.testing.fixtures import *`` in a ``conftest.py``
after :func:`inferops.testing.configure_test_env` has run.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from contextlib import suppress
from pathlib import Path

import pytest

from inferops.testing import PROJECT_ROOT, TINY_MODEL_PATH
from inferops.testing.make_tiny_model import build_model

__all__ = ["find_sample_video", "moto_s3_server", "tiny_model"]


def _moto_server_command(port: int) -> list[str]:
    local = PROJECT_ROOT / ".venv" / "bin" / "moto_server"
    exe = str(local) if local.exists() else shutil.which("moto_server")
    if exe is None:
        pytest.fail("moto_server is required for tests that exercise S3 evidence storage")
    return [exe, "-H", "127.0.0.1", "-p", str(port)]


def _wait_for_port(port: int, *, attempts: int = 100) -> bool:
    import socket
    import time

    for _ in range(attempts):
        with suppress(OSError), socket.create_connection(("127.0.0.1", port), timeout=0.1):
            return True
        time.sleep(0.05)
    return False


@pytest.fixture(scope="session", autouse=True)
def moto_s3_server():
    port = int(os.environ["INFEROPS_TEST_MOTO_PORT"])
    with tempfile.TemporaryFile(mode="w+") as log_file:
        proc = subprocess.Popen(_moto_server_command(port), stdout=log_file, stderr=subprocess.STDOUT)
        if not _wait_for_port(port, attempts=300):
            proc.terminate()
            with suppress(Exception):
                proc.wait(timeout=5)
            log_file.seek(0)
            output = log_file.read().strip()
            pytest.fail(f"moto_server did not start on port {port}: {output}")
        yield
        proc.terminate()
        with suppress(Exception):
            proc.wait(timeout=5)
        if proc.poll() is None:
            proc.kill()


@pytest.fixture(scope="session", autouse=True)
def tiny_model():
    """Create one model in process-owned temporary storage, never the checkout."""
    out = TINY_MODEL_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        with tempfile.TemporaryDirectory(dir=out.parent) as directory:
            staged = Path(directory) / "model.onnx"
            build_model(staged)
            os.replace(staged, out)
    # The bootstrap's TemporaryDirectory owns cleanup after this process exits.
    yield


def find_sample_video(*candidates: str | Path) -> Path | None:
    """First existing video among ``INFEROPS_TEST_VIDEO`` and ``candidates``."""
    override = os.environ.get("INFEROPS_TEST_VIDEO")
    paths = [Path(override)] if override else []
    paths += [Path(c) for c in candidates]
    return next((p for p in paths if p.is_file()), None)
