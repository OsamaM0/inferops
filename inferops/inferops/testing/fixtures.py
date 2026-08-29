"""Session fixtures shared by the core and domain test suites.

Import with ``from inferops.testing.fixtures import *`` in a ``conftest.py``
after :func:`inferops.testing.configure_test_env` has run.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import suppress
from pathlib import Path

import pytest

from inferops.testing import PROJECT_ROOT

__all__ = ["clean_test_db", "find_sample_video", "moto_s3_server", "tiny_model"]


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
    log_file = tempfile.TemporaryFile(mode="w+")
    proc = subprocess.Popen(_moto_server_command(port), stdout=log_file, stderr=subprocess.STDOUT)
    if not _wait_for_port(port, attempts=300):
        proc.terminate()
        with suppress(Exception):
            proc.wait(timeout=5)
        log_file.seek(0)
        output = log_file.read().strip()
        log_file.close()
        pytest.fail(f"moto_server did not start on port {port}: {output}")
    yield
    proc.terminate()
    with suppress(Exception):
        proc.wait(timeout=5)
    if proc.poll() is None:
        proc.kill()
    log_file.close()


@pytest.fixture(scope="session", autouse=True)
def tiny_model():
    """Make sure the tiny ONNX exists for tests that need it."""
    out = PROJECT_ROOT / "model_repository" / "tiny_classifier" / "1" / "model.onnx"
    if not out.exists():
        subprocess.check_call([sys.executable, str(PROJECT_ROOT / "scripts" / "make_tiny_model.py"), str(out)])
    yield
    # leave it for repeated runs


@pytest.fixture(scope="session", autouse=True)
def clean_test_db():
    db = PROJECT_ROOT / ".test_inferops.db"
    if db.exists():
        db.unlink()
    yield
    if db.exists():
        with suppress(Exception):
            db.unlink()


def find_sample_video(*candidates: str | Path) -> Path | None:
    """First existing video among ``INFEROPS_TEST_VIDEO`` and ``candidates``."""
    override = os.environ.get("INFEROPS_TEST_VIDEO")
    paths = [Path(override)] if override else []
    paths += [Path(c) for c in candidates]
    return next((p for p in paths if p.is_file()), None)
