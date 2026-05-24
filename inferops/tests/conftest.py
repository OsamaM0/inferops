import os
import shutil
import socket
import subprocess
import sys
import tempfile
from contextlib import suppress
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


MOTO_PORT = _unused_port()

os.environ.setdefault("APP_ENV", "dev")
os.environ.setdefault("INFEROPS_ACTIVE_DOMAIN", "driver_behaviour")
os.environ.setdefault("INFEROPS_TRANSPORT_KIND", "inproc")
os.environ.setdefault("INFEROPS_PG_URL", "sqlite+aiosqlite:///./.test_inferops.db")
os.environ.setdefault("INFEROPS_OBS_LOG_JSON", "false")
os.environ.setdefault("INFEROPS_S3_ENDPOINT", f"http://127.0.0.1:{MOTO_PORT}")
os.environ.setdefault("INFEROPS_S3_PUBLIC_URL", f"http://127.0.0.1:{MOTO_PORT}")
os.environ.setdefault("INFEROPS_TRITON_URL", "127.0.0.1:1")
os.environ.setdefault("INFEROPS_TRITON_HTTP_URL", "http://127.0.0.1:1")

os.chdir(ROOT)  # so configs/, model_repository/ resolve
sys.path.insert(0, str(ROOT))


def _moto_server_command() -> list[str]:
    local = ROOT / ".venv" / "bin" / "moto_server"
    exe = str(local) if local.exists() else shutil.which("moto_server")
    if exe is None:
        pytest.fail("moto_server is required for tests that exercise S3 evidence storage")
    return [exe, "-H", "127.0.0.1", "-p", str(MOTO_PORT)]


def _wait_for_port(port: int, *, attempts: int = 100) -> bool:
    for _ in range(attempts):
        with suppress(OSError), socket.create_connection(("127.0.0.1", port), timeout=0.1):
            return True
        import time

        time.sleep(0.05)
    return False


@pytest.fixture(scope="session", autouse=True)
def _moto_s3_server():
    log_file = tempfile.TemporaryFile(mode="w+")
    proc = subprocess.Popen(
        _moto_server_command(),
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    if not _wait_for_port(MOTO_PORT, attempts=300):
        proc.terminate()
        with suppress(Exception):
            proc.wait(timeout=5)
        log_file.seek(0)
        output = log_file.read().strip()
        log_file.close()
        pytest.fail(f"moto_server did not start on port {MOTO_PORT}: {output}")
    yield
    proc.terminate()
    with suppress(Exception):
        proc.wait(timeout=5)
    if proc.poll() is None:
        proc.kill()
    log_file.close()


@pytest.fixture(scope="session", autouse=True)
def _tiny_model():
    """Make sure the tiny ONNX exists for tests that need it."""
    out = ROOT / "model_repository" / "tiny_classifier" / "1" / "model.onnx"
    if not out.exists():
        subprocess.check_call(
            [sys.executable, str(ROOT / "scripts" / "make_tiny_model.py"), str(out)]
        )
    yield
    # leave it for repeated runs


@pytest.fixture(scope="session", autouse=True)
def _clean_test_db():
    db = ROOT / ".test_inferops.db"
    if db.exists():
        db.unlink()
    yield
    if db.exists():
        with suppress(Exception):
            db.unlink()
