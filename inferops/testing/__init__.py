"""Shared test bootstrap for the core suite and for domain-pack suites.

Settings are computed once per process from the environment, so a pytest
session runs either domain-free (core suite) or with exactly one domain pack
active. Call :func:`configure_test_env` at the very top of a ``conftest.py``,
before anything imports ``inferops.core.config``::

    from inferops.testing import configure_test_env
    configure_test_env(domain="my_domain")        # or domain=None for core
    from inferops.testing.fixtures import *       # noqa: E402,F401,F403
"""
from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # inferops/ (contains configs/, tests/)
_test_storage = TemporaryDirectory(prefix="inferops-tests-")


def _unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def configure_test_env(domain: str | None = None, *, domain_root: str | Path | None = None) -> int:
    """Pin the environment for one test process and return the mock-S3 port."""
    port = int(os.environ.get("INFEROPS_TEST_MOTO_PORT") or _unused_port())
    os.environ["INFEROPS_TEST_MOTO_PORT"] = str(port)

    # Exactly one domain per process — never inherit a stray selection.
    os.environ.pop("INFEROPS_ACTIVE_DOMAIN", None)
    os.environ.pop("INFEROPS_DOMAIN_ROOT", None)
    # Block dotenv from selecting the user's Docker project in core tests.
    os.environ["INFEROPS_PROJECT_ROOT"] = ""
    if domain_root is not None:
        os.environ["INFEROPS_DOMAIN_ROOT"] = str(domain_root)
        os.environ["INFEROPS_ACTIVE_DOMAIN"] = domain or Path(domain_root).name
    elif domain:
        os.environ["INFEROPS_ACTIVE_DOMAIN"] = domain

    os.environ["INFEROPS_CONFIGS_DIR"] = str(
        PROJECT_ROOT / "configs" if domain or domain_root else PROJECT_ROOT / "inferops/testing/configs"
    )
    os.environ["INFEROPS_EXECUTION_MODE"] = "auto"
    os.environ["INFEROPS_MODEL_REPOSITORY"] = "model_repository" if domain or domain_root else str(PROJECT_ROOT / "model_repository")

    os.environ.setdefault("APP_ENV", "dev")
    os.environ.setdefault("INFEROPS_TRANSPORT_KIND", "inproc")
    os.environ["INFEROPS_PG_URL"] = os.environ.get(
        "INFEROPS_TEST_PG_URL", f"sqlite+aiosqlite:///{_test_storage.name}/test.db"
    )
    os.environ.setdefault("INFEROPS_OBS_LOG_JSON", "false")
    os.environ.setdefault("INFEROPS_S3_ENDPOINT", f"http://127.0.0.1:{port}")
    os.environ.setdefault("INFEROPS_S3_PUBLIC_URL", f"http://127.0.0.1:{port}")
    os.environ.setdefault("INFEROPS_TRITON_URL", "127.0.0.1:1")
    os.environ.setdefault("INFEROPS_TRITON_HTTP_URL", "http://127.0.0.1:1")
    # Prewarm is exercised by the domain suites through the manifest; keep the
    # core suite fast and independent of what a manifest asks for.
    os.environ.setdefault("INFEROPS_PREWARM_MODELS", "")
    os.environ.setdefault("INFEROPS_PREWARM_PIPELINES", "")

    os.chdir(PROJECT_ROOT)   # so configs/, model_repository/ resolve
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    return port
