"""Serve the console with disposable configs, model, database and media storage.

Set INFEROPS_TEST_PG_URL to a disposable PostgreSQL instance first.
The browser scenario expects the default port, 18083.
"""
from __future__ import annotations

import argparse
import os
import shutil
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18083)
    args = parser.parse_args()

    # Configure isolated resources before importing the application settings.
    from inferops.testing import PROJECT_ROOT, TINY_MODEL_PATH, configure_test_env
    from inferops.testing.make_tiny_model import build_model

    configure_test_env()
    with TemporaryDirectory(prefix="inferops-console-configs-") as directory:
        shutil.copytree(PROJECT_ROOT / "inferops/testing/configs", directory, dirs_exist_ok=True)
        os.environ["INFEROPS_CONFIGS_DIR"] = directory
        TINY_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        build_model(TINY_MODEL_PATH)

        import uvicorn

        from inferops.adapters.inbound.http.main import create_app

        uvicorn.run(create_app(), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
