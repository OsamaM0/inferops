"""Run the source-control UI with PostgreSQL, persistent disk and MediaMTX.

Start dependencies with ``docker compose up -d postgres mediamtx``, then:
    .venv/bin/python scripts/serve_local.py --runtime cuda
    Open http://127.0.0.1:8081/ui/
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inferops.adapters.inbound.http.app import create_application
from inferops.bootstrap import build_server_engine


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=["cpu", "cuda", "triton"])
    parser.add_argument("--triton-url")
    parser.add_argument("--project")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--data")
    args = parser.parse_args()
    for key, value in {"INFEROPS_EXECUTION_MODE": args.runtime, "INFEROPS_TRITON_URL": args.triton_url,
                       "INFEROPS_DOMAIN_ROOT": str(Path(args.project).resolve()) if args.project else None,
                       "INFEROPS_API_HOST": args.host, "INFEROPS_API_PORT": str(args.port) if args.port else None,
                       "INFEROPS_DATA_PATH": str(Path(args.data).resolve()) if args.data else None}.items():
        if value is not None:
            os.environ[key] = value
    from inferops.adapters.outbound.projects.settings import load_settings
    settings = load_settings()
    root = Path(os.environ.get("INFEROPS_DATA_PATH", ".run/server")).resolve()
    engine, hub = build_server_engine()
    app = create_application(engine, hub=hub, source_root=root / "sources")
    uvicorn.run(app, host=settings.api.host, port=settings.api.port)


if __name__ == "__main__":
    main()
