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
    parser.add_argument("--runtime", choices=["cpu", "cuda", "triton"], default="cuda")
    parser.add_argument("--triton-url", default="127.0.0.1:8001")
    parser.add_argument("--project", default="projects/driver_behaviour")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--data", default=".run/local-control")
    args = parser.parse_args()
    root = Path(args.data).resolve()
    os.environ.update(INFEROPS_EXECUTION_MODE=args.runtime, INFEROPS_TRITON_URL=args.triton_url,
                      INFEROPS_DOMAIN_ROOT=str(Path(args.project).resolve()), INFEROPS_DATA_PATH=str(root),
                      INFEROPS_SPOOL_PATH=str(root / "spool"))
    engine, hub = build_server_engine()
    app = create_application(engine, hub=hub, source_root=root / "sources")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
