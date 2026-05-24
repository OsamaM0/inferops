"""System status routes for browser-friendly service checks."""
from __future__ import annotations

import asyncio
import socket
import subprocess
from urllib.request import urlopen

from fastapi import APIRouter
from sqlalchemy import text

from inferops.adapters.postgres_dao import get_engine
from inferops.adapters.s3_client import s3_client
from inferops.core.config import settings

router = APIRouter()


def _http_status(url: str, timeout: float = 2.0) -> int | None:
    try:
        with urlopen(url, timeout=timeout) as response:
            return int(response.status)
    except Exception:
        return None


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _deepstream_version() -> str | None:
    try:
        out = subprocess.check_output(
            ["deepstream-app", "--version-all"],
            stderr=subprocess.STDOUT,
            timeout=3,
            text=True,
        )
        return out.splitlines()[0] if out else None
    except Exception:
        return None


def _s3_count(bucket: str, prefix: str = "") -> int:
    total = 0
    paginator = s3_client().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        total += len(page.get("Contents", []))
    return total


def _safe_s3_count(bucket: str, prefix: str = "") -> int | None:
    try:
        return _s3_count(bucket, prefix)
    except Exception:
        return None


async def _db_counts() -> dict[str, int | None]:
    try:
        engine = get_engine()
        async with engine.begin() as conn:
            events = (await conn.execute(text("select count(*) from events"))).scalar()
            alerts = (await conn.execute(text("select count(*) from alerts"))).scalar()
        return {"events": int(events or 0), "alerts": int(alerts or 0)}
    except Exception:
        return {"events": None, "alerts": None}


@router.get("/system/status")
async def system_status():
    db_counts, json_count, clip_count = await asyncio.gather(
        _db_counts(),
        asyncio.to_thread(_safe_s3_count, "inferops", "evidence/"),
        asyncio.to_thread(_safe_s3_count, "inferops-clips", ""),
    )
    s3_health = _http_status("http://127.0.0.1:9000/minio/health/live", timeout=1.5)
    s3_backend = "minio" if s3_health == 200 else ("moto" if _port_open("127.0.0.1", 9000) else "down")
    return {
        "services": {
            "api": {"port": 8080, "status": 200, "ok": True},
            "triton_http": {"port": 8000, "status": _http_status("http://127.0.0.1:8000/v2/health/live")},
            "triton_grpc": {"port": 8001, "open": _port_open("127.0.0.1", 8001)},
            "triton_metrics": {"port": 8002, "status": _http_status("http://127.0.0.1:8002/metrics")},
            "s3": {"port": 9000, "open": _port_open("127.0.0.1", 9000), "backend": s3_backend, "endpoint": settings.s3.endpoint},
            "minio_console": {"port": 9001, "open": _port_open("127.0.0.1", 9001)},
            "redpanda_kafka": {"port": 19092, "open": _port_open("127.0.0.1", 19092)},
            "redpanda_admin": {"port": 9644, "open": _port_open("127.0.0.1", 9644)},
        },
        "transport": {"kind": settings.transport.kind, "bootstrap": settings.transport.bootstrap},
        "deepstream": {"version": _deepstream_version()},
        "database": db_counts,
        "evidence": {"json_records": json_count, "clips": clip_count},
    }
