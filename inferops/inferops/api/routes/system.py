"""System status routes for browser-friendly service checks."""
from __future__ import annotations

import asyncio
import socket
import subprocess
from typing import Any
from functools import lru_cache
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


@lru_cache(maxsize=1)
def _deepstream_version() -> str | None:
    try:
        out = subprocess.check_output(
            ["deepstream-app", "--version-all"],
            stderr=subprocess.STDOUT,
            timeout=1.5,
            text=True,
        )
        return out.splitlines()[0] if out else None
    except Exception:
        return None


def _safe_s3_probe(bucket: str, prefix: str = "") -> dict[str, Any]:
    try:
        response = s3_client().list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        key_count = int(response.get("KeyCount") or 0)
        return {"available": True, "count": 0 if key_count == 0 else None}
    except Exception:
        return {"available": False, "count": None}


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
    (
        db_counts,
        json_probe,
        clip_probe,
        triton_http_status,
        triton_grpc_open,
        triton_metrics_status,
        s3_health,
        s3_open,
        minio_console_open,
        redpanda_kafka_open,
        redpanda_admin_open,
        deepstream_version,
    ) = await asyncio.gather(
        _db_counts(),
        asyncio.to_thread(_safe_s3_probe, "inferops", "evidence/"),
        asyncio.to_thread(_safe_s3_probe, "inferops-clips", ""),
        asyncio.to_thread(_http_status, "http://127.0.0.1:8000/v2/health/live", 0.75),
        asyncio.to_thread(_port_open, "127.0.0.1", 8001, 0.25),
        asyncio.to_thread(_http_status, "http://127.0.0.1:8002/metrics", 0.75),
        asyncio.to_thread(_http_status, "http://127.0.0.1:9000/minio/health/live", 0.75),
        asyncio.to_thread(_port_open, "127.0.0.1", 9000, 0.25),
        asyncio.to_thread(_port_open, "127.0.0.1", 9001, 0.25),
        asyncio.to_thread(_port_open, "127.0.0.1", 19092, 0.25),
        asyncio.to_thread(_port_open, "127.0.0.1", 19644, 0.25),
        asyncio.to_thread(_deepstream_version),
    )
    s3_backend = "minio" if s3_health == 200 else ("moto" if _port_open("127.0.0.1", 9000) else "down")
    s3_service = {"port": 9000, "open": s3_open, "backend": s3_backend, "endpoint": settings.s3.endpoint}
    return {
        "services": {
            "api": {"port": 8080, "status": 200, "ok": True},
            "triton_http": {"port": 8000, "status": triton_http_status},
            "triton_grpc": {"port": 8001, "open": triton_grpc_open},
            "triton_metrics": {"port": 8002, "status": triton_metrics_status},
            "s3": s3_service,
            "minio_s3": s3_service,
            "minio_console": {"port": 9001, "open": minio_console_open},
            "redpanda_kafka": {"port": 19092, "open": redpanda_kafka_open},
            "redpanda_admin": {"port": 19644, "open": redpanda_admin_open},
        },
        "transport": {"kind": settings.transport.kind, "bootstrap": settings.transport.bootstrap},
        "deepstream": {"version": deepstream_version},
        "database": db_counts,
        "evidence": {
            "json_records": json_probe["count"],
            "clips": clip_probe["count"],
            "available": bool(json_probe["available"] or clip_probe["available"]),
        },
    }
