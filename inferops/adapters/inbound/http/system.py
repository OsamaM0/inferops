"""System status routes for browser-friendly service checks."""
from __future__ import annotations

import asyncio
import os
import socket
import subprocess
from contextlib import suppress
from functools import lru_cache
from typing import Any
from urllib.parse import urlsplit
from urllib.request import urlopen

from fastapi import APIRouter, Request
from sqlalchemy import func, select

from inferops.adapters.outbound.projects.settings import settings

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


def _safe_s3_probe(bucket: str, prefix: str = "", client=None) -> dict[str, Any]:
    try:
        pages = client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
        return {"available": True, "count": sum(int(page.get("KeyCount") or 0) for page in pages)}
    except Exception:
        return {"available": False, "count": None}


def _runtime_info(registry=None) -> dict[str, Any]:
    """Report what device inference is actually running on.

    Nothing previously surfaced this, so a spec asking for ``preferred: gpu``
    that silently degraded to CPU — missing CUDA libs, no visible device, an
    ONNX Runtime built without the CUDA provider — looked identical to a
    healthy GPU deployment. ``downgraded`` is the flag to alert on.
    """
    from inferops.adapters.outbound.inference.devices import describe_runtime

    info: dict[str, Any] = describe_runtime()
    info["execution_mode"] = settings.execution_mode
    info["video_decoder"] = "pyav"
    models: list[dict[str, Any]] = []
    try:
        for name, backend in (registry.loaded_backends() if registry is not None else {}).items():
            device_info = getattr(backend, "device_info", None) or {}
            models.append({
                "model": name,
                "backend": type(backend).__name__,
                "device": getattr(backend, "device", None),
                "requested_device": device_info.get("requested"),
                "downgraded": bool(device_info.get("downgraded", False)),
                "reason": device_info.get("reason"),
                "active_providers": device_info.get("active_providers"),
            })
    except Exception as exc:                      # never let status 500
        info["models_error"] = str(exc)
    info["models"] = models
    info["models_downgraded"] = [m["model"] for m in models if m["downgraded"]]
    return info


async def _db_counts(repository) -> dict[str, int | None]:
    try:
        async with repository.engine.begin() as conn:
            events = (await conn.execute(select(func.count()).select_from(repository.events))).scalar()
            alerts = (await conn.execute(select(func.count()).select_from(repository.incidents))).scalar()
        return {"events": int(events or 0), "alerts": int(alerts or 0)}
    except Exception:
        return {"events": None, "alerts": None}


@router.get("/system/status")
async def system_status(request: Request):
    application = request.app.state.engine.application
    storage = application.evidence.storage
    with suppress(Exception):
        await storage.start()
    client = getattr(storage, "client", None)
    s3_url = urlsplit(settings.s3.endpoint or "http://localhost:9000")
    s3_host, s3_port = s3_url.hostname or "localhost", s3_url.port or (443 if s3_url.scheme == "https" else 80)
    triton_url = urlsplit(settings.triton.http_url)
    triton_host = triton_url.hostname or "localhost"
    grpc_url = urlsplit(f"//{settings.triton.url}")
    kafka_url = urlsplit(f"//{settings.transport.bootstrap.split(',')[0]}")
    console_port = int(os.getenv("INFEROPS_MINIO_CONSOLE_PORT", str(s3_port + 1)))
    console_url = urlsplit(os.getenv("INFEROPS_MINIO_CONSOLE_INTERNAL_URL", f"http://{s3_host}:{console_port}"))
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
        grafana_status,
        prometheus_status,
    ) = await asyncio.gather(
        _db_counts(application.repository),
        asyncio.to_thread(_safe_s3_probe, getattr(storage, "bucket", settings.s3.bucket), "evidence/", client),
        asyncio.to_thread(_safe_s3_probe, "inferops-clips", "", client),
        asyncio.to_thread(_http_status, f"{settings.triton.http_url.rstrip('/')}/v2/health/ready", 0.75),
        asyncio.to_thread(_port_open, grpc_url.hostname or triton_host, grpc_url.port or 8001, 0.25),
        asyncio.to_thread(_http_status, f"{triton_url.scheme}://{triton_host}:8002/metrics", 0.75),
        asyncio.to_thread(_http_status, f"{settings.s3.endpoint}/minio/health/live", 0.75),
        asyncio.to_thread(_port_open, s3_host, s3_port, 0.25),
        asyncio.to_thread(_port_open, console_url.hostname or s3_host, console_url.port or 9001, 0.25),
        asyncio.to_thread(_port_open, kafka_url.hostname or "localhost", kafka_url.port or 19092, 0.25),
        asyncio.to_thread(_port_open, kafka_url.hostname or "localhost", 9644, 0.25),
        asyncio.to_thread(_deepstream_version),
        asyncio.to_thread(_http_status, os.getenv("INFEROPS_GRAFANA_INTERNAL_URL", "http://localhost:3000").rstrip("/") + "/api/health", 0.75),
        asyncio.to_thread(_http_status, os.getenv("INFEROPS_PROMETHEUS_INTERNAL_URL", "http://localhost:9090").rstrip("/") + "/-/ready", 0.75),
    )
    s3_backend = "minio" if s3_health == 200 else ("s3" if json_probe["available"] or clip_probe["available"] else "down")
    s3_service = {"port": s3_port, "open": s3_open, "backend": s3_backend, "endpoint": settings.s3.endpoint}
    return {
        "services": {
            "api": {"port": request.url.port or (443 if request.url.scheme == "https" else 80), "status": 200, "ok": True},
            "triton_http": {"port": 8000, "status": triton_http_status,
                            "enabled": settings.execution_mode in {"triton", "deepstream"}},
            "triton_grpc": {"port": 8001, "open": triton_grpc_open,
                            "enabled": settings.execution_mode in {"triton", "deepstream"}},
            "triton_metrics": {"port": 8002, "status": triton_metrics_status,
                               "enabled": settings.execution_mode in {"triton", "deepstream"}},
            "s3": s3_service,
            "minio_s3": s3_service,
            "minio_console": {"port": console_port, "open": minio_console_open},
            "grafana": {"port": int(os.getenv("INFEROPS_GRAFANA_PORT", "3000")), "status": grafana_status},
            "prometheus": {"port": int(os.getenv("INFEROPS_PROMETHEUS_PORT", "9090")), "status": prometheus_status},
            "redpanda_kafka": {"port": 19092, "open": redpanda_kafka_open, "enabled": settings.transport.kind == "redpanda"},
            "redpanda_admin": {"port": 9644, "open": redpanda_admin_open, "enabled": settings.transport.kind == "redpanda"},
        },
        "transport": {"kind": settings.transport.kind, "bootstrap": settings.transport.bootstrap},
        "runtime": _runtime_info(getattr(getattr(request.app.state, "engine", None), "application", None).models
                                 if getattr(request.app.state, "engine", None) else None),
        "deepstream": {"version": deepstream_version},
        "database": db_counts,
        "evidence": {
            "json_records": json_probe["count"],
            "clips": clip_probe["count"],
            "available": bool(json_probe["available"] or clip_probe["available"]),
        },
    }
