"""One packaged operations console and browser-safe service discovery."""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

API_DIR = Path(__file__).resolve().parent

# Public ports and optional URLs are separate from Docker's internal endpoints.
SERVICE_INTERFACES = {
    "minio": ("MinIO Console", "MINIO_CONSOLE", 9001, "/", "Buckets, objects, users and storage administration"),
    "grafana": ("Grafana", "GRAFANA", 3000, "/d/inferops-overview/inferops-overview", "Metrics dashboards and exploration"),
    "prometheus": ("Prometheus", "PROMETHEUS", 9090, "/graph", "Metrics queries, scrape targets and alerts"),
}
API_PAGES = {
    "cameras": ("Sources", "/ui/cameras", "Register RTSP, video and image sources; annotations, history and measured capacity"),
    "dashboard": ("InferOps", "/ui/", "Project controls, live preview, activity and configuration forms"),
    "guide": ("Interface guide", "/ui/guide", "All interfaces, ports and API routes in this running instance"),
    "swagger": ("Swagger API docs", "/docs", "Interactive HTTP API documentation"),
    "redoc": ("ReDoc", "/redoc", "Readable HTTP API reference"),
    "openapi": ("OpenAPI schema", "/openapi.json", "Machine-readable API specification"),
    "health": ("Health", "/healthz", "API process health"),
    "readiness": ("Readiness", "/readyz", "API startup readiness"),
    "metrics": ("API metrics", "/metrics", "Prometheus text exposition"),
}
PORTS = [
    ("API", "8080", "8080", "HTTP / WebSocket", "INFEROPS_API_PORT"),
    ("MinIO S3", "9000", "9000", "S3 HTTP API", "INFEROPS_S3_PORT"),
    ("MinIO Console", "9001", "9001", "Original web console", "INFEROPS_MINIO_CONSOLE_PORT"),
    ("Grafana", "3000", "3000", "Original web UI / HTTP API", "INFEROPS_GRAFANA_PORT"),
    ("Prometheus", "9090", "9090", "Original web UI / HTTP API", "INFEROPS_PROMETHEUS_PORT"),
    ("PostgreSQL", "5432", "5432", "PostgreSQL protocol; no web UI", "INFEROPS_PG_PORT"),
    ("Redpanda Kafka", "19092", "19092 / 9092", "Kafka protocol; no web UI", "INFEROPS_KAFKA_PORT"),
    ("Triton HTTP", "8000", "8000", "Inference API; Triton mode", "INFEROPS_TRITON_HTTP_PORT"),
    ("Triton gRPC", "8001", "8001", "Inference gRPC; Triton mode", "INFEROPS_TRITON_GRPC_PORT"),
    ("Triton metrics", "8002", "8002", "Metrics; Triton mode", "INFEROPS_TRITON_METRICS_PORT"),
    ("MediaMTX RTSP", "8554", "8554", "Default shared camera relay", "INFEROPS_MEDIAMTX_RTSP_PORT"),
    ("MediaMTX WebRTC", "8889", "8889", "Live video player", "INFEROPS_MEDIAMTX_WEBRTC_PORT"),
    ("MediaMTX HLS", "8888", "8888", "HLS player", "INFEROPS_MEDIAMTX_HLS_PORT"),
    ("MediaMTX playback", "9996", "9996", "Original stream recordings", "INFEROPS_MEDIAMTX_PLAYBACK_PORT"),
    ("MediaMTX ICE", "8189/udp", "8189/udp", "WebRTC media", ""),
    ("MediaMTX control", "9997 (loopback)", "9997", "Authenticated internal API", "INFEROPS_MEDIAMTX_API_PORT"),
    ("Redpanda admin", "unpublished", "9644", "Internal HTTP admin / metrics API", ""),
    ("Redpanda Schema Registry", "unpublished", "8081", "Internal HTTP API", ""),
    ("Redpanda Pandaproxy", "unpublished", "8082", "Internal HTTP Kafka proxy", ""),
    ("Redpanda RPC", "unpublished", "33145", "Internal broker RPC", ""),
]


def service_url(request: Request, name: str) -> str:
    """Return a browser address, never a container hostname or a storage secret."""
    _, prefix, default_port, path, _ = SERVICE_INTERFACES[name]
    configured = os.getenv(f"INFEROPS_{prefix}_PUBLIC_URL", "").strip()
    if configured:
        parsed = urlsplit(configured)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise HTTPException(503, f"INFEROPS_{prefix}_PUBLIC_URL must be an HTTP(S) URL without credentials")
        return configured.rstrip("/") + "/"
    port = int(os.getenv(f"INFEROPS_{prefix}_PORT", str(default_port)))
    return str(request.url.replace(port=port, path=path, query="", fragment=""))


def interface_catalog(request: Request) -> list[dict]:
    interfaces = [
        {"id": name, "title": title, "url": service_url(request, name),
         "description": description, "port": int(os.getenv(f"INFEROPS_{prefix}_PORT", str(port))),
         "native": True}
        for name, (title, prefix, port, _, description) in SERVICE_INTERFACES.items()
    ]
    interfaces.extend(
        {"id": name, "title": title, "url": str(request.url.replace(path=path, query="", fragment="")),
         "description": description, "port": request.url.port, "native": False}
        for name, (title, path, description) in API_PAGES.items()
    )
    return interfaces


def mount_ui(app: FastAPI) -> None:
    """Keep application HTML, CSS and JavaScript in the installed API package."""
    @app.get("/ui/cameras", include_in_schema=False)
    @app.get("/ui/guide", include_in_schema=False)
    @app.get("/ui", include_in_schema=False)
    @app.get("/ui/", include_in_schema=False)
    @app.get("/frontend", include_in_schema=False)
    @app.get("/frontend/", include_in_schema=False)
    async def dashboard():
        return FileResponse(API_DIR / "templates/dashboard.html", headers={"Cache-Control": "no-cache"})

    app.mount("/ui/static", StaticFiles(directory=API_DIR / "static"), name="ui-assets")

    @app.get("/v1/interfaces", tags=["interfaces"])
    async def interfaces(request: Request):
        return {
            "interfaces": interface_catalog(request),
            "ports": [
                {"name": name, "host": os.getenv(variable, host) if variable else host,
                 "internal": internal, "protocol": protocol, "override": variable}
                for name, host, internal, protocol, variable in PORTS
            ],
        }

    @app.get("/v1/interfaces/{name}", tags=["interfaces"])
    async def open_interface(name: str, request: Request):
        for item in interface_catalog(request):
            if item["id"] == name:
                return RedirectResponse(item["url"], status_code=307)
        raise HTTPException(404, "Unknown interface")
