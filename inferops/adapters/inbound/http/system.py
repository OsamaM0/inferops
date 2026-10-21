"""System status routes for browser-friendly service checks."""
from __future__ import annotations

import asyncio
import os
from typing import Any
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
    storage = application.evidence.storage if application.evidence else None
    mode = application.definition.mode
    remote = mode in {"triton"}
    is_s3 = storage is not None and hasattr(storage, "bucket")
    monitoring = os.getenv("INFEROPS_MONITORING_ENABLED", "false").lower() == "true"
    services = {"api": {"port": request.url.port or 80, "status": 200, "ok": True}}

    async def probe(name, url, enabled):
        services[name] = {"enabled": enabled, "status": await asyncio.to_thread(_http_status, url, .75) if enabled else None}

    await asyncio.gather(
        probe("triton_http", f"{settings.triton.http_url.rstrip('/')}/v2/health/ready", remote),
        probe("triton_metrics", os.getenv("INFEROPS_TRITON_METRICS_URL", "http://localhost:8002") + "/metrics", remote),
        probe("minio_s3", f"{settings.s3.endpoint}/minio/health/live", is_s3),
        probe("minio_console", os.getenv("INFEROPS_MINIO_CONSOLE_INTERNAL_URL", "http://localhost:9001"), is_s3),
        probe("grafana", os.getenv("INFEROPS_GRAFANA_INTERNAL_URL", "http://localhost:3000") + "/api/health", monitoring),
        probe("prometheus", os.getenv("INFEROPS_PROMETHEUS_INTERNAL_URL", "http://localhost:9090") + "/-/ready", monitoring),
    )
    services["s3"] = services["minio_s3"]
    relay = getattr(getattr(request.app.state, "sources", None), "relay", None)
    services["mediamtx"] = await relay.status() if relay else {"enabled": False, "status": None}
    runtime = await asyncio.to_thread(_runtime_info, application.models)
    runtime["execution_mode"] = mode
    records = await application.repository.list_evidence(None, 100)
    return {
        "services": services,
        "transport": {"kind": type(application.transport).__name__ if application.transport else "disabled"},
        "runtime": runtime,
        "database": await _db_counts(application.repository),
        "evidence": {"available": storage is not None, "backend": type(storage).__name__ if storage else "disabled",
                     "json_records": len(records), "clips": sum("clip" in row.get("artifacts", {}) for row in records),
                     "counts_scope": "latest 100 evidence records"},
    }


@router.get("/system/workspace")
async def workspace(request: Request):
    from inferops.application.execution import execution_declarations
    application = request.app.state.engine.application
    definition = application.definition
    loaded = application.models.loaded_backends()
    pipelines = []
    for spec in definition.pipelines.values():
        steps = [{"name": "source", "kind": "capture", "detail": "RTSP / video / image → paced frames; latest frame for live inputs"}]
        for decl in execution_declarations(spec):
            references = {key: decl.params[key] for key in ("model", "models", "fusion", "alerts") if key in decl.params}
            steps.append({"name": decl.name, "kind": decl.stage.split(":")[-1], "references": references})
        steps.extend([
            {"name": "commit", "kind": "persistence", "detail": "Events + incidents + evidence intent + outbox in one transaction"},
            {"name": "evidence", "kind": "capture", "detail": "Bounded ring → fsynced spool → encode → storage → Saved / Incomplete / Failed"},
            {"name": "preview", "kind": "output", "detail": "Camera ID + session ID + results + shared annotated JPEG"},
        ])
        pipelines.append({"name": spec.name, "label": spec.tags.get("label", spec.name),
                          "description": spec.tags.get("description", ""),
                          "steps": steps, "timeout_ms": spec.limits.timeout_ms})
    models = []
    for name, spec in definition.models.items():
        if definition.resolve_model(name) != name:
            continue
        backend = loaded.get(name)
        info = backend.device_info if backend else {}
        logical = next((key for key, variants in definition.resolver.variants.items() if name in variants), name)
        models.append({"name": name, "logical_name": logical,
                       "selected": definition.resolver.ports.get(logical, {definition.mode: name}).get(definition.mode) == name,
                       "backend": spec.backend, "loaded": backend is not None,
                       "device": backend.device if backend else None, "placement": info,
                       "input_shapes": [tensor.shape for tensor in spec.inputs],
                       "batching": {"server_max_batch_size": info.get("max_batch_size"),
                                    "dynamic": info.get("dynamic_batching", False)}})
    storage = application.evidence.storage if application.evidence else None
    dialect = getattr(getattr(application.repository, "engine", None), "dialect", None)
    return {"project": definition.name, "runtime": definition.mode, "pipelines": pipelines, "models": models,
            "admission": {"local_gpu_concurrency": application.models.max_concurrency,
                          "triton_outstanding_requests": application.models.triton_concurrency,
                          "max_pending": application.models.admission_pending,
                          "max_queue_age_ms": application.models.admission_queue_ms,
                          "policy": "round-robin by session",
                          "remote": application.models._rpc_slots.diagnostics()},
            "components": [
                {"name": "Session owner", "state": getattr(application, "role", "combined"), "purpose": "Camera registry, ordered pipeline state and authenticated worker control"},
                {"name": "Database", "state": getattr(dialect, "name", type(application.repository).__name__), "purpose": "Sessions, incidents, evidence and durable outbox"},
                {"name": "Evidence storage", "state": type(storage).__name__ if storage else "disabled", "purpose": "Snapshots, clips and manifests"},
                {"name": "Redpanda", "state": "enabled" if type(application.transport).__name__ == "RedpandaTransport" else "optional / disabled", "purpose": "External Kafka consumers and topic workers"},
                {"name": "Triton", "state": "enabled" if definition.mode in {"triton"} else "optional / disabled", "purpose": "Shared model serving and qualified dynamic batching"},
                {"name": "Monitoring", "state": "enabled" if os.getenv("INFEROPS_MONITORING_ENABLED", "false").lower() == "true" else "optional / disabled", "purpose": "Prometheus + Grafana; API metrics remain available"},
            ]}


@router.get("/system/triton")
async def triton_statistics(request: Request):
    application = request.app.state.engine.application
    remote = {name: backend for name, backend in application.models.loaded_backends().items()
              if backend.device == "remote"}
    results = await asyncio.gather(*(asyncio.wait_for(backend.statistics(), 2) for backend in remote.values()),
                                   return_exceptions=True)
    return {"enabled": application.definition.mode in {"triton"},
            "models": [{"model": name, "available": not isinstance(result, BaseException),
                        "statistics": result if not isinstance(result, BaseException) else None}
                       for name, result in zip(remote, results, strict=True)]}
