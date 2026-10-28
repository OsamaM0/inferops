"""Portable workload identity for comparing capacity reports with a server."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import suppress
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def model_fingerprints(application):
    values = {}
    for name, spec in application.definition.models.items():
        path = Path(spec.artifact.path or "")
        if path.is_file():
            h = hashlib.sha256()
            with path.open("rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(block)
            values[name] = {"sha256": h.hexdigest(), "bytes": path.stat().st_size}
    return values


def workload_fingerprint(application):
    definition = application.definition
    payload = {
        "pipelines": {n: v.model_dump(mode="json") for n, v in definition.pipelines.items()},
        "models": {n: v.model_dump(mode="json") for n, v in definition.models.items()},
    }
    root = getattr(definition, "root", None)
    raw = json.dumps(payload, sort_keys=True)
    if root:
        raw = raw.replace(str(Path(root).resolve()), "$PROJECT")
    h = hashlib.sha256(raw.encode())
    if root:
        for path in sorted(Path(root).rglob("*.py")):
            if "tests" not in path.parts and "__pycache__" not in path.parts:
                h.update(str(path.relative_to(root)).encode())
                h.update(path.read_bytes())
    return h.hexdigest()


def software_fingerprint():
    root = Path(__file__).resolve().parents[3]
    h = hashlib.sha256()
    for package in ("onnxruntime", "onnxruntime-gpu", "av", "opencv-python-headless", "numpy"):
        with suppress(PackageNotFoundError):
            h.update(f"{package}:{version(package)}".encode())
    for path in sorted(root.rglob("*.py")):
        if "testing" not in path.relative_to(root).parts:
            h.update(str(path.relative_to(root)).encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def service_profile(application, *, media_router="direct", media_recording=False):
    """Do not promote embedded-memory or filesystem tests to SQL/S3 capacity."""
    import cv2

    repository = application.repository
    engine = getattr(repository, "engine", None)
    storage = application.evidence.storage if application.evidence else None
    return {
        "media_router": media_router,
        "media_recording": media_recording,
        "source_catalog": "postgresql" if engine is not None else "memory",
        "repository": type(repository).__name__,
        "sql_dialect": engine.dialect.name if engine is not None else None,
        "evidence_storage": type(storage).__name__ if storage is not None else None,
        "model_concurrency": application.models.max_concurrency,
        "auto_scale": application.models.auto_scale,
        "triton_concurrency": application.models.triton_concurrency,
        "admission_pending": application.models.admission_pending,
        "admission_queue_ms": application.models.admission_queue_ms,
        "triton_configurations": {name: backend.device_info.get("configuration_sha256")
                                   for name, backend in application.models.loaded_backends().items()
                                   if backend.device == "remote"},
        "opencv_threads": cv2.getNumThreads(),
        "ort_threads": int(os.getenv("INFEROPS_ORT_THREADS", "4")),
        "cudnn_max_workspace": int(os.getenv("INFEROPS_CUDNN_MAX_WORKSPACE", "0")),
        "preview_fps": 2,
        "preview_width": 640,
        "jpeg_quality": 75,
    }
