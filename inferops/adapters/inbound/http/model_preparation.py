"""Checkpoint prerequisites and validated, versioned ONNX exports."""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

router = APIRouter()
OptionalCheckpointFile = File(None)


@router.get("/model-preparation")
async def preparations(request: Request):
    jobs = request.app.state.checkpoints
    return {"configured": bool(jobs.python and Path(jobs.python).is_file()),
            "adapters": ["rfdetr"], "jobs": list(reversed(list(jobs.records.values()))),
            "prerequisite": "Trained .pth/.pt, supported RF-DETR architecture, class order matching the project. CPU export environment: INFEROPS_EXPORT_PYTHON.",
            "activation": "Validated exports are staged separately; use Configuration to select their artifact path. Triton additionally requires repository preparation and explicit model reload."}


@router.post("/model-preparation", status_code=202)
async def prepare(request: Request, file: UploadFile | None = OptionalCheckpointFile,
                  server_path: str = Form(""), variant: str = Form("auto"),
                  resolution: int = Form(576), max_batch: int = Form(4)):
    jobs = request.app.state.checkpoints
    if not jobs.python or not Path(jobs.python).is_file():
        raise HTTPException(400, "Configure INFEROPS_EXPORT_PYTHON before uploading a checkpoint")
    if len(jobs.tasks) >= 2:
        raise HTTPException(409, "Export queue is full")
    if bool(file) == bool(server_path.strip()):
        raise HTTPException(400, "Choose a checkpoint upload or a project checkpoint path")
    uploaded = None
    try:
        if file:
            suffix = Path(file.filename or "").suffix.lower()
            if suffix not in {".pt", ".pth"}:
                raise HTTPException(400, "Expected a .pth/.pt checkpoint")
            folder = jobs.root / "uploads"
            folder.mkdir(exist_ok=True)
            uploaded = folder / f"{uuid.uuid4().hex}{suffix}"
            size = 0
            with uploaded.open("wb") as stream:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > int(os.getenv("INFEROPS_CHECKPOINT_MAX_BYTES", str(512 * 1024**2))):
                        raise HTTPException(413, "Checkpoint exceeds configured upload limit")
                    await asyncio.to_thread(stream.write, chunk)
            path = uploaded
        else:
            root = request.app.state.engine.application.definition.root
            path = Path(server_path).resolve()
            if root is None or not path.is_relative_to(root.resolve()):
                raise HTTPException(400, "Server checkpoint must be inside the active project")
        return await jobs.submit(path, variant=variant, resolution=resolution, max_batch=max_batch)
    except BaseException:
        if uploaded:
            uploaded.unlink(missing_ok=True)
        raise


@router.get("/model-preparation/{job_id}/{asset}")
async def asset(request: Request, job_id: str, asset: str):
    jobs = request.app.state.checkpoints
    record = jobs.records.get(job_id)
    if record is None or asset not in {"report.json", "export.log", "model.onnx"}:
        raise HTTPException(404, "Export asset not found")
    if asset in {"report.json", "model.onnx"} and record["status"] != "validated":
        raise HTTPException(409, "Export has not passed validation")
    path = jobs.root / job_id / asset
    if not path.is_file():
        raise HTTPException(404, "Export asset not available")
    return FileResponse(path, filename=asset)
