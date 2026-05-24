"""Health + metrics routes."""
from __future__ import annotations

from fastapi import APIRouter, Response

from inferops.obs import metrics_text

router = APIRouter()


@router.get("/healthz")
async def healthz():
    return {"status": "ok"}


@router.get("/readyz")
async def readyz():
    return {"status": "ready"}


@router.get("/metrics")
async def metrics():
    return Response(content=metrics_text(), media_type="text/plain; version=0.0.4")
