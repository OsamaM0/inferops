"""Reviewable retention plans; every deletion is routed to the current owner."""
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Request

from inferops.adapters.inbound.http.cameras import forward_owned
from inferops.application.retention import RetentionService

router = APIRouter()


@router.get("/retention/outbox")
async def outbox_plan(request: Request, before: datetime, limit: int = Query(100, ge=1, le=1000)):
    if before.tzinfo is None:
        raise HTTPException(422, "cutoff must include a timezone")
    if not hasattr(request.app.state.engine.application.repository, "prune_delivered_outbox"):
        raise HTTPException(409, "this repository does not implement retention")
    return await request.app.state.engine.application.repository.prune_delivered_outbox(before, limit=limit)


@router.post("/retention/outbox")
async def outbox_apply(request: Request, before: datetime, limit: int = Query(100, ge=1, le=1000)):
    if before.tzinfo is None:
        raise HTTPException(422, "cutoff must include a timezone")
    if not hasattr(request.app.state.engine.application.repository, "prune_delivered_outbox"):
        raise HTTPException(409, "this repository does not implement retention")
    return await request.app.state.engine.application.repository.prune_delivered_outbox(before, dry_run=False, limit=limit)


@router.get("/retention/evidence")
async def plan(request: Request, before: datetime, limit: int = Query(100, ge=1, le=1000), worker_id: str | None = None):
    if before.tzinfo is None:
        raise HTTPException(422, "cutoff must include a timezone")
    if worker_id:
        forwarded = await forward_owned(request, "camera", "", worker_id=worker_id)
        if forwarded is not None:
            return forwarded
    application = request.app.state.engine.application
    if not hasattr(application.repository, "retention_candidates"):
        raise HTTPException(409, "this repository does not implement retention")
    return await RetentionService(application.repository, application.evidence).plan(before, limit=limit)


@router.post("/retention/evidence/{evidence_id}")
async def apply(request: Request, evidence_id: str, before: datetime):
    if before.tzinfo is None:
        raise HTTPException(422, "cutoff must include a timezone")
    forwarded = await forward_owned(request, "evidence", evidence_id)
    if forwarded is not None:
        return forwarded
    application = request.app.state.engine.application
    if application.evidence is None:
        raise HTTPException(409, "evidence storage is disabled")
    return await RetentionService(application.repository, application.evidence).apply([evidence_id], before)
