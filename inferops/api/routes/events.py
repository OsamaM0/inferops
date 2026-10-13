"""Events + Alerts listing."""
from __future__ import annotations

from fastapi import APIRouter

from inferops.adapters.postgres_dao import list_alerts, list_events

router = APIRouter()


@router.get("/events")
async def get_events(pipeline: str | None = None, limit: int = 100):
    try:
        return await list_events(pipeline, limit)
    except Exception as e:
        return {"error": str(e), "events": []}


@router.get("/alerts")
async def get_alerts(limit: int = 100):
    try:
        return await list_alerts(limit)
    except Exception as e:
        return {"error": str(e), "alerts": []}
