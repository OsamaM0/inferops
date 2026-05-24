"""WebSocket live feed."""
from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from inferops.pipeline.stages.sink_ws import HUB

router = APIRouter()


@router.websocket("/ws/{channel}")
async def ws(websocket: WebSocket, channel: str):
    await websocket.accept()
    HUB.subscribe(channel, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        HUB.unsubscribe(channel, websocket)
