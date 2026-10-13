"""Low-latency WebSocket live-inference endpoint.

Replaces the previous "POST a JPEG every N ms" dashboard loop with a single
bidirectional WebSocket per browser session:

  ws://host/v1/live/{pipeline}?stream_id=<id>&model=<override>

Protocol
========
- The client sends **binary** frames (JPEG / PNG). One binary message = one
  frame. Frames arriving while another frame is still being inferred are
  *dropped* (oldest-wins back-pressure) so we never build a queue and never
  pay catch-up latency on the wire.
- The server replies with a small JSON object per processed frame:

      {
        "ts_ms": <client timestamp>,
        "seq": <monotonic counter>,
        "latency_ms": <e2e ms inside the server>,
        "detections": [{"label", "score", "box":[x1,y1,x2,y2], "track_id"}],
        "results": [...],
        "fusion_events": [...],
        "alerts": [...],
        "model": "<model used>",
      }

- Text messages from the client are interpreted as small JSON control
  packets, e.g. ``{"ts_ms": 1234567890, "seq": 42}`` carried *before* the
  next binary frame; the values are attached to the next frame.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import orjson
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect

from inferops.api.deps import pipelines as pipelines_dep
from inferops.core.errors import RegistryError
from inferops.core.types import StageItem, new_id, now_ms
from inferops.obs import E2E_LATENCY, FRAMES_DROPPED, WS_CLIENTS, get_logger

log = get_logger(__name__)
router = APIRouter()

def _slim(item: StageItem) -> dict[str, Any]:
    """Pre-build the JSON payload by hand so we never serialize numpy frames.

    Domain stages expose extra per-frame state through two generic payload
    channels that the dashboard renders without knowing the domain:

    - ``payload["signals"]``: flat ``{name: number | string}`` shown as chips
    - ``payload["overlay_boxes"]``: ``[{"label", "box": [x1,y1,x2,y2], "tone"}]``
      normalised boxes drawn on the preview in addition to detections
    """
    out: dict[str, Any] = {
        "id": item.id,
        "stream_id": item.stream_id,
        "ts_ms": item.ts_ms,
        "seq": item.seq,
        "results": [],
        "detections": [],
        "classifications": [],
        "fusion_events": item.payload.get("fusion_events", []),
        "alerts": item.payload.get("alerts", []),
        "signals": item.payload.get("signals", {}),
        "overlay_boxes": item.payload.get("overlay_boxes", []),
        "model_selection": item.payload.get("model_selection"),
    }
    last_latency = 0.0
    for r in item.results:
        out["results"].append({
            "kind": r.kind,
            "model": r.model,
            "latency_ms": r.latency_ms,
        })
        last_latency = r.latency_ms or last_latency
        if r.detections:
            for d in r.detections:
                out["detections"].append({
                    "label": d.label,
                    "score": d.score,
                    "box": list(d.box),
                    "track_id": d.attrs.get("track_id"),
                })
        if r.classifications:
            for c in r.classifications:
                out["classifications"].append({
                    "label": c.label, "score": c.score, "index": c.index,
                })
    out["model"] = (item.payload.get("model_selection") or {}).get("used", "")
    out["model_latency_ms"] = last_latency
    return out


@router.websocket("/live/{pipeline}")
async def live_infer(
    websocket: WebSocket,
    pipeline: str,
    stream_id: str = Query("dashboard-live"),
    model: str | None = Query(None),
):
    try:
        store = pipelines_dep()
        runner = await store.get_runner(pipeline)
    except RegistryError as exc:
        await websocket.close(code=4404, reason=str(exc))
        return
    except HTTPException as exc:
        await websocket.close(code=4000, reason=exc.detail)
        return

    await websocket.accept()
    WS_CLIENTS.labels(channel=f"live:{pipeline}").inc()

    seq = 0
    inflight: asyncio.Task | None = None
    pending_meta: dict[str, Any] = {}
    sender_lock = asyncio.Lock()

    async def _process_and_reply(payload: bytes, meta: dict[str, Any]) -> None:
        nonlocal seq
        seq += 1
        t_recv_ms = meta.get("client_ts_ms") or now_ms()
        t0 = time.perf_counter()
        item = StageItem(
            id=new_id(),
            stream_id=stream_id,
            ts_ms=t_recv_ms,
            seq=seq,
            payload={
                "image_bytes": payload,
                "live_mode": True,           # let stages skip heavy sinks
                "client_ts_ms": t_recv_ms,
            },
        )
        if model:
            item.payload["model_override"] = model
        try:
            outs = await runner.run_one(item)
        except Exception as exc:
            log.warning("live_infer_failed", pipeline=pipeline, err=str(exc))
            async with sender_lock:
                with _suppress_disconnect():
                    await websocket.send_text(
                        orjson.dumps({"error": str(exc), "seq": seq}).decode()
                    )
            return
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        E2E_LATENCY.labels(pipeline=pipeline).observe(elapsed_ms / 1000.0)

        if not outs:
            return
        msg = _slim(outs[0])
        msg["latency_ms"] = round(elapsed_ms, 2)
        async with sender_lock:
            with _suppress_disconnect():
                await websocket.send_bytes(orjson.dumps(msg))

    try:
        while True:
            event = await websocket.receive()
            if event["type"] == "websocket.disconnect":
                break
            if "bytes" in event and event["bytes"] is not None:
                if inflight and not inflight.done():
                    # Drop this frame: server is still working on a previous
                    # one. This is the back-pressure that keeps latency low.
                    FRAMES_DROPPED.labels(pipeline=pipeline, reason="busy").inc()
                    continue
                meta, pending_meta = pending_meta, {}
                inflight = asyncio.create_task(_process_and_reply(event["bytes"], meta))
            elif event.get("text"):
                # Optional control message preceding the next binary frame.
                try:
                    pending_meta = orjson.loads(event["text"]) or {}
                except orjson.JSONDecodeError:
                    pending_meta = {}
    except WebSocketDisconnect:
        pass
    finally:
        if inflight and not inflight.done():
            inflight.cancel()
        WS_CLIENTS.labels(channel=f"live:{pipeline}").dec()


class _suppress_disconnect:
    def __enter__(self) -> _suppress_disconnect:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return exc_type is not None and issubclass(exc_type, (WebSocketDisconnect, RuntimeError))
