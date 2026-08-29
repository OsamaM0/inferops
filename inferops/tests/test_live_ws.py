"""Smoke test for the WS live-infer endpoint (core tiny_classifier pipeline)."""
import cv2
import numpy as np
import orjson
import pytest
from fastapi.testclient import TestClient

from inferops.api.main import create_app


@pytest.mark.asyncio
async def test_live_ws_roundtrip():
    app = create_app()
    # TestClient drives lifespan on enter/exit.
    with TestClient(app) as client:
        with client.websocket_connect("/v1/live/tiny_classifier?stream_id=t") as ws:
            img = (np.random.rand(96, 96, 3) * 255).astype("uint8")
            ok, buf = cv2.imencode(".jpg", img)
            assert ok
            ws.send_bytes(buf.tobytes())

            # Server replies with binary orjson payload.
            data = ws.receive_bytes()
            msg = orjson.loads(data)
            assert msg["stream_id"] == "t"
            assert msg["seq"] == 1
            assert "latency_ms" in msg
            assert isinstance(msg["detections"], list)
            assert msg["classifications"], "classifier pipeline should return top-k classes"
            # Generic per-frame channels are always present so the dashboard
            # can render any domain without special cases.
            assert isinstance(msg["signals"], dict)
            assert isinstance(msg["overlay_boxes"], list)
            assert msg["model"] == "tiny_classifier"


def test_live_ws_unknown_pipeline_closes():
    app = create_app()
    with TestClient(app) as client:
        with pytest.raises(Exception):
            with client.websocket_connect("/v1/live/no_such_pipeline") as ws:
                ws.receive_bytes()
