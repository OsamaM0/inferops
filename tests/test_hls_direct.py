"""HLS direct capture regression: inference need not depend on MediaMTX HLS."""
import asyncio

import httpx
import pytest

from inferops.adapters.inbound.http.app import create_application
from inferops.adapters.outbound.media import mediamtx
from inferops.adapters.outbound.media.camera import _input_options, _capture_status
from tests.test_media_relay import ControlAPI, declaration, relay
from tests.test_redesign_engine import engine


SIGNED = "https://example.invalid/live/index.m3u8?token=private-token"

def test_hls_uses_safe_per_context_options():
    hls = declaration(kind="hls", uri=SIGNED)
    options = _input_options(hls)
    assert "rtsp_transport" not in options
    assert options["reconnect"] == "1"
    assert options["probesize"] == "1048576"
    assert "private-token" not in _capture_status(Exception(f"Server returned 403 for {SIGNED}"))
    assert _input_options(declaration())["rtsp_transport"] == "tcp"


async def test_hls_direct_inference_does_not_call_media_router(tmp_path, monkeypatch):
    api = ControlAPI()
    gateway = relay(api)
    monkeypatch.setattr(mediamtx, "configured_relay", lambda **kwargs: gateway)
    app = create_application(engine(tmp_path), source_root=tmp_path / "sources")
    observed, ready = [], asyncio.Event()

    async def frames(spec, state):
        observed.append((spec.kind, spec.uri))
        ready.set()
        await asyncio.Event().wait()
        yield None

    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://localhost"
    ) as client:
        app.state.sources.frames = frames
        result = await client.post("/v1/cameras", json=declaration(kind="hls", uri=SIGNED).model_dump())
        assert result.status_code == 201, result.text
        assert result.json()["live_video_available"] is False
        assert not api.mutations
        await client.post("/v1/cameras/front-1/start")
        await asyncio.wait_for(ready.wait(), timeout=2)
        assert observed == [("hls", SIGNED)]
        assert (await client.get("/v1/cameras/front-1/live")).status_code == 409
        assert not api.mutations
        await client.post("/v1/cameras/front-1/stop")
        assert (await client.delete("/v1/cameras/front-1")).status_code == 200
        assert not api.mutations


async def test_hls_relay_is_opt_in(tmp_path, monkeypatch):
    api = ControlAPI()
    gateway = relay(api)
    monkeypatch.setattr(mediamtx, "configured_relay", lambda **kwargs: gateway)
    app = create_application(engine(tmp_path), source_root=tmp_path / "sources")

    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://localhost"
    ) as client:
        payload = declaration(kind="hls", uri=SIGNED, hls_mode="relay").model_dump()
        result = await client.post("/v1/cameras", json=payload)
        assert result.status_code == 201, result.text
        assert result.json()["live_video_available"] is True
        assert api.paths["inferops/combined/front-1"]["source"] == SIGNED
        edited = await client.put("/v1/cameras/front-1", json={"hls_mode": "direct"})
        assert edited.status_code == 200
        assert edited.json()["live_video_available"] is False
        assert not api.paths
