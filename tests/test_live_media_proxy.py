"""Live viewing stays on the API origin, including HLS assets and WHEP sessions."""
import asyncio
from types import SimpleNamespace

import httpx
from fastapi import FastAPI

from inferops.adapters.inbound.http import cameras
from tests.test_media_relay import ControlAPI, declaration, relay


async def test_live_player_proxies_hls_assets_and_webrtc_session_locations(monkeypatch):
    selected = relay(ControlAPI())
    app = FastAPI()
    app.include_router(cameras.router, prefix="/v1")
    app.state.sources = SimpleNamespace(_lock=asyncio.Lock(), specs={"front-1": declaration()}, relay=selected)
    monkeypatch.setenv("INFEROPS_MEDIAMTX_API_URL", "http://media:9997")
    seen = []

    def upstream(request):
        seen.append(request)
        assert request.url.host == "media"
        assert request.url.path.startswith("/inferops/combined/front-1/")
        if request.method == "POST":
            assert request.content == b"offer"
            return httpx.Response(201, text="answer", headers={
                "Content-Type": "application/sdp", "Location": "/inferops/combined/front-1/whep/session"})
        if request.url.path.endswith("hls.min.js"):
            return httpx.Response(200, text="player code", headers={"Content-Type": "application/javascript"})
        if request.url.path.endswith("index.m3u8"):
            return httpx.Response(200, text="#EXTM3U\nsegment.mp4", headers={"Content-Type": "application/vnd.apple.mpegurl"})
        return httpx.Response(200, text='<video></video><script src="hls.min.js"></script>',
                              headers={"Content-Type": "text/html"})

    monkeypatch.setattr(cameras, "_live_http_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(upstream)))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://pod.proxy.example") as client:
            info = (await client.get("/v1/cameras/front-1/live")).json()
            assert info["hls_proxy_url"].startswith("/v1/cameras/front-1/live/hls/")
            assert "secret" not in str(info)
            base = "/v1/cameras/front-1/live/hls/"
            page = await client.get(base)
            assert page.status_code == 200 and 'src="hls.min.js"' in page.text
            assert (await client.get(base + "hls.min.js")).text == "player code"
            playlist = await client.get(base + "index.m3u8?token=viewer")
            assert playlist.text.startswith("#EXTM3U")
            assert seen[-1].url.params["token"] == "viewer"
            offer = await client.post("/v1/cameras/front-1/live/webrtc/whep", content="offer",
                                      headers={"Content-Type": "application/sdp"})
            assert offer.status_code == 201 and offer.text == "answer"
            assert offer.headers["location"] == "/v1/cameras/front-1/live/webrtc/whep/session"
            assert (await client.get(base + "%2e%2e/index.m3u8")).status_code == 400
            assert (await client.get("/v1/cameras/front-1/live/unknown/")).status_code == 404
    finally:
        await selected.close()


async def test_live_proxy_connection_failure_is_actionable_and_redacted(monkeypatch):
    selected = relay(ControlAPI())
    app = FastAPI()
    app.include_router(cameras.router, prefix="/v1")
    app.state.sources = SimpleNamespace(_lock=asyncio.Lock(), specs={"front-1": declaration()}, relay=selected)

    def unavailable(request):
        raise httpx.ConnectError("camera-secret control-secret", request=request)

    monkeypatch.setattr(cameras, "_live_http_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(unavailable)))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://pod.example") as client:
            response = await client.get("/v1/cameras/front-1/live/hls/")
            assert response.status_code == 502
            assert "MediaMTX" in response.text and "secret" not in response.text
    finally:
        await selected.close()
