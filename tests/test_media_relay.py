"""Relay contracts, credential boundaries, owner routing and decoder selection."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import Response

from inferops.adapters.inbound.http.app import create_application
from inferops.adapters.inbound.http.cameras import router
from inferops.adapters.outbound.media.mediamtx import MediaMTX, MediaRelayError
from inferops.adapters.outbound.persistence.source_catalog import PostgresSourceStore
from inferops.application.sources import SourceManager
from inferops.domain.sources import SourceSpec
from inferops.testing.postgres import test_database_url
from tests.test_redesign_engine import engine


class ControlAPI:
    def __init__(self):
        self.paths, self.mutations = {}, []

    def __call__(self, request):
        prefix = "/v3/config/paths/"
        operation, name = request.url.path.removeprefix(prefix).split("/", 1)
        if operation == "get":
            return httpx.Response(200, json=self.paths[name]) if name in self.paths else httpx.Response(404)
        self.mutations.append((operation, name))
        if operation == "delete":
            return httpx.Response(200 if self.paths.pop(name, None) is not None else 404)
        self.paths[name] = json.loads(request.content)
        return httpx.Response(200, json={"status": "ok"})


def declaration(**patch):
    return SourceSpec.model_validate({"camera_id": "front-1", "name": "Front", "kind": "rtsp",
                                      "uri": "rtsp://operator:camera-secret@camera/live", "pipeline": "test", **patch})


def relay(api, **kwargs):
    return MediaMTX("http://media:9997", "rtsp://media:8554", password="control-secret",
                    transport=httpx.MockTransport(api), **kwargs)


async def test_relay_is_idempotent_updates_credentials_and_restores_after_router_restart():
    api = ControlAPI()
    selected = relay(api, scope="camera-a")
    try:
        spec = declaration(timeout_s=30)
        routed = await selected.prepare(spec)
        name = "inferops/camera-a/front-1"
        assert routed.uri == f"rtsp://media:8554/{name}"
        assert "camera-secret" not in routed.uri and "camera-secret" in spec.uri
        assert api.paths[name]["sourceOnDemand"] is True
        assert api.paths[name]["alwaysAvailable"] is False  # offline placeholders must not become AI input
        assert api.paths[name]["record"] is True  # original recordings supplement annotated incident evidence
        await selected.prepare(spec)
        assert api.mutations == [("add", name)]  # no reconnect-inducing update for unchanged configuration
        await selected.prepare(declaration(uri="rtsp://operator:new-secret@camera/live"))
        assert api.mutations[-1] == ("patch", name)
        assert "new-secret" in api.paths[name]["source"]
        api.paths.clear()  # Control API changes do not survive a MediaMTX restart
        await selected.prepare(spec)
        assert api.mutations[-1] == ("add", name)
        image = declaration(kind="image", uri="file.jpg")
        assert await selected.prepare(image) is image
        await selected.remove(spec.camera_id)
        await selected.remove(spec.camera_id)
        assert not api.paths
    finally:
        await selected.close()


@pytest.mark.parametrize("mode", ["http", "network", "malformed", "missing-write"])
async def test_upstream_errors_are_actionable_and_do_not_expose_credentials(mode):
    def fail(request):
        if mode == "network":
            raise httpx.ConnectError("camera-secret control-secret", request=request)
        if mode == "malformed":
            return httpx.Response(200, text="camera-secret")
        if mode == "missing-write":
            return httpx.Response(404)
        return httpx.Response(403, text="camera-secret control-secret")

    selected = relay(fail)
    try:
        with pytest.raises(MediaRelayError) as error:
            await selected.prepare(declaration())
        assert "MediaMTX" in str(error.value) and "secret" not in str(error.value)
    finally:
        await selected.close()


async def test_http_registry_retains_original_uri_but_decoder_reads_relay_and_remove_cleans_path(tmp_path, monkeypatch):
    from inferops.adapters.outbound.media import mediamtx

    api = ControlAPI()
    selected = relay(api)
    monkeypatch.setattr(mediamtx, "configured_relay", lambda **kwargs: selected)
    monkeypatch.setenv("INFEROPS_MEDIAMTX_WEBRTC_PUBLIC_URL", "https://video.example/webrtc")
    monkeypatch.setenv("INFEROPS_MEDIAMTX_HLS_PUBLIC_URL", "https://video.example/hls")
    app = create_application(engine(tmp_path), source_root=tmp_path / "sources")
    seen, reading = [], asyncio.Event()

    async def frames(spec, state):
        seen.append(spec.uri)
        reading.set()
        await asyncio.Event().wait()
        yield None

    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://public.example:8081"
    ) as client:
        app.state.sources.frames = frames
        created = await client.post("/v1/cameras", json=declaration().model_dump())
        assert created.status_code == 201 and created.json()["live_video_available"]
        assert "secret" not in created.text
        assert (await app.state.sources.store.list_sources())[0].uri == declaration().uri
        live = await client.get("/v1/cameras/front-1/live")
        assert live.status_code == 200 and live.json()["annotations"] is False
        assert live.json()["webrtc_url"].startswith("https://video.example/webrtc/inferops/combined/front-1?")
        assert "secret" not in live.text and "rtsp://" not in live.text
        assert (await client.post("/v1/cameras/front-1/start")).status_code == 200
        await asyncio.wait_for(reading.wait(), 1)
        assert seen == ["rtsp://media:8554/inferops/combined/front-1"]
        await client.post("/v1/cameras/front-1/stop")
        updated = await client.put("/v1/cameras/front-1", json={"uri": "rtsp://other/live"})
        assert updated.status_code == 200
        assert api.paths["inferops/combined/front-1"]["source"] == "rtsp://other/live"
        monkeypatch.setenv("INFEROPS_MEDIAMTX_WEBRTC_PUBLIC_URL", "https://user:private@video.example")
        invalid = await client.get("/v1/cameras/front-1/live")
        assert invalid.status_code == 503 and "private" not in invalid.text
        assert (await client.delete("/v1/cameras/front-1")).status_code == 200
        assert not api.paths


async def test_worker_live_urls_use_api_browser_host_instead_of_internal_worker():
    async def record_scope(kind, record_id):
        return "camera-a"

    class Control:
        async def forward(self, *args, **kwargs):
            return Response(json.dumps({"path": "inferops/camera-a/front-1",
                                        "webrtc_url": "http://internal-worker:8889/unused"}),
                            media_type="application/json")

    app = FastAPI()
    app.include_router(router, prefix="/v1")
    app.state.role = "api"
    app.state.engine = SimpleNamespace(application=SimpleNamespace(repository=SimpleNamespace(record_scope=record_scope)))
    app.state.worker_controls = {"a": Control()}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://browser.example:8080") as client:
        response = await client.get("/v1/cameras/front-1/live")
        assert response.status_code == 200
        assert response.json()["webrtc_url"].startswith("http://browser.example:8889/inferops/camera-a/front-1?")
        assert "internal-worker" not in response.text


@pytest.mark.parametrize("manual_stop", [False, True])
async def test_initial_router_outage_retries_autostart_but_respects_manual_stop(tmp_path, manual_stop):
    restored = asyncio.Event()

    class Relay:
        available = False

        async def prepare(self, spec):
            if not self.available:
                raise MediaRelayError("MediaMTX is unavailable")
            restored.set()
            return spec

        async def close(self):
            pass

    class Session:
        id = "autostart-session"

        async def close(self, reason):
            pass

    async def open_session(*args, **kwargs):
        return Session()

    async def frames(spec, stats):
        await asyncio.Event().wait()
        yield None

    store = PostgresSourceStore(test_database_url(tmp_path), preview_root=tmp_path / "previews")
    await store.start()
    await store.put_source(declaration(autostart=True))
    await store.close()
    gateway = Relay()
    selected = SourceManager(SimpleNamespace(definition=SimpleNamespace(pipelines={"test": None}),
        open_session=open_session), store, frames=frames, render=None, serialize=None, relay=gateway)
    try:
        await selected.start()
        assert not selected.tasks
        assert selected.snapshot("front-1")["status"] == "waiting"
        await asyncio.sleep(.01)  # let the first failed reconciliation complete
        if manual_stop:
            await selected.stop("front-1")
            assert selected.snapshot("front-1")["status"] == "stopped"
        gateway.available = True
        await asyncio.wait_for(restored.wait(), 12)
        if not manual_stop:
            async def restarted():
                while "front-1" not in selected.tasks:
                    await asyncio.sleep(.01)
            await asyncio.wait_for(restarted(), 2)
        assert ("front-1" in selected.tasks) is not manual_stop
    finally:
        await selected.close()


async def test_recordings_use_native_playback_and_remove_internal_urls(monkeypatch):
    from urllib.parse import parse_qs, urlsplit

    from inferops.adapters.outbound.media.mediamtx import configured_relay

    calls = []

    def playback(request):
        calls.append(request)
        return httpx.Response(200, json=[{"start": "2026-10-20T10:00:00Z", "duration": 12.5,
                                         "url": "http://private-media:9996/get?secret=hidden"}])

    selected = relay(playback, playback_url="http://private-media:9996", scope="camera-a")
    app = FastAPI()
    app.include_router(router, prefix="/v1")
    app.state.role = "combined"
    app.state.sources = SimpleNamespace(specs={"front-1": declaration()}, relay=selected)
    monkeypatch.setenv("INFEROPS_MEDIAMTX_PLAYBACK_PUBLIC_URL", "https://video.example/archive")
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="https://api.example") as client:
            response = await client.get("/v1/cameras/front-1/recordings")
            assert response.status_code == 200
            payload = response.json()
            assert payload["annotations"] is False
            row = payload["recordings"][0]
            parsed = urlsplit(row["url"])
            assert parsed.netloc == "video.example" and parsed.path == "/archive/get"
            assert parse_qs(parsed.query)["path"] == ["inferops/camera-a/front-1"]
            assert "private-media" not in response.text and "secret" not in response.text
            assert calls[0].url.path == "/list"
            assert calls[0].url.params["path"] == "inferops/camera-a/front-1"
            assert (await client.get("/v1/cameras/front-1/recordings?start=2026-10-20T00:00:00")).status_code == 422
            assert (await client.get("/v1/cameras/missing/recordings")).status_code == 404
            app.state.sources.specs["front-1"] = declaration(kind="video", uri="clip.mp4")
            assert (await client.get("/v1/cameras/front-1/recordings")).status_code == 409
    finally:
        await selected.close()
    monkeypatch.delenv("INFEROPS_MEDIA_ROUTER", raising=False)
    monkeypatch.setenv("INFEROPS_MEDIAMTX_RECORD", "false")
    default = configured_relay()
    try:
        assert isinstance(default, MediaMTX) and default.record is False
    finally:
        await default.close()


async def test_recording_errors_do_not_expose_upstream_details():
    selected = relay(lambda request: httpx.Response(503, text="private-camera-secret"))
    try:
        with pytest.raises(MediaRelayError, match="playback is unavailable") as error:
            await selected.recordings("front-1")
        assert "private-camera-secret" not in str(error.value)
    finally:
        await selected.close()
