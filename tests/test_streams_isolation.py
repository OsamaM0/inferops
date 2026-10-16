"""Camera sessions are owned independently of repeated descriptive names."""

import asyncio

from httpx import ASGITransport, AsyncClient

from inferops.adapters.inbound import sources
from inferops.adapters.inbound.http.app import create_application
from inferops.domain.processing import MediaFrame
from tests.test_hexagonal_http_adapter import Engine


async def test_repeated_stream_names_have_isolated_owned_sessions(monkeypatch):
    async def frames(*args, **kwargs):
        await asyncio.Event().wait()
        yield

    monkeypatch.setattr(sources, "camera_frames", frames)
    engine = Engine()
    app = create_application(engine)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app), base_url="http://test") as client,
    ):
        first = (
            await client.post(
                "/v1/streams", params={"pipeline": "driver", "rtsp_url": "rtsp://test", "stream_id": "same"}
            )
        ).json()
        second = (
            await client.post(
                "/v1/streams", params={"pipeline": "driver", "rtsp_url": "rtsp://test", "stream_id": "same"}
            )
        ).json()
        assert first["session_id"] != second["session_id"]
        assert (await client.delete("/v1/streams/" + first["session_id"])).status_code == 403
        stopped = await client.delete(
            "/v1/streams/" + first["session_id"], headers={"X-Session-Capability": first["capability"]}
        )
        assert stopped.status_code == 200
        assert not engine.application.sessions[second["session_id"]].closed
        assert len((await client.get("/v1/streams")).json()) == 1


async def test_finished_streams_are_reaped(monkeypatch):
    async def frames(*args, **kwargs):
        yield MediaFrame(None, 0, 0)

    monkeypatch.setattr(sources, "camera_frames", frames)
    app = create_application(Engine())
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app), base_url="http://test") as client,
    ):
        assert (
            await client.post("/v1/streams", params={"pipeline": "driver", "rtsp_url": "rtsp://test"})
        ).status_code == 200
        await asyncio.sleep(0.05)
        assert (await client.get("/v1/streams")).json() == []
        assert not app.state.engine.application.sessions


async def test_stop_waits_for_owned_session_shutdown(monkeypatch):
    async def frames(*args, **kwargs):
        await asyncio.Event().wait()
        yield

    monkeypatch.setattr(sources, "camera_frames", frames)
    app = create_application(Engine())
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app), base_url="http://test") as client,
    ):
        row = (
            await client.post("/v1/streams", params={"pipeline": "driver", "rtsp_url": "rtsp://test"})
        ).json()
        session = app.state.engine.application.sessions[row["session_id"]]
        response = await client.delete(
            "/v1/streams/" + row["session_id"], headers={"X-Session-Capability": row["capability"]}
        )
        assert response.status_code == 200 and session.closed


async def test_rest_pipeline_requires_explicit_camera_input(tmp_path):
    from inferops import Engine as RealEngine
    from tests.test_redesign_engine import project

    app = create_application(RealEngine.from_project(project(tmp_path)))
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app), base_url="http://test") as client,
    ):
        response = await client.post("/v1/streams", params={"pipeline": "test"})
        assert response.status_code == 400
        assert not app.state.engine.application.sessions
