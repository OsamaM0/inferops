"""Real-time annotated stream uses cached rendered frames, not HLS recording playback."""
import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from inferops.adapters.inbound.http.cameras import annotated_mjpeg_frames, router


async def test_annotated_stream_yields_only_new_encoded_frames_and_exits_on_stop():
    camera_id = "bus-101-cam1"
    state = {"session_id": "s1", "preview_version": 1}
    selected = SimpleNamespace(
        states={camera_id: state},
        specs={camera_id: SimpleNamespace(preview_enabled=True)},
        previews={camera_id: b"JPEG_1"},
        tasks={camera_id: object()},
    )
    frames = annotated_mjpeg_frames(selected, camera_id)
    try:
        first = await asyncio.wait_for(anext(frames), 1)
        assert first.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n")
        assert b"Content-Length: 6" in first and first.endswith(b"JPEG_1\r\n")
        state["preview_version"] = 2
        selected.previews[camera_id] = b"JPEG_2"
        second = await asyncio.wait_for(anext(frames), 1)
        assert second.endswith(b"JPEG_2\r\n")
        del selected.tasks[camera_id]
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(anext(frames), 1)
    finally:
        await frames.aclose()


async def test_annotated_stream_requires_source_preview_and_active_task():
    camera_id = "bus-101-cam1"
    selected = SimpleNamespace(
        specs={}, tasks={}, states={}, previews={},
    )
    app = FastAPI()
    app.include_router(router, prefix="/v1")
    app.state.sources = selected
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(f"/v1/cameras/{camera_id}/annotated.mjpg")
        assert response.status_code == 404
        selected.specs[camera_id] = SimpleNamespace(preview_enabled=False)
        response = await client.get(f"/v1/cameras/{camera_id}/annotated.mjpg")
        assert response.status_code == 409
        selected.specs[camera_id].preview_enabled = True
        response = await client.get(f"/v1/cameras/{camera_id}/annotated.mjpg")
        assert response.status_code == 409
