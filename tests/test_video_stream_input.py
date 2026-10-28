"""Server video paths share the live stream API and preserve file media time."""
import asyncio
import time

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from inferops.adapters.inbound import sources
from inferops.adapters.inbound.http.app import create_application
from inferops.adapters.outbound.stages.source_rtsp import SourceRTSP
from tests.test_hexagonal_http_adapter import Engine


@pytest.fixture
def video(tmp_path):
    av = pytest.importorskip("av")
    path = tmp_path / "video with spaces.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
        for sequence in range(3):
            frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), sequence * 50, np.uint8), format="bgr24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


async def test_local_video_is_paced_and_completes_at_eof(video):
    started = time.monotonic()
    frames = [frame async for frame in sources.camera_frames(str(video), fps=10)]
    assert [frame.sequence for frame in frames] == [0, 1, 2]
    assert [frame.media_time_ns for frame in frames] == [0, 100_000_000, 200_000_000]
    assert time.monotonic() - started >= .18
    assert all(frame.timestamp_provenance == "presentation_timestamp" for frame in frames)


async def test_loop_replays_with_monotonic_sequence_and_media_time(video):
    started = time.monotonic()
    frames = sources.camera_frames(str(video), fps=10, loop=True)
    try:
        values = [await anext(frames) for _ in range(5)]
    finally:
        await frames.aclose()
    assert time.monotonic() - started >= .38
    assert [frame.sequence for frame in values] == list(range(5))
    assert [frame.media_time_ns for frame in values] == [i * 100_000_000 for i in range(5)]


async def test_compatibility_stage_accepts_local_video_and_reports_decoding(video):
    stage = SourceRTSP(str(video), fps=10)
    items = [item async for item in stage.run()]
    assert len(items) == 3 and items[0].payload["image_np"].shape == (64, 64, 3)
    assert stage.stats["decoded"] == 3


@pytest.mark.parametrize("parameter", ["video_path", "rtsp_url"])
async def test_stream_api_routes_server_paths_and_finishes(video, parameter):
    app = create_application(Engine())
    async with app.router.lifespan_context(app), AsyncClient(
        transport=ASGITransport(app), base_url="http://test"
    ) as client:
        response = await client.post("/v1/streams", params={"pipeline": "driver", parameter: str(video), "fps": 10})
        assert response.status_code == 200, response.text
        for _ in range(100):
            if not app.state.engine.application.sessions:
                break
            await asyncio.sleep(.01)
        assert not app.state.engine.application.sessions
        assert (await client.get("/v1/streams")).json() == []


async def test_invalid_file_and_conflicting_inputs_do_not_create_sessions(tmp_path):
    app = create_application(Engine())
    async with app.router.lifespan_context(app), AsyncClient(
        transport=ASGITransport(app), base_url="http://test"
    ) as client:
        for params in [
            {"video_path": str(tmp_path / "missing.mp4")},
            {"video_path": "https://example.com/video.mp4"},
            {"video_path": str(tmp_path)},
            {"video_path": "a.mp4", "rtsp_url": "rtsp://test"},
            {"video_path": "a.mp4", "mjpeg_url": "http://test"},
        ]:
            response = await client.post("/v1/streams", params={"pipeline": "driver", **params})
            assert response.status_code == 400, response.text
        assert not app.state.engine.application.sessions
