"""HLS source compatibility tests."""
import pytest
from pydantic import ValidationError
from inferops.domain.sources import SourceSpec
from tests.test_media_relay import ControlAPI, declaration, relay

SIGNED = "https://camera.example/live/index.m3u8?token=example-signed-token"

def test_hls_validation():
    spec = declaration(kind="hls", uri=SIGNED)
    assert spec.kind == "hls"
    assert declaration().kind == "rtsp"
    for uri in ("rtsp://camera/live", "http://camera/video.mp4", "file:///tmp/a.m3u8"):
        with pytest.raises(ValidationError):
            declaration(kind="hls", uri=uri)

async def test_hls_relay_and_rtsp_transition():
    api = ControlAPI()
    media = relay(api)
    try:
        name = "inferops/combined/front-1"
        routed = await media.prepare(declaration(kind="hls", uri=SIGNED))
        assert routed.uri == f"rtsp://media:8554/{name}"
        assert routed.kind == "rtsp" and routed.rtsp_transport == "tcp"
        assert api.paths[name]["source"] == SIGNED
        assert "rtspTransport" not in api.paths[name]
        await media.prepare(declaration(kind="hls", uri=SIGNED))
        assert api.mutations == [("add", name)]
        await media.prepare(declaration())
        assert api.paths[name]["rtspTransport"] == "tcp"
        assert api.paths[name]["source"].startswith("rtsp://")
        await media.remove("front-1")
        assert name not in api.paths
    finally:
        await media.close()
