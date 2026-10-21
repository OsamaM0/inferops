"""HTTP MJPEG uses the same bounded decoder and shutdown contract as RTSP."""
from inferops.adapters.outbound.stages.source_rtsp import SourceRTSP


class SourceMJPEG(SourceRTSP):
    kind = "mjpeg"

    def __init__(self, url: str, stream_id: str = "mjpeg", fps: float = 5, timeout_s: float = 5):
        super().__init__(url=url, stream_id=stream_id, fps=fps, timeout_s=timeout_s)
