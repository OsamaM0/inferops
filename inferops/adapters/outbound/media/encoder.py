"""Image and clip encoding at the infrastructure boundary."""
from __future__ import annotations

import os
import tempfile
from contextlib import suppress
from fractions import Fraction
from typing import Any

from inferops.domain.evidence import EncodedFrame


class OpenCVMediaEncoder:
    def snapshot(self, image: Any, item: Any = None, annotate: Any = None) -> bytes:
        import cv2

        if item is not None and hasattr(item, "results"):
            # Keep the established generic and project-specific overlay algorithm.
            from inferops.adapters.outbound.media.annotations import _draw_annotations

            image = _draw_annotations(image, item, annotate)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValueError("Snapshot encoding failed")
        return encoded.tobytes()

    def annotate(self, jpeg: bytes, item: Any, annotate: Any = None) -> bytes:
        import cv2
        import numpy as np

        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Undecodable annotation frame")
        return self.snapshot(image, item, annotate)

    def clip(self, frames: list[EncodedFrame], fps: float) -> bytes:
        import cv2
        import numpy as np

        if not frames:
            raise ValueError("No decodable frames for evidence clip")

        def decode(frame):
            image = cv2.imdecode(np.frombuffer(frame.jpeg, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Undecodable evidence frame")
            return image

        first = decode(frames[0])
        height, width = first.shape[:2]

        def images():
            for frame in frames:
                image = decode(frame)
                if image.shape[:2] != (height, width):
                    raise ValueError("Evidence frames changed dimensions within one timeline")
                yield image

        try:
            import av
        except ImportError:
            # Preserve the established minimal-install MP4 fallback. Decode
            # incrementally instead of collecting uncompressed video in RAM.
            from inferops.adapters.outbound.media.annotations import _encode_mp4

            return _encode_mp4(images(), width, height, fps)

        # Retain the working H.264/faststart settings while carrying source
        # presentation intervals through variable-frame-rate captures.
        descriptor, path = tempfile.mkstemp(suffix=".mp4")
        os.close(descriptor)
        try:
            with av.open(path, "w", options={"movflags": "+faststart"}) as container:
                stream = container.add_stream("libx264", rate=Fraction(str(fps)).limit_denominator(10000))
                stream.width, stream.height = width, height
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "23", "preset": "fast"}
                stream.time_base = stream.codec_context.time_base = Fraction(1, 90000)
                previous = -1
                for frame, image in zip(frames, images(), strict=True):
                    video = av.VideoFrame.from_ndarray(image, format="bgr24")
                    video.time_base = Fraction(1, 90000)
                    video.pts = max(previous + 1, (frame.media_time_ns - frames[0].media_time_ns) * 90000 // 10**9)
                    previous = video.pts
                    for packet in stream.encode(video):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            with open(path, "rb") as handle:
                return handle.read()
        finally:
            with suppress(OSError):
                os.unlink(path)
