"""Established detection overlays and portable clip encoding helpers."""
from __future__ import annotations

import os
import tempfile
from contextlib import suppress
from fractions import Fraction

import cv2
import numpy as np

_DEFAULT_COLOR = (200, 80, 200)

def _draw_annotations(frame: np.ndarray, item, annotator=None) -> np.ndarray:
    """Return a copy of *frame* with detection boxes + labels drawn.

    Generic detection boxes are always drawn.  An optional *annotator*
    (a ``FrameAnnotator`` instance supplied by the domain) is called
    afterwards to add any domain-specific overlays.
    """
    out = frame.copy()
    h, w = out.shape[:2]
    for result in (item.results or []):
        for det in (result.detections or []):
            # Prefer pre-computed pixel coords, else de-normalise
            px = (det.attrs or {}).get("xyxy_pixels")
            if px and len(px) == 4:
                x1, y1, x2, y2 = (int(v) for v in px)
            else:
                x1 = int(det.box[0] * w)
                y1 = int(det.box[1] * h)
                x2 = int(det.box[2] * w)
                y2 = int(det.box[3] * h)
            cv2.rectangle(out, (x1, y1), (x2, y2), _DEFAULT_COLOR, 2)
            label = f"{det.label} {det.score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            by = max(y1 - 4, th + 6)
            cv2.rectangle(out, (x1, by - th - 4), (x1 + tw + 4, by), _DEFAULT_COLOR, cv2.FILLED)
            cv2.putText(out, label, (x1 + 2, by - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    if annotator is not None:
        out = annotator(out, item)
    return out

def _encode_mp4(frames, w, h, fps):
    """Encode BGR numpy frames to MP4 bytes (runs in thread pool)."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmppath = f.name
    try:
        try:
            import av
        except ImportError:
            # Minimal installs can still export clips for download.
            vw = cv2.VideoWriter(tmppath, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if not vw.isOpened():
                raise RuntimeError("Cannot create MP4 evidence encoder") from None
            for frame in frames:
                vw.write(frame)
            vw.release()
        else:
            # H.264 + faststart lets the evidence browser play clips in Chrome.
            with av.open(tmppath, "w", options={"movflags": "+faststart"}) as container:
                stream = container.add_stream("libx264", rate=Fraction(str(fps)).limit_denominator(10000))
                stream.width, stream.height = w, h
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "23", "preset": "fast"}
                for image in frames:
                    for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="bgr24")):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
        with open(tmppath, "rb") as f:
            return f.read()
    finally:
        with suppress(OSError):
            os.unlink(tmppath)
