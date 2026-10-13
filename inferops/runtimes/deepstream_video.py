"""Optional DeepStream file decoder; inference remains in the Triton backend.

Frames cross into host memory at appsink so the existing tracking, fusion,
alert and evidence stages receive the same BGR contract as CPU/CUDA mode.
Requires the DeepStream SDK and system Python's GObject bindings.
"""
from __future__ import annotations

import time
from itertools import pairwise
from pathlib import Path

import numpy as np

from inferops.core.errors import BackendError
from inferops.core.types import StageItem


def decode_video(item: StageItem, path: str, every_n: int = 1) -> list[StageItem]:
    try:
        import gi
        gi.require_version("Gst", "1.0")
        gi.require_version("GstVideo", "1.0")
        from gi.repository import Gst, GstVideo
    except ImportError as exc:
        raise BackendError("DeepStream mode requires the DeepStream container and python3-gi") from exc

    Gst.init(None)
    pipeline = Gst.Pipeline.new("inferops-video")
    elements = []
    for factory, name in [
        ("nvurisrcbin", "source"), ("nvvideoconvert", "gpu-convert"),
        ("capsfilter", "host-caps"), ("videoconvert", "bgr-convert"),
        ("capsfilter", "bgr-caps"), ("appsink", "frames"),
    ]:
        element = Gst.ElementFactory.make(factory, name)
        if element is None:
            raise BackendError(f"DeepStream plugin unavailable: {factory}")
        pipeline.add(element)
        elements.append(element)
    source, convert, host_caps, _bgr_convert, bgr_caps, sink = elements
    source.set_property("uri", Path(path).resolve().as_uri())
    host_caps.set_property("caps", Gst.Caps.from_string("video/x-raw,format=BGRx"))
    bgr_caps.set_property("caps", Gst.Caps.from_string("video/x-raw,format=BGR"))
    sink.set_property("sync", False)
    sink.set_property("async", False)
    sink.set_property("max-buffers", 2)
    sink.set_property("drop", False)

    def link_video(_source, pad):
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps.get_structure(0).get_name().startswith("video/"):
            pad.link(convert.get_static_pad("sink"))

    source.connect("pad-added", link_video)
    for left, right in pairwise(elements[1:]):
        if not left.link(right):
            raise BackendError(f"Cannot link DeepStream {left.name} to {right.name}")
    bus = pipeline.get_bus()
    frames = []
    seq = 0
    last_sample_at = time.monotonic()
    try:
        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise BackendError("DeepStream decoder failed to start")
        while True:
            sample = sink.emit("try-pull-sample", Gst.SECOND)
            if sample is None:
                message = bus.pop_filtered(Gst.MessageType.ERROR)
                if message:
                    error, detail = message.parse_error()
                    raise BackendError(f"DeepStream decode failed: {error}: {detail}")
                if sink.is_eos():
                    break
                if time.monotonic() - last_sample_at > 60:
                    raise BackendError("DeepStream decoder timed out waiting for a frame")
                continue
            last_sample_at = time.monotonic()
            seq += 1
            if seq % every_n:
                continue
            info = GstVideo.VideoInfo.new_from_caps(sample.get_caps())
            buffer = sample.get_buffer()
            ok, mapped = buffer.map(Gst.MapFlags.READ)
            if not ok:
                raise BackendError("Cannot map decoded DeepStream frame")
            try:
                image = np.ndarray(
                    (info.height, info.width, 3), dtype=np.uint8,
                    buffer=mapped.data, offset=info.offset[0], strides=(info.stride[0], 3, 1),
                ).copy()
            finally:
                buffer.unmap(mapped)
            fps = info.fps_n / info.fps_d if info.fps_d else 25.0
            pts_ms = buffer.pts / Gst.MSECOND if buffer.pts != Gst.CLOCK_TIME_NONE else seq * 1000 / fps
            frames.append(StageItem(
                stream_id=item.stream_id, ts_ms=item.ts_ms + round(pts_ms), seq=seq,
                payload={"image_np": image, "width": info.width, "height": info.height,
                         "video_fps": fps / every_n, "video_decoder": "deepstream"},
            ))
    finally:
        pipeline.set_state(Gst.State.NULL)
    if not frames:
        raise BackendError("DeepStream decoded no frames")
    return frames
