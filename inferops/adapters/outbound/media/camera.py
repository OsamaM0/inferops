"""Shared per-context PyAV acquisition, one bounded latest slot per camera."""
from __future__ import annotations

import asyncio
import random
import re
import threading
import time

from inferops.domain.processing import MediaFrame


def bgr_array(frame, *, threads=1):
    import av
    # PyAV 17 introduced an automatic swscale pool per conversion context.
    # Sixty streams must not each create a CPU-sized color conversion pool.
    options = {"threads": threads} if int(av.__version__.split(".")[0]) >= 17 else {}
    return frame.to_ndarray(format="bgr24", **options)


def _resize(image, spec):
    import cv2

    h, w = image.shape[:2]
    scale = min(1, spec.max_width / w, spec.max_height / h)
    return cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale)))) if scale < 1 else image


def _input_options(spec):
    """Per-camera demuxer options: retain the existing RTSP behavior."""
    options = {"analyzeduration": "1000000", "probesize": "32768"}
    if spec.kind == "rtsp":
        options["rtsp_transport"] = spec.rtsp_transport
    elif spec.kind == "hls":
        # Segmented HTTP sources need a longer probe than low-latency RTSP.
        # Retry temporary HTTP/network failures, but not expired 401/403 tokens.
        options.update(
            analyzeduration="5000000", probesize="1048576",
            reconnect="1", reconnect_streamed="1",
            reconnect_on_network_error="1",
            reconnect_on_http_error="429,500,502,503,504",
            reconnect_delay_max="2",
        )
    return options


def _capture_status(error):
    """Never return a signed URL (including its query string) to clients."""
    message = str(error)
    status = re.search(r"(?:HTTP|Server returned)\\s*(?:error\\s*)?(401|403|404|410|429|5\\d\\d)\\b", message, re.I)
    if status:
        return f"Upstream HTTP {status.group(1)}; check URL validity or camera access"
    return "Cannot decode HLS stream; check playlist, network, and codec"


async def capture_frames(spec, stats):
    import av

    stop = threading.Event()
    lock = threading.Lock()
    slot = [None]

    def read():
        initial_delay = getattr(spec, "reconnect_initial_s", 0.5)
        sequence, generation, delay = 0, 0, initial_delay
        while not stop.is_set():
            try:
                # Options are per context, never process-global environment.
                options = _input_options(spec)
                with av.open(
                    spec.uri,
                    options=options,
                    timeout=(spec.timeout_s, spec.timeout_s),
                ) as container:
                    if not container.streams.video:
                        raise ValueError("stream has no video track")
                    container.streams.video[0].thread_count = getattr(spec, "decoder_threads", 1)
                    generation += 1
                    origin = time.monotonic_ns()
                    stats.pop("source_error", None)
                    # A reconnect starts a new media clock/timeline, causing
                    # the session to reset temporal state and finish old capture.
                    for frame in container.decode(video=0):
                        if stop.is_set():
                            return
                        captured_at = time.monotonic_ns()
                        image = _resize(bgr_array(frame, threads=getattr(spec, "decoder_threads", 1)), spec)
                        received = captured_at - origin
                        stats.update(
                            decoded=stats["decoded"] + 1,
                            source_width=frame.width,
                            source_height=frame.height,
                            connection="connected",
                            reconnects=generation - 1,
                            last_frame_at=time.time(),
                        )
                        envelope = MediaFrame(
                            image,
                            received,
                            sequence,
                            "local_receive_monotonic",
                            payload={"camera_id": spec.camera_id, "connection_generation": generation,
                                     "_captured_monotonic_ns": captured_at,
                                     "source_pts_ns": int(frame.pts * frame.time_base * 1e9)
                                     if frame.pts is not None and frame.time_base else None},
                        )
                        sequence += 1
                        delay = initial_delay
                        with lock:
                            if slot[0] is not None:
                                stats["capture_overwrites"] = stats.get("capture_overwrites", 0) + 1
                            slot[0] = envelope
                if not stop.is_set():
                    raise OSError("source ended")
            except Exception as error:
                if spec.kind == "hls":
                    stats["source_error"] = _capture_status(error)
                stats["decode_errors"] = stats.get("decode_errors", 0) + 1
                if stop.is_set():
                    return
                stats["connection"] = "reconnecting"
                # Do not retain/display a stale frame after a disconnect.
                with lock:
                    slot[0] = None
                stop.wait(delay * random.uniform(0.8, 1.2))
                delay = min(8, delay * 2)

    thread = threading.Thread(target=read, name=f"{spec.kind}-{spec.camera_id}", daemon=True)
    thread.start()
    last = -1
    try:
        while True:
            await asyncio.sleep(1 / max(spec.inference_fps, spec.evidence_fps or 0))
            with lock:
                frame, slot[0] = slot[0], None
            if frame is not None and frame.sequence != last:
                stats["dropped"] = stats.get("dropped", 0) + max(0, frame.sequence - last - 1)
                last = frame.sequence
                yield frame
    finally:
        stop.set()
        await asyncio.to_thread(thread.join, spec.timeout_s * 2 + 1)
        if thread.is_alive():
            raise RuntimeError("Camera reader did not stop within its I/O deadline")
