"""Provisional memory sizing from measured encoded frame sizes."""
import math


def evidence_budget(*, cameras, jpeg_p95_bytes, width, height, sample_fps=10,
                    pre_seconds=5, post_seconds=5, lag_seconds=5, overlapping=3, overhead=1.2):
    if min(cameras, jpeg_p95_bytes, width, height, sample_fps, overlapping) <= 0:
        raise ValueError("camera count, frame sizes, rates and overlaps must be positive")
    if min(pre_seconds, post_seconds, lag_seconds) < 0 or overhead < 1:
        raise ValueError("invalid windows or overhead factor")
    per_history = math.ceil(sample_fps * (pre_seconds + lag_seconds) * jpeg_p95_bytes * overhead)
    burst = math.ceil(cameras * overlapping * sample_fps * (pre_seconds + post_seconds)
                      * jpeg_p95_bytes * overhead)
    return {"cameras": cameras, "jpeg_p95_bytes": jpeg_p95_bytes,
            "per_session_history_bytes": per_history, "global_history_bytes": cameras * per_history,
            "decoded_slots_bytes": cameras * width * height * 3 * 3,
            "burst_material_bytes": burst, "max_active_captures": cameras * overlapping,
            "excluded": ["decoder buffers", "model/VRAM", "encoders", "uploads", "runtime", "disk reserve"],
            "qualification": "sizing estimate; measure RSS, spool use and terminal outcomes under bursts"}
