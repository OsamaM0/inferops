"""Frame cache backed by Redis (stores JPEG bytes by key)."""
from __future__ import annotations

import cv2
import numpy as np

from inferops.adapters.redis_client import get_redis
from inferops.core.config import settings

_PREFIX = "frame"


def make_ref(stream_id: str, seq: int) -> str:
    return f"redis://{_PREFIX}/{stream_id}/{seq}"


def parse_ref(ref: str) -> tuple[str, int] | None:
    if not ref.startswith(f"redis://{_PREFIX}/"):
        return None
    rest = ref[len(f"redis://{_PREFIX}/"):]
    sid, _, sq = rest.rpartition("/")
    return sid, int(sq)


async def put_frame(stream_id: str, seq: int, image: np.ndarray, *,
                    quality: int = 80, ttl_s: int | None = None) -> str:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    r = get_redis()
    key = f"{_PREFIX}:{stream_id}:{seq}"
    await r.set(key, bytes(buf), ex=ttl_s or settings.redis.frame_ttl_s)
    return make_ref(stream_id, seq)


async def get_frame(ref: str) -> np.ndarray | None:
    parsed = parse_ref(ref)
    if parsed is None:
        return None
    sid, seq = parsed
    r = get_redis()
    raw = await r.get(f"{_PREFIX}:{sid}:{seq}")
    if not raw:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


async def delete_frame(stream_id: str, seq: int) -> None:
    await get_redis().delete(f"{_PREFIX}:{stream_id}:{seq}")
