"""Upload evidence (JSON + optional JPEG images + MP4 video clips) to S3/MinIO.

evidence_mode controls what gets stored for *alert* frames:
  "json"  - JSON metadata only (always written for every frame)
  "image" - JSON + per-frame JPEG for frames that trigger an alert
  "video" - JSON + MP4 clip around each alert window  (default)
  "both"  - JSON + per-frame JPEG + MP4 clip around each alert window
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from collections import defaultdict
from contextlib import suppress

import cv2
import numpy as np
import orjson

from inferops.adapters.s3_client import put_bytes
from inferops.pipeline.stage import Stage

# ── Annotation colours (BGR) ───────────────────────────────────────────────
_CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "seatbelt":  (50, 200,  50),   # green
    "phone":     (40,  40, 220),   # red
    "cigarette": ( 0, 130, 255),   # orange
    "foodItem":  ( 0, 210, 210),   # yellow
    "person":    (220, 150,  40),  # blue
}
_DEFAULT_COLOR: tuple[int, int, int] = (200, 80, 200)   # purple


def _draw_annotations(frame: np.ndarray, item) -> np.ndarray:
    """Return a copy of *frame* with detection boxes + labels drawn."""
    out = frame.copy()
    h, w = out.shape[:2]
    for result in (item.results or []):
        for det in (result.detections or []):
            color = _CLASS_COLORS.get(det.label, _DEFAULT_COLOR)
            # Prefer pre-computed pixel coords, else de-normalise
            px = (det.attrs or {}).get("xyxy_pixels")
            if px and len(px) == 4:
                x1, y1, x2, y2 = (int(v) for v in px)
            else:
                x1 = int(det.box[0] * w)
                y1 = int(det.box[1] * h)
                x2 = int(det.box[2] * w)
                y2 = int(det.box[3] * h)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{det.label} {det.score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            by = max(y1 - 4, th + 6)
            cv2.rectangle(out, (x1, by - th - 4), (x1 + tw + 4, by), color, cv2.FILLED)
            cv2.putText(out, label, (x1 + 2, by - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return out


class SinkS3(Stage):
    """Upload frame evidence to S3/MinIO.

    Params
    ------
    prefix         : S3 key prefix for JSON records (default "evidence/")
    bucket         : override default S3 bucket for JSON records
    evidence_mode  : "json" | "image" | "video" | "both"
    clips_bucket   : bucket for image / video clips (default "inferops-clips")
    pre_frames     : frames before the alert to include in a video clip
    post_frames    : frames after the alert to include in a video clip
    fps            : frame-rate for encoded video clips (default 25.0)
    """

    def __init__(
        self,
        prefix: str = "evidence/",
        bucket=None,
        evidence_mode: str = "video",
        clips_bucket: str = "inferops-clips",
        pre_frames: int = 30,
        post_frames: int = 30,
        fps: float = 25.0,
        # legacy compat aliases
        upload_image: bool = False,
        clips_on_alert: bool = True,
    ) -> None:
        # Legacy shim: map old params to new evidence_mode
        if evidence_mode == "video" and upload_image:
            evidence_mode = "both"
        elif evidence_mode == "video" and not clips_on_alert:
            evidence_mode = "json"

        super().__init__(prefix=prefix, bucket=bucket, evidence_mode=evidence_mode,
                         clips_bucket=clips_bucket, pre_frames=pre_frames,
                         post_frames=post_frames, fps=fps)
        self.prefix = prefix
        self.bucket = bucket
        self.evidence_mode = evidence_mode.lower()
        self.clips_bucket = clips_bucket
        self.pre_frames = pre_frames
        self.post_frames = post_frames
        self.fps = fps

        # per-stream buffers: list of (ts_ms, seq, image_np, alert_types_or_None, item_or_None)
        self._buffers: dict = defaultdict(list)

    async def process(self, item):
        uris = []

        alerts = item.payload.get("alerts", [])
        image_np = item.payload.get("image_np")

        # JSON metadata record (every frame, always)
        json_key = f"{self.prefix}{item.stream_id}/{item.id}.json"
        uri = await put_bytes(
            json_key,
            orjson.dumps({
                "id": item.id,
                "stream_id": item.stream_id,
                "ts_ms": item.ts_ms,
                "seq": item.seq,
                "results": [r.model_dump(mode="json") for r in item.results],
                "alerts": alerts,
                "fusion_events": item.payload.get("fusion_events", []),
            }),
            bucket=self.bucket,
            content_type="application/json",
        )
        uris.append(uri)

        # Per-frame JPEG for alert frames (mode: image or both)
        if self.evidence_mode in ("image", "both") and alerts and image_np is not None:
            annotated = _draw_annotations(image_np, item)
            ok, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                alert_tag = "+".join(sorted({a.get("type", "ALERT") for a in alerts}))
                img_uri = await put_bytes(
                    f"clips/{item.stream_id}/{item.ts_ms}_{alert_tag}_{item.id[:8]}.jpg",
                    jpg.tobytes(), bucket=self.clips_bucket, content_type="image/jpeg")
                uris.append(img_uri)

        # Buffer frame for video clip assembly (mode: video or both)
        if self.evidence_mode in ("video", "both") and image_np is not None:
            alert_types = "+".join(
                sorted({a.get("type", "ALERT") for a in alerts})
            ) if alerts else None
            self._buffers[item.stream_id].append(
                (item.ts_ms, item.seq, image_np.copy(), alert_types, item)
            )

        item.payload.setdefault("evidence_uris", []).extend(uris)
        return item

    async def finalize(self) -> None:
        """Encode + upload MP4 clips for each alert window, then clear buffers."""
        if self.evidence_mode not in ("video", "both") or not self._buffers:
            return

        tasks = []
        for stream_id, frames in self._buffers.items():
            # Sort by ts_ms in case concurrent processing reordered frames
            frames.sort(key=lambda f: (f[0], f[1]))
            alert_indices = [i for i, f in enumerate(frames) if f[3] is not None]
            if not alert_indices:
                continue
            windows = _merge_windows(
                alert_indices,
                [frames[i][3] for i in alert_indices],
                self.pre_frames,
                self.post_frames,
                len(frames),
            )
            for start, end, types_set, alert_idxs in windows:
                clip_frames = [_draw_annotations(f[2], f[4]) for f in frames[start : end + 1]]
                ts_ms = frames[start][0]
                alert_tag = "+".join(sorted(types_set))
                alert_items = [frames[i][4] for i in alert_idxs if frames[i][4] is not None]
                tasks.append(self._upload_clip_and_attach(stream_id, clip_frames, ts_ms, alert_tag, alert_items))

        if tasks:
            await asyncio.gather(*tasks)
        self._buffers.clear()

    async def _upload_clip(self, stream_id, frames, ts_ms, alert_tag):
        h, w = frames[0].shape[:2]
        key = f"clips/{stream_id}/{ts_ms}_{alert_tag}.mp4"
        mp4_bytes = await asyncio.to_thread(_encode_mp4, frames, w, h, self.fps)
        return await put_bytes(key, mp4_bytes, bucket=self.clips_bucket,
                               content_type="video/mp4")

    async def _upload_clip_and_attach(self, stream_id, frames, ts_ms, alert_tag, alert_items):
        uri = await self._upload_clip(stream_id, frames, ts_ms, alert_tag)
        for item in alert_items:
            item.payload.setdefault("evidence_uris", []).append(uri)
            item.payload.setdefault("evidence_ready", []).append({
                "uri": uri,
                "type": "video/mp4",
                "alert_types": alert_tag.split("+") if alert_tag else [],
            })
        return uri


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _merge_windows(indices, alert_types_list, pre, post, total):
    """Merge overlapping [i-pre .. i+post] windows, collecting all alert types."""
    spans = [(max(0, i - pre), min(total - 1, i + post), atype, [i])
             for i, atype in zip(indices, alert_types_list, strict=True)]
    merged = []
    for s, e, atype, alert_idxs in sorted(spans):
        types = {t for t in atype.split("+") if t}
        if not merged or s > merged[-1][1] + 1:
            merged.append((s, e, types, alert_idxs))
        else:
            prev = merged[-1]
            merged[-1] = (prev[0], max(prev[1], e), prev[2] | types, prev[3] + alert_idxs)
    return merged


def _encode_mp4(frames, w, h, fps):
    """Encode BGR numpy frames to MP4 bytes (runs in thread pool)."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmppath = f.name
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(tmppath, fourcc, fps, (w, h))
        for frame in frames:
            vw.write(frame)
        vw.release()
        with open(tmppath, "rb") as f:
            return f.read()
    finally:
        with suppress(OSError):
            os.unlink(tmppath)
