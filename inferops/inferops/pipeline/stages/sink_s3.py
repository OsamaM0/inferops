"""Upload evidence (JSON + optional JPEG images + MP4 video clips) to S3/MinIO.

evidence_mode controls what gets stored for *alert* frames:
  "json"  - JSON metadata only (always written for every frame)
  "image" - JSON + per-frame JPEG for frames that trigger an alert
  "video" - JSON + MP4 clip around each alert window  (default)
  "both"  - JSON + per-frame JPEG + MP4 clip around each alert window

Sustained-alert evidence (controlled by trigger_secs):
  Only save evidence once the same alert condition has been continuously active
  for at least *trigger_secs* seconds.  A ring-buffer of the last *pre_secs*
  seconds is always kept so the video can start before the trigger moment.
  A JPEG snapshot is automatically saved at the exact trigger frame.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from collections import defaultdict, deque
from contextlib import suppress
from typing import Any

import cv2
import numpy as np
import orjson

from inferops.adapters.s3_client import put_bytes
from inferops.core.plugins import instantiate
from inferops.core.types import now_ms
from inferops.obs import EVIDENCE_UPLOAD_LATENCY, EVIDENCE_UPLOADS
from inferops.pipeline.stage import Stage


def _current_alert_flags(item) -> set[str]:
    """
    Derive the set of alert-type strings that are TRUE *right now* directly
    from per-frame payload signals and detection results — not from throttled
    alert/fusion events.  This lets the sustained-trigger tracker work
    accurately even when the alert engine is in cooldown.
    """
    flags: set[str] = set()
    payload = item.payload

    # Phone in frame with enough confidence
    for r in (item.results or []):
        if any(d.get("score", 0.0) >= 0.45
               for d in (r.raw or {}).get("by_class", {}).get("phone", [])):
            flags.add("phone_use")
            break

    # Cigarette near mouth (face-mesh intersection score)
    if (payload.get("smoking_face_score") or 0.0) >= 0.15:
        flags.add("smoking")
    else:
        for r in (item.results or []):
            if any(d.get("score", 0.0) >= 0.60
                   for d in (r.raw or {}).get("by_class", {}).get("cigarette", [])):
                flags.add("smoking")
                break

    # Food / drink near mouth
    if (payload.get("eating_face_score") or 0.0) >= 0.40:
        flags.add("eating")

    # Drowsiness via EAR
    if (payload.get("drowsy_ear") or 0.0) >= 0.5:
        flags.add("drowsiness")

    # Head-pose distraction
    if (payload.get("distracted_score") or 0.0) >= 0.5:
        flags.add("distraction")

    # Seatbelt detected with low confidence (≈ not wearing)
    for r in (item.results or []):
        sb = (r.raw or {}).get("by_class", {}).get("seatbelt", [])
        if sb and max(d.get("score", 0.0) for d in sb) < 0.4:
            flags.add("no_seatbelt")
            break

    # Hands off wheel
    if (payload.get("hand_on_wheel", 1.0) or 1.0) < 0.5:
        flags.add("hands_off_wheel")

    return flags

_DEFAULT_COLOR: tuple[int, int, int] = (200, 80, 200)   # purple


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


class SinkS3(Stage):
    """Upload frame evidence to S3/MinIO.

    Params
    ------
    prefix         : S3 key prefix for JSON records (default "evidence/")
    bucket         : override default S3 bucket for JSON records
    evidence_mode  : "json" | "image" | "video" | "both"
    clips_bucket   : bucket for image / video clips (default "inferops-clips")
    pre_frames     : frames before the trigger to include in a video clip
    post_frames    : frames after the trigger to include in a video clip
    fps            : frame-rate for encoded video clips (default 25.0)
    trigger_secs   : seconds of *continuous* alert before saving evidence
                     (default 5.0 — set to 0.0 to save on every alert frame)
    """

    def __init__(
        self,
        prefix: str = "evidence/",
        bucket=None,
        evidence_mode: str = "video",
        clips_bucket: str = "inferops-clips",
        pre_frames: int = 125,
        post_frames: int = 125,
        fps: float = 25.0,
        annotator: str | None = None,
        annotator_params: dict | None = None,
        trigger_secs: float = 5.0,
    ) -> None:
        super().__init__(prefix=prefix, bucket=bucket, evidence_mode=evidence_mode,
                         clips_bucket=clips_bucket, pre_frames=pre_frames,
                         post_frames=post_frames, fps=fps,
                         annotator=annotator, annotator_params=annotator_params,
                         trigger_secs=trigger_secs)
        self.prefix = prefix
        self.bucket = bucket
        self.evidence_mode = evidence_mode.lower()
        self.clips_bucket = clips_bucket
        self.pre_frames = pre_frames
        self.post_frames = post_frames
        self.fps = fps
        self.trigger_ms = max(0, int(trigger_secs * 1000))
        self._annotator = instantiate(annotator, annotator_params or {}) if annotator else None

        # per-stream ring buffers — capped at (pre + post + overhead) frames
        # Each entry: (ts_ms, seq, image_np, alert_type_tag_or_None, item)
        self._max_buf = self.pre_frames + self.post_frames + 60
        self._buffers: dict[str, list] = defaultdict(list)

        # Sustained-alert tracking (per stream × alert-type)
        # Maps (stream_id, alert_type) → ts_ms when this alert first became active
        self._alert_onset: dict[tuple[str, str], int] = {}
        # Tracks which (stream_id, alert_type) episodes already had a snapshot taken
        self._snapshot_taken: set[tuple[str, str]] = set()

    async def process(self, item):
        # In live-streaming mode we explicitly skip *all* S3 work — the
        # WebSocket client is waiting for bounding boxes, not durable
        # evidence; per-frame PUTs would add ~50–200 ms each.
        if item.payload.get("live_mode"):
            return item

        uris = []

        alerts = item.payload.get("alerts", [])
        image_np = item.payload.get("image_np")
        stream_id = item.stream_id or ""
        ts_ms = item.ts_ms or now_ms()

        # ── sustained-alert tracking ───────────────────────────────────────
        # Derive current flag set directly from raw signals (not from throttled
        # alert events) so the onset timer runs every frame.
        current_flags = _current_alert_flags(item)

        # Clear onset entries for flags that are no longer active
        stale = {
            k for k in self._alert_onset
            if k[0] == stream_id and k[1] not in current_flags
        }
        for k in stale:
            self._alert_onset.pop(k, None)
            self._snapshot_taken.discard(k)

        # Record onset for newly active flags
        for flag in current_flags:
            self._alert_onset.setdefault((stream_id, flag), ts_ms)

        # Flags that have been active continuously for >= trigger_ms
        sustained_flags = {
            flag for flag in current_flags
            if (self.trigger_ms == 0
                or ts_ms - self._alert_onset.get((stream_id, flag), ts_ms) >= self.trigger_ms)
        }

        # On the FIRST triggered frame per episode: save a JPEG snapshot
        if sustained_flags and image_np is not None and self.evidence_mode in ("image", "both"):
            new_triggers = sustained_flags - {
                flag for (sid, flag) in self._snapshot_taken if sid == stream_id
            }
            if new_triggers:
                for flag in new_triggers:
                    self._snapshot_taken.add((stream_id, flag))
                # Fire-and-forget snapshot upload
                asyncio.create_task(
                    self._upload_snapshot(image_np, item, sorted(new_triggers))
                )

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

        # Per-frame JPEG for alert frames is handled via _upload_snapshot
        # at the trigger moment (above). Legacy per-alert-event images are
        # skipped here to avoid double-saving.

        # Buffer frame for video clip assembly (mode: video or both)
        if self.evidence_mode in ("video", "both") and image_np is not None:
            # Only mark as an alert frame if the alert has been SUSTAINED
            buf_alert_tag = (
                "+".join(sorted(sustained_flags)) if sustained_flags else None
            )
            buf = self._buffers[stream_id]
            buf.append((ts_ms, item.seq, image_np.copy(), buf_alert_tag, item))
            # Trim ring buffer: drop oldest non-alert frames when over capacity
            while len(buf) > self._max_buf and buf[0][3] is None:
                buf.pop(0)

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
                clip_frames = [_draw_annotations(f[2], f[4], self._annotator) for f in frames[start : end + 1]]
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
        bucket = self.clips_bucket
        with EVIDENCE_UPLOAD_LATENCY.labels(bucket=bucket, kind="video").time():
            mp4_bytes = await asyncio.to_thread(_encode_mp4, frames, w, h, self.fps)
            try:
                uri = await put_bytes(key, mp4_bytes, bucket=bucket,
                                      content_type="video/mp4")
            except Exception:
                EVIDENCE_UPLOADS.labels(bucket=bucket, kind="video", result="error").inc()
                raise
        EVIDENCE_UPLOADS.labels(bucket=bucket, kind="video", result="ok").inc()
        return uri

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

    async def _upload_snapshot(self, image_np: np.ndarray, item, alert_types: list[str]) -> None:
        """Upload a JPEG snapshot taken at the trigger moment."""
        annotated = _draw_annotations(image_np, item, self._annotator)
        ok, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return
        alert_tag = "+".join(sorted(alert_types))
        key = f"clips/{item.stream_id}/{item.ts_ms}_{alert_tag}_trigger.jpg"
        bucket = self.clips_bucket
        with EVIDENCE_UPLOAD_LATENCY.labels(bucket=bucket, kind="image").time():
            with suppress(Exception):
                await put_bytes(key, jpg.tobytes(), bucket=bucket,
                                content_type="image/jpeg")
        EVIDENCE_UPLOADS.labels(bucket=bucket, kind="image", result="ok").inc()


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
