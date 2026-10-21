"""Persistent source identity, independent of a processing session."""

from __future__ import annotations

from math import isfinite
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrameCadence:
    """Select a fixed media-time cadence without accumulating arrival jitter."""

    def __init__(self, fps):
        if not isfinite(fps) or not 0 < fps <= 120:
            raise ValueError("inference_fps must be in (0, 120]")
        self.period_ns = int(1e9 / fps)
        self.next_ns = self.last_ns = self.generation = None

    def due(self, media_ns, generation=None):
        if generation != self.generation or (self.last_ns is not None and media_ns < self.last_ns):
            self.next_ns = None
        self.generation, self.last_ns = generation, media_ns
        if self.next_ns is None:
            self.next_ns = media_ns + self.period_ns
            return True
        if media_ns < self.next_ns:
            return False
        self.next_ns += ((media_ns - self.next_ns) // self.period_ns + 1) * self.period_ns
        return True

class SourceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
    name: str = Field(min_length=1, max_length=120)
    kind: Literal["rtsp", "video", "image"]
    uri: str = Field(min_length=1, max_length=4096)
    pipeline: str = Field(min_length=1, max_length=120)
    inference_fps: float = Field(default=5, gt=0, le=120)
    evidence_fps: float | None = Field(default=None, gt=0, le=120)
    max_frame_age_ms: float = Field(default=500, gt=0, le=30000)
    max_width: int = Field(default=1920, ge=64, le=7680)
    max_height: int = Field(default=1080, ge=64, le=4320)
    preview_fps: float = Field(default=2, gt=0, le=15)
    preview_enabled: bool = True
    preview_width: int = Field(default=640, ge=160, le=1920)
    jpeg_quality: int = Field(default=75, ge=30, le=95)
    loop: bool = False
    autostart: bool = False
    rtsp_transport: Literal["tcp", "udp"] = "tcp"
    timeout_s: float = Field(default=5, ge=1, le=30)
    decoder_threads: int = Field(default=1, ge=1, le=16)

    @model_validator(mode="after")
    def validate_location(self):
        if self.kind == "rtsp":
            parsed = urlsplit(self.uri)
            if parsed.scheme not in {"rtsp", "rtsps"} or not parsed.hostname:
                raise ValueError("RTSP sources require an rtsp:// or rtsps:// URL")
        elif "://" in self.uri:
            raise ValueError("image/video sources must use uploaded files or server-local paths")
        if self.kind == "image" and self.loop:
            raise ValueError("an image is processed once; loop is only for video")
        return self
