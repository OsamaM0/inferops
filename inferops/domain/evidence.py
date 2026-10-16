"""Evidence policy and value objects, independent of infrastructure."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite


class EvidenceState(str, Enum):
    PENDING = "Pending"
    SAVED = "Saved"
    INCOMPLETE = "Incomplete"
    FAILED = "Failed"


@dataclass(frozen=True)
class EvidenceConfig:
    """Provisional operational settings; production values require measurement."""

    pre_seconds: float = 5.0
    post_seconds: float = 5.0
    sample_fps: float = 10.0
    processing_lag_seconds: float = 5.0
    per_session_bytes: int = 64 * 1024**2
    global_bytes: int = 256 * 1024**2
    max_active_captures: int = 64
    workers: int = 2
    max_spool_bytes: int = 10 * 1024**3
    disk_reserve_bytes: int = 1024**3
    retry_delays: tuple[float, ...] = (1.0, 5.0, 30.0)
    capture_timeout_seconds: float = 60.0
    shutdown_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.pre_seconds < 0 or self.post_seconds < 0 or self.processing_lag_seconds < 0:
            raise ValueError("Evidence windows must be nonnegative")
        if not all(isfinite(value) for value in (self.pre_seconds, self.post_seconds, self.sample_fps, self.processing_lag_seconds,
                self.capture_timeout_seconds, self.shutdown_timeout_seconds, *self.retry_delays)):
            raise ValueError("Evidence time settings must be finite")
        if self.sample_fps <= 0 or self.capture_timeout_seconds <= 0 or self.shutdown_timeout_seconds <= 0:
            raise ValueError("Evidence sampling and timeout must be positive")
        for name in ("per_session_bytes", "global_bytes", "max_active_captures", "workers", "max_spool_bytes"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.disk_reserve_bytes < 0 or any(delay < 0 for delay in self.retry_delays):
            raise ValueError("Disk reserve and retry delays must be nonnegative")


@dataclass(frozen=True)
class EncodedFrame:
    timeline_id: str
    media_time_ns: int
    sequence: int
    jpeg: bytes

