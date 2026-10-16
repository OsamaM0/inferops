"""Time and result contracts shared by all input adapters."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MediaFrame:
    image: Any
    media_time_ns: int | None = None
    sequence: int = 0
    timestamp_provenance: str = "source"
    presentation_ts_ms: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunSummary:
    pipeline: str
    session_id: str
    frames: int = 0
    detections_by_class: dict[str, int] = field(default_factory=dict)
    models_used: set[str] = field(default_factory=set)
    inference_total_ms: float = 0
    inference_count: int = 0
    incident_count: int = 0
    event_count: int = 0

    def add(self, items) -> None:
        for item in items:
            self.frames += 1
            self.incident_count += len(item.payload.get("alerts", []))
            self.event_count += len(item.payload.get("fusion_events", []))
            for result in item.results:
                self.models_used.add(result.model)
                self.inference_total_ms += result.latency_ms
                self.inference_count += 1
                for detection in result.detections or []:
                    self.detections_by_class[detection.label] = self.detections_by_class.get(detection.label, 0) + 1

    def asdict(self) -> dict:
        return {"pipeline": self.pipeline, "session_id": self.session_id, "frames": self.frames,
                "detections_by_class": self.detections_by_class, "models_used": sorted(self.models_used),
                "mean_inference_ms": self.inference_total_ms / self.inference_count if self.inference_count else None,
                "incident_count": self.incident_count, "event_count": self.event_count}
