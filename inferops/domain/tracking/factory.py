"""TrackingSpec: pydantic schema for the `tracking:` block in PipelineSpec."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from inferops.domain.tracking.tracker import Tracker, make_tracker


class TrackingSpec(BaseModel):
    """Per-pipeline tracking configuration.

    YAML example:
        tracking:
          enabled: true
          algorithm: iou_kalman   # or bytetrack
          by: stream_id            # tracker scope key (stream_id|pipeline|global)
          max_age: 30
          min_hits: 3
          iou_threshold: 0.3
          low_threshold: 0.5
          score_high: 0.5
          score_low: 0.1
          per_class: true
          model_filter: []         # optional list of model names to track
          insert_after: ""         # stage name to insert after (defaults to first infer)
    """

    enabled: bool = False
    algorithm: Literal["iou_kalman", "bytetrack"] = "iou_kalman"
    by: Literal["stream_id", "pipeline", "global"] = "stream_id"
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3
    low_threshold: float = 0.5
    score_high: float = 0.5
    score_low: float = 0.1
    per_class: bool = True
    model_filter: list[str] = Field(default_factory=list)
    insert_after: str = ""


def build_tracker(spec: TrackingSpec) -> Tracker:
    return make_tracker(
        algorithm=spec.algorithm,
        max_age=spec.max_age,
        min_hits=spec.min_hits,
        iou_threshold=spec.iou_threshold,
        low_threshold=spec.low_threshold,
        score_high=spec.score_high,
        score_low=spec.score_low,
        per_class=spec.per_class,
    )
