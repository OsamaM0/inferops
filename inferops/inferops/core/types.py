"""Core types used throughout InferOps."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field


def now_ms() -> int:
    return int(time.time() * 1000)


def new_id() -> str:
    return uuid.uuid4().hex


class Task(str, Enum):
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    OCR = "ocr"
    EMBEDDING = "embedding"
    LANDMARKS = "landmarks"
    CUSTOM = "custom"


@dataclass(slots=True)
class Frame:
    """A raw or decoded frame travelling through the pipeline."""
    stream_id: str
    ts_ms: int
    image: np.ndarray | None = None       # HxWxC uint8 BGR or None if only ref
    ref: str | None = None                # redis://, s3://, file://
    width: int = 0
    height: int = 0
    seq: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


class Detection(BaseModel):
    label: str
    score: float
    box: tuple[float, float, float, float]   # xyxy normalized 0..1
    attrs: dict[str, Any] = Field(default_factory=dict)


class Classification(BaseModel):
    label: str
    score: float
    index: int = 0


class Embedding(BaseModel):
    vector: list[float]
    dim: int


class OCRText(BaseModel):
    text: str
    score: float = 1.0


class Result(BaseModel):
    """Tagged union of inference results."""
    kind: Literal["detections", "classifications", "embedding", "ocr", "raw"]
    detections: list[Detection] | None = None
    classifications: list[Classification] | None = None
    embedding: Embedding | None = None
    ocr: OCRText | None = None
    raw: dict[str, Any] | None = None
    model: str = ""
    latency_ms: float = 0.0


class StageItem(BaseModel):
    """Envelope passed between pipeline stages."""
    id: str = Field(default_factory=new_id)
    stream_id: str = ""
    ts_ms: int = Field(default_factory=now_ms)
    seq: int = 0
    frame_ref: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    results: list[Result] = Field(default_factory=list)
    trace: dict[str, str] = Field(default_factory=dict)   # traceparent etc.

    model_config = {"arbitrary_types_allowed": True}
