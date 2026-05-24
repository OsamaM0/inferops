"""Signal types."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from inferops.core.types import now_ms


class Signal(BaseModel):
    name: str
    entity_id: str
    ts_ms: int = Field(default_factory=now_ms)
    value: float = 0.0
    attrs: dict[str, Any] = Field(default_factory=dict)


class FusionEvent(BaseModel):
    fusion: str
    entity_id: str
    ts_ms: int = Field(default_factory=now_ms)
    type: str
    severity: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
