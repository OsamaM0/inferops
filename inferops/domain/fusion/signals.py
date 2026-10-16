"""Signal types."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from inferops.domain.types import new_id, now_ms


class Signal(BaseModel):
    name: str
    entity_id: str
    ts_ms: int = Field(default_factory=now_ms)
    value: float = 0.0
    attrs: dict[str, Any] = Field(default_factory=dict)


class FusionEvent(BaseModel):
    # Events are stored in the same generic table as everything else, which is
    # keyed by id. Alert already carried one; FusionEvent did not, so events had
    # no stable identity to persist or de-duplicate against.
    id: str = Field(default_factory=new_id)
    fusion: str
    entity_id: str
    ts_ms: int = Field(default_factory=now_ms)
    type: str
    severity: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
