"""AlertSpec."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SeverityLevel(BaseModel):
    name: str
    rank: int
    color: str | None = None
    notify_channels: list[str] = Field(default_factory=list)


def _default_levels() -> list[SeverityLevel]:
    return [SeverityLevel(name="low", rank=1, color="#9cf"), SeverityLevel(name="medium", rank=2, color="#fc6"),
            SeverityLevel(name="high", rank=3, color="#f63"), SeverityLevel(name="critical", rank=4, color="#f00")]


class SeverityMapSpec(BaseModel):
    levels: list[SeverityLevel] = Field(default_factory=_default_levels)
    mapping: dict[str, str] = Field(default_factory=dict)   # type -> level name
    default: str = "low"


class DedupeSpec(BaseModel):
    strategy: Literal["by_type_entity", "by_payload_hash", "by_key_template"] = "by_type_entity"
    key_template: str = "{type}:{entity_id}"
    window_ms: int = 60_000


class CooldownSpec(BaseModel):
    per_dedupe_key_ms: int = 30_000
    per_type_ms: int = 0
    global_ms: int = 0


class SinkSpec(BaseModel):
    name: str
    kind: Literal["webhook", "kafka", "postgres", "ws", "stdout", "slack", "plugin"]
    params: dict[str, Any] = Field(default_factory=dict)
    min_severity: str | None = None


class AlertSpec(BaseModel):
    name: str
    severity: SeverityMapSpec = Field(default_factory=SeverityMapSpec)
    dedupe: DedupeSpec = Field(default_factory=DedupeSpec)
    cooldown: CooldownSpec = Field(default_factory=CooldownSpec)
    sinks: list[SinkSpec] = Field(default_factory=list)
    persistence_enabled: bool = True
    event_types: list[str] | None = None
    # Only events with a severity (from their fusion rule or severity.mapping)
    # become incidents, so new rules need no separate alert registration.
    rated_only: bool = False


class AlertCandidate(BaseModel):
    type: str
    entity_id: str
    ts_ms: int
    payload: dict[str, Any] = Field(default_factory=dict)
    severity_hint: str | None = None
    dedupe_key: str | None = None


class Alert(BaseModel):
    id: str
    type: str
    entity_id: str
    ts_ms: int
    severity: str
    dedupe_key: str
    payload: dict[str, Any] = Field(default_factory=dict)
    status: str = "open"
