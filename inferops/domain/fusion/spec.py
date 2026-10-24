"""Fusion specs (windows, rules, state machines)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class WindowSpec(BaseModel):
    kind: Literal["rolling_time", "rolling_count", "ewma"] = "rolling_time"
    duration_ms: int | None = 5000
    count: int | None = None
    alpha: float | None = None
    reducers: list[str] = Field(default_factory=lambda: ["latest"])
    reducer_params: dict[str, Any] = Field(default_factory=dict)


class RuleSpec(BaseModel):
    name: str
    when: str
    emit: dict[str, Any] | None = None
    cooldown_ms: int = 0
    severity: str | None = None
    confirm_ms: int = Field(default=0, ge=0)
    min_samples: int = Field(default=1, ge=1)
    max_gap_ms: int = Field(default=1500, gt=0)


class TransitionSpec(BaseModel):
    from_state: str
    to_state: str
    when: str
    min_dwell_ms: int = 0
    confirm_ms: int = Field(default=0, ge=0)
    min_samples: int = Field(default=1, ge=1)
    max_gap_ms: int = Field(default=1500, gt=0)


class StateMachineSpec(BaseModel):
    initial: str
    states: list[str]
    transitions: list[TransitionSpec] = Field(default_factory=list)
    on_enter: dict[str, list[str]] = Field(default_factory=dict)
    on_exit: dict[str, list[str]] = Field(default_factory=dict)


class FusionOutputs(BaseModel):
    events_topic: str | None = None
    alerts_topic: str | None = None


class FusionSpec(BaseModel):
    name: str
    entity_key: str = "stream_id"
    signals: list[str]
    windows: dict[str, WindowSpec] = Field(default_factory=dict)
    derived: dict[str, str] = Field(default_factory=dict)
    rules: list[RuleSpec] = Field(default_factory=list)
    state_machine: StateMachineSpec | None = None
    outputs: FusionOutputs = Field(default_factory=FusionOutputs)
    # Mirrors AlertSpec.persistence_enabled. Without this the engine emitted
    # events but never stored them, so GET /v1/events was permanently empty
    # unless a pipeline happened to declare a `sink_postgres` stage — while
    # alerts persisted automatically. The dashboard's Events panel reads that
    # endpoint, so it could never populate.
    persistence_enabled: bool = True
    # How to extract signals from a StageItem (signal_name -> python expression)
    extract: dict[str, str] = Field(default_factory=dict)
