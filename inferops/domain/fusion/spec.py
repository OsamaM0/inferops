"""Fusion specs (windows, rules, state machines)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from inferops.domain.errors import ConfigError


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
    # A list is expanded into one transition per source state before validation.
    from_state: str | list[str]
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


class FusionDefaults(BaseModel):
    """Shared values written once instead of on every rule, transition and window."""

    rule: dict[str, Any] = Field(default_factory=dict)
    transition: dict[str, Any] = Field(default_factory=dict)
    # Every declared signal gets this window unless `windows` names its own.
    window: dict[str, Any] | None = None
    # Extract expression for signals without one; `{name}` is the signal name.
    extract: str | None = None


def _expand(data: dict[str, Any]) -> dict[str, Any]:
    defaults = FusionDefaults(**(data.get("defaults") or {}))
    data = dict(data)
    if "signals" not in data:
        data["signals"] = list(dict.fromkeys([*data.get("extract", {}), *data.get("windows", {})]))
    if defaults.window is not None:
        data["windows"] = {signal: {**defaults.window, **data.get("windows", {}).get(signal, {})}
                           for signal in dict.fromkeys([*data["signals"], *data.get("windows", {})])}
    if defaults.extract:
        data["extract"] = {**{signal: defaults.extract.format(name=signal) for signal in data["signals"]},
                           **data.get("extract", {})}
    data["rules"] = [{**defaults.rule, **rule} for rule in data.get("rules", [])]
    machine = data.get("state_machine")
    if machine:
        transitions = []
        for transition in machine.get("transitions", []):
            sources = transition.get("from_state")
            for source in sources if isinstance(sources, list) else [sources]:
                transitions.append({**defaults.transition, **transition, "from_state": source})
        data["state_machine"] = {**machine, "transitions": transitions}
    return data


class FusionSpec(BaseModel):
    name: str
    entity_key: str = "stream_id"
    # Optional: inferred from `extract` and `windows` when omitted.
    signals: list[str]
    defaults: FusionDefaults = Field(default_factory=FusionDefaults)
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
    # Rules switched off because the models producing their signals are not selected.
    inactive_rules: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _apply_defaults(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        try:
            return _expand(data)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ConfigError(f"fusion {data.get('name')!r}: invalid defaults: {exc}") from exc
