"""PipelineSpec Pydantic schema."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from inferops.core.errors import ConfigError
from inferops.core.tracking import TrackingSpec


class StageDecl(BaseModel):
    name: str
    stage: str                                 # builtin alias OR "module:Class"
    params: dict[str, Any] = Field(default_factory=dict)


class LimitsSpec(BaseModel):
    max_concurrency: int = 4
    queue_size: int = 32
    timeout_ms: int | None = None


class PipelineSpec(BaseModel):
    name: str
    version: int = 1
    source: StageDecl | None = None
    stages: list[StageDecl] = Field(default_factory=list)
    sinks: list[StageDecl] = Field(default_factory=list)
    edges: list[list[str]] = Field(default_factory=list)  # if empty, infer linear order
    limits: LimitsSpec = Field(default_factory=LimitsSpec)
    tracking: TrackingSpec | None = None
    tags: dict[str, str] = Field(default_factory=dict)

    def for_mode(self, mode: str, resolver=None) -> PipelineSpec:
        """Resolve model choices identically for startup, preflight and workers."""
        resolved = self.model_copy(deep=True)
        from inferops.registry.resolver import canonical_mode

        mode = canonical_mode(mode)
        declarations = ([resolved.source] if resolved.source else []) + resolved.stages + resolved.sinks
        for stage in declarations:
            choices = stage.params.pop("model_modes", {})
            if "auto" in choices and "cpu" not in choices:
                choices["cpu"] = choices["auto"]
            if choices and mode not in choices:
                raise ConfigError(
                    f"pipeline {self.name!r}, stage {stage.name!r}: no model configured for mode {mode!r}; "
                    f"available modes: {', '.join(choices)}"
                )
            if mode in choices:
                stage.params["model"] = choices[mode]
            if stage.params.get("model"):
                if resolver is not None:
                    stage.params["model"] = resolver.resolve(stage.params["model"], mode)
                stage.params["fallback_model"] = None
        if resolved.tracking and resolver is not None:
            resolved.tracking.model_filter = [resolver.resolve(name, mode) for name in resolved.tracking.model_filter]
        return resolved
