"""PipelineSpec Pydantic schema."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


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
    tags: dict[str, str] = Field(default_factory=dict)
