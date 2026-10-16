"""Explicit project definition and loader contracts."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from inferops.domain.fusion.spec import FusionSpec
from inferops.domain.incident_spec import AlertSpec
from inferops.domain.models import ModelSpec
from inferops.domain.pipeline import PipelineSpec
from inferops.ports.inference import InferenceBackend


class PluginResolver(Protocol):
    def resolve(self, reference: str) -> str: ...


@dataclass(frozen=True)
class ProjectDefinition:
    root: Path | None
    name: str
    mode: str
    revision: str
    models: Mapping[str, ModelSpec]
    pipelines: Mapping[str, PipelineSpec]
    fusion: Mapping[str, FusionSpec]
    alerts: Mapping[str, AlertSpec]
    source_dirs: Mapping[str, Path]
    plugin_resolver: PluginResolver
    object_loader: Callable[[str], Any]
    backend_factory: Callable[[ModelSpec, Path | None], InferenceBackend]
    resolver: Any = None
    manifest: Any = None
    platform: Mapping[str, Any] = field(default_factory=dict)

    def resolve_model(self, name: str) -> str:
        return self.resolver.resolve(name, self.mode) if self.resolver is not None else name


class ProjectLoader(Protocol):
    def load(self) -> ProjectDefinition: ...
