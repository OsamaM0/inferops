"""ModelSpec pydantic schema."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from inferops.core.types import Task


class IOSpec(BaseModel):
    name: str
    shape: list[int] = Field(default_factory=list)
    dtype: str = "float32"
    layout: Literal["NCHW", "NHWC", "ND"] = "NCHW"
    color: Literal["RGB", "BGR", "GRAY", "NONE"] = "RGB"
    role: str | None = None


class ArtifactSource(BaseModel):
    """A downloadable weight file, with optional extraction from an archive."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["url", "google_drive", "huggingface", "installer"]
    url: str | None = None
    file_id: str | None = None
    repo_id: str | None = None
    filename: str | None = None
    revision: str | None = None
    archive_member: str | None = None
    installer: str | None = None

    @model_validator(mode="after")
    def validate_source(self):
        required = {"url": ["url"], "google_drive": ["file_id"],
                    "huggingface": ["repo_id", "filename"], "installer": ["installer"]}
        for field in required[self.kind]:
            if not getattr(self, field):
                raise ValueError(f"{self.kind} source requires {field}")
        if self.url and not self.url.startswith(("https://", "http://")):
            raise ValueError("download URL must use HTTP or HTTPS")
        return self


class ArtifactSpec(BaseModel):
    path: str | None = None
    triton_model_name: str | None = None
    url: str | None = None
    sha256: str | None = None
    source: ArtifactSource | None = None


class PluginRef(BaseModel):
    plugin: str
    params: dict[str, Any] = Field(default_factory=dict)


class LabelsSpec(BaseModel):
    inline: list[str] | None = None
    from_file: str | None = None

    def resolve(self, base_dir: str | Path | None = None) -> list[str]:
        if self.inline:
            return list(self.inline)
        if self.from_file:
            p = Path(self.from_file)
            if base_dir and not p.is_absolute():
                p = Path(base_dir) / p
            if p.exists():
                return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return []


class BatchingSpec(BaseModel):
    enabled: bool = False
    max_batch_size: int = 1
    max_delay_ms: int = 0


class DeviceSpec(BaseModel):
    preferred: Literal["gpu", "cpu"] = "cpu"
    gpu_id: int = 0


class WarmupSpec(BaseModel):
    iterations: int = 0
    dummy_input_seed: int = 0


class ModelSpec(BaseModel):
    name: str
    version: int = 1
    task: Task = Task.CUSTOM
    # A built-in name (see inferops.runtimes.BUILTIN_BACKENDS) or a
    # "module:Class" plugin reference to a domain-provided InferenceBackend.
    backend: str = "onnxruntime"
    artifact: ArtifactSpec = Field(default_factory=ArtifactSpec)
    inputs: list[IOSpec]
    outputs: list[IOSpec] = Field(default_factory=list)
    labels: LabelsSpec | list[str] | None = None
    preprocess: PluginRef | None = None
    postprocess: PluginRef | None = None
    batching: BatchingSpec = Field(default_factory=BatchingSpec)
    device: DeviceSpec = Field(default_factory=DeviceSpec)
    warmup: WarmupSpec = Field(default_factory=WarmupSpec)
    tags: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _coerce_labels(self) -> ModelSpec:
        if isinstance(self.labels, list):
            self.labels = LabelsSpec(inline=self.labels)
        return self

    def label_list(self, base_dir: str | Path | None = None) -> list[str]:
        if isinstance(self.labels, LabelsSpec):
            return self.labels.resolve(base_dir)
        return []
