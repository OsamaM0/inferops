"""Validated static assignments; the manifest contains secret references only."""
from pydantic import BaseModel, ConfigDict, Field, model_validator

from inferops.domain.sources import SourceSpec


class WorkerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    worker_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
    endpoint: str = Field(pattern=r"^https?://[^/]+/?$")
    max_active_sources: int = Field(default=12, ge=1, le=1024)


class CameraAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    worker_id: str
    source: SourceSpec


class WorkerManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int = Field(default=1, ge=1, le=1)
    workers: list[WorkerSpec]
    cameras: list[CameraAssignment] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_assignments(self):
        names = [worker.worker_id for worker in self.workers]
        cameras = [entry.source.camera_id for entry in self.cameras]
        if len(set(names)) != len(names) or len(set(cameras)) != len(cameras):
            raise ValueError("worker IDs and camera IDs must be unique")
        if any(entry.worker_id not in names for entry in self.cameras):
            raise ValueError("camera assignment references an unknown worker")
        return self
