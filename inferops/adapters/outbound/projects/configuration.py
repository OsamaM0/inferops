"""Shared configuration discovery, schemas and platform precedence.

The UI edits these same documents; it does not maintain a separate settings store.
This module has no environment/settings side effects during project loading.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from inferops.adapters.outbound.projects.manifest import DomainManifest, manifest_path
from inferops.adapters.outbound.projects.models import ModelPortSpec, model_config_files
from inferops.adapters.outbound.projects.yaml import deep_merge, load_yaml
from inferops.domain.errors import ConfigError
from inferops.domain.fusion.spec import FusionSpec
from inferops.domain.incident_spec import AlertSpec
from inferops.domain.models import ModelSpec
from inferops.domain.pipeline import PipelineSpec

SECTIONS = {
    "settings": "Workspace settings",
    "project": "Project",
    "models": "Models",
    "pipelines": "Pipelines",
    "fusion": "Signals & rules",
    "alerts": "Alerts & actions",
    "platform": "Platform",
}
SCHEMAS = {"project": DomainManifest, "models": ModelSpec, "pipelines": PipelineSpec,
           "fusion": FusionSpec, "alerts": AlertSpec}


def platform_paths(configs_dir: Path, root: Path | None) -> list[Path]:
    paths = [configs_dir / "platform.yaml"]
    if root and root / "configs/platform.yaml" not in paths:
        paths.append(root / "configs/platform.yaml")
    return paths


def load_platform(configs_dir: Path, root: Path | None, manifest, *, overrides=None) -> dict:
    overrides = overrides or {}
    merged = {}
    for path in platform_paths(configs_dir, root):
        if path in overrides or path.is_file():
            data = overrides[path] if path in overrides else load_yaml(path)
            if not isinstance(data, dict):
                raise ConfigError(f"{path}: platform configuration must be a mapping")
            merged = deep_merge(merged, data)
    merged = deep_merge(merged, manifest.platform) if manifest else merged
    # The workspace is authoritative over packaged project platform defaults.
    central_platform = overrides.get((configs_dir / "platform.yaml").resolve())
    return deep_merge(merged, central_platform) if central_platform is not None else merged


def configuration_paths(definition) -> dict[tuple[str, str], Path]:
    """Stable editor IDs are independent of logical names and runtime variants."""
    found = {}
    for section in ("models", "pipelines", "fusion", "alerts"):
        directory = definition.source_dirs[section]
        paths = model_config_files(directory) if section == "models" else sorted(directory.glob("*.yaml"))
        for path in paths:
            name = path.parent.name if path.name == "model.yaml" else path.stem
            key = (section, name)
            if key in found:
                raise ConfigError(f"ambiguous configuration ID {section}/{name}")
            found[key] = path
    if definition.root:
        path = manifest_path(definition.root)
        if path.is_file():
            found["project", "project"] = path
    configs_dir = definition.source_dirs.get("platform_configs", definition.source_dirs["pipelines"].parent)
    for index, path in enumerate(platform_paths(configs_dir, definition.root)):
        if path.is_file() or (index == 0 and "workspace_config" in definition.source_dirs):
            found["platform", "platform" if index == 0 else "project"] = path
    central_path = definition.source_dirs.get("workspace_config")
    if central_path:
        from inferops.adapters.outbound.projects import central
        data = central.read_config(central_path)
        found["settings", "inferops"] = central_path
        for path in central.documents(central_path, data, definition.root, configs_dir):
            for section in ("models", "pipelines", "fusion", "alerts"):
                directory = definition.source_dirs[section]
                if path.parent == directory or (section == "models" and path.parent.parent == directory and path.name == "model.yaml"):
                    found[section, path.parent.name if path.name == "model.yaml" else path.stem] = path
    return found


def configuration_schema(section: str, data: dict) -> dict:
    """Use runtime schemas for forms, including partial model variant overrides."""
    if section == "settings":
        return {"type": "object", "title": "Workspace configuration", "additionalProperties": True,
                "properties": {key: {"type": "object"} for key in
                               ("runtime", "platform", "sources", "evidence", "media", "storage", "docker", "projects", "workers")}}
    if section == "platform":
        # Imported only by the HTTP editor, never by the library loader.
        from inferops.adapters.outbound.projects.settings import Settings
        schema = Settings.model_json_schema(by_alias=False)
        for key in ("configs_dir", "active_domain", "domain_root", "model_repository", "admission"):
            schema["properties"].pop(key, None)
        schema["properties"]["runtime"] = {
            "type": "object", "properties": {"admission": {"$ref": "#/$defs/AdmissionSettings"}},
        }
        return schema
    if section == "models" and "variants" in data:
        schema = ModelPortSpec.model_json_schema()
        model = ModelSpec.model_json_schema()
        schema["$defs"] = model.pop("$defs", {})
        model.pop("required", None)
        schema["properties"]["defaults"] = deepcopy(model)
        schema["properties"]["defaults"]["title"] = "Shared model settings"
        model.pop("title", None)
        schema["properties"]["variants"]["additionalProperties"] = model
        schema["properties"]["variants"]["title"] = "Runtime overrides"
        return schema
    schema = SCHEMAS[section].model_json_schema()
    if section == "pipelines":
        from inferops.adapters.outbound.stages.registry import stage_configuration

        schema["x-stages"] = stage_configuration()
        schema["$defs"]["StageDecl"]["x-stage-declaration"] = True
    if section == "project":
        platform = configuration_schema("platform", {})
        schema.setdefault("$defs", {}).update(platform.pop("$defs", {}))
        schema["properties"]["platform"] = platform
        schema["properties"]["platform"]["title"] = "Project platform overrides"
    return schema


def validate_document(section: str, data: dict) -> None:
    """Validate supplied platform fields without consulting environment overrides."""
    if section == "settings":
        from pydantic import TypeAdapter

        from inferops.adapters.outbound.projects.central import validate_config
        from inferops.domain.evidence import EvidenceConfig
        from inferops.domain.sources import SourceSpec
        from inferops.domain.workers import WorkerManifest
        validate_config(data)
        validate_document("platform", data.get("platform", {}))
        TypeAdapter(EvidenceConfig).validate_python(data.get("evidence", {}))
        if data.get("workers"):
            WorkerManifest.model_validate({**data["workers"], "cameras": []})
        defaults = data.get("sources", {})
        allowed = set(SourceSpec.model_fields) - {"camera_id", "name", "kind", "uri", "pipeline", "group", "location"}
        if set(defaults) - allowed:
            raise ConfigError(f"unknown source defaults: {sorted(set(defaults) - allowed)}")
        SourceSpec(camera_id="validation", name="Validation", kind="rtsp", uri="rtsp://validation/live", pipeline="validation", **defaults)
    elif section == "platform":
        # BaseSettings init reads environment; TypeAdapter validates only this data.
        from pydantic import TypeAdapter

        from inferops.adapters.outbound.projects.settings import Settings
        fields = dict(Settings.model_fields)
        values = dict(data)
        runtime = values.pop("runtime", {})
        if not isinstance(runtime, dict):
            raise ConfigError("runtime must be a mapping")
        if set(runtime) - {"admission"}:
            raise ConfigError(f"unknown runtime settings: {sorted(set(runtime) - {'admission'})}")
        if "admission" in runtime:
            values["admission"] = runtime["admission"]
        for key, value in values.items():
            if key not in fields:
                # Project-specific platform extensions remain supported.
                continue
            field = fields[key]
            annotation = field.annotation
            if isinstance(annotation, type) and hasattr(annotation, "model_fields") and isinstance(value, dict):
                for child, item in value.items():
                    if child not in annotation.model_fields:
                        raise ConfigError(f"unknown platform setting: {key}.{child}")
                    nested = annotation.model_fields[child]
                    TypeAdapter(nested.rebuild_annotation()).validate_python(item)
            else:
                TypeAdapter(field.rebuild_annotation()).validate_python(value)
    elif section == "project":
        manifest = DomainManifest(**data)
        validate_document("platform", manifest.platform)
