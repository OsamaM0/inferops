"""Validate a project's config graph before any runtime resources are opened."""
from __future__ import annotations

from pathlib import Path

from inferops.adapters.outbound.projects.settings import ConfigStack
from inferops.adapters.outbound.projects.yaml import load_yaml
from inferops.domain.errors import ConfigError


def validate_config_stack(stack: ConfigStack, *, overrides: dict[Path, dict] | None = None) -> dict[str, dict]:
    """Load typed specs and reject duplicate names and dangling references."""
    from inferops.adapters.outbound.projects.manifest import DomainManifest, manifest_path
    from inferops.adapters.outbound.projects.models import (
        expand_model_config,
        load_model_variants,
        model_config_files,
    )
    from inferops.adapters.outbound.projects.resolver import ModelResolver
    from inferops.domain.fusion.spec import FusionSpec
    from inferops.domain.incident_spec import AlertSpec
    from inferops.domain.models import ModelSpec
    from inferops.domain.pipeline import PipelineSpec

    schemas = {"models": ModelSpec, "pipelines": PipelineSpec, "fusion": FusionSpec, "alerts": AlertSpec}
    specs: dict[str, dict] = {section: {} for section in schemas}
    errors: list[str] = []
    for section, schema in schemas.items():
        paths = (model_config_files(stack.spec_dir(section)) if section == "models"
                 else sorted(stack.spec_dir(section).glob("*.yaml")))
        for path in paths:
            try:
                if overrides and path in overrides:
                    data = overrides[path]
                    loaded = expand_model_config(data) if section == "models" else [schema(**data)]
                else:
                    loaded = load_model_variants(path) if section == "models" else [schema(**load_yaml(path))]
            except Exception as exc:
                errors.append(f"{path}: {exc}")
                continue
            for spec in loaded:
                if spec.name in specs[section]:
                    errors.append(f"{path}: duplicate {section} name {spec.name!r}")
                specs[section][spec.name] = spec

    if errors:
        raise ConfigError("Invalid project configuration:\n" + "\n".join(errors))
    resolver = ModelResolver(stack.spec_dir("models"), overrides=overrides)

    def check(section: str, name: str, location: str) -> None:
        if not isinstance(name, str):
            errors.append(f"{location}: {section} reference must be a name")
        elif not (resolver.contains(name) if section == "models" else name in specs[section]):
            errors.append(f"{location}: unknown {section} reference {name!r}")

    for pipeline in specs["pipelines"].values():
        declarations = ([pipeline.source] if pipeline.source else []) + pipeline.stages + pipeline.sinks
        for stage in declarations:
            location = f"pipeline {pipeline.name!r}, stage {stage.name!r}"
            for key, section in (("model", "models"), ("fallback_model", "models"),
                                 ("fusion", "fusion"), ("alerts", "alerts")):
                name = stage.params.get(key)
                if name is not None:
                    check(section, name, location)
            choices = stage.params.get("model_modes", {})
            if not isinstance(choices, dict) or any(not isinstance(name, str) for name in choices.values()):
                errors.append(f"{location}: model_modes must map execution modes to model names")
            else:
                for mode, name in choices.items():
                    if mode not in {"auto", "cpu", "cuda", "tensorrt", "triton"}:
                        errors.append(f"{location}: unknown execution mode {mode!r}")
                    check("models", name, f"{location}, mode {mode!r}")
        if pipeline.tracking:
            for name in pipeline.tracking.model_filter:
                check("models", name, f"pipeline {pipeline.name!r} tracking")

    manifest = stack.manifest()
    if stack.domain_root and overrides and manifest_path(stack.domain_root) in overrides:
        manifest = DomainManifest(**overrides[manifest_path(stack.domain_root)])
    if manifest is not None:
        for name in manifest.pipelines or []:
            check("pipelines", name, "project pipelines")
        for name in manifest.prewarm_models:
            check("models", name, "project prewarm_models")
        for name in manifest.prewarm_pipelines:
            check("pipelines", name, "project prewarm_pipelines")
            if manifest.pipelines is not None and name not in manifest.pipelines:
                errors.append(f"prewarm pipeline {name!r} is not enabled in project pipelines")
    if errors:
        raise ConfigError("Invalid project configuration:\n" + "\n".join(errors))
    return specs
