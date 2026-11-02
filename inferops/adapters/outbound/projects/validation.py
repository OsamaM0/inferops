"""Validate a project's config graph before any runtime resources are opened."""
from __future__ import annotations

from pathlib import Path

from inferops.adapters.outbound.projects.central import apply_settings
from inferops.adapters.outbound.projects.settings import ConfigStack
from inferops.adapters.outbound.projects.yaml import load_yaml
from inferops.domain.errors import ConfigError


def validate_config_stack(stack: ConfigStack, *, overrides: dict[Path, dict] | None = None) -> dict[str, dict]:
    """Load typed specs and reject duplicate names and dangling references."""
    from inferops.adapters.outbound.projects.manifest import DomainManifest, manifest_path
    from inferops.adapters.outbound.projects.models import (
        expand_model_config,
        model_config_files,
    )
    from inferops.adapters.outbound.projects.resolver import ModelResolver
    from inferops.domain.fusion.spec import FusionSpec
    from inferops.domain.incident_spec import AlertSpec
    from inferops.domain.models import ModelSpec
    from inferops.domain.pipeline import PipelineSpec

    if stack.composed():
        # The loader generates composed projects; validate exactly what it serves.
        from inferops.adapters.outbound.projects.loader import ProjectLoader
        definition = ProjectLoader(stack.domain_root, configs_dir=stack.configs_dir).load(overrides)
        return {section: dict(getattr(definition, section)) for section in ("models", "pipelines", "fusion", "alerts")}
    schemas = {"models": ModelSpec, "pipelines": PipelineSpec, "fusion": FusionSpec, "alerts": AlertSpec}
    overrides = {**stack.configuration_overrides(), **(overrides or {})}
    manifest = stack.manifest()
    if stack.domain_root and overrides and manifest_path(stack.domain_root) in overrides:
        manifest = DomainManifest(**overrides[manifest_path(stack.domain_root)])
    try:
        settings = manifest.setting_values() if manifest else {}
    except ValueError as exc:
        raise ConfigError(f"project settings: {exc}") from exc
    specs: dict[str, dict] = {section: {} for section in schemas}
    errors: list[str] = []
    for section, schema in schemas.items():
        paths = (model_config_files(stack.spec_dir(section)) if section == "models"
                 else sorted(stack.spec_dir(section).glob("*.yaml")))
        directory = stack.spec_dir(section).resolve()
        paths = sorted({path.resolve() for path in paths} | {path for path in overrides if path.parent == directory
                        or (section == "models" and path.name == "model.yaml" and path.parent.parent == directory)})
        for path in paths:
            try:
                data = apply_settings(overrides[path] if path in overrides else load_yaml(path), settings)
                loaded = expand_model_config(data) if section == "models" else [schema(**data)]
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
            for key in stage.auxiliary_model_keys():
                check("models", stage.params[key], location)
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
