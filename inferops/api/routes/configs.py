"""Config file routes for dashboard-driven editing and reloads."""
from __future__ import annotations

import re
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, ValidationError

from inferops.alerts import get_alert_store
from inferops.alerts.spec import AlertSpec
from inferops.core.config import get_config_stack, get_spec_dir, settings
from inferops.core.errors import ConfigError
from inferops.core.yaml_loader import parse_yaml
from inferops.fusion import get_fusion_store
from inferops.fusion.spec import FusionSpec
from inferops.pipeline import get_pipeline_store
from inferops.pipeline.spec import PipelineSpec
from inferops.registry import get_registry
from inferops.registry.loader import expand_model_config, load_model_variants, model_config_files

router = APIRouter()

_SECTIONS = ("platform", "models", "pipelines", "fusion", "alerts", "project")
_LIVE_RELOAD_SECTIONS = {"models", "pipelines", "fusion", "alerts"}
_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class ConfigUpdate(BaseModel):
    content: str = Field(min_length=1)
    reload: bool = True


def _configs_root() -> Path:
    return Path(settings.configs_dir)


def _repo_root() -> Path:
    return _configs_root().resolve().parent


def _section_dir(section: str) -> Path:
    if section == "platform":
        return _configs_root()
    return get_spec_dir(section, settings)


def _path_for(section: str, name: str) -> Path:
    if section not in _SECTIONS:
        raise HTTPException(404, f"unknown config section: {section}")
    if section == "project":
        from inferops.core.domain import manifest_path

        stack = get_config_stack(settings)
        if name != "project" or stack.domain_root is None:
            raise HTTPException(404, "no selected project")
        return manifest_path(stack.domain_root)
    if section == "platform":
        if name != "platform":
            raise HTTPException(404, f"unknown config: {section}/{name}")
        return _configs_root() / "platform.yaml"
    if name in {".", ".."} or not _NAME_RE.fullmatch(name):
        raise HTTPException(400, "invalid config name")
    if section == "models":
        port_path = _section_dir(section) / name / "model.yaml"
        if port_path.is_file():
            return port_path
    return _section_dir(section) / f"{name}.yaml"


def _validate_content(section: str, content: str) -> dict:
    try:
        data = parse_yaml(content)
    except yaml.YAMLError as exc:
        raise HTTPException(400, f"invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise HTTPException(400, "config must be a YAML mapping")
    try:
        if section == "models":
            specs = expand_model_config(data)
            if settings.domain_root and any("synthetic" in spec.backend.lower() for spec in specs):
                raise ConfigError("Runnable projects require real models; synthetic backends are not supported")
        elif section == "project":
            from inferops.core.domain import DomainManifest

            DomainManifest(**data)
        elif section == "pipelines":
            PipelineSpec(**data)
        elif section == "fusion":
            FusionSpec(**data)
        elif section == "alerts":
            AlertSpec(**data)
    except ValidationError as exc:
        raise HTTPException(400, exc.errors()) from exc
    except ConfigError as exc:
        raise HTTPException(400, str(exc)) from exc
    return data


def _entry_for(path: Path, section: str) -> dict:
    stat = path.stat()
    resolved = path.resolve()
    try:
        display_path = resolved.relative_to(_repo_root())
    except ValueError:
        display_path = path
    return {
        "section": section,
        "name": (section if section in {"platform", "project"} else
                 path.parent.name if section == "models" and path.name == "model.yaml" else path.stem),
        "path": str(display_path),
        "reload_behavior": "restart_required" if section in {"platform", "project"} else "live",
        "mtime_ms": int(stat.st_mtime * 1000),
        "size_bytes": stat.st_size,
    }


def _pipeline_refs(spec: PipelineSpec) -> dict[str, str]:
    refs: dict[str, str] = {}
    stages = ([spec.source] if spec.source else []) + spec.stages + spec.sinks
    for stage in stages:
        params = stage.params or {}
        for key in ("model", "fusion", "alerts"):
            value = params.get(key)
            if isinstance(value, str) and key not in refs:
                refs[key] = value
    return refs


async def _reload_sections(sections: list[str]) -> list[dict]:
    results: list[dict] = []
    root = _configs_root()
    if any(section in _LIVE_RELOAD_SECTIONS for section in sections):
        await get_pipeline_store().close()

    for section in sections:
        if section in {"platform", "project"}:
            results.append({
                "section": section,
                "reloaded": 0,
                "dir": str(root),
                "restart_required": True,
            })
            continue

        directory = _section_dir(section)
        if section == "models":
            count = await get_registry().reload_from(directory) if directory.exists() else 0
            await get_pipeline_store().reload_from(_section_dir("pipelines"))
        elif section == "pipelines":
            count = await get_pipeline_store().reload_from(directory) if directory.exists() else 0
        elif section == "fusion":
            count = await get_fusion_store().reload_from(directory) if directory.exists() else 0
        elif section == "alerts":
            count = await get_alert_store().reload_from(directory) if directory.exists() else 0
        else:
            raise HTTPException(404, f"unknown config section: {section}")
        results.append({
            "section": section,
            "reloaded": count,
            "dir": str(directory),
            "restart_required": False,
        })

    return results


@router.get("/configs")
async def list_configs():
    root = _configs_root()
    stack = get_config_stack(settings)
    files: list[dict] = []
    model_configs: dict[str, str] = {}
    for section in _SECTIONS:
        if section == "project":
            if stack.domain_root:
                path = _path_for("project", "project")
                if path.is_file():
                    files.append(_entry_for(path, section))
            continue
        if section == "platform":
            path = root / "platform.yaml"
            if path.exists():
                files.append(_entry_for(path, section))
            continue

        directory = _section_dir(section)
        if not directory.exists():
            continue
        paths = model_config_files(directory) if section == "models" else sorted(directory.glob("*.yaml"))
        for path in paths:
            entry = _entry_for(path, section)
            files.append(entry)
            if section == "models":
                for model in load_model_variants(path):
                    model_configs[model.name] = entry["name"]

    domains = []
    for spec in get_pipeline_store().list():
        refs = _pipeline_refs(spec)
        domains.append({
            "id": spec.name,
            "label": spec.tags.get("label") or spec.name.replace("_", " ").title(),
            "pipeline": spec.name,
            "source_stage": spec.source.stage if spec.source else None,
            "config_names": {
                "pipeline": spec.name,
                "model": model_configs.get(refs.get("model"), refs.get("model")),
                "fusion": refs.get("fusion"),
                "alerts": refs.get("alerts"),
            },
            "capabilities": ["camera", "image", "video", "rtsp"],
            "stages": [stage.model_dump(mode="json") for stage in spec.stages],
        })

    manifest = stack.manifest()
    return {
        "configs_dir": str(root.resolve()),
        "active_domain": settings.active_domain,
        "domain_root": str(stack.domain_root.resolve()) if stack.domain_root else None,
        "domain_configs_dir": str(stack.domain_configs_dir.resolve()) if stack.domain_configs_dir else None,
        "domain_pack": manifest.model_dump(mode="json") if manifest else None,
        "sections": [section for section in _SECTIONS if section != "project" or stack.domain_root],
        "files": files,
        "domains": domains,
    }


@router.get("/configs/{section}/{name}")
async def get_config(section: str, name: str):
    path = _path_for(section, name)
    if not path.exists():
        raise HTTPException(404, f"config not found: {section}/{name}")
    return {
        **_entry_for(path, section),
        "content": path.read_text(encoding="utf-8"),
    }


@router.put("/configs/{section}/{name}")
async def update_config(section: str, name: str, payload: ConfigUpdate):
    path = _path_for(section, name)
    if not path.exists():
        raise HTTPException(404, f"config not found: {section}/{name}")
    data = _validate_content(section, payload.content)
    if section in _LIVE_RELOAD_SECTIONS or section == "project":
        from inferops.core.domain import DomainManifest
        from inferops.core.validation import validate_config_stack
        from inferops.registry.resolver import ModelResolver

        try:
            stack = get_config_stack(settings)
            prospective = validate_config_stack(stack, overrides={path: data})
            resolver = ModelResolver(stack.spec_dir("models"), overrides={path: data})
            manifest = DomainManifest(**data) if section == "project" else stack.manifest()
            for spec in prospective["pipelines"].values():
                if manifest is None or manifest.pipelines is None or spec.name in manifest.pipelines:
                    spec.for_mode(settings.execution_mode, resolver)
            for model in manifest.prewarm_models if manifest else []:
                resolver.resolve(model, settings.execution_mode)
        except ConfigError as exc:
            raise HTTPException(400, str(exc)) from exc
    preparation_required = section == "project"
    if section == "models":
        old = expand_model_config(parse_yaml(path.read_text(encoding="utf-8")))
        new = expand_model_config(data)
        def startup_contract(models):
            return [(spec.name, spec.backend, spec.artifact.model_dump(),
                     [tensor.model_dump() for tensor in spec.inputs],
                     [tensor.model_dump() for tensor in spec.outputs]) for spec in models]
        preparation_required = startup_contract(old) != startup_contract(new)
    path.write_text(payload.content.rstrip() + "\n", encoding="utf-8")

    reload_results = None
    if preparation_required:
        reload_results = [{"section": section, "reloaded": 0, "restart_required": True,
                           "preparation_required": True}]
    elif payload.reload:
        reload_results = await _reload_sections([section])

    return {
        **_entry_for(path, section),
        "saved": True,
        "preparation_required": preparation_required,
        "reload": reload_results[0] if reload_results else None,
    }


@router.post("/configs/reload")
async def reload_configs(section: str = Query("all")):
    sections = list(_LIVE_RELOAD_SECTIONS) if section == "all" else [section]
    for item in sections:
        if item not in _SECTIONS:
            raise HTTPException(404, f"unknown config section: {item}")
    from inferops.core.validation import validate_config_stack

    try:
        validate_config_stack(get_config_stack(settings))
    except ConfigError as exc:
        raise HTTPException(400, str(exc)) from exc
    if section == "all":
        ordered = ["models", "fusion", "alerts", "pipelines"]
        results = await _reload_sections(ordered)
        return {"section": "all", "results": results}
    results = await _reload_sections(sections)
    return results[0]
