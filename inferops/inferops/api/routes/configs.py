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
from inferops.fusion import get_fusion_store
from inferops.fusion.spec import FusionSpec
from inferops.pipeline import get_pipeline_store
from inferops.pipeline.spec import PipelineSpec
from inferops.registry import get_registry
from inferops.registry.model_spec import ModelSpec

router = APIRouter()

_SECTIONS = ("platform", "models", "pipelines", "fusion", "alerts")
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
    if section == "platform":
        if name != "platform":
            raise HTTPException(404, f"unknown config: {section}/{name}")
        return _configs_root() / "platform.yaml"
    if not _NAME_RE.fullmatch(name):
        raise HTTPException(400, "invalid config name")
    return _section_dir(section) / f"{name}.yaml"


def _validate_content(section: str, content: str) -> None:
    try:
        data = yaml.safe_load(content) or {}
    except yaml.YAMLError as exc:
        raise HTTPException(400, f"invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise HTTPException(400, "config must be a YAML mapping")
    try:
        if section == "models":
            ModelSpec(**data)
        elif section == "pipelines":
            PipelineSpec(**data)
        elif section == "fusion":
            FusionSpec(**data)
        elif section == "alerts":
            AlertSpec(**data)
    except ValidationError as exc:
        raise HTTPException(400, exc.errors()) from exc


def _entry_for(path: Path, section: str) -> dict:
    stat = path.stat()
    resolved = path.resolve()
    try:
        display_path = resolved.relative_to(_repo_root())
    except ValueError:
        display_path = path
    return {
        "section": section,
        "name": "platform" if section == "platform" else path.stem,
        "path": str(display_path),
        "reload_behavior": "restart_required" if section == "platform" else "live",
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
        if section == "platform":
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
    for section in _SECTIONS:
        if section == "platform":
            path = root / "platform.yaml"
            if path.exists():
                files.append(_entry_for(path, section))
            continue

        directory = _section_dir(section)
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.yaml")):
            files.append(_entry_for(path, section))

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
                "model": refs.get("model"),
                "fusion": refs.get("fusion"),
                "alerts": refs.get("alerts"),
            },
            "capabilities": ["camera", "image", "video", "rtsp"],
            "stages": [stage.model_dump(mode="json") for stage in spec.stages],
        })

    return {
        "configs_dir": str(root.resolve()),
        "active_domain": settings.active_domain,
        "domain_root": str(stack.domain_root.resolve()) if stack.domain_root else None,
        "domain_configs_dir": str(stack.domain_configs_dir.resolve()) if stack.domain_configs_dir else None,
        "sections": list(_SECTIONS),
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
    _validate_content(section, payload.content)
    path.write_text(payload.content.rstrip() + "\n", encoding="utf-8")

    reload_results = None
    if payload.reload:
        reload_results = await _reload_sections([section])

    return {
        **_entry_for(path, section),
        "saved": True,
        "reload": reload_results[0] if reload_results else None,
    }


@router.post("/configs/reload")
async def reload_configs(section: str = Query("all")):
    sections = list(_LIVE_RELOAD_SECTIONS) if section == "all" else [section]
    for item in sections:
        if item not in _SECTIONS:
            raise HTTPException(404, f"unknown config section: {item}")
    if section == "all":
        ordered = ["models", "fusion", "alerts", "pipelines"]
        results = await _reload_sections(ordered)
        return {"section": "all", "results": results}
    results = await _reload_sections(sections)
    return results[0]
