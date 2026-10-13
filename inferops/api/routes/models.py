"""Models registry routes."""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException

from inferops.api.deps import registry
from inferops.core.config import get_config_stack, get_spec_dir, settings
from inferops.core.errors import ConfigError, RegistryError
from inferops.core.validation import validate_config_stack
from inferops.pipeline import get_pipeline_store

router = APIRouter()
RegistryDep = Depends(registry)


@router.get("/models")
async def list_models(view: Literal["variants", "logical"] = "variants", reg = RegistryDep):
    if view == "logical":
        return reg.logical_models()
    return [m.model_dump(mode="json") for m in reg.list_models()]


@router.get("/models/{name}")
async def get_model(name: str, reg = RegistryDep):
    try:
        return reg.get_spec(name).model_dump(mode="json")
    except RegistryError as e:
        raise HTTPException(404, str(e)) from e


@router.post("/models/reload")
async def reload_models(reg = RegistryDep):
    try:
        validate_config_stack(get_config_stack(settings))
    except ConfigError as exc:
        raise HTTPException(400, str(exc)) from exc
    d = get_spec_dir("models", settings)
    await get_pipeline_store().close()
    n = await reg.reload_from(d) if d.exists() else 0
    return {"reloaded": n, "dir": str(d)}
