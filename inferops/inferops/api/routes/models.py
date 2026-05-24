"""Models registry routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from inferops.api.deps import registry
from inferops.core.config import get_spec_dir, settings
from inferops.core.errors import RegistryError
from inferops.pipeline import get_pipeline_store

router = APIRouter()
RegistryDep = Depends(registry)


@router.get("/models")
async def list_models(reg = RegistryDep):
    return [m.model_dump(mode="json") for m in reg.list_models()]


@router.get("/models/{name}")
async def get_model(name: str, reg = RegistryDep):
    try:
        return reg.get_spec(name).model_dump(mode="json")
    except RegistryError as e:
        raise HTTPException(404, str(e)) from e


@router.post("/models/reload")
async def reload_models(reg = RegistryDep):
    d = get_spec_dir("models", settings)
    await get_pipeline_store().close()
    n = await reg.reload_from(d) if d.exists() else 0
    return {"reloaded": n, "dir": str(d)}
