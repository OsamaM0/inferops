"""FastAPI app factory."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from inferops import __version__
from inferops.alerts import autoload as autoload_alerts
from inferops.api.routes import (
    configs,
    events,
    evidence_browser,
    health,
    live,
    models,
    pipelines,
    streams,
    system,
    ws_live,
)
from inferops.core.config import settings
from inferops.fusion import autoload as autoload_fusion
from inferops.obs import get_logger, init_tracing
from inferops.pipeline import autoload as autoload_pipelines
from inferops.pipeline import get_pipeline_store
from inferops.registry import autoload as autoload_models
from inferops.registry import get_registry

log = get_logger(__name__)


def _env_names(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    return [part.strip() for part in raw.split(",") if part.strip()]


async def _prewarm_runtime() -> None:
    model_names = _env_names("INFEROPS_PREWARM_MODELS")
    if model_names:
        registry = get_registry()
        for model_name in model_names:
            try:
                await registry.get_backend(model_name)
                log.info("model_prewarmed", model=model_name)
            except Exception as exc:
                log.warning("model_prewarm_failed", model=model_name, err=str(exc))

    pipeline_names = _env_names("INFEROPS_PREWARM_PIPELINES")
    if pipeline_names:
        store = get_pipeline_store()
        for pipeline_name in pipeline_names:
            try:
                await store.get_runner(pipeline_name)
                log.info("pipeline_prewarmed", pipeline=pipeline_name)
            except Exception as exc:
                log.warning("pipeline_prewarm_failed", pipeline=pipeline_name, err=str(exc))


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_tracing()
    await autoload_models()
    await autoload_pipelines()
    await autoload_fusion()
    await autoload_alerts()
    await _prewarm_runtime()
    # init DB if configured
    try:
        from inferops.adapters.postgres_dao import init_schema
        await init_schema()
    except Exception as e:
        log.warning("pg_init_skipped", err=str(e))
    log.info("inferops_started", version=__version__)
    yield
    log.info("inferops_stopping")


def create_app() -> FastAPI:
    app = FastAPI(title="InferOps", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.api.cors_origins.split(",")],
        allow_methods=["*"], allow_headers=["*"],
    )
    app.include_router(health.router, tags=["health"])
    app.include_router(configs.router, prefix="/v1", tags=["configs"])
    app.include_router(models.router, prefix="/v1", tags=["models"])
    app.include_router(pipelines.router, prefix="/v1", tags=["pipelines"])
    app.include_router(streams.router, prefix="/v1", tags=["streams"])
    app.include_router(system.router, prefix="/v1", tags=["system"])
    app.include_router(events.router, prefix="/v1", tags=["events"])
    app.include_router(evidence_browser.router, prefix="/v1", tags=["evidence"])
    app.include_router(ws_live.router, prefix="/v1", tags=["ws"])
    app.include_router(live.router, prefix="/v1", tags=["live"])

    @app.get("/")
    async def root():
        return {"name": "inferops", "version": __version__,
                "docs": "/docs", "metrics": "/metrics"}

    return app


app = create_app()
