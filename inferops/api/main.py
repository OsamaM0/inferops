"""FastAPI app factory."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response

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
from inferops.api.ui import mount_ui
from inferops.core.config import get_config_stack, settings
from inferops.core.errors import BackendError, PipelineError
from inferops.fusion import autoload as autoload_fusion
from inferops.obs import get_logger, init_tracing
from inferops.pipeline import autoload as autoload_pipelines
from inferops.pipeline import get_pipeline_store
from inferops.registry import autoload as autoload_models
from inferops.registry import get_registry

log = get_logger(__name__)


def _env_names(name: str, default: list[str] | None = None) -> list[str]:
    if name not in os.environ:
        return list(default or [])
    raw = os.environ.get(name, "")
    return [part.strip() for part in raw.split(",") if part.strip()]


async def _prewarm_runtime() -> None:
    # The domain manifest is the default; INFEROPS_PREWARM_* overrides it.
    manifest = get_config_stack(settings).manifest()
    model_names = _env_names("INFEROPS_PREWARM_MODELS", manifest.prewarm_models if manifest else None)
    if "INFEROPS_PREWARM_MODELS" not in os.environ:
        # PipelineStore has already selected each stage's model for this mode.
        # Derive primaries from that surface so new domain packs work too.
        model_names = list(dict.fromkeys(model_names + [
            stage.params["model"]
            for pipeline in get_pipeline_store().list()
            for stage in ([pipeline.source] if pipeline.source else []) + pipeline.stages + pipeline.sinks
            if stage.params.get("model")
        ]))
    if model_names:
        registry = get_registry()
        for model_name in model_names:
            try:
                await registry.get_backend(model_name)
                log.info("model_prewarmed", model=model_name)
            except Exception as exc:
                log.warning("model_prewarm_failed", model=model_name, err=str(exc))
                raise

    pipeline_names = _env_names("INFEROPS_PREWARM_PIPELINES", manifest.prewarm_pipelines if manifest else None)
    if pipeline_names:
        store = get_pipeline_store()
        for pipeline_name in pipeline_names:
            try:
                await store.get_runner(pipeline_name)
                log.info("pipeline_prewarmed", pipeline=pipeline_name)
            except Exception as exc:
                log.warning("pipeline_prewarm_failed", pipeline=pipeline_name, err=str(exc))
                raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_tracing()
    from inferops.core.preparation import selected_surface

    selected_surface(get_config_stack(settings), settings.execution_mode)
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
    try:
        yield
    finally:
        # Startup built pipeline runners and model backends; shutdown used to do
        # nothing at all, so ONNX sessions, Triton gRPC channels, Kafka
        # producers, S3 clients and buffered sink output were dropped on the
        # floor on every restart. Tear down in reverse order of construction:
        # runners hold references to backends, so they must go first.
        log.info("inferops_stopping")
        for name, close in (
            ("pipelines", get_pipeline_store().close),
            ("models", get_registry().close),
        ):
            try:
                await close()
            except Exception as exc:
                # One failing subsystem must not prevent the rest from closing.
                log.warning("shutdown_close_failed", subsystem=name, err=str(exc))
        log.info("inferops_stopped")


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

    async def runtime_error(request, exc):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    app.add_exception_handler(BackendError, runtime_error)
    app.add_exception_handler(PipelineError, runtime_error)

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return Response(status_code=204)

    mount_ui(app)

    @app.get("/")
    async def root(request: Request):
        if "text/html" in request.headers.get("accept", ""):
            return RedirectResponse("/ui/", status_code=307)
        return {"name": "inferops", "version": __version__,
                "docs": "/docs", "frontend": "/frontend/", "ui": "/ui/",
                "interfaces": "/v1/interfaces", "guide": "/ui/guide", "metrics": "/metrics"}

    return app


app = create_app()
