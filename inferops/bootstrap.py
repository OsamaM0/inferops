"""The sole construction boundary for the new engine; no import-time setup."""
from __future__ import annotations

import os
from pathlib import Path


def build_engine(root, *, runtime="cpu", configs_dir="configs", repository=None, storage=None,
                 evidence=False, spool_path=None, evidence_config=None, transport=None, delivery=None,
                 definition=None, **options):
    max_sessions = options.pop("max_sessions", 128)
    idle_timeout_s = options.pop("idle_timeout_s", 300)
    model_concurrency = options.pop("model_concurrency", 1)
    recover_sessions = options.pop("recover_sessions", False)
    prewarm_models = options.pop("prewarm_models", ())
    from inferops.adapters.inbound.sources import source_frames
    from inferops.adapters.outbound.projects.loader import ProjectLoader
    from inferops.adapters.outbound.stages.factory import stage_factory
    from inferops.application.engine import PipelineApplication
    from inferops.application.models import ModelPool
    from inferops.public import Engine

    definition = definition or ProjectLoader(root, configs_dir=configs_dir, mode=runtime, **options).load()
    if repository is None:
        from inferops.adapters.outbound.persistence.memory import MemoryRepository
        repository = MemoryRepository()
    captures = None
    if evidence:
        if storage is None or spool_path is None:
            raise ValueError("evidence capture requires explicit storage and a persistent spool path")
        from inferops.adapters.outbound.media.encoder import OpenCVMediaEncoder
        from inferops.adapters.outbound.media.spool import CrashSafeSpool
        from inferops.application.evidence import EvidenceService
        from inferops.domain.evidence import EvidenceConfig
        config = evidence_config or EvidenceConfig()
        captures = EvidenceService(repository, storage, CrashSafeSpool(
            spool_path, max_bytes=config.max_spool_bytes, disk_reserve_bytes=config.disk_reserve_bytes),
            config=config, encoder=OpenCVMediaEncoder())
    annotators = build_annotators(definition)
    application = PipelineApplication(definition, ModelPool(definition, max_concurrency=model_concurrency), repository,
                                      stage_factory=stage_factory(definition), source_factory=source_frames,
                                      evidence=captures, delivery=delivery, transport=transport, annotators=annotators,
                                      max_sessions=max_sessions, idle_timeout_s=idle_timeout_s,
                                      recover_sessions=recover_sessions, prewarm_models=prewarm_models)
    return Engine(application)


def build_annotators(definition):
    annotators = {}
    for pipeline in definition.pipelines.values():
        for sink in pipeline.sinks:
            reference = sink.params.get("annotator")
            if reference:
                annotators[pipeline.name] = definition.object_loader(reference)(**sink.params.get("annotator_params", {}))
    return annotators


def build_server_engine(*, settings=None, hub=None):
    # Environment and .env reads happen only when a server explicitly starts.
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(), override=False)
    from inferops.adapters.outbound.messaging.delivery import DeliveryHub, ServerDelivery
    from inferops.adapters.outbound.persistence.sql import SqlRepository
    from inferops.adapters.outbound.storage.s3 import S3Storage
    from inferops.domain.evidence import EvidenceConfig

    if settings is None:
        from inferops.adapters.outbound.projects.settings import load_settings
        settings = load_settings()
    root = settings.domain_root
    artifact_root = Path(settings.model_repository).expanduser().resolve() if root is None else None
    transport = None
    if settings.transport.kind == "redpanda":
        from inferops.adapters.outbound.messaging.redpanda import RedpandaTransport
        transport = RedpandaTransport(bootstrap=settings.transport.bootstrap,
                                     client_id=settings.transport.client_id,
                                     max_value_bytes=settings.transport.max_value_bytes)
    else:
        from inferops.adapters.outbound.messaging.inproc import InProcTransport
        transport = InProcTransport(max_value_bytes=settings.transport.max_value_bytes)
    hub = hub or DeliveryHub()
    storage = S3Storage(endpoint=settings.s3.endpoint, access_key=settings.s3.access_key,
                        secret_key=settings.s3.secret_key, bucket=settings.s3.bucket, region=settings.s3.region)
    config_fields = EvidenceConfig.__dataclass_fields__
    values = {}
    for name in config_fields:
        value = os.environ.get(f"INFEROPS_EVIDENCE_{name.upper()}")
        if value is not None:
            if name == "retry_delays":
                values[name] = tuple(float(v) for v in value.split(","))
            else:
                default = getattr(EvidenceConfig(), name)
                values[name] = type(default)(value)
    engine = build_engine(root, runtime=settings.execution_mode, configs_dir=settings.configs_dir,
                          repository=SqlRepository(settings.pg.url), storage=storage, evidence=True,
                          spool_path=Path(os.environ.get("INFEROPS_SPOOL_PATH", ".run/spool")),
                          evidence_config=EvidenceConfig(**values), transport=transport,
                          triton_url=settings.triton.url, artifact_root=artifact_root, recover_sessions=True,
                          max_sessions=int(os.environ.get("INFEROPS_MAX_SESSIONS", "128")),
                          idle_timeout_s=float(os.environ.get("INFEROPS_SESSION_IDLE_SECONDS", "300")),
                          model_concurrency=int(os.environ.get("INFEROPS_MODEL_CONCURRENCY", "1")))
    definition = engine.application.definition
    names = {stage.params["model"] for pipeline in definition.pipelines.values() for stage in pipeline.stages
             if stage.params.get("model")}
    names.update(model for pipeline in definition.pipelines.values() for stage in pipeline.stages
                 for model in stage.params.get("models", []))
    if definition.manifest:
        names.update(definition.manifest.prewarm_models)
    if "INFEROPS_PREWARM_MODELS" in os.environ:
        names = {name.strip() for name in os.environ["INFEROPS_PREWARM_MODELS"].split(",") if name.strip()}
    engine.application.prewarm_models = tuple(sorted(names))
    engine.application.delivery = ServerDelivery(hub, transport=transport, definitions=engine.application.definition)
    engine.http_options = {"cors_origins": [origin.strip() for origin in settings.api.cors_origins.split(",")
                                             if origin.strip()]}
    return engine, hub


def build_model_pool(spec, *, base_dir=None, backend_factory=None, object_loader=None):
    """Construct standalone inference with the same service as pipeline engines."""
    from types import MappingProxyType

    from inferops.adapters.outbound.projects.loader import BackendFactory, ProjectPluginResolver
    from inferops.application.models import ModelPool
    from inferops.ports.projects import ProjectDefinition

    spec = spec.model_copy(deep=True)
    base_dir = Path(base_dir).resolve() if base_dir else None
    if spec.artifact.path and base_dir and not Path(spec.artifact.path).is_absolute():
        spec.artifact.path = str((base_dir / spec.artifact.path).resolve())
    plugins = ProjectPluginResolver(None, "standalone")
    mode = "cuda" if spec.device.preferred == "gpu" else "cpu"
    definition = ProjectDefinition(
        root=None, name="standalone", mode=mode, revision="standalone",
        models=MappingProxyType({spec.name: spec}), pipelines=MappingProxyType({}),
        fusion=MappingProxyType({}), alerts=MappingProxyType({}),
        source_dirs=MappingProxyType({spec.name: base_dir}), plugin_resolver=plugins,
        object_loader=object_loader or plugins.load_object,
        backend_factory=backend_factory or BackendFactory(plugins, mode, "localhost:8001", {}),
    )
    return ModelPool(definition)


def build_model_session(root, *, models=None, runtime="cpu"):
    from inferops.adapters.outbound.projects.loader import ProjectLoader
    from inferops.application.models import ModelPool

    definition = ProjectLoader(root, mode=runtime).load()
    resolver = definition.resolver
    grouped = {variant for variants in resolver.variants.values() for variant in variants}
    names = models if models is not None else [*resolver.ports, *sorted(resolver.names - grouped)]
    return ModelPool(definition), {name: definition.resolve_model(name) for name in names}
