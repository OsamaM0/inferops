"""Activate a compiled revision while existing sessions retain their resources."""
from __future__ import annotations


def validate_server_revision(application, definition):
    if not getattr(application, "require_durable_configuration", False):
        return
    from inferops.domain.errors import ConfigError
    if any(not spec.persistence_enabled for spec in (*definition.alerts.values(), *definition.fusion.values())):
        raise ConfigError("persistence_enabled=false is unsupported by the durable server")
    declarations = [decl for pipe in definition.pipelines.values()
                    for decl in ([pipe.source] if pipe.source else []) + pipe.stages + pipe.sinks]
    kafka = any(decl.stage in {"source_kafka", "sink_kafka"} for decl in declarations)
    kafka |= any(sink.kind == "kafka" for spec in definition.alerts.values() for sink in spec.sinks)
    if kafka and not application.broker_enabled:
        raise ConfigError("Kafka sources/sinks require the broker transport")


async def activate_revision(application, *, definition, models, stage_factory, annotators):
    validate_server_revision(application, definition)
    previous = application.models
    application.definition = definition
    application.models = models
    application.stage_factory = stage_factory
    application.annotators = annotators
    if application.delivery is not None and hasattr(application.delivery, "definitions"):
        application.delivery.definitions = definition
    application.retired_models.append(previous)
    await application.collect_retired_models()
