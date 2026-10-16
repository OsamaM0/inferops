"""OpenTelemetry tracing setup (no-op if endpoint not configured)."""
from __future__ import annotations

from opentelemetry import trace

_initialised = False


def init_tracing() -> None:
    from inferops.adapters.outbound.projects.settings import settings
    global _initialised
    if _initialised:
        return
    if not settings.obs.otel_endpoint:
        return
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": settings.obs.service_name}))
    if settings.obs.otel_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.obs.otel_endpoint, insecure=True)))
        except Exception:
            pass
    trace.set_tracer_provider(provider)
    _initialised = True


def tracer(name: str = "inferops") -> trace.Tracer:
    init_tracing()
    return trace.get_tracer(name)
