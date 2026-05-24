"""OpenTelemetry tracing setup (no-op if endpoint not configured)."""
from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from inferops.core.config import settings

_initialised = False


def init_tracing() -> None:
    global _initialised
    if _initialised:
        return
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
