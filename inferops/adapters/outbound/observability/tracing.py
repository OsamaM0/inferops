"""Optional tracing; SDK initialization occurs only at explicit server startup."""
from __future__ import annotations

from contextlib import nullcontext

_initialised = False


class _NoopTracer:
    def start_as_current_span(self, *args, **kwargs):
        return nullcontext()


def init_tracing(*, endpoint=None, service_name="inferops"):
    global _initialised
    if _initialised or not endpoint:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as error:
        raise RuntimeError("tracing endpoint requires inferops[tracing]") from error
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True)))
    trace.set_tracer_provider(provider)
    _initialised = True


def tracer(name="inferops"):
    try:
        from opentelemetry import trace
    except ImportError:
        return _NoopTracer()
    return trace.get_tracer(name)
