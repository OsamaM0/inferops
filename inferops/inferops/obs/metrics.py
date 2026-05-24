"""Prometheus metrics registry."""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry(auto_describe=True)

STAGE_LATENCY = Histogram(
    "inferops_stage_latency_seconds", "Per-stage processing latency",
    ["pipeline", "stage"], registry=REGISTRY,
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
INFER_LATENCY = Histogram(
    "inferops_infer_latency_seconds", "Inference latency",
    ["model"], registry=REGISTRY,
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
INFER_BATCH = Histogram(
    "inferops_infer_batch_size", "Batch size per inference call",
    ["model"], registry=REGISTRY,
    buckets=(1, 2, 4, 8, 16, 32, 64),
)
QUEUE_DEPTH = Gauge(
    "inferops_queue_depth", "Pipeline edge queue depth",
    ["pipeline", "edge"], registry=REGISTRY,
)
ERRORS = Counter(
    "inferops_errors_total", "Errors by kind",
    ["stage", "kind"], registry=REGISTRY,
)
TRANSPORT_PRODUCE = Counter(
    "inferops_transport_produce_total", "Messages produced",
    ["topic", "result"], registry=REGISTRY,
)
TRANSPORT_CONSUME = Counter(
    "inferops_transport_consume_total", "Messages consumed",
    ["topic", "group"], registry=REGISTRY,
)
ALERTS_EMITTED = Counter(
    "inferops_alerts_total", "Alerts fanned out",
    ["type", "severity", "sink", "result"], registry=REGISTRY,
)
FUSION_EVENTS = Counter(
    "inferops_fusion_events_total", "Fusion events emitted",
    ["fusion", "type"], registry=REGISTRY,
)
PIPELINE_RUNS = Counter(
    "inferops_pipeline_runs_total", "Pipeline executions",
    ["pipeline", "result"], registry=REGISTRY,
)


def metrics_text() -> bytes:
    from prometheus_client import generate_latest
    return generate_latest(REGISTRY)
