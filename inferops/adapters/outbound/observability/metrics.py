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

# ── Real-time streaming / live infer / tracking ──────────────────────────────
E2E_LATENCY = Histogram(
    "inferops_e2e_latency_seconds",
    "End-to-end latency from frame arrival to first WS push",
    ["pipeline"], registry=REGISTRY,
    buckets=(0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.5),
)
STREAMS_ACTIVE = Gauge(
    "inferops_streams_active", "Number of currently-active live streams",
    ["pipeline", "source_type"], registry=REGISTRY,
)
STREAM_FPS_IN = Gauge(
    "inferops_stream_fps_in", "Inbound frames per second per stream",
    ["stream_id"], registry=REGISTRY,
)
STREAM_FPS_OUT = Gauge(
    "inferops_stream_fps_out", "Outbound (processed) frames per second per stream",
    ["stream_id"], registry=REGISTRY,
)
FRAMES_DROPPED = Counter(
    "inferops_frames_dropped_total", "Frames dropped due to back-pressure",
    ["pipeline", "reason"], registry=REGISTRY,
)
TRACKS_ACTIVE = Gauge(
    "inferops_tracks_active", "Currently-active object tracks",
    ["pipeline", "label"], registry=REGISTRY,
)
TRACKS_STARTED = Counter(
    "inferops_tracks_started_total", "Tracks born this process lifetime",
    ["pipeline", "label"], registry=REGISTRY,
)
TRACKS_CLOSED = Counter(
    "inferops_tracks_closed_total", "Tracks closed (lost) this process lifetime",
    ["pipeline", "label"], registry=REGISTRY,
)
CLASSIFICATIONS = Counter(
    "inferops_classifications_total", "Classification predictions emitted",
    ["model", "label"], registry=REGISTRY,
)
DETECTIONS = Counter(
    "inferops_detections_total", "Detection predictions emitted",
    ["pipeline", "label"], registry=REGISTRY,
)
EVIDENCE_UPLOADS = Counter(
    "inferops_evidence_uploads_total", "Evidence artifacts uploaded",
    ["bucket", "kind", "result"], registry=REGISTRY,
)
EVIDENCE_UPLOAD_LATENCY = Histogram(
    "inferops_evidence_upload_seconds", "Evidence upload latency",
    ["bucket", "kind"], registry=REGISTRY,
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
WS_CLIENTS = Gauge(
    "inferops_websocket_clients", "Connected WebSocket clients per channel",
    ["channel"], registry=REGISTRY,
)


def metrics_text() -> bytes:
    from prometheus_client import generate_latest
    return generate_latest(REGISTRY)
