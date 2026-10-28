"""Built-in stage alias registry."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from inferops.adapters.outbound.projects.plugins import load_object

BUILTINS: dict[str, str] = {
    # sources
    "source_rest": "inferops.adapters.outbound.stages.source_rest:SourceREST",
    "source_folder": "inferops.adapters.outbound.stages.source_folder:SourceFolder",
    "source_rtsp": "inferops.adapters.outbound.stages.source_rtsp:SourceRTSP",
    "source_mjpeg": "inferops.adapters.outbound.stages.source_mjpeg:SourceMJPEG",
    # transforms
    "decode_any": "inferops.adapters.outbound.stages.decode_any:DecodeAny",
    "decode_image": "inferops.adapters.outbound.stages.decode_image:DecodeImage",
    "decode_video": "inferops.adapters.outbound.stages.decode_video:DecodeVideo",
    "preprocess_generic": "inferops.adapters.outbound.stages.preprocess_generic:PreprocessGeneric",
    "infer": "inferops.adapters.outbound.stages.infer:Infer",
    "postprocess_passthrough": "inferops.adapters.outbound.stages.postprocess_passthrough:PostprocessPassthrough",
    "aggregate_window": "inferops.adapters.outbound.stages.aggregate_window:AggregateWindow",
    "route_by": "inferops.adapters.outbound.stages.route_by:RouteBy",
    "track": "inferops.adapters.outbound.stages.track:Track",
    # sinks
    "sink_file": "inferops.adapters.outbound.stages.sink_file:SinkFile",
    "sink_stdout": "inferops.adapters.outbound.stages.sink_stdout:SinkStdout",
}

# These configuration names select application services, not independently
# constructed stages. Existing YAML names remain accepted during migration.
APPLICATION_STAGES = frozenset({"fuse", "alert", "model_group", "source_kafka",
    "evidence", "sink_s3", "sink_postgres", "sink_kafka", "sink_ws", "sink_webhook", "sink_rest"})


def _parameters(properties=None, required=()):
    return {"type": "object", "properties": properties or {}, "required": list(required),
            "additionalProperties": False}


def _reference(section):
    return {"type": "string", "minLength": 1, "x-reference": section}


# Configuration contracts are data only: schema discovery never imports a stage
# or opens a camera, model backend, broker connection or project plugin.
_TEXT = {"type": "string"}
_PARAMETERS = {
    "source_rest": _parameters(),
    "source_folder": _parameters({"path": _TEXT, "glob": {"type": "string", "default": "*.jpg"},
        "interval_ms": {"type": "integer", "minimum": 0, "default": 0}, "stream_id": _TEXT}, ("path",)),
    "source_rtsp": _parameters({"url": _TEXT, "stream_id": _TEXT,
        "loop": {"type": "boolean", "default": False},
        "fps": {"type": "number", "minimum": 0, "default": 0},
        "transport": {"type": "string", "enum": ["tcp", "udp"], "default": "tcp"},
        "timeout_s": {"type": "number", "exclusiveMinimum": 0, "default": 5},
        "reconnect_s": {"type": "number", "exclusiveMinimum": 0, "default": 0.5},
        "latency_budget_ms": {"type": "integer", "minimum": 0, "default": 100}}, ("url",)),
    "source_mjpeg": _parameters({"url": _TEXT, "stream_id": _TEXT,
        "fps": {"type": "number", "minimum": 0, "default": 5},
        "timeout_s": {"type": "number", "exclusiveMinimum": 0, "default": 5}}, ("url",)),
    "source_kafka": _parameters({"topic": _TEXT}, ("topic",)),
    "decode_any": _parameters({"every_n": {"type": "integer", "minimum": 1, "default": 1}}),
    "decode_image": _parameters(),
    "decode_video": _parameters({"every_n": {"type": "integer", "minimum": 1, "default": 1}}),
    "infer": _parameters({"model": _reference("models"),
        "fallback_model": {**_reference("models"), "deprecated": True},
        "model_modes": {"type": "object", "additionalProperties": _reference("models"), "deprecated": True}}, ("model",)),
    "preprocess_generic": _parameters({"model": _reference("models")}),
    "postprocess_passthrough": _parameters(),
    "aggregate_window": _parameters({"window_ms": {"type": "integer", "default": 5000},
        "key": {"type": "string", "default": "stream_id"}}),
    "route_by": _parameters({"when": {"type": "string", "default": "True"}}),
    "model_group": _parameters({"models": {"type": "array", "minItems": 1,
        "items": _reference("models")}}, ("models",)),
    "fuse": _parameters({"fusion": _reference("fusion")}, ("fusion",)),
    "alert": _parameters({"alerts": _reference("alerts")}, ("alerts",)),
    "evidence": _parameters({"annotator": _TEXT,
        "annotator_params": {"type": "object", "additionalProperties": True}}),
    "sink_s3": _parameters({"annotator": _TEXT,
        "annotator_params": {"type": "object", "additionalProperties": True}}),
    "sink_postgres": _parameters(),
    "sink_ws": _parameters({"channel": {"type": "string", "default": "live.{stream_id}"}}),
    "sink_kafka": _parameters({"topic": _TEXT}, ("topic",)),
    "sink_file": _parameters({"path": _TEXT}, ("path",)),
    "sink_stdout": _parameters(),
    "sink_webhook": _parameters({"url": _TEXT}, ("url",)),
    "sink_rest": _parameters({"url": _TEXT}, ("url",)),
}
_LABELS = {
    "source_rest": "Registered sources / API", "source_folder": "Read a folder",
    "source_rtsp": "Read an RTSP stream or server video", "source_mjpeg": "Read an MJPEG stream",
    "source_kafka": "Read from Kafka", "decode_any": "Decode images or video",
    "decode_image": "Decode an image", "decode_video": "Decode a video",
    "infer": "Run a model", "model_group": "Run independent models",
    "preprocess_generic": "Prepare model input", "postprocess_passthrough": "Pass through results",
    "aggregate_window": "Aggregate over time", "route_by": "Filter with an expression",
    "track": "Track objects", "fuse": "Evaluate signals and rules", "alert": "Apply alert policy",
    "evidence": "Capture incident evidence", "sink_s3": "Capture evidence (legacy)",
    "sink_postgres": "Persist results", "sink_ws": "Publish live results",
    "sink_kafka": "Publish to Kafka", "sink_file": "Write to a file",
    "sink_stdout": "Write to console", "sink_webhook": "Send to a webhook",
    "sink_rest": "Send to a webhook (legacy)",
}


def stage_configuration() -> dict[str, dict]:
    """Return editable parameter schemas for built-in and application stages."""
    from inferops.domain.tracking import TrackingSpec

    parameters = deepcopy(_PARAMETERS)
    tracking = TrackingSpec.model_json_schema()
    for key in ("enabled", "insert_after"):
        tracking["properties"].pop(key, None)
    tracking["properties"]["model_filter"]["items"]["x-reference"] = "models"
    parameters["track"] = tracking
    return {name: {"label": _LABELS.get(name, name.replace("_", " ").title()),
                   "category": "source" if name.startswith("source_") else
                       "sink" if name.startswith("sink_") or name == "evidence" else "processor",
                   "deprecated": name in {"sink_rest", "sink_s3"},
                   "params": parameters.get(name, {"type": "object", "additionalProperties": True})}
            for name in sorted(set(BUILTINS) | APPLICATION_STAGES)}


def required_stage_parameters() -> dict[str, list[str]]:
    """The loader and editor share required string parameter names."""
    return {name: [key for key in schema["required"] if schema["properties"][key].get("type") == "string"]
            for name, schema in _PARAMETERS.items()}


def resolve_stage(name: str) -> Any:
    if name in APPLICATION_STAGES:
        raise ValueError(f"{name} is owned by the shared application services")
    return load_object(BUILTINS.get(name, name))
