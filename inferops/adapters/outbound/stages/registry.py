"""Built-in stage alias registry."""
from __future__ import annotations

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


def resolve_stage(name: str) -> Any:
    if name in APPLICATION_STAGES:
        raise ValueError(f"{name} is owned by the shared application services")
    return load_object(BUILTINS.get(name, name))
