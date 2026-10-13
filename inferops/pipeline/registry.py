"""Built-in stage alias registry."""
from __future__ import annotations

from typing import Any

from inferops.core.plugins import load_object

BUILTINS: dict[str, str] = {
    # sources
    "source_rest": "inferops.pipeline.stages.source_rest:SourceREST",
    "source_folder": "inferops.pipeline.stages.source_folder:SourceFolder",
    "source_rtsp": "inferops.pipeline.stages.source_rtsp:SourceRTSP",
    "source_mjpeg": "inferops.pipeline.stages.source_mjpeg:SourceMJPEG",
    "source_kafka": "inferops.pipeline.stages.source_kafka:SourceKafka",
    # transforms
    "decode_any": "inferops.pipeline.stages.decode_any:DecodeAny",
    "decode_image": "inferops.pipeline.stages.decode_image:DecodeImage",
    "decode_video": "inferops.pipeline.stages.decode_video:DecodeVideo",
    "preprocess_generic": "inferops.pipeline.stages.preprocess_generic:PreprocessGeneric",
    "infer": "inferops.pipeline.stages.infer:Infer",
    "postprocess_passthrough": "inferops.pipeline.stages.postprocess_passthrough:PostprocessPassthrough",
    "aggregate_window": "inferops.pipeline.stages.aggregate_window:AggregateWindow",
    "route_by": "inferops.pipeline.stages.route_by:RouteBy",
    "fuse": "inferops.pipeline.stages.fuse:Fuse",
    "alert": "inferops.pipeline.stages.alert:Alert",
    "track": "inferops.pipeline.stages.track:Track",
    # sinks
    "sink_rest": "inferops.pipeline.stages.sink_rest:SinkREST",
    "sink_ws": "inferops.pipeline.stages.sink_ws:SinkWS",
    "sink_kafka": "inferops.pipeline.stages.sink_kafka:SinkKafka",
    "sink_postgres": "inferops.pipeline.stages.sink_postgres:SinkPostgres",
    "sink_s3": "inferops.pipeline.stages.sink_s3:SinkS3",
    "sink_webhook": "inferops.pipeline.stages.sink_webhook:SinkWebhook",
    "sink_file": "inferops.pipeline.stages.sink_file:SinkFile",
    "sink_stdout": "inferops.pipeline.stages.sink_stdout:SinkStdout",
}


def resolve_stage(name: str) -> Any:
    return load_object(BUILTINS.get(name, name))
