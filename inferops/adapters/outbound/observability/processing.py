"""Instrumentation of application completion without inward infrastructure imports."""
from inferops.adapters.outbound.observability.metrics import ALERTS_EMITTED, FUSION_EVENTS, PIPELINE_RUNS


class ProcessingObserver:
    def completed(self, pipeline, result):
        PIPELINE_RUNS.labels(pipeline=pipeline, result=result).inc()

    def accepted(self, items):
        for item in items:
            for event in item.payload.get("fusion_events", []):
                FUSION_EVENTS.labels(fusion=event.get("fusion", "application"), type=event["type"]).inc()
            for alert in item.payload.get("alerts", []):
                ALERTS_EMITTED.labels(type=alert["type"], severity=alert["severity"], sink="accepted", result="ok").inc()
