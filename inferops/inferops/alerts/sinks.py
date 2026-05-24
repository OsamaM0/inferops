"""Alert sink ABC + simple implementations."""
from __future__ import annotations

import orjson

from inferops.alerts.spec import Alert, SinkSpec
from inferops.obs import ALERTS_EMITTED, get_logger

log = get_logger(__name__)


class AlertSink:
    def __init__(self, spec: SinkSpec) -> None:
        self.spec = spec

    async def setup(self) -> None: ...
    async def emit(self, alert: Alert) -> None: ...
    async def close(self) -> None: ...


class StdoutSink(AlertSink):
    async def emit(self, alert):
        print(orjson.dumps({"alert": alert.model_dump(mode="json"), "sink": "stdout"}).decode())


class WebhookSink(AlertSink):
    async def setup(self):
        import httpx
        self._client = httpx.AsyncClient(timeout=5.0)

    async def emit(self, alert):
        try:
            await self._client.post(self.spec.params["url"], json=alert.model_dump(mode="json"))
        except Exception as e:
            log.warning("alert_webhook_failed", err=str(e))
            raise

    async def close(self):
        await self._client.aclose()


class KafkaSink(AlertSink):
    async def setup(self):
        from inferops.adapters.transport import get_transport
        self._prod = await get_transport().producer(f"alerts-{self.spec.name}")

    async def emit(self, alert):
        topic = self.spec.params.get("topic", "inferops.alerts")
        await self._prod.send(topic, orjson.dumps(alert.model_dump(mode="json")),
                              key=alert.entity_id.encode())

    async def close(self):
        await self._prod.close()


class WSSink(AlertSink):
    async def emit(self, alert):
        from inferops.pipeline.stages.sink_ws import HUB
        ch = self.spec.params.get("channel", "alerts").format(
            entity_id=alert.entity_id, type=alert.type, severity=alert.severity)
        await HUB.publish(ch, alert.model_dump(mode="json"))


class PostgresSink(AlertSink):
    async def emit(self, alert):
        from inferops.adapters.postgres_dao import insert_alert
        await insert_alert({
            "id": alert.id, "type": alert.type, "entity_id": alert.entity_id,
            "severity": alert.severity, "dedupe_key": alert.dedupe_key,
            "status": alert.status, "ts_ms": alert.ts_ms,
            "payload": alert.payload,
        })


SinkRegistry = {
    "stdout": StdoutSink,
    "webhook": WebhookSink,
    "kafka": KafkaSink,
    "ws": WSSink,
    "postgres": PostgresSink,
    "slack": WebhookSink,  # treat as webhook with slack-shaped payload
}


def build_sink(spec: SinkSpec) -> AlertSink:
    if spec.kind == "plugin":
        from inferops.core.plugins import load_object
        cls = load_object(spec.params["target"])
        return cls(spec)
    cls = SinkRegistry.get(spec.kind)
    if cls is None:
        raise ValueError(f"unknown alert sink kind: {spec.kind}")
    return cls(spec)


def metric_emit(alert: Alert, sink_name: str, result: str) -> None:
    ALERTS_EMITTED.labels(type=alert.type, severity=alert.severity,
                          sink=sink_name, result=result).inc()
