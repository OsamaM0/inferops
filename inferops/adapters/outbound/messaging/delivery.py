from __future__ import annotations

import json
import logging
from collections import defaultdict

log = logging.getLogger(__name__)


class DeliveryHub:
    def __init__(self):
        self.subscribers = defaultdict(set)

    def subscribe(self, channel, queue):
        self.subscribers[channel].add(queue)

    def unsubscribe(self, channel, queue):
        self.subscribers[channel].discard(queue)
        if not self.subscribers[channel]:
            self.subscribers.pop(channel, None)

    async def publish(self, channel, payload):
        for queue in tuple(self.subscribers.get(channel, ())):
            if queue.full():
                queue.get_nowait()  # Notifications are hints; authoritative records remain queryable.
            queue.put_nowait(payload)


class ServerDelivery:
    def __init__(self, hub, *, transport=None, definitions=None):
        self.hub, self.transport = hub, transport
        self.definitions = definitions
        self.producer = None
        self.client = None
        self.plugins = {}

    async def publish(self, topic, key, payload, event_id):
        if topic == "delivery.webhook":
            if self.client is None:
                import httpx
                self.client = httpx.AsyncClient(timeout=5)
            response = await self.client.post(payload["url"], json=payload["body"], headers={"Idempotency-Key": event_id})
            response.raise_for_status()
            return
        if self.transport:
            if self.producer is None:
                self.producer = await self.transport.producer("outbox")
            await self.producer.send(topic, json.dumps({**payload, "event_id": event_id}).encode(), key=key.encode())
        if topic == "incident.accepted":
            alert = payload.get("alert", payload)
            await self.hub.publish(f"alerts.{key}", alert)
            # Legacy external stream channels remain notification-only aliases.
            stream = alert.get("payload", {}).get("stream_name")
            if stream:
                await self.hub.publish(f"alerts.{stream}", alert)
            if self.definitions:
                from inferops.domain.incident_spec import AlertSpec
                policies = ([AlertSpec.model_validate(payload["delivery_policy"])]
                            if payload.get("delivery_policy") else self.definitions.alerts.values())
                for spec in policies:
                    if payload.get("policy_name") and payload["policy_name"] != spec.name:
                        continue
                    for sink in spec.sinks:
                        ranks = {level.name: level.rank for level in spec.severity.levels}
                        if sink.min_severity and ranks.get(alert["severity"], 0) < ranks.get(sink.min_severity, 0):
                            continue
                        if sink.kind in {"webhook", "slack"}:
                            if self.client is None:
                                import httpx
                                self.client = httpx.AsyncClient(timeout=5)
                            response = await self.client.post(sink.params["url"], json=alert,
                                                              headers={"Idempotency-Key": event_id})
                            response.raise_for_status()
                        elif sink.kind == "stdout":
                            log.info("incident_accepted", extra={"incident": alert})
                        elif sink.kind == "ws":
                            channel = sink.params.get("channel", "alerts.{stream_id}").format(stream_id=key, entity_id=key)
                            await self.hub.publish(channel, alert)
                        elif sink.kind == "kafka" and self.transport:
                            await self.producer.send(sink.params.get("topic", "inferops.alerts"),
                                                     json.dumps({**alert, "event_id": event_id}).encode(), key=key.encode())
                        elif sink.kind == "plugin":
                            from inferops.domain.incident_spec import Alert
                            identity = json.dumps(sink.model_dump(), sort_keys=True)
                            if identity not in self.plugins:
                                target = sink.params["target"]
                                action = self.definitions.object_loader(target)(sink)
                                await action.setup()
                                self.plugins[identity] = action
                            await self.plugins[identity].emit(Alert(**alert))
        elif topic == "evidence.updated":
            await self.hub.publish(f"evidence.{payload.get('session_id', key)}", payload)

    async def publish_transient(self, channel, payload):
        await self.hub.publish(channel, payload)

    async def close(self):
        if self.client:
            await self.client.aclose()
        for action in self.plugins.values():
            await action.close()
        # Transport owns producer lifetime. Individual outbox relays never close a shared producer.
