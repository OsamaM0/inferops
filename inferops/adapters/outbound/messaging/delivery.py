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
    def __init__(self, hub, *, transport=None, definitions=None, broker_enabled=None):
        self.hub, self.transport = hub, transport
        self.definitions = definitions
        self.broker_enabled = transport is not None if broker_enabled is None else broker_enabled
        self.producer = None
        self.client = None
        self.plugins = {}

    def prepare(self, rows):
        """Split destinations before SQL acceptance, giving each an independent retry."""
        result = []
        for row in rows:
            if row["topic"] not in {"incident.accepted", "evidence.updated"}:
                result.append(row)
                continue
            body = {**row.get("payload", {}), "_delivery_split": True}
            result.append({**row, "payload": body})
            if self.broker_enabled:
                result.append({**row, "id": f"{row['id']}:broker", "topic": "delivery.broker",
                               "payload": {"topic": row["topic"], "body": row.get("payload", {})}})
            if row["topic"] == "incident.accepted":
                alert = body.get("alert", body)
                policy = body.get("delivery_policy")
                if policy:
                    from inferops.domain.incident_spec import AlertSpec
                    policies = [AlertSpec.model_validate(policy)]
                else:
                    policies = self.definitions.alerts.values() if self.definitions else []
                for spec in policies:
                    if body.get("policy_name") and body["policy_name"] != spec.name:
                        continue
                    ranks = {level.name: level.rank for level in spec.severity.levels}
                    for sink in spec.sinks:
                        if sink.min_severity and ranks.get(alert["severity"], 0) < ranks.get(sink.min_severity, 0):
                            continue
                        if sink.kind == "postgres":
                            continue  # Acceptance already persisted the incident.
                        result.append({**row, "id": f"{row['id']}:action:{spec.name}:{sink.name}",
                                       "topic": "delivery.action", "payload": {"alert": alert,
                                       "action": sink.model_dump(mode="json")}})
        return result

    async def _send(self, topic, key, payload, event_id):
        if self.transport is None:
            raise RuntimeError("required broker destination has no configured transport")
        if self.producer is None:
            self.producer = await self.transport.producer("outbox")
        await self.producer.send(topic, json.dumps({**payload, "event_id": event_id}).encode(),
                                 key=(key or "").encode())

    async def _post(self, url, body, event_id):
        if self.client is None:
            import httpx
            self.client = httpx.AsyncClient(timeout=5)
        response = await self.client.post(url, json=body, headers={"Idempotency-Key": event_id})
        response.raise_for_status()

    async def _action(self, sink, alert, key, event_id):
        if sink.kind in {"webhook", "slack"}:
            await self._post(sink.params["url"], alert, event_id)
        elif sink.kind == "stdout":
            log.info("incident_accepted", extra={"incident": alert})
        elif sink.kind == "ws":
            channel = sink.params.get("channel", "alerts.{stream_id}").format(stream_id=key, entity_id=key)
            await self.hub.publish(channel, alert)
        elif sink.kind == "kafka":
            await self._send(sink.params.get("topic", "inferops.alerts"), key, alert, event_id)
        elif sink.kind == "plugin":
            from inferops.domain.incident_spec import Alert
            identity = json.dumps(sink.model_dump(), sort_keys=True)
            if identity not in self.plugins:
                if self.definitions is None:
                    raise RuntimeError("plugin delivery requires project definitions")
                action = self.definitions.object_loader(sink.params["target"])(sink)
                await action.setup()
                self.plugins[identity] = action
            await self.plugins[identity].emit(Alert(**alert))

    async def publish(self, topic, key, payload, event_id):
        if topic == "delivery.webhook":
            return await self._post(payload["url"], payload["body"], event_id)
        if topic == "delivery.broker":
            return await self._send(payload["topic"], key, payload["body"], event_id)
        if topic == "delivery.action":
            from inferops.domain.incident_spec import SinkSpec
            return await self._action(SinkSpec.model_validate(payload["action"]), payload["alert"], key, event_id)
        if topic == "incident.accepted":
            alert = payload.get("alert", payload)
            await self.hub.publish(f"alerts.{key}", alert)
            stream = alert.get("payload", {}).get("stream_name")
            if stream:
                await self.hub.publish(f"alerts.{stream}", alert)
        elif topic == "evidence.updated":
            await self.hub.publish(f"evidence.{payload.get('session_id', key)}",
                                   {k: v for k, v in payload.items() if k != "_delivery_split"})
        if payload.get("_delivery_split"):
            return
        # Old outbox rows remain deliverable during the compatibility period.
        if self.transport:
            await self._send(topic, key, payload, event_id)
        if topic == "incident.accepted":
            for row in self.prepare([{"id": event_id, "topic": topic, "key": key, "payload": payload}]):
                if row["topic"] == "delivery.action":
                    await self.publish(row["topic"], key, row["payload"], row["id"])

    async def publish_transient(self, channel, payload):
        await self.hub.publish(channel, payload)

    async def close(self):
        if self.client:
            await self.client.aclose()
        for action in self.plugins.values():
            await action.close()
        # Transport owns producer lifetime. Individual outbox relays never close a shared producer.
