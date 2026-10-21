"""Transactional incident acceptance, evidence requests and outbox construction."""
from __future__ import annotations

import time
import uuid


def wire_result(item):
    wire = item.model_dump(mode="json", exclude={"payload"})
    wire["payload"] = {key: value for key, value in item.payload.items()
                       if key not in {"image_np", "image_bytes", "feeds", "frame_meta"} and not key.startswith("_")}
    return wire


async def accept_incidents(items, *, repository, evidence, definition, pipeline, session_id,
                           timeline_id, stream_name, fusions, broker_context=None, camera_id=None):
    events, incidents, captures, outbox = [], [], [], []
    broker = broker_context or next((item.payload.get("_broker_event") for item in items if item.payload.get("_broker_event")), None)
    for output_index, item in enumerate(items):
        identity = f"{session_id}:{timeline_id}:{item.seq}:{output_index}"
        for index, event in enumerate(item.payload.get("fusion_events", [])):
            event["id"] = uuid.uuid5(uuid.NAMESPACE_URL, f"{identity}:event:{index}").hex
            events.append({"id": event["id"], "pipeline": pipeline.name, "type": event["type"],
                           "entity_id": session_id, "ts_ms": time.time_ns() // 1_000_000,
                           "payload": {**event.get("payload", {}), "session_id": session_id,
                                       "timeline_id": timeline_id, "media_time_ns": item.payload["media_time_ns"],
                                       "stream_name": stream_name}})
            topic = next((engine.spec.outputs.events_topic for engine in fusions.values()
                          if engine.spec.outputs.events_topic), None)
            if topic:
                outbox.append({"id": f"{event['id']}:publish", "topic": topic, "key": session_id,
                               "payload": event})
        for index, alert in enumerate(item.payload.get("alerts", [])):
            alert["id"] = uuid.uuid5(uuid.NAMESPACE_URL, f"{identity}:incident:{index}").hex
            alert["entity_id"] = session_id
            alert["ts_ms"] = time.time_ns() // 1_000_000
            alert["payload"] = {**alert.get("payload", {}), "session_id": session_id,
                                "timeline_id": timeline_id, "media_time_ns": item.payload["media_time_ns"],
                                "pipeline": pipeline.name, "stream_name": stream_name}
            incidents.append(alert)
            outbox.append({"id": alert["id"], "topic": "incident.accepted", "key": session_id,
                           "payload": {"event_type": "incident.accepted", "alert": alert,
                                       "policy_name": alert["payload"].get("policy_name"),
                                       "delivery_policy": definition.alerts[alert["payload"]["policy_name"]].model_dump(mode="json")
                                       if alert["payload"].get("policy_name") in definition.alerts else None}})
            if evidence:
                config = evidence.config
                evidence_id = uuid.uuid5(uuid.NAMESPACE_URL, f"{alert['id']}:evidence").hex
                captures.append({"id": evidence_id, "incident_id": alert["id"], "session_id": session_id,
                                 "camera_id": camera_id, "stream_name": stream_name,
                                 "timeline_id": timeline_id, "pipeline": pipeline.name,
                                 "media_time_ns": item.payload["media_time_ns"], "state": "Pending",
                                 "request_start_ns": item.payload["media_time_ns"] - int(config.pre_seconds * 1e9),
                                 "request_end_ns": item.payload["media_time_ns"] + int(config.post_seconds * 1e9),
                                 "artifacts": {}, "errors": []})
                alert["payload"]["evidence_id"] = evidence_id
                alert["payload"]["evidence_state"] = "Pending"
        if broker and broker.get("output_topic"):
            wire = wire_result(item)
            outbox.append({"id": f"{broker['consumer']}:{broker['event_id']}:output:{output_index}",
                           "topic": broker["output_topic"], "key": broker.get("output_key") or session_id,
                           "payload": wire})
        for declaration in pipeline.sinks:
            if declaration.stage == "sink_kafka":
                outbox.append({"id": f"{identity}:{declaration.name}", "topic": declaration.params["topic"],
                               "key": session_id, "payload": wire_result(item)})
            elif declaration.stage == "sink_webhook":
                outbox.append({"id": f"{identity}:{declaration.name}", "topic": "delivery.webhook",
                               "key": session_id, "payload": {"url": declaration.params["url"], "body": wire_result(item)}})
            elif declaration.stage == "sink_postgres":
                events.append({"id": uuid.uuid5(uuid.NAMESPACE_URL, f"{identity}:{declaration.name}").hex,
                               "pipeline": pipeline.name, "entity_id": session_id, "ts_ms": item.ts_ms,
                               "type": declaration.params.get("type_field", "event"), "payload": wire_result(item)})
    if events or incidents or broker or outbox:
        try:
            if broker:
                accepted = await repository.accept(events, incidents, captures, outbox,
                                                                    inbox=(broker["event_id"], broker["consumer"]))
                if accepted is False:
                    for item in items:
                        item.payload["alerts"] = []
                        item.payload["fusion_events"] = []
                    return
            else:
                await repository.accept(events, incidents, captures, outbox)
        except Exception as exc:
            for item in items:
                item.payload["alerts"] = []
                item.payload["incident_error"] = {"code": "persistence_unavailable", "detail": str(exc)}
            return
    for capture in captures:
        await evidence.request(capture)
        row = await repository.get_evidence(capture["id"])
        for item in items:
            item.payload.setdefault("evidence", []).append(row)

