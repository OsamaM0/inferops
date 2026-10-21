"""Ordered pipeline construction and explicit independent model groups."""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

from inferops.domain.fusion import FusionEngine
from inferops.domain.incidents import IncidentPolicy


def execution_declarations(spec):
    """The same ordered stages used by execution and the dashboard inspector."""
    declarations = list(spec.stages)
    tracking = spec.tracking
    if tracking and tracking.enabled and not any(d.stage == "track" for d in declarations):
        from inferops.domain.pipeline import StageDecl
        anchor = tracking.insert_after or next((d.name for d in declarations if d.stage == "infer"), "")
        if anchor:
            values = tracking.model_dump(exclude={"enabled", "insert_after"})
            declarations.insert(next(i for i, d in enumerate(declarations) if d.name == anchor) + 1,
                                StageDecl(name="auto_track", stage="track", params=values))
    return [d for d in declarations if d.stage not in {"decode_any", "decode_image", "decode_video"}]


async def build_steps(session):
    session.steps = []
    session.fusions = {}
    session.policies = {}
    declarations = execution_declarations(session.spec)
    for declaration in declarations:
        alias = declaration.stage
        if alias == "fuse":
            name = declaration.params["fusion"]
            session.fusions[declaration.name] = FusionEngine(session.definition.fusion[name])
            session.steps.append((declaration.name, "fusion", session.fusions[declaration.name]))
        elif alias == "alert":
            name = declaration.params["alerts"]
            session.policies[declaration.name] = IncidentPolicy(session.definition.alerts[name])
            session.steps.append((declaration.name, "incident", session.policies[declaration.name]))
        elif alias == "model_group":
            session.steps.append((declaration.name, "model_group", declaration.params["models"]))
        else:
            stage = session.stage_factory(declaration)
            try:
                await stage.setup(SimpleNamespace(pipeline=session.spec.name, name=declaration.name,
                                              params=declaration.params, registry=session.models,
                                              transport=session.application.transport, extra={"session_id": session.id}))
            except BaseException:
                await stage.teardown()
                raise
            session.steps.append((declaration.name, "stage", stage))
    session.sinks = []
    for declaration in session.spec.sinks:
        if declaration.stage in {"evidence", "sink_s3", "sink_postgres", "sink_kafka", "sink_ws", "sink_webhook", "sink_rest"}:
            continue  # Application-owned durable effects and transient delivery.
        sink = session.stage_factory(declaration)
        await sink.setup(SimpleNamespace(pipeline=session.spec.name, name=declaration.name,
                                         params=declaration.params, registry=session.models,
                                         transport=session.application.transport, extra={"session_id": session.id}))
        session.sinks.append(sink)


async def execute_steps(session, current):
    for name, kind, stage in session.steps:
        next_items = []
        for value in current:
            started = time.monotonic()
            if kind == "fusion":
                events = await stage.ingest_item(value, pipeline=session.spec.name)
                value.payload.setdefault("fusion_events", []).extend(e.model_dump(mode="json") for e in events)
                out = value
            elif kind == "incident":
                incidents = await stage.ingest_item(value)
                for incident in incidents:
                    incident.payload["policy_name"] = stage.spec.name
                value.payload.setdefault("alerts", []).extend(a.model_dump(mode="json") for a in incidents)
                out = value
            elif kind == "model_group":
                results = await asyncio.gather(*(session.models.predict(
                    model, value.payload["image_np"], stream_id=session.id, ts_ms=value.ts_ms, seq=value.seq)
                    for model in stage), return_exceptions=True)
                for result in results:
                    if isinstance(result, BaseException):
                        raise result
                value.results.extend(results)
                out = value
            else:
                out = await stage(value)
            value.trace[f"stage_ms.{name}"] = str((time.monotonic() - started) * 1000)
            next_items.extend(out if isinstance(out, list) else [out] if out is not None else [])
        current = next_items
    return current
