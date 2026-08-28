"""Fusion engine: ingests signals, runs windows + rules + state machine."""
from __future__ import annotations

from typing import Any

import orjson

from inferops.adapters.transport import get_transport
from inferops.core.types import StageItem, now_ms
from inferops.fusion.dsl import safe_eval
from inferops.fusion.signals import FusionEvent, Signal
from inferops.fusion.spec import FusionSpec
from inferops.fusion.state_machine import StateMachine
from inferops.fusion.window import Window
from inferops.obs import FUSION_EVENTS, get_logger

log = get_logger(__name__)


class FusionEngine:
    # Class-level so a process with many engines still warns only once.
    _persist_warned = False

    def __init__(self, spec: FusionSpec) -> None:
        self.spec = spec
        self._windows: dict[tuple[str, str], Window] = {}   # (entity, signal) -> Window
        self._cooldowns: dict[tuple[str, str], int] = {}    # (entity, rule_name) -> until_ms
        self._fsm = StateMachine(spec.state_machine) if spec.state_machine else None
        self._extractors = dict(spec.extract)
        self._producer = None

    async def _get_producer(self):
        if self._producer is None and self.spec.outputs.events_topic:
            t = get_transport()
            self._producer = await t.producer(f"fusion-{self.spec.name}")
        return self._producer

    def _window(self, entity: str, signal: str) -> Window | None:
        ws = self.spec.windows.get(signal)
        if ws is None:
            return None
        key = (entity, signal)
        if key not in self._windows:
            self._windows[key] = Window(ws)
        return self._windows[key]

    def _namespace(self, entity: str) -> dict:
        ns: dict[str, Any] = {}
        for sig in self.spec.signals:
            w = self._windows.get((entity, sig))
            ns[sig] = w if w is not None else _NullWindow()
        for dname, dexpr in self.spec.derived.items():
            try:
                ns[dname] = safe_eval(dexpr, ns)
            except Exception:
                ns[dname] = None
        return ns

    async def ingest_signal(self, sig: Signal) -> list[FusionEvent]:
        w = self._window(sig.entity_id, sig.name)
        if w is None:
            return []
        w.add(sig.ts_ms, float(sig.value))
        emitted = await self._evaluate(sig.entity_id, sig.ts_ms)
        if emitted and self.spec.persistence_enabled:
            await self._persist(emitted)
        return emitted

    async def ingest_item(self, item: StageItem, pipeline: str | None = None) -> list[FusionEvent]:
        entity = getattr(item, self.spec.entity_key, None) or item.stream_id
        ts = item.ts_ms or now_ms()
        emitted: list[FusionEvent] = []
        env = {"item": item, "results": item.results, "payload": item.payload,
               "stream_id": item.stream_id}
        # extract signals
        for sig_name, expr in self._extractors.items():
            try:
                val = safe_eval(expr, env)
            except Exception:
                val = None
            if val is None:
                continue
            if isinstance(val, bool):
                val = 1.0 if val else 0.0
            try:
                v = float(val)
            except Exception:
                continue
            w = self._window(entity, sig_name)
            if w is not None:
                w.add(ts, v)
        emitted.extend(await self._evaluate(entity, ts))
        # Persist symmetrically with AlertEngine so GET /v1/events works for
        # every domain, not only those wiring an explicit sink_postgres stage.
        if emitted and self.spec.persistence_enabled:
            await self._persist(emitted, pipeline)
        return emitted

    async def _evaluate(self, entity: str, ts_ms: int) -> list[FusionEvent]:
        ns = self._namespace(entity)
        emitted: list[FusionEvent] = []
        # rules
        for rule in self.spec.rules:
            try:
                ok = bool(safe_eval(rule.when, ns))
            except Exception:
                ok = False
            if not ok:
                continue
            cd_key = (entity, rule.name)
            if self._cooldowns.get(cd_key, 0) > ts_ms:
                continue
            self._cooldowns[cd_key] = ts_ms + rule.cooldown_ms
            etype = (rule.emit or {}).get("type", rule.name)
            ev = FusionEvent(
                fusion=self.spec.name, entity_id=entity, ts_ms=ts_ms,
                type=etype, severity=rule.severity,
                payload={"rule": rule.name, **(rule.emit or {}),
                         "signals": {s: ns[s].asdict() if hasattr(ns[s], "asdict") else ns[s]
                                     for s in self.spec.signals if s in ns},
                         "derived": {name: ns.get(name) for name in self.spec.derived}},
            )
            emitted.append(ev)
            FUSION_EVENTS.labels(fusion=self.spec.name, type=etype).inc()
        # state machine
        if self._fsm is not None:
            new_state, _actions = self._fsm.evaluate(entity, ts_ms, ns)
            if new_state is not None:
                ev = FusionEvent(
                    fusion=self.spec.name, entity_id=entity, ts_ms=ts_ms,
                    type=f"STATE_{new_state}", payload={"state": new_state},
                )
                emitted.append(ev)
                FUSION_EVENTS.labels(fusion=self.spec.name, type=ev.type).inc()
        # publish
        if emitted and self.spec.outputs.events_topic:
            p = await self._get_producer()
            for ev in emitted:
                try:
                    await p.send(self.spec.outputs.events_topic,
                                 orjson.dumps(ev.model_dump(mode="json")),
                                 key=entity.encode())
                except Exception as e:
                    log.warning("fusion_publish_failed", err=str(e))
        return emitted

    async def _persist(self, events: list[FusionEvent], pipeline: str | None = None) -> None:
        try:
            from inferops.adapters.postgres_dao import insert_event

            for ev in events:
                await insert_event({
                    "id": ev.id,
                    # The column holds the *pipeline* name — that is what
                    # sink_postgres writes and what GET /v1/events filters on.
                    # Falling back to the fusion spec name only when the caller
                    # has no pipeline context (e.g. direct engine use in tests).
                    "pipeline": pipeline or self.spec.name,
                    "type": ev.type,
                    "entity_id": ev.entity_id,
                    "ts_ms": ev.ts_ms,
                    "payload": ev.payload,
                })
        except Exception as e:
            # Warn once, then drop to debug: a sustained database outage must
            # not flood the log at frame rate, but silence would hide data loss.
            if not FusionEngine._persist_warned:
                FusionEngine._persist_warned = True
                log.warning("fusion_persist_failed", err=str(e),
                            hint="events are not being stored; check INFEROPS_PG_URL",
                            note="further failures logged at debug level")
            else:
                log.debug("fusion_persist_failed", err=str(e))


class _NullWindow:
    """Stand-in window for signals lacking a WindowSpec — always returns None/0."""
    def latest(self): return None
    def mean(self): return None
    def max(self): return None
    def min(self): return None
    def count(self): return 0
    def fraction_above(self, _t): return 0.0
    def fraction_below(self, _t): return 0.0
    def slope(self): return 0.0
    def derivative(self): return 0.0
    def asdict(self): return {}
