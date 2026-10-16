"""Pure, session-owned fusion decisions using the established window/DSL algorithms.

Persistence, delivery, clocks, and metrics are application/adapter responsibilities.
The supplied item timestamp is media time, including a valid zero timestamp.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from inferops.domain.fusion.dsl import safe_eval
from inferops.domain.fusion.signals import FusionEvent, Signal
from inferops.domain.fusion.spec import FusionSpec
from inferops.domain.fusion.state_machine import StateMachine
from inferops.domain.fusion.window import Window
from inferops.domain.types import StageItem


class FusionEngine:
    def __init__(self, spec: FusionSpec) -> None:
        self.spec = spec.model_copy(deep=True)
        self.reset()

    def reset(self) -> None:
        self._windows: dict[tuple[str, str], Window] = {}
        self._cooldowns: dict[tuple[str, str], int] = {}
        self._fsm = StateMachine(self.spec.state_machine) if self.spec.state_machine else None

    def checkpoint(self) -> tuple[dict, dict, StateMachine | None]:
        """Snapshot mutable decisions before a durable application transaction.

        This is an in-process rollback point, not persisted session recovery.
        """
        return deepcopy((self._windows, self._cooldowns, self._fsm))

    def restore(self, checkpoint: tuple[dict, dict, StateMachine | None]) -> None:
        # A checkpoint can be reused without subsequent ingestion mutating it.
        self._windows, self._cooldowns, self._fsm = deepcopy(checkpoint)

    def _window(self, entity: str, signal: str) -> Window | None:
        window_spec = self.spec.windows.get(signal)
        if window_spec is None:
            return None
        key = (entity, signal)
        if key not in self._windows:
            self._windows[key] = Window(window_spec)
        return self._windows[key]

    def _namespace(self, entity: str) -> dict[str, Any]:
        namespace: dict[str, Any] = {
            signal: self._windows.get((entity, signal), _NullWindow())
            for signal in self.spec.signals
        }
        for name, expression in self.spec.derived.items():
            try:
                namespace[name] = safe_eval(expression, namespace)
            except Exception:
                namespace[name] = None
        return namespace

    async def ingest_signal(self, signal: Signal) -> list[FusionEvent]:
        window = self._window(signal.entity_id, signal.name)
        if window is None:
            return []
        window.add(signal.ts_ms, float(signal.value))
        return self._evaluate(signal.entity_id, signal.ts_ms)

    async def ingest_item(self, item: StageItem, pipeline: str | None = None) -> list[FusionEvent]:
        entity = getattr(item, self.spec.entity_key, None) or item.stream_id
        timestamp = item.ts_ms
        environment = {"item": item, "results": item.results, "payload": item.payload,
                       "stream_id": item.stream_id}
        for signal, expression in self.spec.extract.items():
            try:
                value = safe_eval(expression, environment)
                if value is None:
                    continue
                value = float(value)
            except Exception:
                continue
            window = self._window(entity, signal)
            if window is not None:
                window.add(timestamp, value)
        return self._evaluate(entity, timestamp)

    def _evaluate(self, entity: str, timestamp: int) -> list[FusionEvent]:
        namespace = self._namespace(entity)
        events: list[FusionEvent] = []
        for rule in self.spec.rules:
            try:
                allowed = bool(safe_eval(rule.when, namespace))
            except Exception:
                allowed = False
            if not allowed:
                continue
            key = (entity, rule.name)
            if self._cooldowns.get(key, 0) > timestamp:
                continue
            self._cooldowns[key] = timestamp + rule.cooldown_ms
            events.append(FusionEvent(
                fusion=self.spec.name, entity_id=entity, ts_ms=timestamp,
                type=(rule.emit or {}).get("type", rule.name), severity=rule.severity,
                payload={"rule": rule.name, **(rule.emit or {}),
                         "signals": {signal: namespace[signal].asdict()
                                     if hasattr(namespace[signal], "asdict") else namespace[signal]
                                     for signal in self.spec.signals},
                         "derived": {name: namespace.get(name) for name in self.spec.derived}},
            ))
        if self._fsm is not None:
            state, _actions = self._fsm.evaluate(entity, timestamp, namespace)
            if state is not None:
                events.append(FusionEvent(fusion=self.spec.name, entity_id=entity, ts_ms=timestamp,
                                          type=f"STATE_{state}", payload={"state": state}))
        return events


class _NullWindow:
    def latest(self): return None
    def mean(self): return None
    def max(self): return None
    def min(self): return None
    def count(self): return 0
    def fraction_above(self, _threshold): return 0.0
    def fraction_below(self, _threshold): return 0.0
    def slope(self): return 0.0
    def derivative(self): return 0.0
    def asdict(self): return {}
