"""Generic FSM with hysteresis."""
from __future__ import annotations

from inferops.domain.fusion.confirmation import Confirmation
from inferops.domain.fusion.dsl import safe_eval
from inferops.domain.fusion.spec import StateMachineSpec


class StateMachine:
    def __init__(self, spec: StateMachineSpec) -> None:
        self.spec = spec
        # per-entity state
        self._state: dict[str, str] = {}
        self._entered_at: dict[str, int] = {}
        self._confirmation = Confirmation()
        self.confirmed = {}

    def state(self, entity: str) -> str:
        return self._state.get(entity, self.spec.initial)

    def evaluate(self, entity: str, ts_ms: int, namespace: dict) -> tuple[str | None, list[str]]:
        """Returns (new_state, actions) — new_state is None if no transition."""
        cur = self.state(entity)
        entered = self._entered_at.setdefault(entity, ts_ms)
        dwell = ts_ms - entered
        ns = {**namespace, "state": cur, "dwell_ms": dwell}
        for index, tr in enumerate(self.spec.transitions):
            if tr.from_state not in ("*", cur):
                continue
            try:
                ok = bool(safe_eval(tr.when, ns))
            except Exception:
                ok = False
            ready = self._confirmation.ready((entity, index), ok, ts_ms, confirm_ms=tr.confirm_ms,
                                             min_samples=tr.min_samples, max_gap_ms=tr.max_gap_ms)
            if ready and dwell >= tr.min_dwell_ms:
                self.confirmed[entity] = self._confirmation.details((entity, index), ts_ms, tr.confirm_ms)
                self._state[entity] = tr.to_state
                self._entered_at[entity] = ts_ms
                self._confirmation.pending = {key: value for key, value in self._confirmation.pending.items()
                                              if key[0] != entity}
                actions = (self.spec.on_exit.get(cur, []) + self.spec.on_enter.get(tr.to_state, []))
                return tr.to_state, actions
        return None, []
