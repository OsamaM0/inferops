"""AlertEngine: severity mapping, dedupe, cooldown, fan-out."""
from __future__ import annotations

import hashlib
from contextlib import suppress

from inferops.alerts.sinks import build_sink, metric_emit
from inferops.alerts.spec import Alert, AlertCandidate, AlertSpec
from inferops.core.types import StageItem, new_id, now_ms
from inferops.obs import get_logger

log = get_logger(__name__)


class AlertEngine:
    # Class-level so a process with many engines still warns only once.
    _persist_warned = False

    def __init__(self, spec: AlertSpec) -> None:
        self.spec = spec
        self._sinks = [build_sink(s) for s in spec.sinks]
        self._setup_done = False
        # dedupe + cooldown state (in-memory; Redis-backed variant TODO)
        self._dedupe: dict[str, int] = {}    # key -> expiry_ms
        self._cd_type: dict[str, int] = {}
        self._cd_global: int = 0
        self._levels = {lvl.name: lvl for lvl in spec.severity.levels}

    async def _ensure(self) -> None:
        if self._setup_done:
            return
        for s in self._sinks:
            try:
                await s.setup()
            except Exception as e:
                log.warning("alert_sink_setup_failed", sink=s.spec.name, err=str(e))
        self._setup_done = True

    def _resolve_severity(self, cand: AlertCandidate) -> str:
        if cand.severity_hint and cand.severity_hint in self._levels:
            return cand.severity_hint
        return self.spec.severity.mapping.get(cand.type, self.spec.severity.default)

    def _dedupe_key(self, cand: AlertCandidate) -> str:
        if cand.dedupe_key:
            return cand.dedupe_key
        st = self.spec.dedupe.strategy
        if st == "by_payload_hash":
            blob = f"{cand.type}|{cand.entity_id}|{sorted(cand.payload.items())}"
            return hashlib.sha1(blob.encode()).hexdigest()[:16]
        if st == "by_key_template":
            payload_kw = {k: v for k, v in cand.payload.items()
                          if k not in {"type", "entity_id"}}
            return self.spec.dedupe.key_template.format(
                type=cand.type, entity_id=cand.entity_id, **payload_kw)
        return f"{cand.type}:{cand.entity_id}"

    def _allowed(self, cand: AlertCandidate, ts_ms: int, sev: str) -> tuple[bool, str]:
        key = self._dedupe_key(cand)
        if self._dedupe.get(key, 0) > ts_ms:
            return False, key
        if self._cd_type.get(cand.type, 0) > ts_ms:
            return False, key
        if self._cd_global > ts_ms:
            return False, key
        # set cooldowns
        self._dedupe[key] = ts_ms + max(self.spec.dedupe.window_ms,
                                        self.spec.cooldown.per_dedupe_key_ms)
        if self.spec.cooldown.per_type_ms:
            self._cd_type[cand.type] = ts_ms + self.spec.cooldown.per_type_ms
        if self.spec.cooldown.global_ms:
            self._cd_global = ts_ms + self.spec.cooldown.global_ms
        return True, key

    async def submit(self, cand: AlertCandidate) -> Alert | None:
        await self._ensure()
        ts = cand.ts_ms or now_ms()
        sev = self._resolve_severity(cand)
        allowed, key = self._allowed(cand, ts, sev)
        if not allowed:
            return None
        alert = Alert(id=new_id(), type=cand.type, entity_id=cand.entity_id,
                      ts_ms=ts, severity=sev, dedupe_key=key, payload=cand.payload)
        # fan-out
        level = self._levels.get(sev)
        for sink in self._sinks:
            if sink.spec.min_severity and level and level.rank < self._levels[sink.spec.min_severity].rank:
                continue
            try:
                await sink.emit(alert)
                metric_emit(alert, sink.spec.name, "ok")
            except Exception:
                metric_emit(alert, sink.spec.name, "error")
        # persistence
        if self.spec.persistence_enabled:
            try:
                from inferops.adapters.postgres_dao import insert_alert
                await insert_alert({
                    "id": alert.id, "type": alert.type, "entity_id": alert.entity_id,
                    "severity": alert.severity, "dedupe_key": alert.dedupe_key,
                    "status": alert.status, "ts_ms": alert.ts_ms,
                    "payload": alert.payload,
                })
            except Exception as e:
                # Losing an alert is a real incident, not a debug detail. This
                # was log.debug, so with no database reachable every alert
                # vanished and the dashboard's activity panels stayed empty with
                # nothing in the logs to explain why. Warn on the first failure,
                # then fall back to debug so a sustained outage cannot flood the
                # log at frame rate.
                if not AlertEngine._persist_warned:
                    AlertEngine._persist_warned = True
                    log.warning("alert_persist_failed", err=str(e),
                                hint="alerts are not being stored; check INFEROPS_PG_URL",
                                note="further failures logged at debug level")
                else:
                    log.debug("alert_persist_failed", err=str(e))
        return alert

    async def ingest_item(self, item: StageItem) -> list[Alert]:
        """Convert StageItem alert payloads into AlertCandidates."""
        out: list[Alert] = []
        events = item.payload.get("fusion_events") or []
        for ev in events:
            cand = AlertCandidate(
                type=ev.get("type", "EVENT"), entity_id=ev.get("entity_id", item.stream_id or ""),
                ts_ms=ev.get("ts_ms", item.ts_ms), payload=ev.get("payload", {}),
                severity_hint=ev.get("severity"),
            )
            a = await self.submit(cand)
            if a is not None:
                out.append(a)

        candidates = item.payload.get("alert_candidates") or []
        if not events:
            candidates = [*candidates, *(item.payload.get("alerts") or [])]
        for raw in candidates:
            if not isinstance(raw, dict) or "id" in raw:
                continue
            cand = AlertCandidate(
                type=raw.get("type", "EVENT"),
                entity_id=raw.get("entity_id", item.stream_id or ""),
                ts_ms=raw.get("ts_ms", item.ts_ms),
                payload=raw.get("payload", raw),
                severity_hint=raw.get("severity") or raw.get("severity_hint"),
                dedupe_key=raw.get("dedupe_key"),
            )
            alert = await self.submit(cand)
            if alert is not None:
                out.append(alert)
        return out

    async def close(self) -> None:
        for s in self._sinks:
            with suppress(Exception):
                await s.close()
