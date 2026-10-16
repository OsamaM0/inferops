"""Pure incident severity, deduplication, and media-time cooldown decisions."""
from __future__ import annotations

import hashlib

from inferops.domain.incident_spec import Alert, AlertCandidate, AlertSpec
from inferops.domain.types import StageItem, new_id


class IncidentPolicy:
    def __init__(self, spec: AlertSpec) -> None:
        self.spec = spec.model_copy(deep=True)
        self._levels = {level.name: level for level in spec.severity.levels}
        self.reset()

    def reset(self) -> None:
        self._dedupe: dict[str, int] = {}
        self._cd_type: dict[str, int] = {}
        self._cd_global = 0

    def checkpoint(self) -> tuple[dict[str, int], dict[str, int], int]:
        """Allow application transactions to roll back uncommitted decisions."""
        return dict(self._dedupe), dict(self._cd_type), self._cd_global

    def restore(self, checkpoint: tuple[dict[str, int], dict[str, int], int]) -> None:
        self._dedupe, self._cd_type, self._cd_global = checkpoint

    def _resolve_severity(self, candidate: AlertCandidate) -> str:
        if candidate.severity_hint and candidate.severity_hint in self._levels:
            return candidate.severity_hint
        return self.spec.severity.mapping.get(candidate.type, self.spec.severity.default)

    def _dedupe_key(self, candidate: AlertCandidate) -> str:
        if candidate.dedupe_key:
            return candidate.dedupe_key
        if self.spec.dedupe.strategy == "by_payload_hash":
            blob = f"{candidate.type}|{candidate.entity_id}|{sorted(candidate.payload.items())}"
            return hashlib.sha1(blob.encode()).hexdigest()[:16]
        if self.spec.dedupe.strategy == "by_key_template":
            payload = {key: value for key, value in candidate.payload.items()
                       if key not in {"type", "entity_id"}}
            return self.spec.dedupe.key_template.format(type=candidate.type,
                                                       entity_id=candidate.entity_id, **payload)
        return f"{candidate.type}:{candidate.entity_id}"

    async def submit(self, candidate: AlertCandidate) -> Alert | None:
        timestamp = candidate.ts_ms
        # Expiry pruning bounds state over long sessions with changing entities.
        self._dedupe = {key: expiry for key, expiry in self._dedupe.items() if expiry > timestamp}
        self._cd_type = {key: expiry for key, expiry in self._cd_type.items() if expiry > timestamp}
        key = self._dedupe_key(candidate)
        if (self._dedupe.get(key, 0) > timestamp or self._cd_type.get(candidate.type, 0) > timestamp
                or self._cd_global > timestamp):
            return None
        self._dedupe[key] = timestamp + max(self.spec.dedupe.window_ms,
                                            self.spec.cooldown.per_dedupe_key_ms)
        if self.spec.cooldown.per_type_ms:
            self._cd_type[candidate.type] = timestamp + self.spec.cooldown.per_type_ms
        if self.spec.cooldown.global_ms:
            self._cd_global = timestamp + self.spec.cooldown.global_ms
        return Alert(id=new_id(), type=candidate.type, entity_id=candidate.entity_id,
                     ts_ms=timestamp, severity=self._resolve_severity(candidate), dedupe_key=key,
                     payload=candidate.payload)

    async def ingest_item(self, item: StageItem) -> list[Alert]:
        events = item.payload.get("fusion_events") or []
        candidates = list(item.payload.get("alert_candidates") or [])
        if not events:
            candidates.extend(item.payload.get("alerts") or [])
        output: list[Alert] = []
        for raw in [*events, *candidates]:
            if not isinstance(raw, dict) or ("id" in raw and raw not in events):
                continue
            candidate = AlertCandidate(
                type=raw.get("type", "EVENT"), entity_id=raw.get("entity_id", item.stream_id or ""),
                ts_ms=raw.get("ts_ms", item.ts_ms), payload=raw.get("payload", raw),
                severity_hint=raw.get("severity") or raw.get("severity_hint"),
                dedupe_key=raw.get("dedupe_key"),
            )
            alert = await self.submit(candidate)
            if alert is not None:
                output.append(alert)
        return output

    async def close(self) -> None:
        self.reset()
