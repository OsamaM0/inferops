"""Alert dedupe & cooldown semantics."""
import pytest

from inferops.domain.incident_spec import (
    AlertCandidate,
    AlertSpec,
    CooldownSpec,
    DedupeSpec,
    SeverityLevel,
    SeverityMapSpec,
    SinkSpec,
)
from inferops.domain.incidents import IncidentPolicy as AlertEngine
from inferops.domain.types import StageItem


def _spec():
    return AlertSpec(
        name="t",
        severity=SeverityMapSpec(
            levels=[SeverityLevel(name="low", rank=1), SeverityLevel(name="high", rank=3)],
            mapping={"PHONE_USE": "high"},
            default="low",
        ),
        dedupe=DedupeSpec(strategy="by_key_template", key_template="{type}:{entity_id}", window_ms=10_000),
        cooldown=CooldownSpec(per_dedupe_key_ms=10_000),
        sinks=[SinkSpec(name="out", kind="stdout", params={})],
        persistence_enabled=False,
    )


@pytest.mark.asyncio
async def test_dedupe_within_window():
    eng = AlertEngine(_spec())
    a1 = await eng.submit(AlertCandidate(type="PHONE_USE", entity_id="cam1", ts_ms=1000, payload={}))
    a2 = await eng.submit(AlertCandidate(type="PHONE_USE", entity_id="cam1", ts_ms=1500, payload={}))
    assert a1 is not None and a1.severity == "high"
    assert a2 is None  # deduped


@pytest.mark.asyncio
async def test_dedupe_releases_after_cooldown():
    eng = AlertEngine(_spec())
    await eng.submit(AlertCandidate(type="PHONE_USE", entity_id="cam1", ts_ms=1000, payload={}))
    a = await eng.submit(AlertCandidate(type="PHONE_USE", entity_id="cam1", ts_ms=20_000, payload={}))
    assert a is not None


@pytest.mark.asyncio
async def test_ingest_item_accepts_alert_candidates():
    eng = AlertEngine(_spec())
    item = StageItem(stream_id="cam2", ts_ms=1000)
    item.payload["alert_candidates"] = [{"type": "PHONE_USE", "payload": {"score": 0.9}}]
    emitted = await eng.ingest_item(item)
    assert len(emitted) == 1
    assert emitted[0].entity_id == "cam2"
