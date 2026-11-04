"""Deleting selected (or all) alerts and evidence from the dashboard."""
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from inferops.adapters.inbound.http.main import create_app
from inferops.adapters.outbound.persistence.memory import MemoryRepository
from inferops.application.activity import delete_alerts, delete_evidence


class FakeCaptures:
    """Evidence service stand-in: pending records refuse deletion like the real one."""

    def __init__(self, repository, pending=()):
        self.repository, self.pending, self.deleted = repository, set(pending), []

    async def delete(self, evidence_id):
        if await self.repository.get_evidence(evidence_id) is None:
            raise KeyError(evidence_id)
        if evidence_id in self.pending:
            raise ValueError("Evidence capture or delivery is still pending")
        self.deleted.append(evidence_id)
        return await self.repository.update_evidence(evidence_id, deleted=True, artifacts={})


async def seeded(alert_ids, *, pending=(), without_evidence=()):
    repository = MemoryRepository()
    alerts = [{"id": a, "type": "new_person", "entity_id": "cam", "severity": "low", "dedupe_key": a, "ts_ms": i,
               "payload": {} if a in without_evidence else {"evidence_id": f"ev-{a}"}}
              for i, a in enumerate(alert_ids)]
    evidence = [{"id": f"ev-{a}", "incident_id": a, "session_id": "s", "state": "Complete", "artifacts": {"clip": "x"}}
                for a in alert_ids if a not in without_evidence]
    await repository.accept([], alerts, evidence, [])
    return SimpleNamespace(repository=repository, evidence=FakeCaptures(repository, {f"ev-{p}" for p in pending}))


@pytest.mark.asyncio
async def test_deleting_selected_alerts_removes_them_with_their_evidence():
    app = await seeded(["a1", "a2", "a3"])
    result = await delete_alerts(app, ["a1", "a3", "missing"])
    assert result["deleted"] == ["a1", "a3"] and result["deleted_count"] == 2
    assert result["evidence_deleted"] == ["ev-a1", "ev-a3"]
    assert result["skipped"] == [{"id": "missing", "reason": "not found"}]
    assert [a["id"] for a in await app.repository.list_incidents()] == ["a2"]
    assert (await app.repository.get_evidence("ev-a2"))["artifacts"] == {"clip": "x"}


@pytest.mark.asyncio
async def test_an_alert_whose_evidence_is_still_saving_is_kept():
    app = await seeded(["a1", "a2"], pending=["a2"], without_evidence=())
    result = await delete_alerts(app, None, everything=True)
    assert result["deleted"] == ["a1"]
    assert result["skipped"] == [{"id": "a2", "reason": "evidence: Evidence capture or delivery is still pending"}]
    assert [a["id"] for a in await app.repository.list_incidents()] == ["a2"]


@pytest.mark.asyncio
async def test_alerts_without_stored_evidence_or_with_evidence_disabled_are_deleted():
    app = await seeded(["a1", "a2"], without_evidence=["a1"])
    assert (await delete_alerts(app, ["a1"]))["deleted"] == ["a1"]
    app.evidence = None
    assert (await delete_alerts(app, ["a2"]))["deleted"] == ["a2"]
    assert await app.repository.list_incidents() == []


@pytest.mark.asyncio
async def test_deleting_evidence_by_selection_or_all_skips_pending_and_unknown():
    app = await seeded(["a1", "a2", "a3"], pending=["a3"])
    result = await delete_evidence(app, ["ev-a1", "nope"])
    assert result == {"deleted": ["ev-a1"], "skipped": [{"id": "nope", "reason": "not found"}]}
    result = await delete_evidence(app, None, everything=True)          # ev-a1 is already a tombstone
    assert result["deleted"] == ["ev-a2"] and [s["id"] for s in result["skipped"]] == ["ev-a3"]
    assert len(await app.repository.list_incidents()) == 3               # alerts stay
    app.evidence = None
    with pytest.raises(ValueError):
        await delete_evidence(app, ["ev-a1"])


@pytest.mark.asyncio
async def test_bulk_delete_endpoints():
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client, \
            app.router.lifespan_context(app):
        repository = app.state.engine.application.repository
        alerts = [{"id": f"bulk-{i}", "type": "new_person", "entity_id": "cam", "severity": "low",
                   "dedupe_key": f"bulk-{i}", "ts_ms": i, "payload": {}} for i in range(4)]
        await repository.accept([], alerts, [], [])
        response = await client.post("/v1/alerts/delete", json={"ids": ["bulk-0", "bulk-1"]})
        assert response.status_code == 200, response.text
        assert response.json()["deleted_count"] == 2
        left = {a["id"] for a in (await client.get("/v1/alerts?limit=100")).json()}
        assert {"bulk-2", "bulk-3"} <= left and not {"bulk-0", "bulk-1"} & left
        assert (await client.post("/v1/alerts/delete", json={"all": True})).json()["deleted_count"] >= 2
        assert (await client.get("/v1/alerts?limit=100")).json() == []
        for body in ({}, {"ids": []}, {"ids": ["x"], "all": True}):
            assert (await client.post("/v1/alerts/delete", json=body)).status_code == 422
        evidence = await client.post("/v1/evidence/records/delete", json={"ids": ["unknown"]})
        assert evidence.status_code in {200, 409}
        if evidence.status_code == 200:
            assert evidence.json() == {"deleted": [], "skipped": [{"id": "unknown", "reason": "not found"}]}
