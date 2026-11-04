"""Manual cleanup of alerts and their evidence (dashboard selection and bulk delete).

Evidence goes through the evidence service's own deletion, which removes the
clip and snapshot but keeps a tombstone record for audit and refuses captures
that are still being written. An alert is removed together with its evidence;
if that evidence cannot be removed yet, the alert is kept so nothing is left
half-deleted, and the reason is reported.
"""
from __future__ import annotations

from typing import Any

# Upper bound for "all": large enough for any dashboard cleanup, bounded for the request.
MAX_BULK = 100_000


async def delete_evidence(application, evidence_ids: list[str] | None, *, everything: bool = False) -> dict[str, Any]:
    """Delete evidence records by id, or every not-yet-deleted record."""
    captures = application.evidence
    if captures is None:
        raise ValueError("evidence capture is disabled")
    if everything:
        records = await application.repository.list_evidence(None, MAX_BULK)
        evidence_ids = [record["id"] for record in records if not record.get("deleted")]
    deleted, skipped = [], []
    for evidence_id in dict.fromkeys(evidence_ids or []):
        reason = await _delete_one(captures, evidence_id)
        if reason is None:
            deleted.append(evidence_id)
        else:
            skipped.append({"id": evidence_id, "reason": reason})
    return {"deleted": deleted, "skipped": skipped}


async def delete_alerts(application, alert_ids: list[str] | None, *, everything: bool = False,
                        with_evidence: bool = True) -> dict[str, Any]:
    """Delete alerts by id (or all of them), each together with its evidence."""
    repository = application.repository
    alerts = await repository.list_incidents(MAX_BULK)
    by_id = {alert["id"]: alert for alert in alerts}
    wanted = list(by_id) if everything else list(dict.fromkeys(alert_ids or []))
    removable, skipped, evidence_deleted = [], [], []
    for alert_id in wanted:
        alert = by_id.get(alert_id)
        if alert is None:
            skipped.append({"id": alert_id, "reason": "not found"})
            continue
        evidence_id = (alert.get("payload") or {}).get("evidence_id")
        if with_evidence and evidence_id and application.evidence is not None:
            reason = await _delete_one(application.evidence, evidence_id)
            if reason == "not found":
                pass  # the alert never had stored evidence; remove it anyway
            elif reason is not None:
                skipped.append({"id": alert_id, "reason": f"evidence: {reason}"})
                continue
            else:
                evidence_deleted.append(evidence_id)
        removable.append(alert_id)
    count = await repository.delete_incidents(removable)
    return {"deleted": removable, "deleted_count": count, "evidence_deleted": evidence_deleted, "skipped": skipped}


async def _delete_one(captures, evidence_id: str) -> str | None:
    """None when deleted (or already deleted), else the reason it was not."""
    try:
        await captures.delete(evidence_id)
    except KeyError:
        return "not found"
    except ValueError as error:
        return str(error)              # e.g. capture or delivery still pending
    except OSError:
        return "artifact deletion pending; retry"
    return None
