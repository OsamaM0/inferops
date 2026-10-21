"""Explicit, reviewable cleanup using the existing evidence tombstone workflow."""
from datetime import datetime


class RetentionService:
    def __init__(self, repository, evidence):
        self.repository, self.evidence = repository, evidence

    async def plan(self, before: datetime, *, limit=100):
        if before.tzinfo is None or not 1 <= limit <= 1000:
            raise ValueError("timezone-aware cutoff and limit 1..1000 are required")
        rows = await self.repository.retention_candidates(before, limit)
        return {"dry_run": True, "before": before.isoformat(), "candidates": rows,
                "policy": "terminal evidence only; pending captures, deletion and delivery excluded"}

    async def apply(self, evidence_ids, before: datetime):
        results = []
        for evidence_id in evidence_ids:
            # Recheck eligibility immediately before the owner's tombstone operation.
            eligible = await self.repository.retention_candidates(before, 1, evidence_id=evidence_id)
            if not eligible:
                results.append({"id": evidence_id, "status": "skipped"})
                continue
            try:
                row = await self.evidence.delete(evidence_id, retention_before=before)
            except ValueError:
                results.append({"id": evidence_id, "status": "skipped"})
                continue
            results.append({"id": evidence_id, "status": "deleted" if not row.get("deletion_pending") else "pending"})
        return results
