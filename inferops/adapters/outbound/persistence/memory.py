"""Service-free persistence for embedded execution and focused tests."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone


class MemoryRepository:
    def __init__(self):
        self.sessions = {}
        self.events = {}
        self.incidents = {}
        self.evidence = {}
        self.outbox = {}
        self.jobs = {}
        self.inbox = set()

    async def start(self): pass
    async def close(self): pass

    async def save_session(self, row):
        self.sessions[row["id"]] = deepcopy(row)
        return deepcopy(row)

    async def get_session(self, session_id):
        return deepcopy(self.sessions.get(session_id))

    async def close_session(self, session_id, status="closed"):
        if session_id in self.sessions:
            self.sessions[session_id]["state"] = status
            self.sessions[session_id]["status"] = status

    async def interrupt_other_sessions(self, current_owner):
        count = 0
        for row in self.sessions.values():
            if row.get("status") == "open" and row.get("owner") != current_owner:
                row.update(status="interrupted", state="interrupted", generation=row.get("generation", 1) + 1)
                count += 1
        return count

    async def accept(self, events, incidents, evidence, outbox, *, inbox=None):
        if inbox is not None:
            if inbox in self.inbox:
                return False
            self.inbox.add(inbox)
        for store, records in ((self.events, events), (self.incidents, incidents),
                               (self.evidence, evidence), (self.outbox, outbox)):
            for row in records:
                store.setdefault(row["id"], deepcopy(row))
                if store is self.outbox:
                    now = datetime.now(timezone.utc)
                    store[row["id"]].setdefault("status", "pending")
                    store[row["id"]].setdefault("created_at", now)
                    store[row["id"]].setdefault("available_at", now)
                    store[row["id"]].setdefault("generation", 0)
                    store[row["id"]].setdefault("attempts", 0)
        return True

    async def list_events(self, pipeline=None, limit=100):
        return deepcopy([v for v in self.events.values() if pipeline is None or v["pipeline"] == pipeline][-limit:])

    async def list_incidents(self, limit=100):
        return deepcopy(list(self.incidents.values())[-limit:])

    async def list_evidence(self, session_id=None, limit=100):
        return deepcopy([v for v in self.evidence.values() if session_id is None or v["session_id"] == session_id][-limit:])

    async def get_evidence(self, evidence_id):
        return deepcopy(self.evidence.get(evidence_id))

    async def update_evidence(self, evidence_id, **changes):
        row = self.evidence[evidence_id]
        row.update(deepcopy(changes))
        return deepcopy(row)

    async def claim_outbox(self, owner, limit=16, lease_s=30):
        now = datetime.now(timezone.utc)
        selected = []
        for row in self.outbox.values():
            if len(selected) >= limit:
                break
            if row["status"] != "pending" or row["available_at"] > now or (row.get("lease_until") and row["lease_until"] > now):
                continue
            row.update(owner=owner, lease_until=now + timedelta(seconds=lease_s),
                       generation=row["generation"] + 1, attempts=row["attempts"] + 1)
            selected.append(deepcopy(row))
        return selected

    def _claim_valid(self, row, owner, generation):
        return bool(row and row["status"] == "pending" and row.get("owner") == owner
                    and row["generation"] == generation and row["lease_until"] > datetime.now(timezone.utc))

    async def ack_outbox(self, message_id, owner, generation):
        row = self.outbox.get(message_id)
        if not self._claim_valid(row, owner, generation):
            return False
        row.update(status="delivered", lease_until=None)
        return True

    async def retry_outbox(self, message_id, owner, generation, error, delay_s):
        row = self.outbox.get(message_id)
        if not self._claim_valid(row, owner, generation):
            return False
        row.update(error=error, available_at=datetime.now(timezone.utc) + timedelta(seconds=delay_s), lease_until=None)
        return True

    async def record_inbox(self, event_id, consumer):
        key = event_id, consumer
        if key in self.inbox:
            return False
        self.inbox.add(key)
        return True

    async def create_job(self, row):
        self.jobs.setdefault(row["id"], deepcopy(row))

    async def update_job(self, job_id, *, status, result=None):
        self.jobs[job_id].update(status=status, result=deepcopy(result))

    async def get_job(self, job_id):
        return deepcopy(self.jobs.get(job_id))

    async def list_jobs(self, limit=100):
        return deepcopy(list(reversed(list(self.jobs.values())))[:max(0, limit)])

    async def list_queued_jobs(self, limit=64):
        return deepcopy([row for row in self.jobs.values() if row["status"] == "queued"][:max(0, limit)])

    async def interrupt_jobs(self):
        count = 0
        for row in self.jobs.values():
            if row["status"] == "running":
                row.update(status="failed", result={"reason": "process_interrupted"})
                count += 1
        return count
