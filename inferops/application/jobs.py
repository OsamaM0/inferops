"""Persistent job admission, cancellation and explicit retries."""
from __future__ import annotations

import asyncio
import uuid

from inferops.domain.errors import PipelineError


class JobService:
    def __init__(self, engine, *, max_running=2, max_queued=64):
        if max_running < 1 or max_queued < 1:
            raise ValueError("job capacities must be positive")
        self.engine = engine
        self.repository = engine.application.repository
        self.max_queued = max_queued
        self.capacity = asyncio.Semaphore(max_running)
        self.tasks = {}
        self.closing = False
        self._admission = asyncio.Lock()

    async def start(self):
        await self.repository.interrupt_jobs()
        for row in await self.repository.list_queued_jobs(limit=self.max_queued):
            self._schedule(row["id"], row["pipeline"], row.get("input") or {})

    def _schedule(self, job_id, pipeline, payload):
        task = asyncio.create_task(self._execute(job_id, pipeline, payload), name=f"job-{job_id}")
        self.tasks[job_id] = task
        task.add_done_callback(lambda done: self.tasks.pop(job_id, None))

    async def submit(self, pipeline, payload):
        if pipeline not in self.engine.application.definition.pipelines:
            raise KeyError(pipeline)
        async with self._admission:
            if self.closing or len(self.tasks) >= self.max_queued:
                raise PipelineError("job capacity exhausted")
            job_id = uuid.uuid4().hex
            await self.repository.create_job({"id": job_id, "pipeline": pipeline, "status": "queued",
                                              "result": None, "input": payload})
            self._schedule(job_id, pipeline, dict(payload))
        return {"job_id": job_id, "pipeline": pipeline, "status": "queued"}

    async def _execute(self, job_id, pipeline, payload):
        running = False
        try:
            async with self.capacity:
                await self.repository.update_job(job_id, status="running")
                running = True
                summary = await self.engine.run(pipeline, payload, stream_name=payload.get("stream_id", ""),
                                                owner_kind="job")
                await self.repository.update_job(job_id, status="succeeded", result=summary)
        except asyncio.CancelledError:
            if not self.closing:
                await self.repository.update_job(job_id, status="cancelled", result={"reason": "cancelled"})
            elif running:
                await self.repository.update_job(job_id, status="failed", result={"reason": "process_interrupted"})
            raise
        except Exception as exc:
            await self.repository.update_job(job_id, status="failed", result={"error": str(exc)})

    async def cancel(self, job_id):
        if await self.repository.get_job(job_id) is None:
            raise KeyError(job_id)
        task = self.tasks.get(job_id)
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        row = await self.repository.get_job(job_id)
        if row and row["status"] in {"queued", "running"}:
            await self.repository.update_job(job_id, status="cancelled", result={"reason": "cancelled"})
        return await self.repository.get_job(job_id)

    async def retry(self, job_id):
        row = await self.repository.get_job(job_id)
        if row is None:
            raise KeyError(job_id)
        if row["status"] not in {"failed", "cancelled"}:
            raise ValueError("only failed or cancelled jobs may be retried")
        return await self.submit(row["pipeline"], {**row.get("input", {}), "retry_of": job_id})

    async def close(self):
        self.closing = True
        for task in list(self.tasks.values()):
            task.cancel()
        await asyncio.gather(*list(self.tasks.values()), return_exceptions=True)
