"""PostgreSQL source catalog and bounded history, scoped to the control owner.

Incident/evidence records remain in the pipeline repository. Preview JPEGs are
replaceable local cache files; camera definitions and history are durable SQL.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import time
from pathlib import Path

from sqlalchemy import BigInteger, Column, Float, Index, MetaData, String, Table, select, text
from sqlalchemy.dialects.postgresql import JSONB, insert

from inferops.adapters.outbound.persistence.sql import SqlRepository
from inferops.domain.sources import SourceSpec


class MemorySourceStore:
    """Explicit embedded/test catalog; durable servers use PostgreSQL."""

    def __init__(self, *, results_per_camera=1000):
        self.results_per_camera = max(1, results_per_camera)
        self.sources, self.runs, self.results, self.previews = {}, {}, {}, {}

    async def start(self):
        pass

    async def close(self):
        pass

    async def list_sources(self):
        return [self.sources[key] for key in sorted(self.sources)]

    async def put_source(self, spec):
        self.sources[spec.camera_id] = spec

    async def delete_source(self, camera_id):
        self.sources.pop(camera_id, None)

    async def save_run(self, row):
        self.runs[row["session_id"]] = dict(row)

    async def list_runs(self, camera_id, limit=50):
        return sorted((row for row in self.runs.values() if row["camera_id"] == camera_id),
                      key=lambda row: row["started_at"], reverse=True)[:limit]

    async def save_result(self, camera_id, session_id, row):
        rows = self.results.setdefault(camera_id, [])
        rows.append(dict(row))
        del rows[:-self.results_per_camera]

    async def list_results(self, camera_id, limit=50):
        return list(reversed(self.results.get(camera_id, [])))[:limit]

    async def save_preview(self, camera_id, jpeg):
        self.previews[camera_id] = jpeg

    async def get_preview(self, camera_id):
        return self.previews.get(camera_id)


class PostgresSourceStore:
    def __init__(self, repository, *, preview_root=".run/previews", results_per_camera=1000, scope=None):
        self._owns_repository = isinstance(repository, str)
        self.repository = SqlRepository(repository) if self._owns_repository else repository
        self.url = self.repository.url
        self.scope = scope or getattr(self.repository, "scope", "combined")
        self.preview_root = Path(preview_root)
        self.results_per_camera = max(1, results_per_camera)
        self._queue = asyncio.Queue(maxsize=256)
        self._writer = None
        self._error = None
        self.metadata = MetaData(schema=self.repository.sessions.schema)
        self.sources = Table("inferops_sources", self.metadata,
            Column("scope", String, primary_key=True), Column("id", String, primary_key=True),
            Column("spec", JSONB, nullable=False))
        self.runs = Table("inferops_source_runs", self.metadata,
            Column("scope", String, primary_key=True), Column("id", String, primary_key=True),
            Column("camera_id", String, nullable=False), Column("started", Float, nullable=False),
            Column("body", JSONB, nullable=False))
        Index("inferops_source_runs_camera", self.runs.c.scope, self.runs.c.camera_id, self.runs.c.started)
        self.results = Table("inferops_source_results", self.metadata,
            Column("id", BigInteger, primary_key=True, autoincrement=True), Column("scope", String, nullable=False),
            Column("camera_id", String, nullable=False), Column("session_id", String, nullable=False),
            Column("body", JSONB, nullable=False))
        Index("inferops_source_results_camera", self.results.c.scope, self.results.c.camera_id, self.results.c.id)

    async def start(self):
        await self.repository.start()
        try:
            async with self.repository._transaction() as conn:
                await conn.execute(text("SELECT pg_advisory_xact_lock(731914215)"))
                await conn.run_sync(self.metadata.create_all)
                rows = (await conn.execute(select(self.runs).where(self.runs.c.scope == self.scope))).mappings()
                for row in rows:
                    body = row["body"]
                    if body.get("status") in {"starting", "running", "reconnecting"}:
                        body = {**body, "status": "interrupted", "finished_at": time.time()}
                        await conn.execute(self.runs.update().where(
                            self.runs.c.scope == self.scope, self.runs.c.id == row["id"]).values(body=body))
        except BaseException:
            if self._owns_repository:
                await self.repository.close()
            raise
        if self._writer is None:
            self._writer = asyncio.create_task(self._write_results(), name="source-history-writer")

    async def close(self):
        await self._queue.join()
        if self._writer:
            self._writer.cancel()
            await asyncio.gather(self._writer, return_exceptions=True)
            self._writer = None
        if self._owns_repository:
            await self.repository.close()
        if self._error:
            raise RuntimeError("source result history could not be persisted") from self._error

    async def list_sources(self):
        async with self.repository.engine.connect() as conn:
            rows = await conn.scalars(select(self.sources.c.spec).where(
                self.sources.c.scope == self.scope).order_by(self.sources.c.id))
            return [SourceSpec.model_validate(row) for row in rows]

    async def put_source(self, spec):
        statement = insert(self.sources).values(scope=self.scope, id=spec.camera_id, spec=spec.model_dump(mode="json"))
        async with self.repository._transaction() as conn:
            await conn.execute(statement.on_conflict_do_update(
                index_elements=[self.sources.c.scope, self.sources.c.id], set_={"spec": statement.excluded.spec}))

    async def delete_source(self, camera_id):
        # Preserve historical runs/results and completed evidence.
        async with self.repository._transaction() as conn:
            await conn.execute(self.sources.delete().where(
                self.sources.c.scope == self.scope, self.sources.c.id == camera_id))

    async def save_run(self, row):
        statement = insert(self.runs).values(scope=self.scope, id=row["session_id"],
            camera_id=row["camera_id"], started=row["started_at"], body=dict(row))
        async with self.repository._transaction() as conn:
            await conn.execute(statement.on_conflict_do_update(
                index_elements=[self.runs.c.scope, self.runs.c.id], set_={"body": statement.excluded.body}))

    async def list_runs(self, camera_id, limit=50):
        async with self.repository.engine.connect() as conn:
            return list(await conn.scalars(select(self.runs.c.body).where(
                self.runs.c.scope == self.scope, self.runs.c.camera_id == camera_id)
                .order_by(self.runs.c.started.desc()).limit(limit)))

    async def save_result(self, camera_id, session_id, row):
        if self._error:
            raise RuntimeError("source result history writer failed") from self._error
        await self._queue.put((camera_id, session_id, dict(row)))

    async def _write_results(self):
        while True:
            batch = [await self._queue.get()]
            await asyncio.sleep(0.25)
            while not self._queue.empty() and len(batch) < 128:
                batch.append(self._queue.get_nowait())
            try:
                rows = [dict(scope=self.scope, camera_id=camera, session_id=session, body=body)
                        for camera, session, body in batch if camera is not None]
                if rows:
                    async with self.repository._transaction() as conn:
                        await conn.execute(self.results.insert(), rows)
                        for camera_id in {row["camera_id"] for row in rows}:
                            condition = (self.results.c.scope == self.scope, self.results.c.camera_id == camera_id)
                            retained = select(self.results.c.id).where(*condition).order_by(
                                self.results.c.id.desc()).limit(self.results_per_camera)
                            await conn.execute(self.results.delete().where(*condition, ~self.results.c.id.in_(retained)))
            except Exception as error:
                self._error = error
            finally:
                for camera_id, _, value in batch:
                    self._queue.task_done()
                    if camera_id is None:
                        value.set()

    async def list_results(self, camera_id, limit=50):
        flushed = asyncio.Event()
        await self._queue.put((None, None, flushed))
        await flushed.wait()
        if self._error:
            raise RuntimeError("source result history writer failed") from self._error
        async with self.repository.engine.connect() as conn:
            return list(await conn.scalars(select(self.results.c.body).where(
                self.results.c.scope == self.scope, self.results.c.camera_id == camera_id)
                .order_by(self.results.c.id.desc()).limit(limit)))

    def _preview_path(self, camera_id):
        key = hashlib.sha256(f"{self.scope}/{camera_id}".encode()).hexdigest()
        return self.preview_root / f"{key}.jpg"

    async def save_preview(self, camera_id, jpeg):
        path = self._preview_path(camera_id)

        def write():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(jpeg)
            os.replace(temporary, path)

        await asyncio.to_thread(write)

    async def get_preview(self, camera_id):
        path = self._preview_path(camera_id)
        return await asyncio.to_thread(path.read_bytes) if path.is_file() else None
