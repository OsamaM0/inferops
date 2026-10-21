"""Disposable PostgreSQL databases for tests; never use the operator's database.

INFEROPS_TEST_PG_URL must point to a disposable PostgreSQL instance whose user
can create databases. Only databases created by this process are removed.
"""
from __future__ import annotations

import asyncio
import atexit
import hashlib
import os
import threading
import uuid

from sqlalchemy.engine import make_url

_nonce = uuid.uuid4().hex[:12]
_databases = {}
_lock = threading.Lock()


def _run(coroutine):
    # A raw thread also works during atexit, after executors have shut down.
    result, errors = [], []

    def run():
        try:
            result.append(asyncio.run(coroutine))
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
    return result[0]


def test_database_url(key="server"):
    """Return a fresh, stable database URL for one process-local test key."""
    base = os.getenv("INFEROPS_TEST_PG_URL")
    if not base or not base.startswith("postgresql"):
        raise RuntimeError("set INFEROPS_TEST_PG_URL to a disposable PostgreSQL instance before running database tests")
    parsed = make_url(base).set(drivername="postgresql+asyncpg")
    digest = hashlib.sha256(str(key).encode()).hexdigest()[:16]
    name = f"inferops_test_{_nonce}_{digest}"
    with _lock:
        if name not in _databases:
            async def create():
                import asyncpg
                conn = await asyncpg.connect(parsed.set(drivername="postgresql").render_as_string(hide_password=False))
                try:
                    await conn.execute(f'CREATE DATABASE "{name}"')
                finally:
                    await conn.close()
            _run(create())
            _databases[name] = parsed
    return parsed.set(database=name).render_as_string(hide_password=False)


# Prevent pytest from collecting the imported helper as a test function.
test_database_url.__test__ = False


@atexit.register
def _cleanup():
    async def drop():
        import asyncpg
        for name, parsed in _databases.items():
            conn = await asyncpg.connect(parsed.set(drivername="postgresql").render_as_string(hide_password=False))
            try:
                await conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
            finally:
                await conn.close()
    if _databases:
        _run(drop())
