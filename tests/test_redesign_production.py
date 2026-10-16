"""Optional production-bundle checks using disposable schemas and S3 buckets.

Set INFEROPS_TEST_PG_URL and INFEROPS_TEST_S3_ENDPOINT explicitly. These tests
never reuse application schemas or buckets and clean their own data on exit.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import subprocess
import sys
from types import SimpleNamespace
from urllib.parse import urlparse
from uuid import uuid4

import cv2
import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import create_async_engine

from inferops import Engine
from inferops.adapters.inbound.http.app import create_application
from inferops.adapters.outbound.persistence.sql import SqlRepository
from inferops.adapters.outbound.storage.s3 import S3Storage
from inferops.domain.evidence import EvidenceConfig
from tests.test_redesign_engine import project


@pytest_asyncio.fixture
async def production_bundle(tmp_path):
    url = os.environ.get("INFEROPS_TEST_PG_URL")
    endpoint = os.environ.get("INFEROPS_TEST_S3_ENDPOINT")
    if not url or not endpoint:
        pytest.skip("set INFEROPS_TEST_PG_URL and INFEROPS_TEST_S3_ENDPOINT for production-bundle checks")
    if not url.startswith("postgresql"):
        pytest.skip("production-bundle checks require PostgreSQL")
    admin = create_async_engine(url.replace("postgresql://", "postgresql+asyncpg://", 1))
    schema = f"inferops_it_{uuid4().hex}"
    bucket = f"inferops-redesign-{uuid4().hex}"
    repository = SqlRepository(url)
    for table in repository.metadata.tables.values():
        table.schema = schema
    storage = S3Storage(
        endpoint=endpoint,
        bucket=bucket,
        access_key=os.environ.get("INFEROPS_TEST_S3_ACCESS_KEY", "minio"),
        secret_key=os.environ.get("INFEROPS_TEST_S3_SECRET_KEY", "minio12345"),
    )
    bucket_created = False
    try:
        async with admin.begin() as conn:
            await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
        await repository.start()
        await storage.start()
        bucket_created = True
        yield SimpleNamespace(
            repository=repository,
            storage=storage,
            project=project(tmp_path),
            spool=tmp_path / "persistent-spool",
            schema=schema,
        )
    finally:
        await repository.close()
        await storage.close()
        if bucket_created:

            def cleanup_bucket():
                import boto3
                from botocore.config import Config

                client = boto3.client("s3", **storage.options, config=Config(signature_version="s3v4"))
                try:
                    pages = client.get_paginator("list_objects_v2").paginate(Bucket=bucket)
                    for page in pages:
                        keys = [{"Key": item["Key"]} for item in page.get("Contents", [])]
                        if keys:
                            client.delete_objects(Bucket=bucket, Delete={"Objects": keys, "Quiet": True})
                    client.delete_bucket(Bucket=bucket)
                finally:
                    client.close()

            await asyncio.to_thread(cleanup_bucket)
        async with admin.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await admin.dispose()


def production_engine(bundle, *, retry_delays=()):
    return Engine.from_project(
        bundle.project,
        repository=bundle.repository,
        storage=bundle.storage,
        evidence=True,
        spool_path=bundle.spool,
        evidence_config=EvidenceConfig(
            pre_seconds=0, post_seconds=0, disk_reserve_bytes=0, retry_delays=retry_delays
        ),
    )


async def test_python_and_http_pipeline_persist_to_postgres_and_minio(production_bundle):
    bundle = production_bundle
    selected = production_engine(bundle)
    image = np.zeros((32, 32, 3), np.uint8)
    _, jpeg = cv2.imencode(".jpg", image)
    app = create_application(selected)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client,
    ):
        direct = await selected.run("test", image, stream_name="same-external-name")
        response = await client.post(
            "/v1/infer/test",
            files={"image": ("test.jpg", jpeg.tobytes())},
            data={"stream_id": "same-external-name", "media_time_ns": "0", "seq": "0"},
        )
        assert response.status_code == 200, response.text
        remote = response.json()[0]
        assert remote["payload"]["session_id"] != direct["session_id"]
        assert len(remote["payload"]["alerts"]) == direct["incident_count"] == 1
        assert remote["payload"]["alerts"][0]["type"] == "hot"
        await selected.application.evidence.wait_idle()
        records = (await client.get("/v1/evidence/records")).json()
        assert len(records) == 2
        assert {record["state"] for record in records} == {"Saved"}
        assert {record["session_id"] for record in records} == {direct["session_id"], remote["session_id"]}
        for record in records:
            assert set(record["artifacts"]) == {"snapshot", "clip"}
            for artifact in record["artifacts"].values():
                assert await bundle.storage.verify(artifact["uri"], artifact["sha256"])
                assert (
                    hashlib.sha256(await bundle.storage.read(artifact["uri"])).hexdigest()
                    == artifact["sha256"]
                )
            media = await client.get(
                f"/v1/evidence/records/{record['id']}/assets/clip", headers={"Range": "bytes=0-15"}
            )
            assert media.status_code == 206 and len(media.content) == 16
        incidents = await bundle.repository.list_incidents()
        assert len(incidents) == 2
        assert all(len(row["payload"]["evidence_uris"]) == 2 for row in incidents)
        assert (await bundle.repository.diagnostics())["pending_outbox"] > 0


async def test_minio_checksum_and_idempotent_retry_preserve_object(production_bundle, monkeypatch):
    storage = production_bundle.storage
    data = b"deterministic production artifact"
    digest = hashlib.sha256(data).hexdigest()
    uri = await storage.put("contract/artifact.bin", data)
    assert await storage.verify(uri, digest)
    assert not await storage.verify(uri, "0" * 64)
    assert hashlib.sha256(await storage.read(uri)).hexdigest() == digest
    assert not await storage.verify(f"s3://{storage.bucket}/absent.bin", digest)
    writes = []
    original = storage.client.put_object

    def count_write(**kwargs):
        writes.append(kwargs["Key"])
        return original(**kwargs)

    monkeypatch.setattr(storage.client, "put_object", count_write)
    assert await storage.put("contract/artifact.bin", data) == uri
    assert writes == []
    assert await storage.read(uri) == data
    await storage.delete(uri)
    assert not await storage.verify(uri, digest)


async def test_restart_recovers_partial_upload_and_legacy_reader_can_serve_new_data(
    production_bundle, monkeypatch
):

    bundle = production_bundle
    # Leave the failed upload actionable until shutdown instead of exhausting
    # its retry budget. Startup resumes Pending material, never terminal states.
    selected = production_engine(bundle, retry_delays=(3600,))
    original_put = bundle.storage.put

    async def fail_clip_once(key, data, content_type="application/octet-stream"):
        if key.endswith("clip.mp4"):
            raise OSError("simulated MinIO interruption after snapshot commit")
        return await original_put(key, data, content_type)

    async with selected:
        monkeypatch.setattr(bundle.storage, "put", fail_clip_once)
        summary = await selected.run("test", np.zeros((32, 32, 3), np.uint8), stream_name="recovery")
        await selected.application.evidence.wait_idle()
        partial = (await bundle.repository.list_evidence())[0]
        assert partial["state"] == "Pending"
        assert set(partial["artifacts"]) == {"snapshot"}
        assert partial["retry_available"] is True
        assert await selected.application.evidence.spool.list_ids() == [partial["id"]]
        evidence_id = partial["id"]
        snapshot_uri = partial["artifacts"]["snapshot"]["uri"]
    monkeypatch.setattr(bundle.storage, "put", original_put)
    restarted = production_engine(bundle)
    app = create_application(restarted)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client,
    ):
        await restarted.application.evidence.wait_idle()
        recovered = await bundle.repository.get_evidence(evidence_id)
        assert recovered["state"] == "Saved"
        assert recovered["artifacts"]["snapshot"]["uri"] == snapshot_uri
        assert recovered["session_id"] == summary["session_id"]
        assert await restarted.application.evidence.spool.list_ids() == []
        assert len(await bundle.repository.list_incidents()) == 1
        async with bundle.repository.engine.begin() as conn:
            legacy = dict((await conn.execute(select(bundle.repository.incidents))).mappings().first())
        assert legacy["payload"]["evidence_id"] == evidence_id
        assert legacy["payload"]["evidence_state"] == "Saved"
        assert set(legacy["payload"]["evidence_uris"]) == {
            artifact["uri"] for artifact in recovered["artifacts"].values()
        }
        response = await client.get(f"/v1/evidence/records/{evidence_id}/assets/clip")
        assert response.status_code == 200
        assert hashlib.sha256(response.content).hexdigest() == recovered["artifacts"]["clip"]["sha256"]
        ranged = await client.get(
            f"/v1/evidence/records/{evidence_id}/assets/clip", headers={"Range": "bytes=0-15"}
        )
        assert ranged.status_code == 206 and ranged.content == response.content[:16]
        snapshot = await client.get(f"/v1/evidence/records/{evidence_id}/assets/snapshot")
        assert snapshot.status_code == 200
        assert hashlib.sha256(snapshot.content).hexdigest() == recovered["artifacts"]["snapshot"]["sha256"]
        claim = (await bundle.repository.claim_outbox("production-relay", limit=100))[0]
        assert await bundle.repository.ack_outbox(claim["id"], "production-relay", claim["generation"])
        assert not await bundle.repository.ack_outbox(claim["id"], "production-relay", claim["generation"])
        assert urlparse(snapshot_uri).netloc == bundle.storage.bucket


async def test_archived_release_can_read_recover_and_process_new_data(production_bundle, monkeypatch, tmp_path):
    """Rollback lives in a previous release artifact, not duplicate current code."""
    from pathlib import Path

    wheel = Path(
        os.environ.get(
            "INFEROPS_TEST_ROLLBACK_WHEEL", ".run/verification/pre-cleanup/inferops-0.1.0-py3-none-any.whl"
        )
    ).resolve()
    if not wheel.is_file():
        pytest.skip("provide INFEROPS_TEST_ROLLBACK_WHEEL to exercise a previous release")
    import zipfile
    release = tmp_path / "rollback-release"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(release)
    bundle = production_bundle
    original_put = bundle.storage.put

    async def unavailable(key, data, content_type="application/octet-stream"):
        if key.endswith("clip.mp4"):
            raise OSError("upload interrupted at cutover")
        return await original_put(key, data, content_type)

    async with production_engine(bundle, retry_delays=(3600,)) as selected:
        monkeypatch.setattr(bundle.storage, "put", unavailable)
        before = await selected.run("test", np.zeros((32, 32, 3), np.uint8))
        evidence_id = before["evidence"][0]["id"]
        assert before["evidence"][0]["state"] == "Pending"
    monkeypatch.setattr(bundle.storage, "put", original_put)
    # A release switch restores matching project imports as well as the wheel.
    import shutil
    rollback_project = tmp_path / "rollback_project"
    shutil.copytree(bundle.project, rollback_project)
    for path in rollback_project.rglob("*.py"):
        content = path.read_text().replace("inferops.adapters.outbound.stages.base", "inferops.pipeline.stage")
        content = content.replace("inferops.domain.types", "inferops.core.types")
        content = content.replace("inferops.domain.models", "inferops.registry.model_spec")
        content = content.replace("inferops.ports.inference", "inferops.runtimes.base")
        path.write_text(content)
    script = r"""import asyncio, os, sys
sys.path.insert(0,sys.argv[1])
import inferops
assert sys.argv[1] in inferops.__file__
from inferops import Engine, bootstrap
from inferops.api.main import create_app
from inferops.adapters import postgres_dao
from inferops.adapters.outbound.persistence.sql import SqlRepository
from inferops.adapters.outbound.storage.s3 import S3Storage
from inferops.adapters.outbound.messaging.delivery import DeliveryHub, ServerDelivery
from inferops.adapters.transport.inproc import InProcTransport
from inferops.domain.evidence import EvidenceConfig
from httpx import ASGITransport, AsyncClient
async def main():
    wheel,root,schema,spool,evidence_id=sys.argv[1:]
    repo=SqlRepository(os.environ['INFEROPS_TEST_PG_URL'])
    for table in repo.metadata.tables.values(): table.schema=schema
    await repo.start()
    postgres_dao._engine=repo.engine.execution_options(schema_translate_map={None:schema})
    storage=S3Storage(endpoint=os.environ['INFEROPS_TEST_S3_ENDPOINT'],bucket=os.environ['INFEROPS_CRASH_BUCKET'],
      access_key=os.environ.get('INFEROPS_TEST_S3_ACCESS_KEY','minio'),secret_key=os.environ.get('INFEROPS_TEST_S3_SECRET_KEY','minio12345'))
    recovery=Engine.from_project(root,repository=repo,storage=storage,evidence=True,spool_path=spool,
      evidence_config=EvidenceConfig(pre_seconds=0,post_seconds=0,disk_reserve_bytes=0,retry_delays=()))
    recovery.application.transport=InProcTransport()
    recovery.application.delivery=ServerDelivery(DeliveryHub(),transport=recovery.application.transport,definitions=recovery.application.definition)
    bootstrap.build_server_engine=lambda **kwargs:(recovery,DeliveryHub())
    from inferops.api.routes import evidence_browser
    evidence_browser.s3_client=lambda:storage.client
    app=create_app(legacy=True)
    async with app.router.lifespan_context(app),AsyncClient(transport=ASGITransport(app),base_url='http://rollback') as client:
        await recovery.application.evidence.wait_idle()
        row=await repo.get_evidence(evidence_id)
        assert row['state']=='Saved'
        alerts=(await client.get('/v1/alerts')).json()
        assert alerts[0]['payload']['evidence_id']==evidence_id
        from urllib.parse import urlparse
        uri=urlparse(row['artifacts']['clip']['uri'])
        media=await client.get('/v1/evidence/file/'+uri.netloc+uri.path,headers={'Range':'bytes=0-15'})
        assert media.status_code==206 and len(media.content)==16
        result=await client.post('/v1/infer/test',data={'stream_id':'rollback-extra'})
        assert result.status_code==200 and result.json()[0]['payload']['alerts']
        assert len((await client.get('/v1/alerts')).json())==2
        await recovery.application.relay.flush()
asyncio.run(main())
"""
    environment = {
        **os.environ,
        "INFEROPS_ENGINE": "legacy",
        "INFEROPS_RECOVERY_ON_LEGACY": "1",
        "INFEROPS_DOMAIN_ROOT": str(rollback_project),
        "INFEROPS_PROJECT_ROOT": "",
        "INFEROPS_ACTIVE_DOMAIN": "",
        "INFEROPS_PREWARM_MODELS": "",
        "INFEROPS_PREWARM_PIPELINES": "",
        "INFEROPS_TRANSPORT_KIND": "inproc",
        "INFEROPS_CRASH_BUCKET": bundle.storage.bucket,
    }
    process = await asyncio.to_thread(
        subprocess.run,
        [
            sys.executable,
            "-c",
            script,
            str(release),
            str(rollback_project),
            bundle.schema,
            str(bundle.spool),
            evidence_id,
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    async with production_engine(bundle) as upgraded:
        row = await bundle.repository.get_evidence(evidence_id)
        assert row["state"] == "Saved"
        preserved = row["artifacts"]
        await upgraded.run("test", np.zeros((32, 32, 3), np.uint8))
        assert len(await bundle.repository.list_incidents()) == 3
        assert (await bundle.repository.get_evidence(evidence_id))["artifacts"] == preserved


@pytest.mark.parametrize(
    ("boundary", "expected"),
    [("acceptance", "Failed"), ("spool", "Incomplete"), ("upload", "Saved"), ("acknowledgment", None)],
)
async def test_process_death_at_real_durability_boundaries(production_bundle, boundary, expected):
    from datetime import timedelta

    from inferops.application.delivery import OutboxRelay

    bundle = production_bundle
    # os._exit bypasses finally blocks, resource shutdown and Python buffering.
    # The child's only database/object destinations are fixture-owned resources.
    script = """import asyncio, os, sys
from pathlib import Path
import numpy as np
from inferops import Engine
from inferops.adapters.outbound.persistence.sql import SqlRepository
from inferops.adapters.outbound.storage.s3 import S3Storage
from inferops.domain.evidence import EvidenceConfig

async def main():
    root, schema, spool, boundary = sys.argv[1:]
    repo = SqlRepository(os.environ['INFEROPS_TEST_PG_URL'])
    for table in repo.metadata.tables.values(): table.schema = schema
    storage = S3Storage(endpoint=os.environ['INFEROPS_TEST_S3_ENDPOINT'],
        bucket=os.environ['INFEROPS_CRASH_BUCKET'],
        access_key=os.environ.get('INFEROPS_TEST_S3_ACCESS_KEY','minio'),
        secret_key=os.environ.get('INFEROPS_TEST_S3_SECRET_KEY','minio12345'))
    engine = Engine.from_project(root, repository=repo, storage=storage,
        evidence=boundary != 'acknowledgment', spool_path=Path(spool),
        evidence_config=EvidenceConfig(pre_seconds=0, post_seconds=0, disk_reserve_bytes=0, retry_delays=()))
    if boundary == 'acceptance':
        async def request(record): os._exit(73)
        engine.application.evidence.request = request
    elif boundary == 'spool':
        write = engine.application.evidence.spool.write
        async def commit(*args):
            await write(*args)
            os._exit(73)
        engine.application.evidence.spool.write = commit
    elif boundary == 'upload':
        put = storage.put
        async def upload(key, data, content_type='application/octet-stream'):
            result = await put(key, data, content_type)
            if key.endswith('clip.mp4'): os._exit(73)
            return result
        storage.put = upload
    else:
        class Delivery:
            async def publish(self, topic, key, payload, event_id):
                await repo.record_inbox(event_id, 'crash-check')
            async def close(self): pass
        engine.application.delivery = Delivery()
        async def ack(*args): os._exit(73)
        repo.ack_outbox = ack
    async with engine:
        await engine.run('test', np.zeros((32,32,3),np.uint8))
        await asyncio.sleep(30)
asyncio.run(main())
"""
    environment = {**os.environ, "INFEROPS_CRASH_BUCKET": bundle.storage.bucket}
    process = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-c", script, str(bundle.project), bundle.schema, str(bundle.spool), boundary],
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert process.returncode == 73, process.stdout + process.stderr
    assert len(await bundle.repository.list_incidents()) == 1
    if boundary == "acknowledgment":

        class Delivery:
            async def publish(self, topic, key, payload, event_id):
                await bundle.repository.record_inbox(event_id, "crash-check")

            async def close(self):
                pass

        from sqlalchemy import func, select

        async with bundle.repository.engine.begin() as connection:
            assert await connection.scalar(select(func.count()).select_from(bundle.repository.inbox)) == 1
            now = await bundle.repository._now(connection)
            await connection.execute(
                bundle.repository.outbox.update().values(lease_until=now - timedelta(seconds=1))
            )
        relay = OutboxRelay(bundle.repository, Delivery())
        assert await relay.flush() == 1
        assert await relay.flush() == 0
        async with bundle.repository.engine.connect() as connection:
            assert await connection.scalar(select(func.count()).select_from(bundle.repository.inbox)) == 1
    else:
        recovered = production_engine(bundle)
        recovered.application.recover_sessions = True
        async with recovered:
            await recovered.application.evidence.wait_idle()
            rows = await bundle.repository.list_evidence()
            assert len(rows) == 1 and rows[0]["state"] == expected
            for artifact in rows[0].get("artifacts", {}).values():
                assert await bundle.storage.verify(artifact["uri"], artifact["sha256"])
