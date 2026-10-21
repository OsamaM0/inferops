"""Shared Python/HTTP execution, media clocks and isolated durable sessions."""
import asyncio

import cv2
import numpy as np
import yaml
from httpx import ASGITransport, AsyncClient

from inferops import Engine
from inferops.adapters.inbound.http.app import create_application
from inferops.adapters.outbound.persistence.sql import SqlRepository
from inferops.adapters.outbound.storage.filesystem import FilesystemStorage
from inferops.domain.evidence import EvidenceConfig
from inferops.domain.processing import MediaFrame
from inferops.testing.postgres import test_database_url


def project(tmp_path):
    root = tmp_path / "project"
    (root / "configs/pipelines").mkdir(parents=True)
    (root / "configs/alerts").mkdir()
    (root / "project.yaml").write_text("name: project\npipelines: [test]\n")
    (root / "features.py").write_text('''from inferops.adapters.outbound.stages.base import Stage
class Features(Stage):
    async def process(self, item):
        item.payload['alert_candidates'] = [{'type': 'hot', 'ts_ms': item.ts_ms}]
        return item
''')
    (root / "configs/pipelines/test.yaml").write_text(yaml.safe_dump({
        "name": "test", "stages": [{"name": "features", "stage": ".features:Features"},
                                     {"name": "alerts", "stage": "alert", "params": {"alerts": "test"}}]}))
    (root / "configs/alerts/test.yaml").write_text(yaml.safe_dump({
        "name": "test", "severity": {"levels": [{"name": "low", "rank": 1}], "default": "low"},
        "dedupe": {"window_ms": 1000}, "cooldown": {"per_dedupe_key_ms": 1000}, "sinks": []}))
    return root


def engine(tmp_path, *, evidence=True):
    return Engine.from_project(project(tmp_path), repository=SqlRepository(test_database_url(tmp_path)),
                               storage=FilesystemStorage(tmp_path / "objects"), evidence=evidence,
                               spool_path=tmp_path / "spool", evidence_config=EvidenceConfig(
                                   pre_seconds=0, post_seconds=0, disk_reserve_bytes=0, retry_delays=()))


async def test_generated_session_names_are_isolated_and_media_cooldown(tmp_path):
    selected = engine(tmp_path)
    image = np.zeros((32, 32, 3), np.uint8)
    async with selected:
        first, second = await asyncio.gather(*(selected.open_session("test", stream_name="same") for _ in range(2)))
        assert first.id != second.id and first.timeline_id != second.timeline_id
        for session in [first, second]:
            outputs = await session.process(MediaFrame(image, 0, 0))
            assert len(outputs[0].payload['alerts']) == 1
            again = await session.process(MediaFrame(image, 0, 0))
            assert again[0].payload['alerts'][0]['id'] == outputs[0].payload['alerts'][0]['id']
        assert len(await selected.application.repository.list_incidents()) == 2
        assert not (await first.process(MediaFrame(image, 500_000_000, 1)))[0].payload.get('alerts')
        # Temporal cooldown follows PTS rather than the speed at which frames are processed.
        assert len((await first.process(MediaFrame(image, 1_100_000_000, 2)))[0].payload['alerts']) == 1
        await selected.application.evidence.wait_idle()
        records = await selected.application.repository.list_evidence()
        assert len(records) == 3 and {row['state'] for row in records} == {'Saved'}
        assert all([await selected.application.evidence.storage.verify(a['uri'], a['sha256'])
                    for row in records for a in row['artifacts'].values()])


async def test_server_default_disk_evidence_survives_reconstruction_and_http_serves_it(tmp_path, monkeypatch):
    from inferops.adapters.outbound.projects.settings import Settings
    from inferops.bootstrap import build_server_engine
    from inferops.testing import PROJECT_ROOT

    # Exercise the bootstrap fallback without reading the operator's .env.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("INFEROPS_STORAGE_KIND", raising=False)
    monkeypatch.setenv("INFEROPS_DATA_PATH", str(tmp_path / "persistent-data"))
    monkeypatch.setenv("INFEROPS_SPOOL_PATH", str(tmp_path / "spool"))
    monkeypatch.setenv("INFEROPS_DATABASE_URL", test_database_url(tmp_path))
    monkeypatch.setenv("INFEROPS_ROLE", "combined")
    for option in ("PRE_SECONDS", "POST_SECONDS", "DISK_RESERVE_BYTES"):
        monkeypatch.setenv(f"INFEROPS_EVIDENCE_{option}", "0")
    settings = Settings(domain_root=str(project(tmp_path)),
                        configs_dir=str(PROJECT_ROOT / "inferops/testing/configs"))
    first, _ = build_server_engine(settings=settings)
    async with first:
        storage = first.application.evidence.storage
        assert isinstance(storage, FilesystemStorage)
        assert storage.root == tmp_path / "persistent-data/evidence"
        await first.run("test", np.zeros((32, 32, 3), np.uint8))
        await first.application.evidence.wait_idle()
        [saved] = await first.application.repository.list_evidence()
        assert saved["state"] == "Saved"
        original = await storage.read(saved["artifacts"]["snapshot"]["uri"])

    second, hub = build_server_engine(settings=settings)
    app = create_application(second, hub=hub)
    async with app.router.lifespan_context(app), AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        records = (await client.get("/v1/evidence/records")).json()
        assert [row["id"] for row in records] == [saved["id"]]
        response = await client.get(f'/v1/evidence/records/{saved["id"]}/assets/snapshot')
        assert response.status_code == 200 and response.content == original
        for artifact in saved["artifacts"].values():
            assert await second.application.evidence.storage.verify(artifact["uri"], artifact["sha256"])


async def test_cancelling_processing_closes_as_cancelled_instead_of_processing_failure(tmp_path):
    selected = engine(tmp_path)
    entered = asyncio.Event()
    reasons = []
    async with selected:
        session = await selected.open_session("test")
        original_finish = selected.application.evidence.finish_session

        async def finish(session_id, reason):
            reasons.append(reason)
            await original_finish(session_id, reason)

        async def blocking(frame, *, observed=False):
            entered.set()
            await asyncio.Event().wait()

        selected.application.evidence.finish_session = finish
        session._process = blocking
        pending = asyncio.create_task(session.process(MediaFrame(np.zeros((8, 8, 3), np.uint8))))
        await entered.wait()
        pending.cancel()
        result = await asyncio.gather(pending, return_exceptions=True)
        assert isinstance(result[0], asyncio.CancelledError)
        assert reasons == ["cancelled"]
        assert session.closed and session.id not in selected.application.sessions


async def test_python_and_http_share_services_and_capability(tmp_path):
    selected = engine(tmp_path)
    image = np.zeros((32, 32, 3), np.uint8)
    _, jpeg = cv2.imencode('.jpg', image)
    app = create_application(selected)
    async with app.router.lifespan_context(app), AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
        direct = await selected.run('test', image, stream_name='same')
        response = await client.post('/v1/infer/test', files={'image': ('test.jpg', jpeg.tobytes())}, data={'stream_id': 'same'})
        assert response.status_code == 200, response.text
        row = response.json()[0]
        assert row['stream_id'] == 'same'
        assert row['payload']['session_id'] != direct['session_id']
        assert len(row['payload']['alerts']) == direct['incident_count'] == 1
        handles = (await client.post('/v1/sessions', json={'pipeline':'test', 'stream_id':'same'})).json()
        denied = await client.post('/v1/infer/test', files={'image':('test.jpg', jpeg.tobytes())},
                                   data={'session_id':handles['session_id']})
        assert denied.status_code == 403
        accepted = await client.post('/v1/infer/test', files={'image':('test.jpg', jpeg.tobytes())},
                                     data={'session_id':handles['session_id'], 'media_time_ns':'0','seq':'0'},
                                     headers={'X-Session-Capability':handles['capability']})
        assert accepted.status_code == 200, accepted.text
        await selected.application.evidence.wait_idle()
        records = (await client.get('/v1/evidence/records')).json()
        assert all(r['state'] == 'Saved' for r in records)
        evidence_id = records[0]['id']
        media = await client.get(f'/v1/evidence/records/{evidence_id}/assets/clip', headers={'Range':'bytes=0-15'})
        assert media.status_code == 206 and len(media.content) == 16
        closed = await client.delete(f"/v1/sessions/{handles['session_id']}", headers={'X-Session-Capability':handles['capability']})
        assert closed.status_code == 200


async def test_discontinuity_resets_temporal_state_and_changes_timeline(tmp_path):
    selected = engine(tmp_path, evidence=False)
    async with selected:
        session = await selected.open_session('test')
        image = np.zeros((8,8,3),np.uint8)
        await session.process(MediaFrame(image,5_000_000_000,0))
        old = session.timeline_id
        assert (await session.process(MediaFrame(image,0,1)))[0].payload['alerts']
        assert old != session.timeline_id
        assert (await session.process(MediaFrame(image,2_000_000_000,2)))[0].payload['alerts']


async def test_incident_failure_visible_and_decisions_retryable(tmp_path, monkeypatch):
    selected = engine(tmp_path, evidence=False)
    async with selected:
        session = await selected.open_session('test')
        accept = selected.application.repository.accept
        async def unavailable(*args, **kwargs): raise OSError('database offline')
        monkeypatch.setattr(selected.application.repository,'accept',unavailable)
        frame = MediaFrame(np.zeros((8,8,3),np.uint8),0,0)
        output = (await session.process(frame))[0]
        assert output.payload['alerts'] == []
        assert output.payload['incident_error']['code'] == 'persistence_unavailable'
        monkeypatch.setattr(selected.application.repository,'accept',accept)
        assert len((await session.process(frame))[0].payload['alerts']) == 1
