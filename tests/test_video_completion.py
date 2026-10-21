"""Offline upload returns durable incident evidence with explicit failures."""

import cv2
import numpy as np
from httpx import ASGITransport, AsyncClient

from inferops.adapters.inbound.http.app import create_application
from inferops.adapters.outbound.media.annotations import _encode_mp4
from inferops.adapters.outbound.persistence.sql import SqlRepository
from inferops.testing.postgres import test_database_url
from tests.test_redesign_engine import engine


def video(tmp_path):
    path = tmp_path / "short.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12.5, (32, 32))
    for value in (20, 100, 180):
        writer.write(np.full((32, 32, 3), value, np.uint8))
    writer.release()
    return path


async def test_upload_returns_only_after_required_evidence_is_readable(tmp_path):
    path = video(tmp_path)
    app = create_application(engine(tmp_path))
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app), base_url="http://test") as client,
    ):
        response = await client.post(
            "/v1/run/test/upload", files={"video": (path.name, path.read_bytes(), "video/mp4")}
        )
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["frames"] == 3 and result["incident_count"] == 1
        assert len(result["evidence_uris"]) == 2
        row = result["evidence"][0]
        assert row["state"] == "Saved"
        for kind in ("snapshot", "clip"):
            asset = await client.get(f"/v1/evidence/records/{row['id']}/assets/{kind}")
            assert asset.status_code == 200 and asset.content


async def test_failed_video_storage_has_explicit_durable_outcome(tmp_path, monkeypatch):
    path = video(tmp_path)
    selected = engine(tmp_path)

    async def failed(*args, **kwargs):
        raise OSError("storage unavailable")

    monkeypatch.setattr(selected.application.evidence.storage, "put", failed)
    app = create_application(selected)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app), base_url="http://test") as client,
    ):
        response = await client.post(
            "/v1/run/test/upload", files={"video": (path.name, path.read_bytes(), "video/mp4")}
        )
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["incident_count"] == 1
        row = result["evidence"][0]
        assert row["state"] == "Failed" and row["errors"] and row["retry_available"]
        assert (await client.get("/v1/evidence/records/" + row["id"])).json()["state"] == "Failed"


def test_evidence_mp4_is_h264_at_the_source_rate(tmp_path):
    import av

    path = tmp_path / "evidence.mp4"
    path.write_bytes(_encode_mp4([np.zeros((32, 32, 3), np.uint8)] * 5, 32, 32, 12.5))
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == "h264"
        assert float(stream.average_rate) == 12.5
        assert len(list(container.decode(video=0))) == 5


async def test_alert_storage_is_idempotent_and_retains_evidence_links(tmp_path):
    repo = SqlRepository(test_database_url(tmp_path))
    await repo.start()
    try:
        row = {
            "id": "incident",
            "type": "phone_use",
            "entity_id": "session",
            "severity": "medium",
            "ts_ms": 1,
            "payload": {},
        }
        request = {
            "id": "capture",
            "incident_id": "incident",
            "session_id": "session",
            "timeline_id": "timeline",
            "state": "Pending",
        }
        await repo.accept([], [row], [request], [])
        await repo.accept([], [row], [request], [])
        artifacts = {"clip": {"uri": "s3://evidence/clip.mp4"}}
        await repo.update_evidence("capture", artifacts=artifacts)
        await repo.update_evidence("capture", artifacts=artifacts)
        records = await repo.list_incidents()
        assert len(records) == 1 and records[0]["payload"]["evidence_uris"] == ["s3://evidence/clip.mp4"]
    finally:
        await repo.close()
