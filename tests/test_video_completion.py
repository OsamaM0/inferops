"""Uploaded videos must have readable evidence when the response arrives."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from inferops.adapters.s3_client import get_bytes, s3_client
from inferops.api.main import create_app
from inferops.core.config import settings
from inferops.pipeline import get_pipeline_store
from inferops.pipeline.stages.sink_s3 import _encode_mp4


@pytest.mark.asyncio
async def test_upload_returns_only_after_all_metadata_is_readable(tmp_path):
    path = tmp_path / "short.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12.5, (32, 32))
    for value in [20, 100, 180]:
        writer.write(np.full((32, 32, 3), value, np.uint8))
    writer.release()
    app = create_app()
    async with app.router.lifespan_context(app), AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/v1/run/tiny_classifier/upload", data={"stream_id": "durable-upload"},
                                     files={"video": (path.name, path.read_bytes(), "video/mp4")})
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["frames"] == 3
        assert len(result["evidence_uris"]) == 3
        assert all([await get_bytes(uri) for uri in result["evidence_uris"]])
        objects = s3_client().list_objects_v2(Bucket=settings.s3.bucket, Prefix="evidence/durable-upload/")
        assert objects["KeyCount"] == 3


@pytest.mark.asyncio
async def test_failed_video_storage_is_an_error_response(tmp_path, monkeypatch):
    path = tmp_path / "broken.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12.5, (32, 32))
    writer.write(np.zeros((32, 32, 3), np.uint8))
    writer.release()
    app = create_app()
    async with app.router.lifespan_context(app), AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        runner = await get_pipeline_store().get_runner("tiny_classifier")

        async def failed_upload(_item):
            raise RuntimeError("storage unavailable")

        monkeypatch.setattr(runner._sinks[0], "process", failed_upload)
        response = await client.post("/v1/run/tiny_classifier/upload",
                                     files={"video": (path.name, path.read_bytes(), "video/mp4")})
        assert response.status_code == 503
        assert "storage unavailable" in response.json()["detail"]


def test_evidence_mp4_is_h264_at_the_source_rate(tmp_path):
    import av
    path = tmp_path / "evidence.mp4"
    path.write_bytes(_encode_mp4([np.zeros((32, 32, 3), np.uint8)] * 5, 32, 32, 12.5))
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == "h264"
        assert float(stream.average_rate) == 12.5
        assert len(list(container.decode(video=0))) == 5


@pytest.mark.asyncio
async def test_alert_storage_is_idempotent_and_retains_evidence_links():
    from inferops.adapters.postgres_dao import attach_alert_evidence, init_schema, insert_alert, list_alerts
    await init_schema()
    row = {"id": "duplicate-incident-test", "type": "phone_use", "entity_id": "driver",
           "severity": "medium", "ts_ms": 1, "payload": {"rule": "phone"}}
    await insert_alert(row)
    await insert_alert(row)
    uri = "s3://inferops-clips/clips/driver/test.mp4"
    await attach_alert_evidence(row["id"], [uri])
    await attach_alert_evidence(row["id"], [uri])
    records = [a for a in await list_alerts() if a["id"] == row["id"]]
    assert len(records) == 1
    assert records[0]["payload"] == {"rule": "phone", "evidence_uris": [uri]}
