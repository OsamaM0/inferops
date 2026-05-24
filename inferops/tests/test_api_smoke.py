"""End-to-end smoke: driver_behaviour pipeline through /v1/infer and /v1/jobs."""
import asyncio
import io

import cv2
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from inferops.api.main import create_app


@pytest.mark.asyncio
async def test_health_and_infer():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac, app.router.lifespan_context(app):
            # health
            r = await ac.get("/healthz")
            assert r.status_code == 200

            # models
            r = await ac.get("/v1/models")
            assert r.status_code == 200
            names = [m["name"] for m in r.json()]
            assert "rfdetr_driver_synth" in names

            # pipelines
            r = await ac.get("/v1/pipelines")
            assert r.status_code == 200
            assert any(p["name"] == "driver_behaviour" for p in r.json())

            # encode a 64x64 RGB blob as JPEG
            img = (np.random.rand(64, 64, 3) * 255).astype("uint8")
            ok, buf = cv2.imencode(".jpg", img)
            assert ok
            files = {"image": ("img.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")}
            data = {"stream_id": "test"}
            r = await ac.post("/v1/infer/driver_behaviour", files=files, data=data)
            assert r.status_code == 200, r.text
            out = r.json()
            assert isinstance(out, list) and out

            r = await ac.post("/v1/jobs/driver_behaviour", json={"stream_id": "job-test"})
            assert r.status_code == 200, r.text
            job_id = r.json()["job_id"]
            job = None
            for _ in range(20):
                r = await ac.get(f"/v1/jobs/{job_id}")
                assert r.status_code == 200, r.text
                job = r.json()
                if job["status"] in {"succeeded", "failed"}:
                    break
                await asyncio.sleep(0.05)
            assert job is not None and job["status"] == "succeeded"

            # metrics
            r = await ac.get("/metrics")
            assert r.status_code == 200
            assert b"inferops" in r.content or b"pipeline" in r.content
