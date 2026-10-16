"""End-to-end smoke: the core tiny_classifier pipeline through /v1/infer and /v1/jobs."""
import asyncio
import base64
import io

import cv2
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from inferops.adapters.inbound.http.main import create_app


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
            assert "tiny_classifier" in names
            logical = (await ac.get("/v1/models?view=logical")).json()
            assert [model["name"] for model in logical] == ["tiny_classifier"]
            assert (await ac.get("/v1/models?view=invalid")).status_code == 422

            # pipelines
            r = await ac.get("/v1/pipelines")
            assert r.status_code == 200
            assert any(p["name"] == "tiny_classifier" for p in r.json())

            # encode a 64x64 RGB blob as JPEG
            img = (np.random.rand(64, 64, 3) * 255).astype("uint8")
            ok, buf = cv2.imencode(".jpg", img)
            assert ok
            files = {"image": ("img.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")}
            data = {"stream_id": "test"}
            r = await ac.post("/v1/infer/tiny_classifier", files=files, data=data)
            assert r.status_code == 200, r.text
            out = r.json()
            assert isinstance(out, list) and out
            results = out[0]["results"]
            assert results and results[0]["kind"] == "classifications"
            assert results[0]["model"] == "tiny_classifier"
            assert out[0]["payload"]["model_selection"]["used"] == "tiny_classifier"
            logical = (await ac.get("/v1/models?view=logical")).json()
            assert logical[0]["loaded_variants"] == ["tiny_classifier"]

            r = await ac.post("/v1/jobs/tiny_classifier", json={"stream_id": "job-test", "image_base64": base64.b64encode(buf.tobytes()).decode()})
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
