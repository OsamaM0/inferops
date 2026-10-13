"""The driver pipeline through the HTTP API: models, pipelines, /infer, /jobs, live signals."""
import asyncio
import io

import cv2
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from inferops.api.main import create_app


def _jpeg() -> bytes:
    img = (np.random.rand(96, 96, 3) * 255).astype("uint8")
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


@pytest.mark.asyncio
async def test_driver_pipeline_via_api():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac, app.router.lifespan_context(app):
        r = await ac.get("/v1/models")
        names = [m["name"] for m in r.json()]
        assert {"rfdetr_driver_onnx", "rfdetr_driver_triton", "face_mesh_onnx"} <= set(names)

        r = await ac.get("/v1/configs")
        data = r.json()
        assert data["active_domain"] == "driver_behaviour"
        assert data["domain_pack"]["name"] == "driver_behaviour"
        assert data["domain_pack"]["pipelines"] == ["driver_behaviour", "driver_video"]
        assert any(d["id"] == "driver_behaviour" for d in data["domains"])

        files = {"image": ("img.jpg", io.BytesIO(_jpeg()), "image/jpeg")}
        r = await ac.post("/v1/infer/driver_behaviour", files=files, data={"stream_id": "t"})
        assert r.status_code == 200, r.text
        out = r.json()
        assert out and out[0]["payload"]["model_selection"]["used"] == "rfdetr_driver_onnx"
        # Face-mesh state is exposed through the generic channels the dashboard reads.
        payload = out[0]["payload"]
        assert isinstance(payload.get("signals", {}), dict)
        assert isinstance(payload.get("overlay_boxes", []), list)

        r = await ac.post("/v1/jobs/driver_behaviour", json={"stream_id": "job-test"})
        assert r.status_code == 200, r.text
        job_id = r.json()["job_id"]
        job = None
        for _ in range(40):
            job = (await ac.get(f"/v1/jobs/{job_id}")).json()
            if job["status"] in {"succeeded", "failed"}:
                break
            await asyncio.sleep(0.05)
        assert job is not None and job["status"] == "succeeded", job
