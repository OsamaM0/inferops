"""Smoke tests for dashboard config-management routes."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from inferops.api.main import create_app
from inferops.pipeline import get_pipeline_store


@pytest.mark.asyncio
async def test_config_routes_round_trip_and_reload_reset_cached_runners():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac, app.router.lifespan_context(app):
        response = await ac.get("/v1/configs")
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["sections"] == ["platform", "models", "pipelines", "fusion", "alerts"]
        assert data["active_domain"] == "driver_behaviour"
        assert data["domain_root"].endswith("examples/driver_behaviour")
        assert any(domain["id"] == "driver_behaviour" for domain in data["domains"])

        response = await ac.get("/v1/configs/pipelines/driver_behaviour")
        assert response.status_code == 200, response.text
        original = response.json()["content"]
        updated = original.rstrip() + "\n# dashboard-config-test\n"

        store = get_pipeline_store()
        runner_before = await store.get_runner("driver_behaviour")

        try:
            response = await ac.put(
                "/v1/configs/pipelines/driver_behaviour",
                json={"content": updated},
            )
            assert response.status_code == 200, response.text
            assert response.json()["reload"]["section"] == "pipelines"

            response = await ac.get("/v1/configs/pipelines/driver_behaviour")
            assert response.status_code == 200, response.text
            assert "# dashboard-config-test" in response.json()["content"]

            runner_after = await store.get_runner("driver_behaviour")
            assert runner_after is not runner_before

            response = await ac.put(
                "/v1/configs/pipelines/driver_behaviour",
                json={"content": "name: ["},
            )
            assert response.status_code == 400
        finally:
            restore = await ac.put(
                "/v1/configs/pipelines/driver_behaviour",
                json={"content": original},
            )
            assert restore.status_code == 200, restore.text
