"""Smoke tests for dashboard config-management routes (core, no domain)."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from inferops.adapters.inbound.http.main import create_app

PIPELINE = "tiny_classifier"


@pytest.mark.asyncio
async def test_config_routes_round_trip_and_reload_reset_cached_runners():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac, app.router.lifespan_context(app):
        response = await ac.get("/v1/configs")
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["sections"] == ["platform", "models", "pipelines", "fusion", "alerts"]
        assert data["active_domain"] is None
        assert data["domain_root"] is None
        assert data["domain_pack"] is None
        assert any(domain["id"] == PIPELINE for domain in data["domains"])
        assert any(f["section"] == "pipelines" and f["name"] == PIPELINE for f in data["files"])
        assert all(f["reload_behavior"] == "live" for f in data["files"] if f["section"] == "pipelines")
        assert all(f["reload_behavior"] == "conditional" for f in data["files"] if f["section"] == "models")

        response = await ac.get(f"/v1/configs/pipelines/{PIPELINE}")
        assert response.status_code == 200, response.text
        original = response.json()["content"]
        updated = original.rstrip() + "\n# dashboard-config-test\n"

        session_before = await app.state.engine.open_session(PIPELINE)

        try:
            response = await ac.put(
                f"/v1/configs/pipelines/{PIPELINE}",
                json={"content": updated},
            )
            assert response.status_code == 200, response.text
            assert response.json()["reload"]["section"] == "pipelines"

            response = await ac.get(f"/v1/configs/pipelines/{PIPELINE}")
            assert response.status_code == 200, response.text
            assert "# dashboard-config-test" in response.json()["content"]

            session_after = await app.state.engine.open_session(PIPELINE)
            assert session_after is not session_before
            assert session_after.models is not session_before.models
            assert not session_before.models._closed
            await session_before.close()
            await session_after.close()

            response = await ac.put(
                f"/v1/configs/pipelines/{PIPELINE}",
                json={"content": "name: ["},
            )
            assert response.status_code == 400
        finally:
            restore = await ac.put(
                f"/v1/configs/pipelines/{PIPELINE}",
                json={"content": original},
            )
            assert restore.status_code == 200, restore.text
