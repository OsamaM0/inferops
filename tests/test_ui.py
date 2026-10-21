"""One packaged console, with compatibility bookmarks and service discovery."""
import pytest
from httpx import ASGITransport, AsyncClient

from inferops.adapters.inbound.http.main import create_app


@pytest.mark.asyncio
async def test_ui_assets_guide_and_legacy_bookmarks():
    async with AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://test") as client:
        shell = (await client.get("/ui/")).text
        for path in ("/ui", "/ui/", "/frontend/", "/frontend", "/ui/cameras", "/ui/guide"):
            response = await client.get(path)
            assert response.status_code == 200
            assert response.text == shell
            assert response.headers["cache-control"] == "no-cache"
            assert "/ui/static/dashboard.js" in response.text
            assert "Service interfaces" in response.text
        for view in ("overview", "sources", "pipeline", "playground", "jobs", "activity-view",
                     "settings", "services", "api", "preferences", "guide"):
            assert f'data-pane="{view}"' in shell
            assert f'data-view="{view}"' in shell
        assert shell.count('rel="stylesheet"') == 1
        for filename in ("console.js", "cameras.js", "dashboard.js", "workspace.js", "console-views.js", "configuration.js"):
            assert (await client.get(f"/ui/static/{filename}")).status_code == 200
        script = await client.get("/ui/static/dashboard.js")
        assert script.status_code == 200
        assert "decodeSocketPayload" in script.text
        assert (await client.get("/ui/static/dashboard.css")).status_code == 200
        # Both the embedded guide and request composer discover this running API.
        schema = (await client.get("/openapi.json")).json()
        assert "/v1/run/{pipeline}/upload" in schema["paths"]
        catalog = (await client.get("/v1/interfaces")).json()
        ports = {row["name"]: row for row in catalog["ports"]}
        assert ports["MinIO Console"]["override"] == "INFEROPS_MINIO_CONSOLE_PORT"
        assert ports["Redpanda admin"]["internal"] == "9644"
        browser = await client.get("/", headers={"accept": "text/html"})
        assert browser.status_code == 307
        assert browser.headers["location"] == "/ui/"
        assert (await client.get("/")).json()["ui"] == "/ui/"


@pytest.mark.asyncio
async def test_service_links_use_browser_host_and_configured_ports(monkeypatch):
    monkeypatch.setenv("INFEROPS_MINIO_CONSOLE_PORT", "19001")
    for name in ("MINIO_CONSOLE", "GRAFANA", "PROMETHEUS"):
        monkeypatch.delenv(f"INFEROPS_{name}_PUBLIC_URL", raising=False)
    async with AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://remote.example:18080") as client:
        catalog = (await client.get("/v1/interfaces")).json()["interfaces"]
        urls = {item["id"]: item["url"] for item in catalog}
        assert urls["minio"] == "http://remote.example:19001/"
        assert urls["grafana"].startswith("http://remote.example:3000/d/")
        assert urls["swagger"] == "http://remote.example:18080/docs"
        ports = (await client.get("/v1/interfaces")).json()["ports"]
        assert next(row for row in ports if row["name"] == "MinIO Console")["host"] == "19001"
        for path in ("/v1/interfaces/minio", "/v1/evidence"):
            response = await client.get(path)
            assert response.status_code == 307
            assert response.headers["location"] == urls["minio"]
        assert (await client.get("/v1/interfaces/missing")).status_code == 404


@pytest.mark.asyncio
async def test_proxy_urls_preserve_service_paths_and_reject_credentials(monkeypatch):
    monkeypatch.setenv("INFEROPS_GRAFANA_PUBLIC_URL", "https://metrics.example/grafana")
    async with AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://test") as client:
        response = await client.get("/v1/interfaces/grafana")
        assert response.headers["location"] == "https://metrics.example/grafana/"
        monkeypatch.setenv("INFEROPS_GRAFANA_PUBLIC_URL", "https://user:secret@metrics.example")
        response = await client.get("/v1/interfaces")
        assert response.status_code == 503
        assert "secret" not in response.text
