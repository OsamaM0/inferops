import asyncio
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from inferops import Engine
from inferops.adapters.inbound.http.app import create_application
from inferops.adapters.outbound.messaging.delivery import DeliveryHub, ServerDelivery
from inferops.adapters.outbound.messaging.worker_control import HTTPWorkerControl
from inferops.domain.workers import WorkerManifest, WorkerSpec
from inferops.ports.ownership import OwnershipLost
from tests.test_redesign_engine import project

pytest_plugins = ["tests.test_worker_ownership"]


def test_manifest_rejects_duplicate_or_unknown_assignments():
    with pytest.raises(ValueError, match="unique"):
        WorkerManifest(workers=[{"worker_id": "a", "endpoint": "http://a"}] * 2)


def test_native_watchdog_can_detect_a_hung_event_loop_without_waiting_on_it():
    from inferops.adapters.inbound.workers.camera_worker import start_watchdog
    expired = threading.Event()
    application = SimpleNamespace(last_ownership_renewal=time.monotonic() - 1)
    stop, thread = start_watchdog(application, timeout_s=.02, exit_process=lambda code: expired.set())
    try:
        assert expired.wait(1)
    finally:
        stop.set()
        thread.join(1)


async def test_api_routes_controls_previews_and_history_without_owning_worker_sessions(owners, tmp_path, monkeypatch):
    monkeypatch.setenv("INFEROPS_WORKER_TOKEN", "test-control-secret")
    root = project(tmp_path)
    engines = [Engine.from_project(root, repository=owners(scope)) for scope in ["camera-a", "camera-b", "api"]]
    for engine, role in zip(engines, ["camera-worker", "camera-worker", "api"], strict=True):
        engine.application.role = role
    apps = [create_application(engine, source_root=tmp_path / "sources") for engine in engines]
    transport_a, transport_b = ASGITransport(app=apps[0]), ASGITransport(app=apps[1])
    async with apps[0].router.lifespan_context(apps[0]), apps[1].router.lifespan_context(apps[1]):
        async with AsyncClient(transport=transport_a, base_url="http://a") as worker_client:
            assert (await worker_client.get("/v1/cameras")).status_code == 403
            assert (await worker_client.get("/readyz")).status_code == 200
            assert (await worker_client.post("/v1/jobs/test", headers={"X-Worker-Token": "test-control-secret"})).status_code == 403
        async with apps[2].router.lifespan_context(apps[2]):
            for worker_id, transport in [("a", transport_a), ("b", transport_b)]:
                apps[2].state.worker_controls[worker_id] = HTTPWorkerControl(
                    WorkerSpec(worker_id=worker_id, endpoint=f"http://{worker_id}"), "test-control-secret", transport=transport)
            _, jpeg = cv2.imencode(".jpg", np.zeros((32, 32, 3), np.uint8))
            path = tmp_path / "image.jpg"
            path.write_bytes(jpeg.tobytes())
            async with AsyncClient(transport=ASGITransport(app=apps[2]), base_url="http://api") as client:
                for name in ["a", "b"]:
                    response = await client.post(f"/v1/cameras?worker_id={name}", json={
                        "camera_id": name, "name": name, "pipeline": "test", "kind": "image", "uri": str(path)})
                    assert response.status_code == 201, response.text
                with pytest.raises(OwnershipLost):
                    await engines[0].application.repository.claim_camera("b")
                assert (await client.post("/v1/cameras/b/start")).status_code == 200
                for _ in range(200):
                    response = await client.get("/v1/cameras/b/results")
                    if response.json():
                        break
                    await asyncio.sleep(.01)
                assert response.json()[0]["camera_id"] == "b"
                assert (await client.get("/v1/cameras/b/preview.jpg")).status_code == 200
                cameras = (await client.get("/v1/cameras")).json()
                assert {row["worker_id"] for row in cameras["cameras"]} == {"a", "b"}
                assert not engines[2].application.sessions
                assert (await client.put("/v1/cameras/a", json={"name": "updated"})).status_code == 200
                assert apps[0].state.sources.specs["a"].name == "updated"
        # The API owner closing leaves both worker leases valid.
        await engines[0].application.repository.renew_ownership()
        await engines[1].application.repository.renew_ownership()


async def test_broker_failure_does_not_block_action_or_local_outbox_destinations(owners):
    repository = owners("api")
    await repository.start()
    hub = DeliveryHub()
    local = asyncio.Queue()
    hub.subscribe("alerts.camera", local)

    class FailedBroker:
        async def producer(self, name):
            raise OSError("broker down")

    delivery = ServerDelivery(hub, transport=FailedBroker())
    repository.outbox_transform = delivery.prepare
    await repository.accept([], [], [], [{"id": "event", "topic": "incident.accepted", "key": "camera", "payload": {
        "alert": {"id": "incident", "severity": "low", "payload": {}},
        "delivery_policy": {"name": "policy", "severity": {"levels": [{"name": "low", "rank": 1}]},
                            "sinks": [{"name": "hint", "kind": "ws", "params": {"channel": "second"}}]}}}])
    from inferops.application.delivery import OutboxRelay
    relay = OutboxRelay(repository, delivery)
    assert await relay.flush() == 2
    assert local.get_nowait()["id"] == "incident"
    assert (await repository.diagnostics())["pending_outbox"] == 1
    await delivery.close()
