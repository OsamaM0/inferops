"""Transport compatibility and lifecycle checks against an explicit fake engine."""
from __future__ import annotations

import asyncio
import secrets
import time
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient

from inferops.adapters.inbound.http.app import _byte_range, _control_packet, create_application
from inferops.adapters.outbound.messaging.delivery import DeliveryHub
from inferops.domain.errors import RegistryError
from inferops.domain.processing import MediaFrame
from inferops.domain.types import StageItem


class Repository:
    def __init__(self):
        self.rows = {"e1": {"id": "e1", "session_id": "s1", "state": "Saved",
                            "artifacts": {"clip": {"uri": "memory://clip"}}}}

    async def get_evidence(self, evidence_id):
        return self.rows.get(evidence_id)


class Storage:
    async def read(self, uri):
        assert uri == "memory://clip"
        return b"0123456789"


class Session:
    def __init__(self, engine, pipeline, stream_name, owner_kind):
        self.engine = engine
        self.id = uuid4().hex
        self.capability = secrets.token_urlsafe(32)
        self.spec = SimpleNamespace(name=pipeline)
        self.timeline_id = "timeline"
        self.stream_name, self.owner_kind = stream_name, owner_kind
        self.origin = time.monotonic_ns()
        self.last_sequence = -1
        self.lock = asyncio.Lock()
        self.closed = False

    def authorize(self, capability):
        if capability != self.capability:
            raise PermissionError("session capability required")

    async def observe(self, frame):
        return frame.media_time_ns

    async def process(self, frame, **kwargs):
        self.last_sequence = frame.sequence
        return [StageItem(stream_id=self.stream_name, ts_ms=frame.presentation_ts_ms or 0,
                          seq=frame.sequence, payload={"session_id": self.id, "timeline_id": self.timeline_id,
                                                     "media_time_ns": frame.media_time_ns,
                                                     "image_np": frame.image, "evidence": []})]

    async def close(self, reason="closed"):
        self.closed = True
        self.engine.application.sessions.pop(self.id, None)


class Engine:
    def __init__(self):
        async def source(value, **kwargs):
            yield MediaFrame(np.zeros((2, 2, 3), dtype=np.uint8))

        self.application = SimpleNamespace(repository=Repository(), sessions={}, started=False,
            evidence=SimpleNamespace(storage=Storage(), dropped_frames=0), source_factory=source)

    async def start(self):
        self.application.started = True

    async def close(self):
        for session in list(self.application.sessions.values()):
            await session.close()

    async def open_session(self, pipeline, *, stream_name="", owner_kind="python"):
        if pipeline != "driver":
            raise RegistryError("unknown pipeline")
        session = Session(self, pipeline, stream_name, owner_kind)
        self.application.sessions[session.id] = session
        return session


@pytest.mark.parametrize(("value", "expected"), [
    ("bytes=0-3", (0, 3)), ("bytes=8-", (8, 9)), ("bytes=-3", (7, 9)),
    ("bytes=-30", (0, 9)), ("bytes=0-99", (0, 9)),
])
def test_byte_ranges_include_suffix_and_open_ended_forms(value, expected):
    assert _byte_range(value, 10) == expected


@pytest.mark.parametrize("value", ["bytes=-0", "bytes=-", "bytes=10-", "bytes=9-2", "bytes=0-1,4-5",
                                          "bytes=abc-def", "items=0-2", "bytes=-1-3"])
def test_invalid_or_unsatisfiable_ranges_rejected(value):
    with pytest.raises(ValueError):
        _byte_range(value, 10)


def test_control_packets_require_typed_objects():
    assert _control_packet('{"media_time_ns":0,"seq":0}') == {"media_time_ns": 0, "seq": 0}
    for value in ("[]", "null", '{"seq":true}', '{"media_time_ns":1.5}', '{"client_ts_ms":-1}'):
        with pytest.raises(ValueError):
            _control_packet(value)


def test_root_compatibility_cors_favicon_and_readonly_ranges():
    app = create_application(Engine())
    with TestClient(app) as client:
        assert client.get("/").json()["name"] == "inferops"
        assert client.get("/", headers={"Accept": "text/html"}, follow_redirects=False).headers["location"] == "/ui/"
        assert client.get("/favicon.ico").status_code == 204
        options = client.options("/v1/sessions", headers={"Origin": "http://example.test",
                                                         "Access-Control-Request-Method": "POST"})
        assert options.status_code == 200 and options.headers["access-control-allow-origin"] == "*"
        suffix = client.get("/v1/evidence/records/e1/assets/clip", headers={"Range": "bytes=-3"})
        assert suffix.status_code == 206 and suffix.content == b"789"
        assert suffix.headers["content-range"] == "bytes 7-9/10"
        assert client.get("/v1/evidence/records/e1/assets/clip", headers={"Range": "bytes=-0"}).status_code == 416
        assert "/v1/system/status" in app.openapi()["paths"]


def test_sessions_require_capability_and_stream_names_do_not_identify_owner():
    engine = Engine()
    with TestClient(create_application(engine)) as client:
        assert client.post("/v1/sessions", json={}).status_code == 400
        first = client.post("/v1/sessions", json={"pipeline": "driver", "stream_id": "same"}).json()
        second = client.post("/v1/sessions", json={"pipeline": "driver", "stream_id": "same"}).json()
        assert first["session_id"] != second["session_id"]
        form = {"session_id": first["session_id"], "media_time_ns": 0}
        assert client.post("/v1/infer/driver", data=form).status_code == 403
        assert client.post("/v1/infer/driver", data=form,
                           headers={"X-Session-Capability": second["capability"]}).status_code == 403
        response = client.post("/v1/infer/driver", data=form,
                               headers={"X-Session-Capability": first["capability"]})
        assert response.status_code == 200 and response.json()[0]["session_id"] == first["session_id"]
        assert client.delete(f"/v1/sessions/{first['session_id']}").status_code == 403
        assert client.delete(f"/v1/sessions/{first['session_id']}",
                             headers={"X-Session-Capability": first["capability"]}).status_code == 200


def test_live_controls_and_quiet_subscription_disconnect_release_resources():
    engine, hub = Engine(), DeliveryHub()
    with TestClient(create_application(engine, hub=hub)) as client:
        with client.websocket_connect("/v1/ws/evidence.e1"):
            assert hub.subscribers["evidence.e1"]
        assert not hub.subscribers
        with client.websocket_connect("/v1/live/driver?stream_id=same") as connection:
            connection.send_text("[]")
            assert connection.receive_json()["error"] == "invalid_control"
            connection.send_text('{"media_time_ns":0,"client_ts_ms":123}')
            connection.send_bytes(b"frame")
            response = connection.receive_json()
            assert response["session_id"] and response["ts_ms"] == 123
            assert response["latency_ms"] >= 0 and response["frames_dropped"] == 0
        assert not engine.application.sessions
