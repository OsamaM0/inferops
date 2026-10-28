"""Registered media identity, lifecycle, bounded history and evidence linkage."""

from __future__ import annotations

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
from inferops.adapters.inbound.sources import process_live_frames, source_frames
from inferops.adapters.outbound.persistence.source_catalog import PostgresSourceStore
from inferops.adapters.outbound.persistence.sql import SqlRepository
from inferops.adapters.outbound.storage.filesystem import FilesystemStorage
from inferops.application.sources import SourceManager
from inferops.domain.capacity import assess_capacity
from inferops.domain.evidence import EvidenceConfig
from inferops.domain.processing import MediaFrame
from inferops.domain.sources import SourceSpec
from inferops.domain.types import StageItem
from inferops.testing.postgres import test_database_url
from tests.test_redesign_engine import engine, project


def spec(**kwargs):
    return SourceSpec(
        camera_id="front-1", name="Front", kind="image", uri="image.jpg", pipeline="test", **kwargs
    )


@pytest.mark.parametrize(
    "patch",
    [
        {"camera_id": "../escape"},
        {"inference_fps": 0},
        {"kind": "image", "loop": True},
        {"kind": "rtsp", "uri": "file:///etc/passwd"},
        {"kind": "video", "uri": "https://example/video.mp4"},
        {"driver_roi": [.8, 0, .2, 1]},
        {"driver_roi": [0, 0, float('nan'), 1]},
        {"driver_roi": [0, 0, 2, 1]},
    ],
)
def test_source_validation(patch):
    values = spec().model_dump()
    with pytest.raises(ValueError):
        SourceSpec.model_validate({**values, **patch})


async def test_catalog_preserves_identity_history_and_bounds_results(tmp_path):
    store = PostgresSourceStore(test_database_url(tmp_path), preview_root=tmp_path / "previews", results_per_camera=2)
    await store.start()
    await store.put_source(spec(group="Fleet A", location="Bus 101"))
    await store.save_run(dict(session_id="s1", camera_id="front-1", status="running", started_at=time.time()))
    for n in range(4):
        await store.save_result("front-1", "s1", {"sequence": n})
    await store.save_result("other", "s2", {"sequence": 99})
    await store.list_results("front-1")
    reopened = PostgresSourceStore(store.url, preview_root=tmp_path / "previews")
    await reopened.start()
    assert (await reopened.list_sources())[0].camera_id == "front-1"
    assert (await reopened.list_sources())[0].group == "Fleet A"
    assert (await reopened.list_sources())[0].location == "Bus 101"
    assert (await reopened.list_runs("front-1"))[0]["status"] == "interrupted"
    assert [r["sequence"] for r in await reopened.list_results("front-1")] == [3, 2]
    await reopened.delete_source("front-1")
    assert not await reopened.list_sources()
    assert len(await reopened.list_runs("front-1")) == 1
    assert len(await reopened.list_results("front-1")) == 2
    await store.close()
    await reopened.close()


async def test_history_read_flushes_prior_results_while_live_writes_continue(tmp_path):
    store = PostgresSourceStore(test_database_url(tmp_path), preview_root=tmp_path / "previews")
    await store.start()

    async def produce():
        sequence = 0
        while True:
            await store.save_result("live", "session", {"sequence": sequence})
            sequence += 1
            await asyncio.sleep(0.01)

    producer = asyncio.create_task(produce())
    try:
        await asyncio.sleep(0.02)
        rows = await asyncio.wait_for(store.list_results("live"), 1)
        assert rows and rows[-1]["sequence"] == 0
    finally:
        producer.cancel()
        await asyncio.gather(producer, return_exceptions=True)
        await store.close()


@pytest.mark.parametrize("preview_enabled", [True, False])
async def test_http_image_upload_runs_results_preview_evidence_and_restart(tmp_path, preview_enabled):
    root = project(tmp_path)

    def engine():
        return Engine.from_project(
            root,
            repository=SqlRepository(test_database_url(tmp_path)),
            storage=FilesystemStorage(tmp_path / "objects"),
            evidence=True,
            spool_path=tmp_path / "spool",
            evidence_config=EvidenceConfig(
                pre_seconds=0, post_seconds=0, disk_reserve_bytes=0, retry_delays=()
            ),
        )

    app = create_application(engine(), source_root=tmp_path / "registry")
    _, jpg = cv2.imencode(".jpg", np.zeros((80, 120, 3), np.uint8))
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        assert (await client.get("/ui/cameras")).status_code == 200
        upload = await client.post(
            "/v1/cameras/media", files={"file": ("image.jpg", jpg.tobytes(), "image/jpeg")}
        )
        assert upload.status_code == 201, upload.text
        payload = spec().model_dump()
        payload["uri"] = upload.json()["uri"]
        payload["preview_enabled"] = preview_enabled
        payload["driver_roi"] = [.1, .2, .9, .8]
        created = await client.post("/v1/cameras", json=payload)
        assert created.status_code == 201 and "uri" not in created.json()
        assert (await client.post("/v1/cameras", json=payload)).status_code == 400
        sessions = []
        for _ in range(2):
            started = await client.post("/v1/cameras/front-1/start")
            assert started.status_code == 200, started.text
            sessions.append(started.json()["session_id"])
            for _ in range(100):
                current = (await client.get("/v1/cameras")).json()["cameras"][0]
                if not current["active"]:
                    break
                await asyncio.sleep(0.02)
            assert current["status"] == "completed", current
        assert sessions[0] != sessions[1]
        preview = await client.get("/v1/cameras/front-1/preview.jpg")
        assert preview.status_code == (200 if preview_enabled else 404)
        if preview_enabled:
            assert preview.headers["content-type"] == "image/jpeg"
        else:
            assert not app.state.sources.previews
            assert await app.state.sources.store.get_preview("front-1") is None
        results = (await client.get("/v1/cameras/front-1/results")).json()
        assert {r["session_id"] for r in results} == set(sessions)
        assert all(r["camera_id"] == "front-1" for r in results)
        assert all(r["results"][0]["stream_id"] == "front-1" for r in results)
        assert all(r["results"][0]["payload"]["driver_roi"] == [.1, .2, .9, .8] for r in results)
        await app.state.engine.application.evidence.wait_idle()
        evidence = (await client.get("/v1/evidence/records")).json()
        assert {r["session_id"] for r in evidence} == set(sessions)
        assert {r["state"] for r in evidence} == {"Saved"}
        assert {r["camera_id"] for r in evidence} == {"front-1"}
        status = (await client.get("/v1/system/status")).json()
        assert status["runtime"]["execution_mode"] == "cpu"
        assert status["evidence"]["backend"] == "FilesystemStorage"
        assert status["services"]["minio_s3"]["enabled"] is False
        topology = (await client.get("/v1/system/workspace")).json()
        assert topology["runtime"] == "cpu" and topology["pipelines"][0]["steps"][0]["name"] == "source"
        assert (await client.get("/v1/system/triton")).json() == {"enabled": False, "models": []}
        assert (await client.put("/v1/cameras/front-1", json={"camera_id": "renamed"})).status_code == 400
    restarted = create_application(engine(), source_root=tmp_path / "registry")
    async with (
        restarted.router.lifespan_context(restarted),
        AsyncClient(transport=ASGITransport(app=restarted), base_url="http://test") as client,
    ):
        assert (await client.get("/v1/cameras")).json()["cameras"][0]["camera_id"] == "front-1"
        assert (await client.get("/v1/cameras/front-1/preview.jpg")).status_code == (200 if preview_enabled else 404)
        assert len((await client.get("/v1/cameras/front-1/runs")).json()) == 2
        assert (await client.delete("/v1/cameras/front-1")).status_code == 200
        assert not (await client.get("/v1/cameras")).json()["cameras"]
        assert len((await client.get("/v1/cameras/front-1/results")).json()) == 2


async def test_source_limit_immediate_stop_and_credential_redaction(tmp_path):
    sessions = []

    class Session:
        annotator = None

        def __init__(self):
            self.id = f"s{len(sessions)}"
            self.closed = False
            sessions.append(self)

        async def close(self, reason):
            self.closed = True

    async def open_session(*a, **kw):
        return Session()

    async def frames(spec, stats):
        await asyncio.Event().wait()
        yield None

    app = SimpleNamespace(definition=SimpleNamespace(pipelines={"test": None}), open_session=open_session)
    selected = SourceManager(
        app,
        PostgresSourceStore(test_database_url(tmp_path), preview_root=tmp_path / "previews"),
        frames=frames,
        render=None,
        serialize=None,
        max_active=1,
    )
    await selected.start()
    a = spec().model_copy(update={"kind": "rtsp", "uri": "rtsp://user:secret@localhost/live"})
    await selected.put(a)
    await selected.put(a.model_copy(update={"camera_id": "second"}))
    assert "secret" not in str(selected.snapshot(a.camera_id))
    await selected.run(a.camera_id)
    with pytest.raises(ValueError, match="active source limit"):
        await selected.run("second")
    await selected.stop(a.camera_id)
    assert sessions[0].closed and not selected.tasks
    await selected.run("second")
    await selected.close()
    assert all(s.closed for s in sessions)


async def test_edit_running_stream_applies_fps_resolution_and_metadata(tmp_path):
    app = create_application(engine(tmp_path, evidence=False), source_root=tmp_path / "registry")
    seen = []

    async def frames(source, state):
        seen.append(source)
        await asyncio.Event().wait()
        yield None

    async with app.router.lifespan_context(app), AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        app.state.sources.frames = frames
        payload = spec().model_copy(update={"kind": "rtsp", "uri": "rtsp://camera/live"}).model_dump()
        assert (await client.post("/v1/cameras", json=payload)).status_code == 201
        first = (await client.post("/v1/cameras/front-1/start")).json()
        response = await client.put("/v1/cameras/front-1", json={
            "inference_fps": 30, "preview_fps": 10, "max_width": 1280, "max_height": 720,
            "group": "Fleet A", "location": "Bus 101", "uri": ""})
        assert response.status_code == 200, response.text
        await asyncio.sleep(0)
        saved = response.json()
        assert saved["active"] and saved["session_id"] != first["session_id"]
        assert seen[-1].inference_fps == 30 and seen[-1].max_width == 1280
        assert seen[-1].preview_fps == 10 and seen[-1].max_height == 720
        assert saved["group"] == "Fleet A" and saved["location"] == "Bus 101"
        assert saved["effective_settings"]["inference_fps"] == 30
        assert "uri" not in saved["effective_settings"]
        assert (await app.state.sources.store.list_sources())[0].group == "Fleet A"
        await client.post("/v1/cameras/front-1/stop")
        saved = await client.put("/v1/cameras/front-1", json={"group": "Fleet B", "location": "Bus 102"})
        assert saved.json()["group"] == "Fleet B"

    restarted = create_application(
        Engine.from_project(tmp_path / "project", repository=SqlRepository(test_database_url(tmp_path)), evidence=False),
        source_root=tmp_path / "registry")
    async with restarted.router.lifespan_context(restarted), AsyncClient(
        transport=ASGITransport(app=restarted), base_url="http://test"
    ) as client:
        current = (await client.get("/v1/cameras")).json()["cameras"][0]
        # Historical runs retain old metadata; the catalog shows the current settings.
        assert current["group"] == "Fleet B" and current["location"] == "Bus 102"
        assert current["inference_fps"] == 30


async def test_slow_preview_does_not_block_analysis_and_keeps_one_pending_frame():
    from inferops.adapters.outbound.persistence.source_catalog import MemorySourceStore

    preview_started, release_preview = asyncio.Event(), asyncio.Event()

    class Session:
        id = "preview-test"
        annotator = None

        async def observe(self, frame):
            pass

        async def process(self, frame, **kwargs):
            return [StageItem(trace={"stage_ms.detect": "1.5"})]

        async def close(self, reason):
            pass

    async def open_session(*args, **kwargs):
        return Session()

    async def frames(source, state):
        for sequence in range(6):
            yield MediaFrame(np.zeros((8, 8, 3), np.uint8), sequence * 100_000_000, sequence)
            await asyncio.sleep(.01)

    async def render(*args):
        preview_started.set()
        await release_preview.wait()
        return b"jpeg"

    app = SimpleNamespace(definition=SimpleNamespace(pipelines={"test": None}), open_session=open_session)
    selected = SourceManager(app, MemorySourceStore(), frames=frames, render=render, serialize=lambda _: {})
    await selected.start()
    await selected.put(spec(inference_fps=30, preview_fps=15))
    await selected.run("front-1")
    try:
        await asyncio.wait_for(preview_started.wait(), 1)
        for _ in range(50):
            if selected.states["front-1"]["processed"] >= 4:
                break
            await asyncio.sleep(.01)
        assert selected.states["front-1"]["processed"] >= 4
        assert selected.states["front-1"]["stage_timings_ms"] == {"detect": 1.5}
    finally:
        release_preview.set()
        await selected.close()


async def test_decoder_cancellation_waits_for_native_read_before_close(monkeypatch):
    from inferops.adapters.inbound import sources

    reading = threading.Event()

    def slow(path, every_n):
        reading.set()
        time.sleep(0.07)
        yield MediaFrame(np.zeros((2, 2, 3), np.uint8))

    monkeypatch.setattr(sources, "_video_frames", slow)
    frames = source_frames("file.mp4")
    task = asyncio.create_task(anext(frames))
    await asyncio.to_thread(reading.wait)
    task.cancel()
    result = await asyncio.gather(task, return_exceptions=True)
    assert isinstance(result[0], asyncio.CancelledError)
    await frames.aclose()


async def test_registered_rtsp_closes_its_nested_reader(monkeypatch):
    from inferops.adapters.inbound import registered_sources

    closed = []

    async def reader(spec, stats):
        try:
            yield MediaFrame(np.zeros((2, 2, 3), np.uint8))
        finally:
            closed.append(True)

    monkeypatch.setattr(registered_sources, "capture_frames", reader)
    declaration = spec().model_copy(update={"kind": "rtsp", "uri": "rtsp://localhost/test"})
    frames = registered_sources.registered_frames(declaration, {"decoded": 0, "dropped": 0})
    await anext(frames)
    await frames.aclose()
    assert closed == [True]


async def test_failed_capture_cannot_deadlock_full_latest_frame_queue():
    processing = asyncio.Event()

    class Session:
        id = "test"

        async def observe(self, frame):
            pass

        async def process(self, frame, **kwargs):
            processing.set()
            await asyncio.Event().wait()

    async def frames():
        yield MediaFrame(None, sequence=0)
        await processing.wait()
        yield MediaFrame(None, sequence=1)
        raise ValueError("decoder failed with full queue")

    task = asyncio.create_task(process_live_frames(Session(), frames()))
    await processing.wait()
    await asyncio.sleep(0.02)
    task.cancel()
    await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 1)


@pytest.mark.parametrize(("step_ns", "expected"), [(100_000_000, [0, 5, 10, 15]),
                                                   (108_000_000, [0, 5, 10, 14, 19])])
async def test_live_evidence_observes_faster_than_inference_and_expired_work_is_not_a_result(step_ns, expected):
    observed, processed, reported = [], [], []

    class Session:
        id = "sampled"

        async def observe(self, frame):
            observed.append(frame.sequence)

        async def process(self, frame, **kwargs):
            processed.append(frame.sequence)
            if frame.sequence == 0:
                frame.payload["_admission_expired"] = True
                return []
            return [frame.sequence]

    async def frames():
        for n in range(20):
            yield MediaFrame(None, media_time_ns=n * step_ns, sequence=n)
            await asyncio.sleep(0)

    async def result(outputs, latency):
        reported.extend(outputs)

    dropped = await process_live_frames(Session(), frames(), inference_fps=2, on_result=result)
    assert observed == list(range(20))
    assert processed == expected
    assert reported == expected[1:]
    assert dropped == 1


def test_capacity_requires_matching_measured_profile_and_never_invents_maximum():
    hardware = {"cpu_model": "cpu", "ram_bytes": 16, "gpus": [{"uuid": "gpu1"}]}
    report = {
        "completed": True,
        "pipeline": "test",
        "runtime": "cuda",
        "model_artifacts": {"a": "checksum"},
        "hardware": hardware,
        "evidence_enabled": True,
        "preview_enabled": True,
        "cases": [{"width": 1920, "height": 1080, "inference_fps": 5, "cameras": 4, "passed": True}],
    }
    kwargs = dict(
        pipeline="test",
        runtime="cuda",
        model_artifacts={"a": "checksum"},
        width=1920,
        height=1080,
        fps=5,
        evidence=True,
    )
    result = assess_capacity([report], hardware, **kwargs)
    assert result["tested_passing_cameras"] == 4
    assert result["suggested_cameras_with_20_percent_headroom"] == 3
    assert result["maximum_cameras"] is None
    direct_report = {**report, "service_profile": {"media_router": "direct"}}
    assert assess_capacity([direct_report], hardware, **kwargs,
                           service_profile={"media_router": "direct"})["tested_passing_cameras"] == 4
    assert assess_capacity([direct_report], hardware, **kwargs,
                           service_profile={"media_router": "mediamtx"})["status"] == "unmeasured"
    for key, value in [
        ("fps", 10),
        ("model_artifacts", {"a": "changed"}),
        ("width", 1280),
        ("evidence", False),
        ("evidence_fps", 10),
    ]:
        assert assess_capacity([report], hardware, **{**kwargs, key: value})["status"] == "unmeasured"
    assert assess_capacity([report], {**hardware, "gpus": []}, **kwargs)["status"] == "unmeasured"
    faster_history = {**report, "cases": [{**report["cases"][0], "evidence_fps": 10}]}
    assert assess_capacity([faster_history], hardware, **kwargs)["status"] == "unmeasured"
    assert assess_capacity([faster_history], hardware, **kwargs, evidence_fps=10)["tested_passing_cameras"] == 4
    assert assess_capacity([{**report, "completed": False}], hardware, **kwargs)["status"] == "unmeasured"
    assert (
        assess_capacity([report], hardware, **kwargs, software_fingerprint="changed")["status"]
        == "unmeasured"
    )
    assert (
        assess_capacity([report], hardware, **kwargs, evidence_policy={"pre_seconds": 10})["status"]
        == "unmeasured"
    )
    assert (
        assess_capacity([report], {**hardware, "data_filesystem": "ext4"}, **kwargs)["status"] == "unmeasured"
    )
    assert (
        assess_capacity([report], hardware, **kwargs, service_profile={"repository": "SqlRepository"})[
            "status"
        ]
        == "unmeasured"
    )


def test_capacity_allows_ram_page_drift_but_rejects_different_ram_or_cpu_count():
    hardware = {"cpu_model": "cpu", "cpu_count": 24, "ram_bytes": 16 * 1024**3}
    report = {"completed": True, "pipeline": "test", "runtime": "cpu", "model_artifacts": {},
              "hardware": hardware, "evidence_enabled": False, "preview_enabled": True,
              "cases": [{"width": 640, "height": 480, "inference_fps": 2,
                         "cameras": 2, "passed": True}]}
    kwargs = dict(pipeline="test", runtime="cpu", model_artifacts={}, width=640, height=480,
                  fps=2, evidence=False)
    assert assess_capacity([report], {**hardware, "ram_bytes": hardware["ram_bytes"] + 8192}, **kwargs)["tested_passing_cameras"] == 2
    for changed in ({"ram_bytes": 8 * 1024**3}, {"cpu_count": 8}):
        assert assess_capacity([report], {**hardware, **changed}, **kwargs)["status"] == "unmeasured"


async def test_postgres_source_scopes_isolate_history_and_restart_recovery(tmp_path):
    url = test_database_url(tmp_path)
    a = PostgresSourceStore(url, scope="camera-a", preview_root=tmp_path / "previews")
    b = PostgresSourceStore(url, scope="camera-b", preview_root=tmp_path / "previews")
    await a.start()
    await b.start()
    try:
        await a.put_source(spec())
        await b.put_source(spec().model_copy(update={"name": "Other owner"}))
        for store, value in ((a, 1), (b, 2)):
            await store.save_run(dict(session_id="same-id", camera_id="front-1", status="running", started_at=time.time()))
            await store.save_result("front-1", "same-id", {"sequence": value})
            await store.save_preview("front-1", bytes([value]))
        assert (await a.list_sources())[0].name != (await b.list_sources())[0].name
        assert (await a.list_results("front-1"))[0]["sequence"] == 1
        assert (await b.list_results("front-1"))[0]["sequence"] == 2
        assert await a.get_preview("front-1") == b"\x01"
        assert await b.get_preview("front-1") == b"\x02"
        await a.close()
        await a.start()
        assert (await a.list_runs("front-1"))[0]["status"] == "interrupted"
        assert (await b.list_runs("front-1"))[0]["status"] == "running"
        await a.delete_source("front-1")
        assert len(await b.list_sources()) == 1
    finally:
        await a.close()
        await b.close()
