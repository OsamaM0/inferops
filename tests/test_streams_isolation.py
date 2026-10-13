"""Regression: a stream must never tear down the shared cached runner.

POST /v1/streams without rtsp_url/mjpeg_url used to hand back the pipeline
store's shared PipelineRunner and then tear it down when the stream ended,
leaving /v1/infer and /v1/live using closed stages for the rest of the process.
"""
from __future__ import annotations

import asyncio

import pytest

from inferops.api.routes import streams as streams_mod
from inferops.pipeline.spec import PipelineSpec
from inferops.pipeline.store import PipelineStore

_SPEC = {
    "name": "stream_pipe",
    "version": 1,
    "source": {"name": "src", "stage": "source_rest"},
    "stages": [{"name": "decode", "stage": "decode_any"}],
    "sinks": [],
}


@pytest.fixture
def store() -> PipelineStore:
    s = PipelineStore()
    spec = PipelineSpec(**_SPEC)
    s._specs[spec.name] = spec
    return s


@pytest.fixture(autouse=True)
def _clean_registry():
    streams_mod._RUNNING.clear()
    streams_mod._META.clear()
    yield
    for task in list(streams_mod._RUNNING.values()):
        task.cancel()
    streams_mod._RUNNING.clear()
    streams_mod._META.clear()


@pytest.mark.asyncio
async def test_default_source_stream_gets_its_own_runner(store):
    shared = await store.get_runner("stream_pipe")
    assert shared._setup_done

    result = await streams_mod.start_stream(
        pipeline="stream_pipe", stream_id="s1", store=store
    )
    assert result["status"] == "started"
    await asyncio.sleep(0.05)          # let the stream task run and finish

    await streams_mod.stop_stream("s1")

    # The shared runner must be untouched and still usable.
    assert store._runners["stream_pipe"] is shared
    assert shared._setup_done, "the shared runner was torn down by a stream"
    assert shared._stages, "the shared runner's stages were closed and dropped"


@pytest.mark.asyncio
async def test_reusing_a_live_stream_id_is_rejected_not_orphaned(store, monkeypatch):
    from fastapi import HTTPException

    # A source that never finishes, so the first stream stays live.
    async def forever(self, max_concurrency=None):
        await asyncio.Event().wait()

    monkeypatch.setattr(streams_mod.PipelineRunner, "start", forever)

    await streams_mod.start_stream(pipeline="stream_pipe", stream_id="dup", store=store)
    first = streams_mod._RUNNING["dup"]

    with pytest.raises(HTTPException) as exc:
        await streams_mod.start_stream(pipeline="stream_pipe", stream_id="dup", store=store)
    assert exc.value.status_code == 409

    assert streams_mod._RUNNING["dup"] is first, "the live task was orphaned"
    await streams_mod.stop_stream("dup")


@pytest.mark.asyncio
async def test_finished_streams_are_reaped(store):
    await streams_mod.start_stream(pipeline="stream_pipe", stream_id="short", store=store)
    await asyncio.sleep(0.05)          # source_rest yields nothing; task completes
    listing = await streams_mod.list_streams()
    assert all(row["stream_id"] != "short" for row in listing), (
        "finished streams were never evicted, so /v1/streams grows forever"
    )


@pytest.mark.asyncio
async def test_stop_waits_for_teardown(store, monkeypatch):
    torn_down = asyncio.Event()

    async def forever(self, max_concurrency=None):
        await asyncio.Event().wait()

    real_teardown = streams_mod.PipelineRunner.teardown

    async def tracking_teardown(self):
        await real_teardown(self)
        torn_down.set()

    monkeypatch.setattr(streams_mod.PipelineRunner, "start", forever)
    monkeypatch.setattr(streams_mod.PipelineRunner, "teardown", tracking_teardown)

    await streams_mod.start_stream(pipeline="stream_pipe", stream_id="slow", store=store)
    result = await streams_mod.stop_stream("slow")
    assert result["status"] == "stopped"
    assert torn_down.is_set(), "stop returned before the runner had been torn down"
