"""Release checks for resource limits, restart behavior and capture scheduling."""
import asyncio
import subprocess
import sys
from fractions import Fraction
from types import SimpleNamespace

import av
import numpy as np
import pytest

from inferops.adapters.inbound.sources import process_live_frames, source_frames
from inferops.adapters.outbound.persistence.memory import MemoryRepository
from inferops.application.jobs import JobService
from inferops.domain.errors import PipelineError
from inferops.domain.processing import MediaFrame
from tests.test_redesign_engine import engine


async def test_session_admission_limit_covers_concurrent_setup(tmp_path):
    selected = engine(tmp_path, evidence=False)
    selected.application.max_sessions = 1
    async with selected:
        results = await asyncio.gather(*(selected.open_session("test") for _ in range(8)), return_exceptions=True)
        assert sum(not isinstance(result, Exception) for result in results) == 1
        assert sum(isinstance(result, PipelineError) for result in results) == 7
        assert len(selected.application.sessions) == 1


async def test_failed_startup_closes_all_owned_resources(tmp_path, monkeypatch):
    selected = engine(tmp_path, evidence=False)
    selected.application.prewarm_models = ("broken",)

    async def failed(*args):
        raise ValueError("model could not start")

    monkeypatch.setattr(selected.application.models, "get_backend", failed)
    with pytest.raises(ValueError, match="model could not start"):
        await selected.start()
    assert selected.application.repository._engine is None
    assert selected.application.models._closed
    with pytest.raises(RuntimeError, match="closed"):
        await selected.open_session("test")


async def test_shutdown_preserves_queued_jobs_and_marks_running_interrupted():
    repository = MemoryRepository()
    started = asyncio.Event()

    async def wait(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    application = SimpleNamespace(repository=repository, definition=SimpleNamespace(pipelines={"test": object()}))
    selected = SimpleNamespace(application=application, run=wait)
    jobs = JobService(selected, max_running=1)
    running = await jobs.submit("test", {})
    await started.wait()
    queued = await jobs.submit("test", {"stream_id": "queued"})
    await asyncio.sleep(0)
    await jobs.close()
    assert (await repository.get_job(running["job_id"]))["status"] == "failed"
    assert (await repository.get_job(queued["job_id"]))["status"] == "queued"

    async def finish(*args, **kwargs):
        return {"frames": 1}

    selected.run = finish
    recovered = JobService(selected)
    await recovered.start()
    await asyncio.gather(*list(recovered.tasks.values()))
    assert (await repository.get_job(queued["job_id"]))["status"] == "succeeded"
    await recovered.close()


async def test_live_capture_observes_frames_before_latest_frame_drops():
    observed, processed = [], []

    async def capture(frame):
        observed.append(frame.sequence)

    async def process(frame, **kwargs):
        await asyncio.sleep(.02)
        processed.append(frame.sequence)
        return []

    session = SimpleNamespace(id="capture", observe=capture, process=process)

    async def frames():
        for sequence in range(10):
            yield MediaFrame(None, sequence * 1_000_000, sequence)
            await asyncio.sleep(.001)

    dropped = await process_live_frames(session, frames())
    assert observed == list(range(10))
    assert dropped > 0 and len(processed) + dropped == 10
    assert processed[-1] == 9


async def test_live_capture_shutdown_does_not_block_on_full_queue():
    async def noop(*args, **kwargs):
        pass

    async def blocked(*args, **kwargs):
        await asyncio.Event().wait()

    async def frames():
        for sequence in range(100):
            yield MediaFrame(None, sequence, sequence)
        await asyncio.Event().wait()

    session = SimpleNamespace(id="cancel", observe=noop, process=blocked)
    task = asyncio.create_task(process_live_frames(session, frames()))
    await asyncio.sleep(.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)


async def test_incremental_video_source_keeps_variable_presentation_timestamps(tmp_path):
    path = tmp_path / "variable.mp4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
        for timestamp in (0, 100, 500):
            frame = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), np.uint8), format="bgr24")
            frame.pts, frame.time_base = timestamp, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    decoded = [frame async for frame in source_frames(path)]
    assert [frame.media_time_ns for frame in decoded] == [0, 100_000_000, 500_000_000]
    assert {frame.timestamp_provenance for frame in decoded} == {"presentation_timestamp"}


def test_asgi_import_does_not_load_deployment_configuration():
    code = """from unittest.mock import patch
with patch('inferops.adapters.outbound.projects.settings.load_settings', side_effect=AssertionError('import reads settings')):
    import inferops.adapters.inbound.http.main
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            env={"PATH": "/usr/bin:/bin", "INFEROPS_ENGINE": "hexagonal"})
    assert result.returncode == 0, result.stderr
