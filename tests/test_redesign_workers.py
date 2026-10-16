"""Worker entry points use generated sessions and acknowledge durable processing."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from types import SimpleNamespace

import pytest

from inferops.adapters.inbound.workers.execution import process_source
from inferops.adapters.inbound.workers.topic_worker import run_topic_worker
from inferops.domain.types import StageItem
from inferops.ports.messaging import Message


class FakeConsumer:
    def __init__(self, messages):
        self.messages, self.commits, self.seen = messages, [], []
        self.closed = False

    async def subscribe(self, topics, group):
        self.subscription = topics, group

    async def stream(self):
        for message in self.messages:
            self.seen.append(message)
            yield message

    async def commit(self, message):
        self.commits.append(message)

    async def close(self):
        self.closed = True


class FakeTransport:
    def __init__(self, messages):
        self.input = FakeConsumer(messages)
        self.topics = []

    async def admin_ensure_topic(self, topic):
        self.topics.append(topic)

    async def consumer(self, group):
        return self.input

    async def producer(self, name):
        raise AssertionError("workers must publish through the transactional outbox")

    async def close(self):
        await self.input.close()


class FakeSession:
    def __init__(self, *, fail=False, slow=False):
        self.id = uuid.uuid4().hex
        self.last_active = time.monotonic()
        self.frames, self.observed = [], []
        self.fail, self.slow, self.closed = fail, slow, False

    async def process(self, frame, *, observed=False):
        self.frames.append(frame)
        self.last_active = time.monotonic()
        if self.slow:
            await asyncio.sleep(0.01)
        if self.fail:
            return [StageItem(payload={"incident_error": {"code": "persistence_unavailable"}})]
        return [StageItem(payload={"session_id": self.id})]

    async def observe(self, frame):
        self.observed.append(frame)

    async def close(self, reason="source_closed"):
        self.closed = True


class FakeEngine:
    def __init__(self, messages=(), *, fail=False, slow=False):
        self.application = SimpleNamespace(transport=FakeTransport(messages), evidence=None)
        self.sessions = []
        self.fail, self.slow, self.closed = fail, slow, False

    async def open_session(self, pipeline, **options):
        session = FakeSession(fail=self.fail, slow=self.slow)
        session.options = options
        self.sessions.append(session)
        return session

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True


def broker_message(offset, *, key=b"camera", partition=0):
    return Message("frames", json.dumps({"stream_id": "same", "payload": {"value": offset}}).encode(),
                   key=key, partition=partition, offset=offset)


async def test_topic_worker_uses_owned_sessions_and_exact_input_identity():
    messages = [broker_message(0), broker_message(1), broker_message(2, key=b"another"),
                broker_message(0, partition=1)]
    engine = FakeEngine(messages)
    await run_topic_worker("driver", input_topic="frames", output_topic="predictions", engine=engine)
    assert len(engine.sessions) == 3
    assert len({session.id for session in engine.sessions}) == 3
    frames = engine.sessions[0].frames
    assert [frame.sequence for frame in frames] == [0, 1]
    assert [frame.payload["_broker_event"]["event_id"] for frame in frames] == ["frames:0:0", "frames:0:1"]
    assert frames[0].payload["_broker_event"]["output_topic"] == "predictions"
    assert all(session.options["owner_kind"] == "broker" for session in engine.sessions)
    assert engine.application.transport.input.commits == messages
    assert all(session.closed for session in engine.sessions)
    assert engine.closed


async def test_failed_durable_processing_never_commits_or_skips_to_later_offsets():
    messages = [broker_message(0), broker_message(1)]
    engine = FakeEngine(messages, fail=True)
    with pytest.raises(RuntimeError, match="durable processing failed"):
        await run_topic_worker("driver", input_topic="frames", output_topic="predictions", engine=engine)
    consumer = engine.application.transport.input
    assert consumer.commits == []
    assert consumer.seen == messages[:1]
    assert consumer.closed
    assert engine.sessions[0].closed


async def test_broker_capacity_failure_is_visible_before_offset_commit(monkeypatch):
    monkeypatch.setenv("INFEROPS_WORKER_MAX_SESSIONS", "1")
    messages = [broker_message(0), broker_message(1, key=b"second")]
    engine = FakeEngine(messages)
    with pytest.raises(RuntimeError, match="capacity exhausted"):
        await run_topic_worker("driver", input_topic="frames", output_topic=None, engine=engine)
    assert engine.application.transport.input.commits == messages[:1]


class Source:
    stream_id = "same"

    def __init__(self, *, live=False):
        self.params = {"live": live}

    async def run(self):
        for number in range(20):
            yield StageItem(stream_id=self.stream_id, payload={"value": number})
            await asyncio.sleep(0)


async def test_offline_worker_applies_backpressure_and_keeps_every_frame():
    engine = FakeEngine(slow=True)
    await process_source(engine, "driver", Source())
    session = engine.sessions[0]
    assert [frame.sequence for frame in session.frames] == list(range(20))
    assert session.closed


async def test_live_worker_observes_before_dropping_and_bounds_pending_frames():
    engine = FakeEngine(slow=True)
    await process_source(engine, "driver", Source(live=True))
    session = engine.sessions[0]
    assert len(session.observed) == 20
    assert len(session.frames) < 20
    assert session.frames[-1].sequence == 19
    assert session.frames[-1].payload["source_frames_dropped"] > 0
    assert all(frame.payload["live_mode"] for frame in session.observed)
    assert all(frame.media_time_ns is not None for frame in session.observed)
    assert session.closed


def test_cli_pipeline_run_uses_shared_engine(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from inferops.adapters.inbound.cli.__main__ import app
    from inferops.adapters.inbound.workers import bootstrap

    source = tmp_path / "input.jpg"
    source.write_bytes(b"fixture")
    engine = FakeEngine()
    calls = []

    async def run(name, path, **options):
        calls.append((name, path, options))
        return {"frames": 1}

    engine.run = run
    monkeypatch.setenv("INFEROPS_ENGINE", "hexagonal")
    monkeypatch.setattr(bootstrap, "server_engine", lambda: engine)
    result = CliRunner().invoke(app, ["pipelines", "run", "driver", str(source)])
    assert result.exit_code == 0, result.output
    assert calls == [("driver", source, {"stream_name": "cli", "owner_kind": "cli"})]
    assert engine.closed


async def test_redpanda_commits_only_the_processed_partition_offset():
    from aiokafka.structs import TopicPartition

    from inferops.adapters.outbound.messaging.redpanda import _RPConsumer
    from inferops.domain.errors import TransportError

    commits = []

    async def commit(offsets):
        commits.append(offsets)

    consumer = _RPConsumer("localhost:9092", "test")
    consumer._c = SimpleNamespace(commit=commit)
    await consumer.commit(broker_message(7, partition=3))
    assert commits == [{TopicPartition("frames", 3): 8}]
    with pytest.raises(TransportError, match="partition and offset"):
        await consumer.commit(Message("frames", b"{}"))
    assert len(commits) == 1


async def test_real_pipeline_broker_outputs_and_inbox_commit_together(tmp_path):
    from sqlalchemy import func, select

    from inferops.adapters.outbound.persistence.sql import SqlRepository
    from inferops.bootstrap import build_engine

    root = tmp_path / "sample"
    pipeline = root / "configs" / "pipelines"
    pipeline.mkdir(parents=True)
    (root / "project.yaml").write_text("name: sample\n")
    (pipeline / "driver.yaml").write_text("name: driver\nstages: []\n")
    repository = SqlRepository(f"sqlite+aiosqlite:///{tmp_path / 'worker.db'}")
    messages = [broker_message(0), broker_message(0), broker_message(1)]
    transport = FakeTransport(messages)
    engine = build_engine(root, repository=repository, transport=transport)
    async with engine:
        await run_topic_worker("driver", input_topic="frames", output_topic="predictions", engine=engine,
                               manage_engine=False)
        async with repository.engine.connect() as connection:
            assert await connection.scalar(select(func.count()).select_from(repository.inbox)) == 2
            assert await connection.scalar(select(func.count()).select_from(repository.outbox)) == 2
        records = await repository.claim_outbox("test-reader")
        assert all(record["topic"] == "predictions" for record in records)
        assert [record["payload"]["payload"]["value"] for record in records] == [0, 1]
        assert all("_broker_event" not in record["payload"]["payload"] for record in records)
    assert transport.input.commits == messages


def test_new_cli_server_does_not_activate_legacy_preflight(monkeypatch):
    import uvicorn

    from inferops.adapters.inbound.cli import __main__ as cli
    calls = []

    def legacy_preflight(**_options):
        raise AssertionError("new server startup must not activate legacy stores")

    monkeypatch.setenv("INFEROPS_ENGINE", "hexagonal")
    monkeypatch.setattr(cli, "preflight", legacy_preflight)
    monkeypatch.setattr(uvicorn, "run", lambda target, **options: calls.append((target, options)))
    cli.serve(host="127.0.0.1", port=8089)
    assert calls == [("inferops.adapters.inbound.http.main:app", {"host": "127.0.0.1", "port": 8089, "reload": False})]


async def test_configured_folder_source_uses_shared_decoder_and_owned_context(tmp_path):
    import cv2
    import numpy as np

    from inferops.adapters.inbound.sources import source_frames
    from inferops.adapters.inbound.workers.pipeline_worker import run_pipeline_worker
    from inferops.adapters.outbound.projects import ProjectLoader
    from inferops.adapters.outbound.stages.factory import stage_factory

    folder = tmp_path / "frames"
    folder.mkdir()
    cv2.imwrite(str(folder / "one.jpg"), np.zeros((3, 4, 3), dtype=np.uint8))
    root = tmp_path / "project"
    pipelines = root / "configs" / "pipelines"
    pipelines.mkdir(parents=True)
    (root / "project.yaml").write_text("name: project\n")
    (pipelines / "driver.yaml").write_text(
        "name: driver\nsource:\n  name: folder\n  stage: source_folder\n"
        f"  params: {{path: '{folder}'}}\nstages: [{{name: decode, stage: decode_any}}]\n")
    definition = ProjectLoader(root).load()
    engine = FakeEngine()
    engine.application.definition = definition
    engine.application.stage_factory = stage_factory(definition)
    engine.application.source_factory = source_frames
    engine.application.models = SimpleNamespace(max_concurrency=1)
    engine.application.owner = "worker"
    await run_pipeline_worker("driver", engine=engine)
    session = engine.sessions[0]
    assert len(session.frames) == 1
    assert session.frames[0].image.shape == (3, 4, 3)
    assert session.frames[0].payload["filename"] == "one.jpg"
    assert session.frames[0].media_time_ns is not None
    assert session.closed
