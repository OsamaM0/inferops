"""Caching and resource cleanup through engine-owned pools and sessions."""

import asyncio
from dataclasses import replace

import pytest

from inferops.adapters.outbound.projects.loader import ProjectLoader
from inferops.adapters.outbound.stages.base import Stage
from inferops.application.models import ModelPool
from inferops.domain.errors import BackendError, RegistryError
from tests.test_redesign_engine import engine
from tests.test_redesign_projects_models import project


def pool(tmp_path, factory=None):
    definition = ProjectLoader(project(tmp_path / "sample")).load()
    return ModelPool(replace(definition, backend_factory=factory) if factory else definition)


async def test_backend_is_built_once_and_cached(tmp_path):
    selected = pool(tmp_path)
    assert await selected.get_backend("identity") is await selected.get_backend("identity")
    await selected.close()


async def test_concurrent_first_use_builds_exactly_one_backend(tmp_path):
    selected = pool(tmp_path)
    backends = await asyncio.gather(*(selected.get_backend("identity") for _ in range(5)))
    assert len({id(backend) for backend in backends}) == 1
    assert type(backends[0]._backend).loads == 1
    await selected.close()


async def test_failed_load_is_not_cached_and_is_closed(tmp_path):
    selected = pool(tmp_path)
    original = selected.definition.backend_factory
    closed = []

    def factory(spec, path):
        backend = original(spec, path)

        async def fail(_spec):
            raise BackendError("load exploded")

        async def close():
            closed.append(backend)

        backend.load, backend.close = fail, close
        return backend

    selected.definition = replace(selected.definition, backend_factory=factory)
    with pytest.raises(BackendError):
        await selected.get_backend("identity")
    assert len(closed) == 1 and selected._backends == {}
    selected.definition = replace(selected.definition, backend_factory=original)
    assert await selected.get_backend("identity") is not None
    await selected.close()


async def test_close_releases_every_backend(tmp_path):
    selected = pool(tmp_path)
    backend = await selected.get_backend("identity")
    calls = []

    async def close():
        calls.append(True)

    backend._backend.close = close
    await selected.close()
    await selected.close()
    assert calls == [True] and selected._backends == {}


async def test_unknown_model_raises_registry_error(tmp_path):
    selected = pool(tmp_path)
    with pytest.raises(RegistryError):
        await selected.get_backend("nope")
    await selected.close()


async def test_sessions_share_resources_but_never_mutable_steps(tmp_path):
    selected = engine(tmp_path, evidence=False)
    async with selected:
        sessions = await asyncio.gather(
            *(selected.open_session("test", stream_name="same") for _ in range(5))
        )
        assert len({session.id for session in sessions}) == 5
        assert len({id(session.steps[0][2]) for session in sessions}) == 5
        assert all(session.models is selected.application.models for session in sessions)
        await sessions[0].close()
        assert not sessions[1].closed and not selected.application.models._closed


async def test_failed_stage_setup_is_torn_down_and_session_released(tmp_path):
    selected = engine(tmp_path, evidence=False)
    torn_down = []

    class Broken(Stage):
        async def setup(self, ctx):
            raise RuntimeError("stage setup exploded")

        async def process(self, item):
            return item

        async def teardown(self):
            torn_down.append(True)

    async with selected:
        factory = selected.application.stage_factory
        selected.application.stage_factory = lambda declaration: Broken()
        with pytest.raises(RuntimeError, match="setup exploded"):
            await selected.open_session("test")
        assert torn_down == [True] and not selected.application.sessions
        selected.application.stage_factory = factory
        assert not (await selected.open_session("test")).closed


async def test_close_tears_down_every_session_even_if_one_raises(tmp_path):
    selected = engine(tmp_path, evidence=False)
    await selected.start()
    first, second = await selected.open_session("test"), await selected.open_session("test")

    async def fail():
        raise RuntimeError("teardown exploded")

    first.steps[0][2].teardown = fail
    with pytest.raises(RuntimeError, match="shutdown failures"):
        await selected.close()
    assert first.closed and second.closed and not selected.application.sessions
    assert selected.application.models._closed
