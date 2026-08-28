"""Lifecycle: caching, concurrency, failure cleanup, and teardown.

The registry and pipeline store both cached objects with a check-then-act that
had no lock, and neither released a half-built object when construction failed.
These tests pin the corrected behaviour so it cannot regress.
"""
from __future__ import annotations

import asyncio

import pytest

from inferops.core.errors import BackendError, RegistryError
from inferops.pipeline.spec import PipelineSpec
from inferops.pipeline.store import PipelineStore
from inferops.registry.model_spec import ModelSpec
from inferops.registry.store import ModelRegistry

# ── model registry ───────────────────────────────────────────────────────────


def _synth_spec(name: str = "synth") -> ModelSpec:
    return ModelSpec(name=name, backend="synthetic", inputs=[],
                     tags={"classes": "6", "queries": "20"})


async def _registry_with(spec: ModelSpec) -> ModelRegistry:
    reg = ModelRegistry()
    reg._specs[spec.name] = spec
    return reg


@pytest.mark.asyncio
async def test_backend_is_built_once_and_cached():
    reg = await _registry_with(_synth_spec())
    first = await reg.get_backend("synth")
    second = await reg.get_backend("synth")
    assert first is second
    await reg.close()


@pytest.mark.asyncio
async def test_concurrent_first_use_builds_exactly_one_backend(monkeypatch):
    """Without the double-checked lock each caller built its own backend."""
    reg = await _registry_with(_synth_spec())
    built = []

    import inferops.runtimes as runtimes_mod

    real_build = runtimes_mod.build_backend

    def counting_build(spec, base_dir=None):
        backend = real_build(spec, base_dir=base_dir)
        built.append(backend)
        real_load = backend.load

        async def slow_load(s):
            await asyncio.sleep(0.01)      # any real IO during load does this
            return await real_load(s)

        backend.load = slow_load
        return backend

    monkeypatch.setattr(runtimes_mod, "build_backend", counting_build)

    backends = await asyncio.gather(*[reg.get_backend("synth") for _ in range(5)])
    assert len(built) == 1, f"built {len(built)} backends for one model"
    assert len({id(b) for b in backends}) == 1
    await reg.close()


@pytest.mark.asyncio
async def test_failed_load_is_not_cached_and_is_closed(monkeypatch):
    """A backend that dies mid-load must not be handed to the next caller."""
    reg = await _registry_with(_synth_spec())
    closed = []

    import inferops.runtimes as runtimes_mod

    real_build = runtimes_mod.build_backend

    def failing_build(spec, base_dir=None):
        backend = real_build(spec, base_dir=base_dir)

        async def boom(_s):
            raise BackendError("load exploded")

        async def track_close():
            closed.append(backend)

        backend.load = boom
        backend.close = track_close
        return backend

    monkeypatch.setattr(runtimes_mod, "build_backend", failing_build)

    with pytest.raises(BackendError):
        await reg.get_backend("synth")

    assert closed, "half-built backend was leaked instead of closed"
    assert "synth" not in reg._backends, "broken backend was cached"

    # The next call must be free to retry rather than inherit the failure.
    monkeypatch.setattr(runtimes_mod, "build_backend", real_build)
    backend = await reg.get_backend("synth")
    assert backend is not None
    await reg.close()


@pytest.mark.asyncio
async def test_close_releases_every_backend():
    reg = await _registry_with(_synth_spec())
    backend = await reg.get_backend("synth")
    closed = []
    backend.close = lambda: closed.append(True) or asyncio.sleep(0)
    await reg.close()
    assert reg._backends == {}
    assert reg._specs == {}


@pytest.mark.asyncio
async def test_unknown_model_raises_registry_error():
    reg = ModelRegistry()
    with pytest.raises(RegistryError):
        await reg.get_backend("nope")


# ── pipeline store ───────────────────────────────────────────────────────────

_MINIMAL_PIPELINE = {
    "name": "unit_pipe",
    "version": 1,
    "source": {"name": "src", "stage": "source_rest"},
    "stages": [{"name": "decode", "stage": "decode_any", "params": {"every_n": 1}}],
    "sinks": [],
}


def _store_with_spec() -> PipelineStore:
    store = PipelineStore()
    spec = PipelineSpec(**_MINIMAL_PIPELINE)
    store._specs[spec.name] = spec
    return store


@pytest.mark.asyncio
async def test_runner_is_built_once_and_cached():
    store = _store_with_spec()
    first = await store.get_runner("unit_pipe")
    second = await store.get_runner("unit_pipe")
    assert first is second
    await store.close()


@pytest.mark.asyncio
async def test_concurrent_first_use_builds_exactly_one_runner(monkeypatch):
    """Two concurrent requests used to orphan a fully set-up runner."""
    store = _store_with_spec()
    constructed = []

    import inferops.pipeline.store as store_mod

    RealRunner = store_mod.PipelineRunner

    class SlowRunner(RealRunner):
        async def setup(self):
            constructed.append(self)
            await asyncio.sleep(0.01)
            return await super().setup()

    monkeypatch.setattr(store_mod, "PipelineRunner", SlowRunner)

    runners = await asyncio.gather(*[store.get_runner("unit_pipe") for _ in range(5)])
    cached = store._runners["unit_pipe"]

    assert len(constructed) == 1, f"{len(constructed)} runners set up for one pipeline"
    assert all(r is cached for r in runners), "a caller got an orphaned runner"
    await store.close()


@pytest.mark.asyncio
async def test_failed_setup_is_torn_down_and_not_cached(monkeypatch):
    store = _store_with_spec()
    torn_down = []

    import inferops.pipeline.store as store_mod

    RealRunner = store_mod.PipelineRunner

    class BrokenRunner(RealRunner):
        async def setup(self):
            raise RuntimeError("stage setup exploded")

        async def teardown(self):
            torn_down.append(self)

    monkeypatch.setattr(store_mod, "PipelineRunner", BrokenRunner)

    with pytest.raises(RuntimeError):
        await store.get_runner("unit_pipe")

    assert torn_down, "half-built runner was leaked instead of torn down"
    assert "unit_pipe" not in store._runners, "broken runner was cached"


@pytest.mark.asyncio
async def test_close_tears_down_every_runner_even_if_one_raises():
    store = _store_with_spec()
    runner = await store.get_runner("unit_pipe")

    calls = []

    async def angry_teardown():
        calls.append("first")
        raise RuntimeError("teardown exploded")

    runner.teardown = angry_teardown
    await store.close()                      # must not propagate
    assert calls == ["first"]
    assert store._runners == {}


@pytest.mark.asyncio
async def test_reload_replaces_specs_and_drops_stale_runners(tmp_path):
    store = _store_with_spec()
    await store.get_runner("unit_pipe")
    assert store._runners

    other = tmp_path / "pipelines"
    other.mkdir()
    (other / "fresh.yaml").write_text(
        "name: fresh_pipe\nversion: 1\n"
        "source: {name: src, stage: source_rest}\n"
        "stages: [{name: decode, stage: decode_any}]\n",
        encoding="utf-8",
    )

    count = await store.reload_from(other)
    assert count == 1
    assert store._runners == {}, "stale runners survived a config reload"
    assert [s.name for s in store.list()] == ["fresh_pipe"]
    await store.close()
