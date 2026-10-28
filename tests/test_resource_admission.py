"""Global GPU admission must remain fair and safe during cancellation."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from inferops.application.models import ModelPool
from inferops.domain.models import ModelSpec


def pool(*, remote=False, concurrency=1):
    started, releases = [], {}
    specs = {
        name: ModelSpec(
            name=name,
            backend="triton" if remote else "onnxruntime",
            inputs=[],
            device={"preferred": "gpu", "gpu_id": 0},
        )
        for name in ("a", "b")
    }

    class Backend:
        device = "remote" if remote else "gpu"

        @property
        def device_info(self):
            return {}

        async def load(self, spec):
            self.spec = spec

        async def infer(self, inputs):
            token = int(inputs["token"][0])
            started.append(token)
            releases[token] = asyncio.Event()
            await releases[token].wait()
            return inputs

        async def close(self):
            pass

    definition = SimpleNamespace(
        models=specs,
        resolver=None,
        source_dirs={},
        backend_factory=lambda *_: Backend(),
        resolve_model=lambda name: name,
    )
    return ModelPool(definition, max_concurrency=concurrency, triton_concurrency=3), started, releases


async def turn_until(predicate):
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("task did not reach expected state")


def request(models, model, token):
    return asyncio.create_task(models.infer(model, {"token": np.array([token])}))


async def test_models_share_gpu_and_waiters_are_fair():
    models, started, releases = pool()
    first = request(models, "a", 0)
    await turn_until(lambda: started == [0])
    second = request(models, "b", 1)
    await asyncio.sleep(0)
    third = request(models, "a", 2)
    await asyncio.sleep(0)
    assert started == [0]
    releases[0].set()
    await turn_until(lambda: len(started) == 2)
    assert started == [0, 1]
    releases[1].set()
    await turn_until(lambda: len(started) == 3)
    releases[2].set()
    await asyncio.gather(first, second, third)
    await models.close()


async def test_cancelled_native_inference_keeps_slot_until_it_finishes():
    models, started, releases = pool()
    first = request(models, "a", 0)
    await turn_until(lambda: bool(started))
    first.cancel()
    second = request(models, "b", 1)
    queued = request(models, "a", 2)
    await asyncio.sleep(0)
    queued.cancel()
    await asyncio.gather(queued, return_exceptions=True)
    assert started == [0] and not first.done()
    releases[0].set()
    await turn_until(lambda: len(started) == 2)
    releases[1].set()
    await asyncio.gather(first, second, return_exceptions=True)
    assert started == [0, 1]
    await models.close()


async def test_triton_requests_overlap_despite_one_local_gpu_slot():
    models, started, releases = pool(remote=True)
    tasks = [request(models, "a" if i % 2 else "b", i) for i in range(4)]
    await turn_until(lambda: len(started) == 3)
    assert len(started) == 3
    releases[started[0]].set()
    await turn_until(lambda: len(started) == 4)
    for event in releases.values():
        event.set()
    await asyncio.gather(*tasks)
    await models.close()


@pytest.mark.parametrize("local,remote", [(0, 1), (1, 0), (-1, 3)])
def test_admission_limits_must_be_positive(local, remote):
    with pytest.raises(ValueError):
        ModelPool(None, max_concurrency=local, triton_concurrency=remote)


async def test_warmup_uses_shared_gpu_admission_and_drains_on_cancel():
    models, started, releases = pool()
    first = request(models, "a", 0)
    await turn_until(lambda: started == [0])
    original = models.definition.backend_factory
    models.definition.models["b"].warmup.iterations = 1

    def factory(*args):
        backend = original(*args)
        async def warmup():
            started.append(99)
            releases[99] = asyncio.Event()
            await releases[99].wait()
        backend.warmup = warmup
        return backend

    models.definition.backend_factory = factory
    loading = asyncio.create_task(models.get_backend("b"))
    await asyncio.sleep(0)
    assert started == [0]
    releases[0].set()
    await turn_until(lambda: started == [0, 99])
    loading.cancel()
    second = request(models, "a", 1)
    await asyncio.sleep(0)
    assert not loading.done() and started == [0, 99]
    releases[99].set()
    await turn_until(lambda: started == [0, 99, 1])
    releases[1].set()
    await asyncio.gather(first, loading, second, return_exceptions=True)
    await models.close()


async def test_auto_gpu_capacity_tracks_sessions_and_shares_models():
    models, started, releases = pool(concurrency=3)
    models.auto_scale = True
    models.register_session("one")
    first = request(models, "a", 0)
    await turn_until(lambda: started == [0])
    second = request(models, "b", 1)
    await turn_until(lambda: models._gpu_slots[0].pending == 1)
    assert models.gpu_concurrency == 1
    models.register_session("two")
    await turn_until(lambda: started == [0, 1])
    models.register_session("three")
    models.register_session("four")
    assert models.gpu_concurrency == 3
    models.unregister_session("two")
    models.unregister_session("three")
    models.unregister_session("four")
    assert models.gpu_concurrency == 1
    assert models.runtime_status()[0]["concurrency"] == 1
    third = request(models, "a", 2)
    await turn_until(lambda: models._gpu_slots[0].pending == 1)
    releases[0].set()
    await first
    assert started == [0, 1]
    releases[1].set()
    await turn_until(lambda: started == [0, 1, 2])
    releases[2].set()
    await asyncio.gather(second, third)
    models.unregister_session("one")
    assert models._gpu_slots[0].available == 1
    await models.close()


async def test_session_lifecycle_resizes_its_model_pool(tmp_path):
    from inferops import Engine
    from tests.test_redesign_engine import project

    engine = Engine.from_project(project(tmp_path), model_concurrency=3, auto_scale=True)
    async with engine:
        first = await engine.open_session("test")
        second = await engine.open_session("test")
        assert engine.application.models.gpu_concurrency == 2
        await first.close()
        assert engine.application.models.gpu_concurrency == 1
        await second.close()
        assert engine.application.models._session_ids == set()
