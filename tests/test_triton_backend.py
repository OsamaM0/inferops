import sys
import types

import numpy as np
import pytest

from inferops.adapters.outbound.inference.triton_backend import TritonBackend
from inferops.domain.errors import BackendError
from inferops.domain.models import ModelSpec


@pytest.mark.asyncio
@pytest.mark.parametrize("dynamic", [False, True])
async def test_triton_backend_reuses_loaded_output_requests(monkeypatch, dynamic):
    requested_outputs: list[str] = []
    infer_outputs: list[list[object]] = []

    class FakeInferenceServerException(Exception):
        pass

    class FakeInferInput:
        def __init__(self, name, shape, datatype):
            self.name = name
            self.shape = shape
            self.datatype = datatype
            self.array = None

        def set_data_from_numpy(self, array):
            self.array = array

    class FakeInferRequestedOutput:
        def __init__(self, name):
            requested_outputs.append(name)
            self.name = name

    class FakeResult:
        def as_numpy(self, name):
            return np.array([name])

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def get_model_metadata(self, *, model_name, as_json):
            return {
                "inputs": [{"name": "input", "datatype": "FP32", "shape": [1, 3, 2, 2]}],
                "outputs": [{"name": "dets"}, {"name": "labels"}],
            }

        async def get_model_config(self, *, model_name, as_json):
            config = {"input": [{"name": "input", "dims": [3, 2, 2] if dynamic else [1, 3, 2, 2]}]}
            if dynamic:
                config.update(max_batch_size=4, dynamic_batching={})
            return {"config": config}

        async def get_inference_statistics(self, **kwargs):
            return {"model_stats": [{"name": "rfdetr", "version": "1", "inference_count": "32", "execution_count": "8"}]}

        async def infer(self, *, model_name, inputs, outputs, request_id, timeout, client_timeout):
            assert len(request_id) == 32
            assert timeout == 10_000_000
            assert client_timeout == 10
            infer_outputs.append(outputs)
            assert inputs[0].datatype == "FP32"
            return FakeResult()

    triton_mod = types.ModuleType("tritonclient")
    grpc_mod = types.ModuleType("tritonclient.grpc")
    aio_mod = types.ModuleType("tritonclient.grpc.aio")
    aio_mod.InferenceServerClient = FakeClient
    aio_mod.InferInput = FakeInferInput
    aio_mod.InferRequestedOutput = FakeInferRequestedOutput
    utils_mod = types.ModuleType("tritonclient.utils")
    utils_mod.InferenceServerException = FakeInferenceServerException
    grpc_mod.aio = aio_mod
    triton_mod.grpc = grpc_mod
    triton_mod.utils = utils_mod

    monkeypatch.setitem(sys.modules, "tritonclient", triton_mod)
    monkeypatch.setitem(sys.modules, "tritonclient.grpc", grpc_mod)
    monkeypatch.setitem(sys.modules, "tritonclient.grpc.aio", aio_mod)
    monkeypatch.setitem(sys.modules, "tritonclient.utils", utils_mod)

    backend = TritonBackend(url="localhost:8001")
    spec = ModelSpec(
        name="fast_triton",
        backend="triton",
        artifact={"triton_model_name": "rfdetr"},
        inputs=[{"name": "input", "shape": [1, 3, 2, 2], "dtype": "float32"}],
    )
    await backend.load(spec)

    assert backend._client.kwargs["channel_args"] == [
        ("grpc.keepalive_time_ms", 300_000),
        ("grpc.keepalive_timeout_ms", 20_000),
        ("grpc.keepalive_permit_without_calls", False),
        ("grpc.http2.max_pings_without_data", 2),
        ("grpc.http2.min_time_between_pings_ms", 300_000),
        ("grpc.http2.min_ping_interval_without_data_ms", 300_000),
    ]

    feeds = {"input": np.zeros((1, 3, 2, 2), dtype=np.float32)}
    await backend.infer(feeds)
    await backend.infer(feeds)

    assert requested_outputs == ["dets", "labels"]
    assert len(infer_outputs) == 2
    assert infer_outputs[0] is infer_outputs[1]
    for invalid in ({"wrong": feeds["input"]}, {"input": feeds["input"].astype("float64")},
                    {"input": np.zeros((5, 3, 2, 2), np.float32)}):
        with pytest.raises(BackendError):
            await backend.infer(invalid)
    assert len(infer_outputs) == 2
    stats = await backend.statistics()
    assert stats["dynamic_batching"] is dynamic
    assert stats["versions"][0]["average_execution_batch_size"] == 4
