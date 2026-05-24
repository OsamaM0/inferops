"""Triton backend (lazy import)."""
from __future__ import annotations

import time
from contextlib import suppress
from typing import Any

from inferops.core.config import settings
from inferops.core.errors import BackendError
from inferops.obs import INFER_BATCH, INFER_LATENCY
from inferops.registry.model_spec import ModelSpec
from inferops.runtimes.base import InferenceBackend


class TritonBackend(InferenceBackend):
    def __init__(self, url: str | None = None) -> None:
        self._url = url or settings.triton.url
        self._client = None
        self._model_name: str = ""
        self._inputs_meta: list[tuple[str, str, list[int]]] = []
        self._outputs_meta: list[str] = []

    async def load(self, spec: ModelSpec) -> None:
        try:
            import tritonclient.grpc.aio as triton_aio
            from tritonclient.utils import InferenceServerException
        except ImportError as e:
            raise BackendError("tritonclient not installed (pip install 'inferops[triton]')") from e
        self.spec = spec
        if not spec.artifact.triton_model_name:
            raise BackendError(f"model {spec.name}: triton backend requires artifact.triton_model_name")
        self._model_name = spec.artifact.triton_model_name
        self._client = triton_aio.InferenceServerClient(url=self._url, verbose=False)
        try:
            meta: dict[str, Any] = await self._client.get_model_metadata(model_name=self._model_name, as_json=True)
            cfg = await self._client.get_model_config(model_name=self._model_name, as_json=True)
        except InferenceServerException as exc:
            if "unknown model" not in str(exc).lower():
                raise BackendError(str(exc)) from exc
            try:
                await self._client.load_model(self._model_name)
                meta = await self._client.get_model_metadata(model_name=self._model_name, as_json=True)
                cfg = await self._client.get_model_config(model_name=self._model_name, as_json=True)
            except Exception as load_exc:
                raise BackendError(f"failed to load Triton model {self._model_name}: {load_exc}") from load_exc
        cfg_inputs = {i["name"]: i for i in cfg.get("config", {}).get("input", [])}
        self._inputs_meta = [
            (i["name"], i["datatype"],
             [int(d) for d in cfg_inputs[i["name"]]["dims"]] if i["name"] in cfg_inputs else list(i["shape"]))
            for i in meta["inputs"]
        ]
        self._outputs_meta = [o["name"] for o in meta["outputs"]]

    async def infer(self, inputs):
        import tritonclient.grpc.aio as triton_aio
        from tritonclient.utils import np_to_triton_dtype
        if self._client is None:
            raise BackendError("triton backend not loaded")
        triton_inputs = []
        first = None
        for name, _dt, _ in self._inputs_meta:
            arr = inputs[name]
            if first is None:
                first = arr
            ti = triton_aio.InferInput(name, list(arr.shape), np_to_triton_dtype(arr.dtype))
            ti.set_data_from_numpy(arr)
            triton_inputs.append(ti)
        triton_outputs = [triton_aio.InferRequestedOutput(o) for o in self._outputs_meta]
        INFER_BATCH.labels(model=self.spec.name).observe(first.shape[0] if first is not None else 1)
        t0 = time.perf_counter()
        res = await self._client.infer(model_name=self._model_name, inputs=triton_inputs, outputs=triton_outputs)
        INFER_LATENCY.labels(model=self.spec.name).observe(time.perf_counter() - t0)
        return {o: res.as_numpy(o) for o in self._outputs_meta}

    async def close(self) -> None:
        if self._client is not None:
            with suppress(Exception):
                await self._client.close()
            self._client = None
