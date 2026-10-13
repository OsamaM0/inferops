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

# Use conservative keepalive settings so the client does not trigger Triton's
# ENHANCE_YOUR_CALM / too_many_pings protection when the channel is idle.
_GRPC_CHANNEL_ARGS = [
    ("grpc.keepalive_time_ms", 300_000),             # 5 min between keepalive pings
    ("grpc.keepalive_timeout_ms", 20_000),           # give server 20 s to respond
    ("grpc.keepalive_permit_without_calls", False),  # do not ping idle channels
    ("grpc.http2.max_pings_without_data", 2),        # match server-side defaults
    ("grpc.http2.min_time_between_pings_ms", 300_000),
    ("grpc.http2.min_ping_interval_without_data_ms", 300_000),
]


class TritonBackend(InferenceBackend):
    def __init__(self, url: str | None = None) -> None:
        self._url = url or settings.triton.url
        self._client = None
        self._infer_input_cls: Any | None = None
        self._model_name: str = ""
        self._inputs_meta: list[tuple[str, str, list[int]]] = []
        self._outputs_meta: list[str] = []
        self._triton_outputs: list[Any] = []
        self._batch_observer: Any | None = None
        self._latency_observer: Any | None = None

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
        self._infer_input_cls = triton_aio.InferInput
        self._client = triton_aio.InferenceServerClient(
            url=self._url, verbose=False, channel_args=_GRPC_CHANNEL_ARGS,
        )
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
        self._triton_outputs = [triton_aio.InferRequestedOutput(o) for o in self._outputs_meta]
        self._batch_observer = INFER_BATCH.labels(model=spec.name)
        self._latency_observer = INFER_LATENCY.labels(model=spec.name)

        # Triton owns device placement via instance_group in config.pbtxt, so
        # report what the server told us rather than guessing from DeviceSpec.
        instance_groups = cfg.get("config", {}).get("instance_group", []) or []
        kinds = [g.get("kind", "") for g in instance_groups]
        self.device = "remote"
        self.device_info = {
            "requested": spec.device.preferred,
            "effective": "remote",
            "server": self._url,
            "triton_model": self._model_name,
            "instance_kinds": kinds,
            "downgraded": bool(kinds) and spec.device.preferred == "gpu"
                          and not any("GPU" in k for k in kinds),
            "reason": f"placement decided by Triton instance_group ({', '.join(kinds) or 'unspecified'})",
        }

    async def infer(self, inputs):
        if self._client is None or self._infer_input_cls is None:
            raise BackendError("triton backend not loaded")
        triton_inputs = []
        first = None
        for name, datatype, _ in self._inputs_meta:
            arr = inputs[name]
            if first is None:
                first = arr
            ti = self._infer_input_cls(name, list(arr.shape), datatype)
            ti.set_data_from_numpy(arr)
            triton_inputs.append(ti)
        if self._batch_observer is not None:
            self._batch_observer.observe(first.shape[0] if first is not None else 1)
        t0 = time.perf_counter()
        res = await self._client.infer(model_name=self._model_name, inputs=triton_inputs, outputs=self._triton_outputs)
        if self._latency_observer is not None:
            self._latency_observer.observe(time.perf_counter() - t0)
        return {o: res.as_numpy(o) for o in self._outputs_meta}

    async def close(self) -> None:
        if self._client is not None:
            with suppress(Exception):
                await self._client.close()
            self._client = None
