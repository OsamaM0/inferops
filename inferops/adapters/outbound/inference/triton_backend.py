"""Triton backend (lazy import)."""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from contextlib import suppress
from typing import Any

import numpy as np

from inferops.adapters.outbound.observability import INFER_BATCH, INFER_LATENCY
from inferops.domain.errors import BackendError
from inferops.domain.models import ModelSpec
from inferops.ports.inference import InferenceBackend

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
    def __init__(self, url: str | None = None, *, request_timeout_s: float | None = None) -> None:
        if url is None:
            from inferops.adapters.outbound.projects.settings import settings
            url = settings.triton.url
        self._url = url
        self._timeout = float(request_timeout_s if request_timeout_s is not None else os.getenv("INFEROPS_TRITON_TIMEOUT_SECONDS", "10"))
        if not np.isfinite(self._timeout) or self._timeout <= 0:
            raise ValueError("Triton request timeout must be finite and positive")
        self._client = None
        self._infer_input_cls: Any | None = None
        self._model_name: str = ""
        self._inputs_meta: list[tuple[str, str, list[int]]] = []
        self._outputs_meta: list[str] = []
        self._triton_outputs: list[Any] = []
        self._batch_observer: Any | None = None
        self._latency_observer: Any | None = None
        self._max_batch_size = 0

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
        self._max_batch_size = int(cfg.get("config", {}).get("max_batch_size", 0))
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
            "max_batch_size": self._max_batch_size,
            "dynamic_batching": "dynamic_batching" in cfg.get("config", {}),
            "artifact_sha256": cfg.get("config", {}).get("parameters", {}).get("inferops_artifact_sha256", {}).get("string_value"),
            "request_timeout_s": self._timeout,
            "configuration_sha256": hashlib.sha256(json.dumps({"metadata": meta, "config": cfg}, sort_keys=True).encode()).hexdigest(),
            "downgraded": bool(kinds) and spec.device.preferred == "gpu"
                          and not any("GPU" in k for k in kinds),
            "reason": f"placement decided by Triton instance_group ({', '.join(kinds) or 'unspecified'})",
        }

    async def infer(self, inputs):
        if self._client is None or self._infer_input_cls is None:
            raise BackendError("triton backend not loaded")
        triton_inputs = []
        first = None
        if set(inputs) != {name for name, _, _ in self._inputs_meta}:
            raise BackendError(f"model {self._model_name}: input names do not match Triton metadata")
        for name, datatype, dims in self._inputs_meta:
            arr = np.asarray(inputs[name])
            shape = ([-1] if self._max_batch_size else []) + dims
            if len(arr.shape) != len(shape) or any(d > 0 and d != a for d, a in zip(shape, arr.shape, strict=True)):
                raise BackendError(f"model {self._model_name}: {name} shape {arr.shape} does not match {shape}")
            if self._max_batch_size and not 1 <= arr.shape[0] <= self._max_batch_size:
                raise BackendError(f"model {self._model_name}: request batch exceeds {self._max_batch_size}")
            expected = {"FP16":"float16", "FP32":"float32", "FP64":"float64", "INT64":"int64", "INT32":"int32", "UINT8":"uint8", "BOOL":"bool"}.get(datatype)
            if expected and arr.dtype != np.dtype(expected):
                raise BackendError(f"model {self._model_name}: {name} requires {expected}, received {arr.dtype}")
            if first is None:
                first = arr
            ti = self._infer_input_cls(name, list(arr.shape), datatype)
            ti.set_data_from_numpy(arr)
            triton_inputs.append(ti)
        if self._batch_observer is not None:
            self._batch_observer.observe(first.shape[0] if first is not None else 1)
        t0 = time.perf_counter()
        res = await self._client.infer(model_name=self._model_name, inputs=triton_inputs,
                                       outputs=self._triton_outputs, request_id=uuid.uuid4().hex,
                                       timeout=int(self._timeout * 1_000_000), client_timeout=self._timeout)
        if self._latency_observer is not None:
            self._latency_observer.observe(time.perf_counter() - t0)
        return {o: res.as_numpy(o) for o in self._outputs_meta}

    async def statistics(self):
        if self._client is None:
            raise BackendError("triton backend not loaded")
        response = await self._client.get_inference_statistics(model_name=self._model_name, as_json=True)
        rows = []
        for row in response.get("model_stats", []):
            inference = int(row.get("inference_count", 0))
            execution = int(row.get("execution_count", 0))
            rows.append({"name": row.get("name"), "version": row.get("version"),
                         "inference_count": inference, "execution_count": execution,
                         "average_execution_batch_size": inference / execution if execution else None,
                         "batch_stats": row.get("batch_stats", []), "inference_stats": row.get("inference_stats", {})})
        return {"model": self._model_name, "max_batch_size": self._max_batch_size,
                "dynamic_batching": self.device_info["dynamic_batching"], "versions": rows}

    async def close(self) -> None:
        if self._client is not None:
            with suppress(Exception):
                await self._client.close()
            self._client = None
