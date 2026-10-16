"""ONNX Runtime backend (CPU + CUDA)."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from inferops.adapters.outbound.inference.devices import resolve_onnx_providers
from inferops.adapters.outbound.observability import INFER_BATCH, INFER_LATENCY, get_logger
from inferops.adapters.outbound.projects.artifacts import resolve_artifact
from inferops.domain.errors import BackendError
from inferops.domain.models import ModelSpec
from inferops.ports.inference import InferenceBackend

log = get_logger(__name__)


class OnnxRuntimeBackend(InferenceBackend):
    def __init__(self, base_dir: Path | None = None, *, execution_mode: str | None = None,
                 respect_env: bool = True, threads: int | None = None) -> None:
        self.base_dir = base_dir
        self.execution_mode = execution_mode
        self.respect_env = respect_env
        self.threads = threads
        self._sess = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        self.device: str = "cpu"
        self.device_info: dict[str, Any] = {}

    def _resolve_model_path(self, spec: ModelSpec) -> Path:
        if not spec.artifact.path:
            raise BackendError(f"model {spec.name}: onnxruntime backend requires artifact.path")
        # Explicit project loading already resolved paths. Avoid consulting any
        # process settings for these independent engines/library models.
        artifact = Path(spec.artifact.path)
        if artifact.is_absolute():
            if not artifact.is_file():
                raise BackendError(f"model {spec.name}: file not found: {artifact}")
            return artifact
        if self.execution_mode is not None:
            artifact = (self.base_dir or Path.cwd()) / artifact
            if not artifact.is_file():
                raise BackendError(f"model {spec.name}: file not found: {artifact}")
            return artifact
        from inferops.adapters.outbound.projects.settings import get_model_repository_root
        from inferops.adapters.outbound.projects.settings import settings as _s

        found, tried = resolve_artifact(
            spec.artifact.path,
            base_dir=self.base_dir,
            model_repository=get_model_repository_root(_s),
            domain_root=_s.domain_root,
            configs_dir=_s.configs_dir,
        )
        if found is None:
            raise BackendError(
                f"model {spec.name}: file not found, tried: {[str(p) for p in tried]}"
            )
        return found

    def _provider_resolution(self, spec: ModelSpec):
        return resolve_onnx_providers(spec.device.preferred, spec.device.gpu_id,
                                      respect_env=self.respect_env)

    def _validate_providers(self, active: list[str]) -> None:
        return None

    async def load(self, spec: ModelSpec) -> None:
        try:
            import onnxruntime as ort
        except Exception as exc:
            # onnxruntime-gpu wheels raise ImportError on CPU-only hosts when
            # their CUDA shared objects are missing. Surface that as a backend
            # error naming the install fix rather than a bare ImportError.
            raise BackendError(
                f"model {spec.name}: onnxruntime is not importable ({type(exc).__name__}: {exc}). "
                "Install the CPU wheel with `pip install onnxruntime`, or install matching CUDA "
                "runtime libraries for onnxruntime-gpu."
            ) from exc

        self.spec = spec
        path = self._resolve_model_path(spec)

        # Load the optional pip-provided CUDA/cuDNN libraries before building a
        # session. A CPU-only installation does not expose this function.
        if spec.device.preferred == "gpu" and hasattr(ort, "preload_dlls"):
            ort.preload_dlls(directory="")

        resolution = self._provider_resolution(spec)
        self.device = resolution.effective_device
        providers = resolution.providers

        def _create_session(target: str, provider_list: list[Any]) -> ort.InferenceSession:
            opts = ort.SessionOptions()
            # Enable all graph optimisations (constant folding, node fusion, ...).
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # Sequential execution: nodes always run in the same order, which
            # avoids floating-point non-determinism from parallel thread
            # scheduling, so identical input always yields identical output.
            # (The previous value, ORT_PARALELL, was a typo for ORT_PARALLEL and
            # raised AttributeError on every load.)
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            # Avoid oversubscribing laptop CPUs with ORT's default thread pool.
            import os
            opts.intra_op_num_threads = (self.threads if self.threads is not None
                                        else int(os.getenv("INFEROPS_ORT_THREADS", "4")))
            return ort.InferenceSession(target, providers=provider_list, sess_options=opts)

        # Run the blocking session-creation in a thread so the event loop stays live.
        try:
            self._sess = await asyncio.to_thread(_create_session, str(path), providers)
        except Exception as exc:
            if resolution.effective_device != "gpu":
                raise BackendError(
                    f"model {spec.name}: failed to create ONNX session from {path}: {exc}"
                ) from exc
            # CUDA looked available but session creation still failed (driver
            # mismatch, OOM, unsupported op). Retry on CPU instead of taking the
            # whole pipeline down.
            log.warning("onnx_gpu_session_failed_falling_back_to_cpu",
                        model=spec.name, err=str(exc))
            self.device = "cpu"
            try:
                self._sess = await asyncio.to_thread(
                    _create_session, str(path), ["CPUExecutionProvider"]
                )
            except Exception as cpu_exc:
                raise BackendError(
                    f"model {spec.name}: failed to create ONNX session from {path} "
                    f"on GPU ({exc}) and on CPU ({cpu_exc})"
                ) from cpu_exc

        self._input_names = [i.name for i in self._sess.get_inputs()]
        self._output_names = [o.name for o in self._sess.get_outputs()]
        active = self._sess.get_providers()
        try:
            self._validate_providers(active)
        except Exception:
            await self.close()
            raise
        gpu_active = any("CUDA" in ep or "TensorRT" in ep for ep in active)
        mode = self.execution_mode
        if mode is None:
            from inferops.adapters.outbound.projects.settings import settings
            mode = settings.execution_mode
        if mode == "cuda" and not gpu_active:
            self._sess = None
            raise BackendError(f"model {spec.name}: CUDA mode requires an active CUDA execution provider")
        # get_providers() is authoritative: trust it over our own prediction.
        self.device = "gpu" if gpu_active else "cpu"
        self.device_info = {
            "requested": resolution.requested_device,
            "effective": self.device,
            "gpu_id": resolution.gpu_id,
            "downgraded": resolution.requested_device == "gpu" and self.device != "gpu",
            "reason": resolution.reason,
            "active_providers": list(active),
        }
        emit = log.warning if self.device_info["downgraded"] else log.info
        emit("onnx_session_loaded", model=spec.name, path=str(path),
             requested_device=resolution.requested_device, effective_device=self.device,
             gpu_id=resolution.gpu_id, downgraded=self.device_info["downgraded"],
             reason=resolution.reason, active_providers=active)

    async def infer(self, inputs):
        if self._sess is None:
            raise BackendError("backend not loaded")
        feeds = {n: inputs[n] for n in self._input_names if n in inputs}
        if not feeds:
            raise BackendError(
                f"model {self.spec.name}: none of the model inputs {self._input_names} "
                f"were supplied (got: {sorted(inputs)})"
            )
        missing = [n for n in self._input_names if n not in feeds]
        if missing:
            raise BackendError(
                f"model {self.spec.name}: missing required input(s) {missing}; "
                f"got: {sorted(inputs)}"
            )
        first = next(iter(feeds.values()))
        INFER_BATCH.labels(model=self.spec.name).observe(first.shape[0] if first.ndim > 0 else 1)
        t0 = time.perf_counter()
        outs = await asyncio.to_thread(self._sess.run, self._output_names, feeds)
        INFER_LATENCY.labels(model=self.spec.name).observe(time.perf_counter() - t0)
        return dict(zip(self._output_names, outs, strict=True))

    async def close(self) -> None:
        # Drop the session and let ORT release its device memory. Without an
        # explicit del the arena can outlive a hot reload and leak GPU memory.
        sess, self._sess = self._sess, None
        if sess is not None:
            del sess
        self._input_names = []
        self._output_names = []
