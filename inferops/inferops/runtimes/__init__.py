from pathlib import Path

from inferops.core.errors import BackendError
from inferops.registry.model_spec import ModelSpec
from inferops.runtimes.base import InferenceBackend


def build_backend(spec: ModelSpec, base_dir: Path | None = None) -> InferenceBackend:
    if spec.backend == "onnxruntime":
        from inferops.runtimes.onnxruntime_backend import OnnxRuntimeBackend
        return OnnxRuntimeBackend(base_dir=base_dir)
    if spec.backend == "triton":
        from inferops.runtimes.triton_backend import TritonBackend
        return TritonBackend()
    if spec.backend == "tensorrt":
        from inferops.runtimes.tensorrt_backend import TensorRTBackend
        return TensorRTBackend()
    if spec.backend == "synthetic":
        from inferops.runtimes.synthetic_backend import SyntheticDetectorBackend
        return SyntheticDetectorBackend(base_dir=base_dir)
    if spec.backend == "synthetic_face_mesh":
        from inferops.runtimes.synthetic_face_mesh_backend import SyntheticFaceMeshBackend
        return SyntheticFaceMeshBackend(base_dir=base_dir)
    raise BackendError(f"unknown backend: {spec.backend}")


__all__ = ["InferenceBackend", "build_backend"]
