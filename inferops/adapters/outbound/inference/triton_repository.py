"""Build a separate Triton repository, admitting batching only after parity.

Widening external ONNX axes is a candidate, never evidence of batch support.
The original model is kept whenever a graph cannot execute independent samples
as a batch. Stateful tracking and incident logic remain in Python sessions.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def qualify(original, candidate, *, batches=(1, 2, 4), providers=None, detector=False, sample_video=None):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    providers = providers or ["CPUExecutionProvider"]
    reference = ort.InferenceSession(str(original), sess_options=options, providers=providers)
    tested = ort.InferenceSession(str(candidate), sess_options=options, providers=providers)
    if "CUDAExecutionProvider" in str(providers) and not all(
        "CUDAExecutionProvider" in s.get_providers() for s in (reference, tested)
    ):
        raise ValueError("CUDA qualification cannot use CPU fallback")
    rng = np.random.default_rng(20261019)
    checks = []
    samples = []
    if detector:
        import cv2
        if not sample_video:
            raise ValueError("RF-DETR qualification requires representative sample video")
        capture = cv2.VideoCapture(str(sample_video))
        try:
            for index in range(max(batches)):
                capture.set(cv2.CAP_PROP_POS_MSEC, index * 1000)
                ok, frame = capture.read()
                if not ok:
                    raise ValueError("Representative video has too few decodable samples")
                samples.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
    for batch in sorted(set(batches)):
        feeds = {}
        for item in reference.get_inputs():
            if item.type != "tensor(float)" or any(not isinstance(d, int) or d <= 0 for d in item.shape[1:]):
                raise ValueError("Qualification requires FP32 inputs with fixed spatial dimensions")
            if isinstance(item.shape[0], int) and item.shape[0] != 1:
                raise ValueError("Reference must accept individual samples")
            # Different images in every slot detect accidental broadcasting or
            # reuse of sample zero. Three input distributions exercise the graph.
            feeds[item.name] = rng.uniform(-1, 1, (batch, *item.shape[1:])).astype(np.float32)
            feeds[item.name][0] = 0
            if batch > 1:
                feeds[item.name][-1] = rng.uniform(0, 1, feeds[item.name][-1].shape)
            if samples:
                height, width = item.shape[-2:]
                feeds[item.name] = np.stack([cv2.resize(s, (width, height)).transpose(2, 0, 1)
                                            for s in samples[:batch]]).astype(np.float32) / 255
        actual = tested.run(None, feeds)
        raw_parity = True
        foreground_counts = []
        for slot in range(batch):
            expected = reference.run(None, {key: value[slot : slot + 1] for key, value in feeds.items()})
            for name, result, single in zip(
                (o.name for o in tested.get_outputs()), actual, expected, strict=True
            ):
                if result.shape != (batch, *single.shape[1:]):
                    raise ValueError(f"{name}: output shape {result.shape} is not batch-separable")
                if detector:
                    if not np.isfinite(result).all():
                        raise ValueError("Nonfinite detector output")
                    raw_parity &= bool(np.allclose(result[slot:slot + 1], single, rtol=1e-3, atol=1e-4))
                else:
                    np.testing.assert_allclose(result[slot:slot + 1], single, rtol=1e-3, atol=1e-4,
                                               err_msg=f"{name}, batch {batch}, sample {slot}")
            if detector:
                from inferops.adapters.outbound.inference.checkpoint_export import compare_detections
                foreground_counts.append(compare_detections([x[slot:slot + 1] for x in actual], expected,
                                                             classes=expected[1].shape[-1] - 1))
        checks.append({"batch": batch, "independent_samples_match": True,
                       **({"raw_tensor_parity": raw_parity, "foreground_counts": foreground_counts} if detector else {})})
    if detector and not any(sum(check["foreground_counts"]) for check in checks):
        raise ValueError("Representative input produced no foreground detections; cannot qualify detector")
    return {"providers": tested.get_providers(), "checks": checks, "rtol": 1e-3, "atol": 1e-4,
            "output_shapes": {item.name: [1, *value.shape[1:]]
                              for item, value in zip(tested.get_outputs(), actual, strict=True)},
            "comparison": "RF-DETR foreground detections at 0.35; raw parity reported separately" if detector else "raw tensors"}


def configuration(model, name, *, batch_size, queue_delay_us=2000, queue_size=128, instance_count=1,
                  artifact_sha256="", output_shapes=None):
    import onnx

    def tensors(values):
        rows = []
        for value in values:
            if value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
                raise ValueError("Repository generator currently supports FP32 model I/O")
            shape = [d.dim_value if d.HasField("dim_value") else -1 for d in value.type.tensor_type.shape.dim]
            shape = (output_shapes or {}).get(value.name, shape)
            if batch_size:
                shape = shape[1:]
            rows.append(f'{{ name: "{value.name}" data_type: TYPE_FP32 dims: {shape} }}')
        return ",\n".join(rows)

    scheduling = ""
    if batch_size:
        scheduling = f"""dynamic_batching {{
  max_queue_delay_microseconds: {queue_delay_us}
  default_queue_policy {{ timeout_action: REJECT default_timeout_microseconds: 5000000 max_queue_size: {queue_size} }}
}}
"""
    return f'''# Generated after executing batch parity checks. See ../qualification.json.
name: "{name}"
platform: "onnxruntime_onnx"
max_batch_size: {batch_size}
version_policy {{ specific {{ versions: [1] }} }}
input [ {tensors(model.graph.input)} ]
output [ {tensors(model.graph.output)} ]
instance_group [ {{ kind: KIND_GPU count: {instance_count} gpus: [0]
  rate_limiter {{ resources [ {{ name: "GPU_EXECUTION" count: 1 }} ] }}
}} ]
{scheduling}
optimization {{ execution_accelerators {{ gpu_execution_accelerator [ {{ name: "cuda"
  parameters {{ key: "use_tf32" value: "0" }}
  parameters {{ key: "arena_extend_strategy" value: "kSameAsRequested" }}
  parameters {{ key: "cudnn_conv_use_max_workspace" value: "0" }}
}} ] }} }}
parameters {{ key: "intra_op_thread_count" value: {{ string_value: "2" }} }}
parameters {{ key: "inter_op_thread_count" value: {{ string_value: "1" }} }}
parameters {{ key: "inferops_artifact_sha256" value: {{ string_value: "{artifact_sha256}" }} }}
model_warmup [ {{ name: "batch_one" batch_size: {1 if batch_size else 0}
  inputs {{ key: "{model.graph.input[0].name}" value {{ data_type: TYPE_FP32
    dims: {[d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim][1:] if batch_size else [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]} zero_data: true }} }}
}} ]
'''


def prepare_repository(source, destination, *, max_batch_size=4, providers=None, queue_delay_us=2000,
                       detector_models=(), sample_video=None):
    import onnx

    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Use a separate output repository; original weights must stay unchanged")
    if max_batch_size not in {2, 4, 8}:
        raise ValueError("Start with a bounded maximum batch size of 2, 4 or 8")
    if queue_delay_us < 0 or queue_delay_us > 100000:
        raise ValueError("Queue delay must be between 0 and 100000 microseconds")
    destination.mkdir(parents=True, exist_ok=True)
    report = {"source": str(source), "models": [], "server_image": "nvcr.io/nvidia/tritonserver:26.09-py3"}
    for path in sorted(source.glob("*/1/model.onnx")):
        name = path.parent.parent.name
        original = onnx.load(path)
        candidate = onnx.ModelProto()
        candidate.CopyFrom(original)
        for value in [*candidate.graph.input, *candidate.graph.output]:
            shape = value.type.tensor_type.shape.dim
            if not shape or (shape[0].HasField("dim_value") and shape[0].dim_value != 1):
                raise ValueError(f"{name}: expected a graph accepting individual samples")
            shape[0].dim_param = "batch"
        # Stale intermediate shape annotations must not assert batch=1.
        del candidate.graph.value_info[:]
        result = {"model": name, "original_sha256": digest(path), "max_batch_size": 0}
        with tempfile.TemporaryDirectory(prefix="inferops-onnx-") as temporary:
            trial = Path(temporary) / "model.onnx"
            onnx.save(candidate, trial)
            try:
                onnx.checker.check_model(candidate)
                result["qualification"] = qualify(
                    path, trial, batches=(1, 2, max_batch_size), providers=providers,
                    detector=name in detector_models, sample_video=sample_video,
                )
                result["qualification"]["provider_options"] = providers
                result["max_batch_size"] = max_batch_size
                selected, graph = trial, candidate
            except Exception as error:
                result["batching_disabled_reason"] = f"{type(error).__name__}: {str(error)[:1200]}"
                # A failed dynamic source is bounded to individual requests too.
                graph = onnx.ModelProto()
                graph.CopyFrom(original)
                for value in [*graph.graph.input, *graph.graph.output]:
                    value.type.tensor_type.shape.dim[0].dim_value = 1
                if graph == original:
                    selected = path
                else:
                    selected = Path(temporary) / "fallback.onnx"
                    onnx.save(graph, selected)
            target = destination / name / "1"
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(selected, target / "model.onnx")
            result["serving_sha256"] = digest(target / "model.onnx")
            (target.parent / "config.pbtxt").write_text(
                configuration(graph, name, batch_size=result["max_batch_size"], queue_delay_us=queue_delay_us,
                              artifact_sha256=result["serving_sha256"],
                              output_shapes=result.get("qualification", {}).get("output_shapes"))
            )
        report["models"].append(result)
    if not report["models"]:
        raise ValueError("No ONNX models found")
    (destination / "qualification.json").write_text(json.dumps(report, indent=2))
    return report
