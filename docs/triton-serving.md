# Triton serving and batching

The selected current release is **26.09 / Triton 2.73.0**, pinned as `nvcr.io/nvidia/tritonserver:26.09-py3`, with the Python client constrained to `>=2.73,<2.74`. NVIDIA's [26.09 release notes](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-26-09.html) list CUDA 13.4.1, TensorRT 11.3 and ONNX Runtime 1.28.0, with compute capability 7.5 or newer. The local RTX 4060 Laptop GPU is Ada, compute capability 8.9, with driver 580.178.04. CUDA 13.x supports R580 through [minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html); newer toolkit/PTX features can still require newer drivers. This release was actually loaded and executed on that host, including dynamic batches for both models. Local ORT CUDA wheels remain a separate CUDA 12 environment; the remote server's TensorRT ABI is isolated inside its container.

## Prepare without overwriting original weights

```bash
pip install -e '.[server,triton,cuda,video]'
python scripts/prepare_triton.py --device cuda --max-batch-size 4 --output .run/triton/models
INFEROPS_EXECUTION_MODE=triton INFEROPS_TRITON_REPOSITORY=./.run/triton/models \
  docker compose --profile triton up -d --build api
# Alternatively use the native API and a Docker model server:
INFEROPS_TRITON_REPOSITORY=./.run/triton/models make run MODE=triton
```

The tool widens external batch axes in a **candidate copy**, executes batch 1/2/4 against independent original single-sample outputs, checks every output's shape and value, and only then emits a dynamic config. It does not fix internal static reshape/broadcast operators. A failure retains the original batch-one graph and reports the reason in `qualification.json`. Qualification is numerical batch separation, not model accuracy validation. Production data and threshold-sensitive downstream decisions need representative parity checks too.

Initial qualification of the original ONNX artifacts (preserved as references):

| Model | CPU qualification | GPU qualification with FP32 CUDA | Serving choice |
| --- | --- | --- | --- |
| FaceMesh | Independent batches 1/2/4 match | Independent batches 1/2/4 match with TF32 disabled | Dynamic maximum 4 in the generated repository |
| RF-DETR driver detector | Batch 2 fails an internal positional-embedding broadcast | Same structural failure | Original graph, `max_batch_size: 0`; requires genuine re-export from training source/checkpoint |

The subsequently supplied `.pth` was identified as **RF-DETR Medium, six foreground classes** and genuinely re-exported using RF-DETR 1.11.2 and CPU PyTorch 2.10.0. Both models in `.run/triton/checkpoint-models` now have maximum batch 4. The new detector passed real-video GPU batch 1/2/4 comparison, including nonempty foreground detections. Raw synthetic/background proposal parity can fail; the RF-DETR foreground qualification is explicit and records raw parity separately. See [the conversion guide](checkpoint-conversion.md) for prerequisites, comparison scope and activation.

Live Triton verification of 32 concurrent requests per model observed **32 inferences / 12 executions = 2.67 average detector batch**, and **32 / 9 = 3.56 FaceMesh batch**. It verified response isolation, foreground detections and single-response postprocessors. These model-only results are distinct from the much smaller actual camera capacity of the current CPU-heavy face localization pipeline. Under paced full-pipeline load, arrival timing can leave average batches close to one.

The first GPU FaceMesh trial with default TF32 failed strict parity. The generated CUDA execution accelerator explicitly disables TF32 and limits cuDNN workspace. This precision choice is kept consistent between GPU qualification and serving. Original reference configs remain unbatched. No original ONNX file is modified. See the [ONNX Runtime backend configuration](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/onnxruntime_backend/README.html) for accelerator parameters.

## Features used for this workload

| Feature | Configuration / reason |
| --- | --- |
| Shared model serving | One model instance per model on GPU 0; cameras do not each load models |
| Dynamic batching | Only models passing batch separation; maximum 4, 2 ms collection delay |
| Queue policy | Dynamic queue ≤128; 5-second default queue deadline, reject expired requests |
| Client admission | ≤64 total outstanding RPCs per application owner by default |
| Client deadlines | `INFEROPS_TRITON_TIMEOUT_SECONDS=10`, queue hint plus client RPC timeout |
| GPU rate limiter | Both models share one per-device `GPU_EXECUTION` resource, execution-count scheduling |
| Model control/version | Explicit startup loading, strict configuration, specific model version 1; no repository polling |
| Warmup/readiness | Batch-one zero-input warmup before readiness; startup fails on model errors |
| Memory pools | Bounded pinned and CUDA pools; one instance per model conserves 8 GB VRAM |
| Observability | Readiness, GPU metrics, inference statistics, actual execution batch sizes in dashboard |
| Stateful processing | Tracking, fusion and incident state stay isolated in ordered Python sessions |

Server batching joins independent batch-one requests and splits responses, so existing single-image postprocessors receive one sample. Passing a multi-image client request to a single-image pipeline is a separate contract and is not enabled by this work.

TensorRT FP16/INT8 engines, CUDA shared memory, response caching, extra model instances, ensembles, sequence batching and decoupled responses remain optional experiments. They are not all useful for this pipeline. FP16/INT8 need accuracy checks; extra instances consume VRAM; shared memory helps only with the correct local topology/lifetimes; dynamic camera pixels usually offer little cache reuse. Stateful sequence batching would require moving session state into the server. The old unmeasured claim of a 2–4× TensorRT speedup has been removed.

## Verify actual execution

`/v1/system/triton` and the Services view report model counters. Average executed batch size is inference count divided by execution count; inspect the interval under concurrent load rather than counting a client request's leading dimension. A dynamic config alone does not demonstrate batching. The [NVIDIA batching guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html) describes the scheduler.

```bash
python scripts/verify_triton.py --url 127.0.0.1:8001 --output .run/triton/verification.json
```

This checks live server metadata, individual versus concurrent responses, response separation, execution batch counters and local CPU reference parity. Use the full pipeline sweep from [the dashboard guide](unified-dashboard.md) to measure camera capacity, including decode, Haar fallback, evidence, SQL and previews. A model-only batch result is not camera capacity.
