# InferOps Python library

InferOps separates inference mechanics from application logic. The core owns model
contracts, runtime backends, image transforms, pipelines, tracking and fusion.
A project adapter owns its models, labels, domain plugins, pipelines and rules.
Driver behaviour is distributed as `inferops-driver-behaviour`, independently of
`inferops`; it is never bundled into the core wheel.

## Installation

Once the packages are published to your package index:

```bash
pip install 'inferops[cpu]'
pip install 'inferops[cuda]'
pip install 'inferops[triton]'
pip install 'inferops[tensorrt]'
pip install 'inferops[deepstream]'
pip install inferops-driver-behaviour
```

Choose one local runtime environment. `cpu` and `cuda` install different wheels
that share the same `onnxruntime` Python module. Do not combine those extras.
The CUDA wheel already includes the CPU execution provider. To switch:

```bash
pip uninstall -y onnxruntime onnxruntime-gpu
pip install 'inferops[cuda]'
```

Extras use lowercase names (`triton` refers to NVIDIA Triton Inference Server).
`gpu` remains an alias for `cuda`. Bare `inferops` installs shared library code;
a model runtime is optional. Add `server` for the API, workers and storage clients,
`artifacts` for model preparation/downloads, `video` for PyAV video support, and
`dev` for development tools. For example, `inferops[server,cpu,dev]`.

For this checkout, from the repository root (the directory containing `pyproject.toml`):

```bash
pip install -e '.[cpu]'
pip install -e projects/driver_behaviour
inferops adapters list
inferops adapters init driver_behaviour ./driver_workspace
```

`adapters init` copies configuration and Python plugins to a writable project.
It does not copy weights or overwrite existing destinations. Download weights
into that workspace, rather than modifying installed packages:

```bash
pip install -e '.[cpu,artifacts]'
inferops project prepare --domain-root ./driver_workspace
inferops validate --domain-root ./driver_workspace
```

The existing Compose and Make workflows remain available; setup installs the
`server` extra and the selected CPU/CUDA extra. Packages have not been uploaded
to PyPI by this refactor. Build distributions locally with `python -m build`, and
build the adapter from `projects/driver_behaviour` separately.

## Run a model in Python

The local API needs no web server, Redis, Kafka, PostgreSQL or object store.
It is asynchronous; call `asyncio.run` at your application's entry point, or
`await` directly inside an existing async application.

```python
import asyncio
import cv2
from inferops import Model

async def main():
    async with Model.from_yaml('my_model.yaml') as model:
        result = await model.predict(cv2.imread('frame.jpg'))
        print(result.model_dump())

asyncio.run(main())
```

`Model.from_yaml` loads a single flat `ModelSpec`. For an existing logical port
with several variants, use `InferenceSession.from_project`. `Model(ModelSpec(...))`
also accepts a programmatic spec. `Model.infer({'tensor_name': numpy_array})`
runs raw tensors, bypassing image preprocessing/decoding. Backend loading is lazy,
and async context managers release resources on exit. Each model instance should
be used within one event loop. Close sessions after in-flight predictions finish.

Images are `uint8` HWC **BGR**, as returned by OpenCV. Image preprocessors determine
normalization, resizing and channel order. Results contain model names, latency,
and normalized `xyxy` detections, classifications, or raw output tensors.

## Run several models together

```python
import asyncio
import cv2
from inferops import InferenceSession

async def main():
    async with InferenceSession.from_project(
        './my_project', models=['detector', 'classifier'], runtime='cpu'
    ) as session:
        results = await session.predict(cv2.imread('frame.jpg'), parallel=True)
        for name, result in results.items():
            print(name, result.model_dump())

asyncio.run(main())
```

This example assumes your project defines image preprocessors for both models.
The driver adapter's face-mesh model depends on an RF-DETR face crop and is run
by its custom `FaceMeshStage`; use its existing `driver_behaviour` pipeline for
the complete crop/landmarks/fusion workflow, rather than calling `predict` on the
face-mesh model directly.

Use the logical port names from `models/*/model.yaml`. You may also construct a
session from a dictionary of `Model` objects, including models from different
projects. Models get separate frame metadata so different input transforms do
not interfere. Sequential execution is the default; `parallel=True` submits
independent predictions concurrently and waits for all before reporting errors.
This returns predictions keyed by model name; domain fusion belongs in the
existing pipeline stages and rules.

A pipeline can chain multiple `infer` stages on a frame, then `track`, `fuse` and
`alert`. Each infer stage appends its result, allowing fusion rules to combine
signals from all preceding models. See [project blocks](project-blocks.md) and
[API guide](api.md) for the service workflows.

## Model formats and runtimes

| Model contract | Preprocessing | Decoder |
| --- | --- | --- |
| YOLO ONNX, raw pre-NMS xywh + class scores | `image_letterbox:Letterbox` | `yolo_decode:YoloDecode` |
| DETR ONNX with softmax logits | `resize_normalize:ResizeNormalize` | `detr_decode:DetrDecode` |
| RF-DETR ONNX with sigmoid logits | `resize_normalize:ResizeNormalize`, `normalize: true` | `rfdetr_decode:RFDetrDecode` |
| Any ONNX tensor contract | Custom plugin or raw feeds | Custom plugin or raw results |

Plugin modules above are prefixed with `inferops.plugins.builtin.`. Match the
export's actual tensor names, input size, label order, box format and background
slot. RF-DETR defaults to outputs `dets` and `labels`; override `boxes_out` and
`logits_out` for other exports. Set `background_index` explicitly if your export
includes a background column. Generic DETR supports `activation` and `box_format`
parameters. YOLO supports explicit `layout` (`channels_first` or `channels_last`),
`objectness`, and optional `class_agnostic` NMS. Built-in decoders accept one image
per call. Segmentation masks and end-to-end YOLO exports with embedded NMS need a
custom postprocessor.

Training frameworks are not core dependencies: export YOLO/RF-DETR models to ONNX
using their own tools, then describe that tensor contract in a model spec.
Generate a starting model adapter with:

```bash
inferops models init detector --domain-root ./my_project --format rfdetr --labels person,car
```

`backend: onnxruntime` runs ONNX on CPU/CUDA. `backend: tensorrt` runs an ONNX graph
through ONNX Runtime's TensorRT execution provider with CUDA/CPU fallback for
unsupported graph nodes; failure to initialize TensorRT raises an error. It does
not load serialized `.plan` files directly. Use `backend: triton` to serve those
engines through a separately running Triton server. A logical model port can add
a `tensorrt` variant and select it with `runtime='tensorrt'` in the Python API.

The TensorRT extra targets CUDA 12 / TensorRT 10. Matching NVIDIA drivers and
runtime libraries must be present. DeepStream uses the existing video execution
path and Triton model serving; its extra installs Python clients/PyAV. Install the
DeepStream SDK, GStreamer plugins and matching Python bindings through NVIDIA's
system/container setup. Installing that extra alone does not provide the SDK or
a standalone `deepstream` model backend. Accelerator execution requires hardware
validation on your deployment machine.

Sources: [ONNX Runtime installation](https://onnxruntime.ai/docs/install/),
[TensorRT pip installation](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-pip.html),
[RF-DETR exports](https://rfdetr.roboflow.com/latest/exports/basics/).

## Structure

```text
inferops/                  reusable Python package
  library.py               Model + InferenceSession public API
  project_adapters.py      installed adapter discovery/workspace copies
  registry/                model contracts and runtime variants
  runtimes/                execution backends
  plugins/builtin/         reusable preprocessing and decoding
  pipeline/ fusion/ alerts/ reusable orchestration
  api/ workers/ adapters/  optional service layer and infrastructure connectors
projects/<adapter>/        independent application packages
  pyproject.toml           adapter dependencies and entry point
  project.yaml             enabled pipelines and application metadata
  models/                  model contracts and custom model plugins
  configs/                 pipelines, fusion and alerts
  plugins/                 application-specific stages and actions
```

The existing `inferops.adapters` directory contains infrastructure connectors;
installed *project adapters* are managed by `inferops.project_adapters`. This keeps
existing imports compatible. See [adapter authoring](adapters.md) to package your
own project or a reusable model integration.
