# InferOps Python library

See the [architecture](architecture.md) for ports and adapters,
[lifecycle](lifecycle.md) for execution modes and resource ownership, and
[documentation index](README.md) for the complete tour.

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
The driver adapter uses SCRFD to select a face, 6DRepNet360 to estimate head
direction, 2D106 to locate the eyes and BlinkCNN to classify closure. Use
`driver_behaviour` for object and face monitoring, or `driver_face` for face-only
monitoring. Install `.[driver_face]` for the image preprocessing dependency.
The face pipeline stages own their tensor preprocessing; they use backend `infer`
rather than assuming these internal model ports have generic image preprocessors.

Use the logical port names from `models/*/model.yaml`. You may also construct a
session from a dictionary of `Model` objects, including models from different
projects. Models get separate frame metadata so different input transforms do
not interfere. Sequential execution is the default; `parallel=True` submits
independent predictions concurrently and waits for all before reporting errors.
This returns predictions keyed by model name; complete tracking/fusion/incident
workflows use the shared `Engine` pipeline services below.

## Shared pipeline engine

`Engine` runs the same application services used by HTTP, without starting a web
server. Embedded construction uses in-memory records and evidence disabled unless
you explicitly supply durable adapters. Project loading is instance-scoped: two
projects with the same external name can coexist without changing global settings
or relative-plugin resolution.

```python
import asyncio
import cv2
from inferops import Engine

async def main():
    async with Engine.from_project('./my_project', runtime='cpu') as engine:
        image = cv2.imread('frame.jpg')
        result = await engine.predict('detector', image)
        summary = await engine.run('inspection', 'clip.mp4', stream_name='camera-1')
        print(result.model_dump())
        print(summary['session_id'], summary['frames'])

asyncio.run(main())
```

Every run gets a generated session identity. `stream_name` is descriptive metadata;
reusing it does not reuse state. Files decode incrementally using media PTS. A
summary contains counts, models, events/incidents and evidence references, rather
than all decoded frames.

For repeated processing, create an explicit session and close it when the source
ends. Temporal state lives inside that session. Pass media capture timestamps in
nanoseconds and increasing sequences; `0` is a valid media timestamp.

```python
from inferops.domain.processing import MediaFrame

async def process_camera(engine, captured_frames):
    session = await engine.open_session('inspection', stream_name='camera-1')
    try:
        async for sequence, media_time_ns, image in captured_frames:
            outputs = await session.process(MediaFrame(
                image=image, media_time_ns=media_time_ns, sequence=sequence,
                timestamp_provenance='source',
            ))
            print(outputs)
    finally:
        await session.close()
```

The production persistence bundle can also be used by a Python process. It reads
the server configuration explicitly and starts owned resources on context entry:

```python
from inferops.bootstrap import build_server_engine

async def durable_run():
    engine, _notifications = build_server_engine()
    async with engine:
        return await engine.run('driver_video', 'clip.mp4')
```

This requires the `server` extra, PostgreSQL, persistent disk and the configured delivery
transport. It enables snapshot/clip capture for accepted incidents and uses the
persistent spool. The supported server deployment has one application owner; do
not start this as an additional owner against a running API's live records.

For durable development runs, use a dedicated PostgreSQL database and persistent disk.
Install the `server` extra and construct these adapters explicitly:

```python
from pathlib import Path
from inferops import Engine
from inferops.adapters.outbound.persistence.sql import SqlRepository
from inferops.adapters.outbound.storage.filesystem import FilesystemStorage

async def development_run():
    Path('.run/dev').mkdir(parents=True, exist_ok=True)
    async with Engine.from_project(
        './my_project',
        repository=SqlRepository('postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/inferops_dev'),
        storage=FilesystemStorage('.run/dev/evidence'),
        evidence=True,
        spool_path='.run/dev/spool',
    ) as engine:
        return await engine.run('inspection', 'clip.mp4')
```

The spool retains recoverable material separately from completed artifacts.
Optional `S3Storage` implements the same `ObjectStore` port; install `.[s3]` and
supply its endpoint, credentials and bucket when shared object storage is needed.
`Model`, `InferenceSession` and `Model.from_yaml()` remain supported, including
raw tensor inference and async cleanup. Relative project plugins require explicit
project loading rather than a process-global active project. See the
[lifecycle guide](lifecycle.md) for states, retries, ownership and release operation.

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

Plugin modules above are prefixed with `inferops.adapters.outbound.inference.plugins.`. Match the
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
runtime libraries must be present. Accelerator execution requires hardware
validation on your deployment machine.

Sources: [ONNX Runtime installation](https://onnxruntime.ai/docs/install/),
[TensorRT pip installation](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-pip.html),
[RF-DETR exports](https://rfdetr.roboflow.com/latest/exports/basics/).

## Structure

```text
inferops/                  reusable Python package
  public.py                Engine, Model and InferenceSession facades
  domain/                  schemas, tracking, fusion and evidence/incident policies
  application/             model pool, execution, sessions, incidents, evidence, jobs and revisions
  ports/                   external capability and extension contracts
  adapters/inbound/        HTTP/UI, CLI, media sources and workers
  adapters/outbound/       inference/plugins, stages, project loading, storage and observability
  bootstrap.py             explicit resource construction
  sdk/ testing/            remote client and reusable testing utilities
projects/<adapter>/        independent application packages
  pyproject.toml           adapter dependencies and entry point
  project.yaml             enabled pipelines and application metadata
  models/                  model contracts and custom model plugins
  configs/                 pipelines, fusion and alerts
  plugins/                 application-specific stages and actions
```

The existing `inferops.adapters` directory contains infrastructure connectors;
installed *project adapters* are managed by `inferops.adapters.outbound.projects.packages`. Import public Python handles from `inferops`; use the documented canonical
paths when authoring plugins. See [adapter authoring](adapters.md) to package your
own project or a reusable model integration.
