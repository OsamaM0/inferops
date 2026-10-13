# Project configuration

One InferOps instance runs one selected project folder. The engine image does
not contain a project's models or behavior, so project edits need no image rebuild.
Shipped projects live in `projects/`. The Python engine source lives in
`inferops/`; Dockerfiles, runtime overlays and monitoring configuration
live in `deploy/`. There are no shipped `domains/` or `examples/` folders.

```text
project/
  project.yaml                   Identity, enabled pipelines, prewarm, platform overrides
  models/<name>/
    model.yaml                   Shared contract, artifacts, sources, runtime variants
    adapter.py                   Model-specific preprocessing/postprocessing or stage
  configs/pipelines/*.yaml       Stage composition, tracking and evidence
  configs/fusion/*.yaml          Temporal signals and rules
  configs/alerts/*.yaml          Severity, cooldown, dedupe and actions
  plugins/                      Project-specific stages, actions and evidence overlays
  model_repository/             Persistent weights and optional Triton configuration
  tests/                        Project tests
```

## Project entry point

```yaml
name: driver_behaviour
label: Driver Behaviour
pipelines: [driver_behaviour, driver_video]
prewarm_pipelines: [driver_behaviour]
# Optional: explicit additional models to prepare/prewarm.
prewarm_models: []
# Optional overrides merged over engine and project configs/platform.yaml.
platform:
  obs: {log_level: INFO}
```

`pipelines: []` explicitly enables no inference pipelines and opens an empty
dashboard. Omitting `pipelines` preserves legacy discovery of all pipeline files.
Enabled pipelines determine which models must be prepared; unused model files
need no downloads. Explicit prewarm models are also prepared.

## Model configuration and sources

Keep one model.yaml beside each model's adapter. Its `defaults` contain the
shared tensor, label and decoder contract; named `variants` override it.
Dictionaries merge recursively and lists replace whole lists.

```yaml
name: detector
defaults:
  task: detection
  inputs: [{name: images, shape: [1, 3, 640, 640], dtype: float32}]
  outputs: [{name: output0, shape: [1, 6, 8400], layout: ND, color: NONE}]
  labels: [person, helmet]
  artifact:
    path: model_repository/detector/1/model.onnx
    source:
      kind: huggingface
      repo_id: owner/model
      filename: onnx/model.onnx
      revision: main             # A commit hash can pin an exact revision.
  preprocess: {plugin: .models.detector.adapter:Preprocess}
  postprocess: {plugin: .models.detector.adapter:Postprocess, params: {conf_threshold: 0.25}}
variants:
  onnx: {backend: onnxruntime}
  triton: {backend: triton, artifact: {triton_model_name: detector}}
```

The example source is a placeholder: choose an exported ONNX file and adjust
its tensor/decoder contract before enabling the pipeline. Downloads do not
train, convert, or guess an arbitrary architecture.

Other `artifact.source` examples:

```yaml
source: {kind: google_drive, file_id: YOUR_PUBLIC_FILE_ID}
source: {kind: url, url: https://example.org/model.onnx}
source:
  kind: url
  url: https://example.org/models.tar.gz
  archive_member: weights/model.onnx
source: {kind: installer, installer: scripts/my_download.py}
```

For Hugging Face private/gated files, provide `HF_TOKEN` in `.env`; accept any
repository access requirements through your Hugging Face account. `artifact.sha256`
can verify an exact checksum. Local weights work without `source`.
Installer scripts accept `--output <temporary-file>`; a `module:function`
installer receives the temporary target Path. Legacy manifest asset scripts
retain their original interface.

Relative artifact paths resolve beside model.yaml first. Shared paths beginning
with `model_repository/` resolve inside the project's repository. Both ONNX and
Triton variants can inherit the same artifact/source without duplication.
Labels from files resolve beside the owning model.yaml.

Preparation uses temporary downloads, graph/tensor validation, file locking,
and atomic replacement. Valid existing weights are reused offline; source
changes or `prepare --force` refresh the artifact. A failed download preserves
the previous artifact and stops preparation. No synthetic substitute is supplied.

## Pipeline model selection

```yaml
- name: detect
  stage: infer
  params:
    model: detector
```

Logical names select ONNX for CPU/CUDA and Triton for Triton/DeepStream. `auto`
is a CPU alias. Tracking filters also accept logical names. Custom stages using
`params.model` receive the same resolved model name.

For advanced control, reference an explicit variant such as `detector_onnx`,
or retain per-stage `model_modes: {cpu: special_cpu, triton: special_remote}`.
When a logical port has no unambiguous implementation for a mode, preparation
fails and asks for an explicit variant. Existing variant `name` overrides remain
supported. Legacy `fallback_model` fields do not enable fallback in any mode.

Triton preparation generates a basic non-batched tensor configuration when
`config.pbtxt` is absent. Supply that file yourself for batching, instance counts,
TensorRT optimization, response caches, or other tuning; it is never overwritten.
Triton weights must reside at `<repository>/<triton_model_name>/<version>/model.onnx`.

## Docker and local commands

Copy `.env.example` once, set the project folder and execution mode, then run
`docker compose up -d --build` from the project root. Absolute external folders
work too. The engine container mounts that folder at `/project`; use relative
plugin references such as `.models.detector.adapter:Postprocess` and
`.plugins.my_action:MyAction` so relocation keeps working.

```bash
make new-project NAME=warehouse
make new-model PROJECT=warehouse NAME=detector LABELS=person,helmet
make validate PROJECT=warehouse
make setup-models PROJECT=warehouse
make preflight PROJECT=warehouse MODE=cpu
make run PROJECT=warehouse MODE=cpu
make run-container PROJECT_ROOT=/path/to/warehouse MODE=cpu
```

`inferops project prepare --domain-root /path/to/project` validates configuration
and prepares selected weights. `setup-assets` remains an alias. Project settings
and model artifact/tensor edits in the dashboard are saved for preparation/restart;
thresholds, fusion, alerts and ordinary pipeline settings support live reload.

## Extension and compatibility contracts

| Extension | Interface | YAML setting |
| --- | --- | --- |
| Input preparation | `Preprocessor.__call__(frame, spec)` | `preprocess.plugin` |
| Output decoding | `Postprocessor.__call__(raw, frame, spec)` | `postprocess.plugin` |
| Custom runtime | `InferenceBackend.load/infer/close` | `backend: module:Class` |
| Custom stage | `Stage.setup/process/teardown` | `stage: module:Class` |
| Evidence overlay | `FrameAnnotator` | sink_s3 `annotator` |
| Alert action | `AlertSink.emit(alert)` | sink `kind: plugin`, `params.target` |

Built-in alert sinks remain stdout, ws, postgres, kafka and webhook. All existing
tracking, rules, evidence and device controls remain configurable.

`project.yaml` takes precedence over `domain.yaml`. Existing projects under
`examples/` or `domains/`, flat `configs/models/*.yaml`, DOMAIN commands and the
INFEROPS_ACTIVE_DOMAIN/INFEROPS_DOMAIN_ROOT variables remain supported. The legacy
project `configs/platform.yaml` layer remains supported; `project.yaml.platform`
overrides it and environment variables take precedence.
