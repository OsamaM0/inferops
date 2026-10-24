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
dashboard. Omitting `pipelines` discovers all pipeline files; this is the default
for new projects, so adding a pipeline does not require a second manifest edit.
Enabled pipelines determine which models must be prepared; unused model files
need no downloads. Explicit prewarm models are also prepared.

## One configuration workspace

Open **Configuration** in `/ui/` to manage the active project. Settings are grouped
by purpose: Project, Models, Pipelines, Signals & rules, Alerts & actions, Platform.
The forms come from the same schemas used by InferOps. Registered workspace
projects save overrides into the central `configs/inferops.yaml` document. They support nested fields, ordered stages, model choices, runtime
overrides, rule expressions and custom plugin parameters. YAML remains available
under **Advanced YAML**; switching editors with unsaved changes asks before discarding them.

Use **Add configuration** to draft a model, pipeline, ruleset or alert policy.
**Validate** checks the candidate against the entire project without writing it.
**Save changes** persists it and can apply live settings to new sessions. Existing
sessions keep their compiled revision. Artifact, project and platform changes
report preparation/restart requirements; a reload cannot bypass pending startup
changes. Concurrent changes to the same document return a conflict instead of
silently overwriting it.

To build a pipeline through the forms, choose **Add configuration → Pipeline**.
Under **Stages**, choose a step such as **Run a model**, then click **Add processing
step**. Model and policy steps offer the project's configured names. Required
parameters and safe defaults appear with the selected step; optional parameters
can be added with **Add field**. **Move up** changes the step order while keeping
its settings. Under **Sinks**, use **Add output** for live results, webhooks,
evidence or other supported outputs. Choose **Custom project stage** for a Python
plugin reference and its own parameters. The stage registry supplies these
choices without importing or executing project plugins.

Forms preserve fields they do not recognize and literal `${ENV}` references.
Form saves normalize YAML formatting and remove comments; use Advanced YAML to
retain comments and formatting. The UI does not create a separate settings database.

Workspace configuration is centralized in [`configs/inferops.yaml`](../configs/inferops.yaml).
Packaged project YAMLs provide baseline contracts; `projects.<key>.parameters`
and `projects.<key>.configuration` customize them. Dashboard edits write only
to the central file. `.env` holds secrets and essential host overrides. See
[central configuration](configuration.md) for precedence, examples and migration.
Standalone adapters without a central workspace keep legacy platform/project
loading for compatibility. Library runtime options remain explicit.

Add `tags.label` and `tags.description` to pipelines to explain their purpose in
the UI. A pipeline refers to logical model names such as `detector`; a model's
runtime variants are implementations of that model, not separate workflows.
New projects and custom plugins use this same editor without changes to InferOps.

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

Logical names select ONNX for CPU/CUDA and Triton for remote serving. `auto`
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

Select the project folder and execution mode in `configs/inferops.yaml`, keep secrets in `.env`, then run
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
