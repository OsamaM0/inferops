# Project configuration

One InferOps instance runs one selected project folder. The engine image does
not contain a project's models or behavior, so project edits need no image rebuild.
Shipped projects live in `projects/`. The Python engine source lives in
`inferops/`; Dockerfiles, runtime overlays and monitoring configuration
live in `deploy/`. There are no shipped `domains/` or `examples/` folders.

```text
models/                          Shared model library, usable by any project
  <name>/
    model.yaml                   Contract, artifacts, variants; its stage, signals and settings
    adapter.py                   Model-specific preprocessing/postprocessing or stage
  model_repository/              Persistent weights and optional Triton configuration
projects/<project>/
  project.yaml                   Identity, selected models, tracking, sinks, settings
  rules.yaml                     Signals, rules, states and alert actions
  plugins/                       Project-specific stages, actions and evidence overlays
  tests/                         Project tests
```

## One pipeline from any model combination

A project selects models from the library; InferOps generates its one pipeline:
decode → each model's stage → fuse → alert → sinks. Change `models` to run any
combination, for example only the detector:

```yaml
name: driver_behaviour
label: Driver Behaviour
models: [rfdetr_driver, face_detector, face_mesh, face_blendshapes]
tracking: {enabled: true}
sinks:
  - {name: evidence, stage: evidence, params: {annotator: .plugins.frame_annotator:DriverAnnotator}}
prewarm_pipelines: [driver_behaviour]
```

A model's `model.yaml` says how it takes part:

```yaml
signals:                        # fusion signals it produces (a list reads payload keys)
  phone_score: "detected('phone')"
stage:                          # optional; without it the model runs as a plain `infer` stage
  plugin: .models.face_mesh.adapter:FaceMeshStage
  uses: {detector_model: face_detector}   # passed only when also selected
  after: [rfdetr_driver]                  # runs after these, when selected
  params: {face_confidence_threshold: "${settings.face_confidence}"}
settings:                       # tunables it adds to the project's Project settings
  face_confidence: {default: 0.8, minimum: 0, maximum: 1}
```

A model listed in another selected model's `uses` runs inside that stage, not on
its own. In `rules.yaml`, a rule that needs a signal from an unselected model is
switched off and listed under **Project settings → Models**; states see that
signal as absent. `rules.yaml` holds fusion fields, extra `signals` no model
produces, and an `alerts:` section. The workspace can override the choice with
`projects.<key>.models` in `configs/inferops.yaml`.

The library is `models/` beside `configs/` (Docker mounts it at `/app/models`),
or `INFEROPS_MODELS_DIR`. A project's own `models/` folder takes precedence, and
`inferops adapters init` copies the selected models into it. Projects without
`rules.yaml` keep explicit `configs/pipelines`, `configs/fusion` and
`configs/alerts` files.

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
Packaged project YAMLs provide baseline contracts; `projects.<key>.settings`
and `projects.<key>.overrides` customize them. Dashboard edits write only
to the central file. `.env` holds secrets and essential host overrides. See
[central configuration](configuration.md) for precedence, examples and migration.
Standalone adapters without a central workspace keep legacy platform/project
loading for compatibility. Library runtime options remain explicit.

Add `tags.label` and `tags.description` to pipelines to explain their purpose in
the UI. A pipeline refers to logical model names such as `detector`; a model's
runtime variants are implementations of that model, not separate workflows.
New projects and custom plugins use this same editor without changes to InferOps.

## Project settings

Declare the values operators should tune in `project.yaml`, then refer to them
from any packaged YAML as `${settings.<name>}`. A value that is only a reference
keeps the setting's type.

```yaml
settings:
  confidence:
    default: 0.5
    label: Detection confidence
    description: Minimum detector score.
    minimum: 0.0
    maximum: 1.0           # also: choices: [a, b]
```

## Signals, rules and alerts

Fusion `defaults` remove repetition: `rule` and `transition` values apply to every
rule/transition, `window` gives every signal a window, and `extract` is the
expression for signals without one (`{name}` is the signal name). `signals` may
be omitted when `extract` lists them all. A transition's `from_state` may be a
list. `derived` expressions run in order, so later ones can reuse named
conditions. Extract expressions can call `detected('<label>', default=0.0)`, the
best detector score for a class, and `signal('<payload key>', default=0.0)`.

```yaml
defaults:
  window: {kind: rolling_time, duration_ms: 5000}
  rule: {confirm_ms: "${settings.confirm_ms}", min_samples: 3, cooldown_ms: 30000}
extract:
  phone_score: "detected('phone')"
derived:
  phone: "phone_score.latest() or 0.0"
  unsafe: "phone >= 0.4"
rules:
  - {name: phone_use, when: "phone >= 0.5", severity: medium}
state_machine:
  initial: NORMAL
  states: [NORMAL, UNSAFE]
  transitions:
    - {from_state: [NORMAL], to_state: UNSAFE, when: unsafe}
```

A rule emits an event typed by its name. In alerts, severity levels default to
low/medium/high/critical, a rule's severity is used directly, and
`rated_only: true` makes only events with a severity (from the rule or
`severity.mapping`) become incidents, so new rules need no alert edits.

## Model configuration and sources

Keep one model.yaml beside each model's adapter. Top-level fields are the shared
tensor, label and decoder contract. `variants` lists the runtimes the model
supports: `onnx`, `tensorrt` and `triton` get their backend automatically, and
Triton's model name comes from `model_repository/<name>/...` in the artifact
path. Use a mapping instead of a list to override fields for one variant;
dictionaries merge recursively and lists replace whole lists. Outputs default
to plain tensors (`layout: ND`, `color: NONE`).

```yaml
name: detector
task: detection
inputs: [{name: images, shape: [1, 3, 640, 640]}]
outputs: [{name: output0, shape: [1, 6, 8400]}]
labels: [person, helmet]
artifact:
  path: model_repository/detector/1/model.onnx
  source:
    kind: huggingface
    repo_id: owner/model
    filename: onnx/model.onnx
    revision: main             # A commit hash can pin an exact revision.
preprocess: {plugin: .models.detector.adapter:Preprocess}
postprocess: {plugin: .models.detector.adapter:Postprocess, params: {conf_threshold: "${settings.confidence}"}}
variants: [onnx, triton]       # or {onnx: null, triton: {batching: {enabled: true}}}
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
