# Central configuration

Edit **[`configs/inferops.yaml`](../configs/inferops.yaml)** for the workspace.
Keep **`.env`** for passwords, tokens and essential host overrides, such as an
external database URL or a project directory outside the checkout.

| YAML section | Controls |
| --- | --- |
| `runtime` | Selected project, CPU/CUDA/Triton mode, API port, optional services and capacity |
| `platform` | Inference admission, logging, transport and service clients |
| `sources` | Defaults for newly registered streams: target FPS, resolution, preview and frame age |
| `evidence` | Clip duration, sampling and optional capacity limits |
| `media` | Routing, recording, retention and public player URLs |
| `storage` | Filesystem or S3 evidence storage |
| `docker` | Engine image, Triton image, published ports and media bind/advertised host |
| `projects` | Optional plugin roots, common parameters and model/pipeline/rule/alert documents |
| `workers` | Optional static worker manifest, in the same YAML file |

For example, select accelerated inference and monitoring with:

```yaml
runtime:
  project: driver_behaviour
  mode: cuda
  api_port: 8080
  services: [monitoring]
```

Keep the other sections in the existing document. To use the bundled S3 service,
set `storage.kind: s3` and add `storage` to `runtime.services`. To use Kafka, add
`broker` and set `platform.transport.kind: redpanda`. The default endpoints are
selected for native or Docker execution. External endpoints belong under the
matching `platform` client section, with credentials in `.env`.

Run `./run.sh` to apply startup settings. `./run.sh configure` renders and
validates Docker configuration without starting application services. Native
launchers (`make run`, `scripts/serve_local.py`) read the same YAML. Explicit
`--mode` / `--port` options override one launch without changing the document.

The launcher generates private, ignored `.run/compose.yaml`, `.run/compose.env`
and `.run/container.env`. These are disposable outputs, including credentials
needed by containers. Do not edit or commit them. After rendering, ordinary
`docker compose` commands use the selected runtime. Re-run configuration after
changing YAML or `.env`. The Docker-only launcher builds a small YAML reader
image, so it needs no Python installation on the host.

## Optional projects

Driver behaviour is one plugin. Register another project under `projects`, give
it a `root`, and select its key in `runtime.project`. No engine code changes are
needed. Project directories and weights remain separately mounted at `/project`.
Relative project roots resolve from the workspace, containing `configs/`.

Project packages provide model/tensor contracts, stage implementations and
baseline definitions. Treat their YAML files as packaged defaults. Configure the
workspace through the central file; the dashboard stores edits under
`projects.<key>.configuration`, including newly created documents. Packaged files
remain unchanged. Existing standalone projects without central configuration
retain their legacy loading and editing behavior.

`parameters` replaces matching parameter keys throughout a project's packaged
definitions. This avoids repeating shared thresholds across model variants and
pipelines. For the bundled driver plugin, confidence is `0.80` and unsafe
behavior confirmation is `3000` milliseconds. A document saved under
`configuration.models`, `configuration.pipelines`, `configuration.fusion` or
`configuration.alerts` is a complete override and takes precedence over those
common parameters. Each mapping key is the editor/document ID, not a runtime
variant name. `configuration.project` overrides the project manifest.

## Saving and applying

The dashboard's **Workspace settings** entry edits the whole central file.
Model, pipeline, rule, alert and platform editors write to the same file.
Validation happens before an atomic save; stale edits return a conflict. Form
saves normalize formatting. Use the whole-file Advanced YAML editor when you
want to preserve your comments and formatting.

Thresholds, rules and ordinary pipeline changes can reload and restart active
managed streams. Explicit Python sessions retain their original revision.
Artifact, backend, tensor, project identity and startup/service changes require
preparation or restart; the API reports this without partially activating them.
Inference admission changes apply live. `sources` defaults apply to new streams;
registered streams keep their saved per-stream values in PostgreSQL. Those
registrations, histories and model weights are application data, rather than
another global settings store.

A target FPS limits sampling. Actual analysis FPS depends on decoding, inference
latency and contention; free VRAM does not establish throughput. Change local
admission under `platform.runtime.admission.local_device_capacity` after measuring
representative streams. With `auto_scale: true`, GPU admission allows one concurrent
call per open processing session, capped by `local_device_capacity`. This workspace
uses a cap of 8. Slots are shared fairly across models on each GPU, and shrink safely
as sessions close. With `auto_scale: false` (the engine default), capacity is fixed.
Remote RPC admission has its own limit. Concurrency increases GPU work overlap;
it does not guarantee 100% utilization or sufficient VRAM for every workload.

For server-local video, use **Sources → Add a source → Video file**, enter the
absolute path (spaces are supported), and enable **Loop** for continuous replay.
The playground's **RTSP / Server video** control accepts the same path. The stream
API accepts `POST /v1/streams?pipeline=...&video_path=...&fps=30&loop=true` (URL-encode
the path), and also accepts local paths through the existing `rtsp_url` parameter.
Files replay using their presentation timestamps, share live inference admission,
and finish at EOF unless looping is enabled. A `source_rtsp` YAML stage also accepts
a local `url` path and `loop: true`. Paths refer to the API/worker filesystem; Docker
deployments must mount the file at that path inside the processing container.

## Advanced settings and migration

Omitted settings use typed engine defaults. Add only the settings you need. Full
client schemas are available from `/v1/configs/schema/platform`; source fields
are documented by `/v1/cameras` and the API schema. Evidence fields correspond to
`EvidenceConfig`. Integrations with an environment-only adapter setting can put
it under `platform.environment`, for example:

```yaml
platform:
  environment:
    INFEROPS_SOURCE_UPLOAD_MAX_BYTES: 536870912
```

Move ordinary values out of an older `.env` into the appropriate YAML section.
Explicit host variables and retained `.env` values override YAML. Remove a stale
override if a YAML change appears not to apply. Never put tokens/passwords in
project overrides or commit generated runtime files.

For the optional two-worker Docker layout, add a `workers` mapping containing
`version: 1`, `workers: [{worker_id: a, endpoint: http://camera-a:8080},
{worker_id: b, endpoint: http://camera-b:8080}]` and `cameras: []`. Select Triton,
shared S3 and its `storage` service, and set `INFEROPS_WORKER_TOKEN` in `.env`.
The renderer includes the worker wiring automatically. The legacy manifest
reader still accepts standalone manifests for independent deployments.

The launcher mounts the repository’s `research` directory read-only at the same
absolute path in API and camera-worker containers. To use another media directory,
set `INFEROPS_MEDIA_ROOT=/absolute/media/directory` in `.env` and recreate services.
