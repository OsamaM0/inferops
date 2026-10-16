# InferOps

A reusable Python library for model inference and multi-model pipelines, with
separately installable application adapters.

```bash
pip install 'inferops[cpu]'        # after publication; use -e '.[cpu]' in this checkout
# Alternatives: inferops[cuda], inferops[triton], inferops[tensorrt], inferops[deepstream]
```

Start with the [Python library guide](docs/python-library.md) for local inference,
runtime installation and multi-model examples, and [adapter authoring](docs/adapters.md)
for independent project packages. Driver behaviour is an optional adapter:
`pip install -e projects/driver_behaviour` from this directory.

Start the [documentation tour](docs/README.md) for the complete system:
[Hexagonal architecture and ports/adapters](docs/architecture.md),
[startup-to-shutdown lifecycle](docs/lifecycle.md),
[Redpanda, outbox and consumer workflows](docs/messaging.md), and the
[offline diagram gallery](docs/architecture/index.html). The
[package structure guide](docs/structure.md) maps the source modules.

The default server now uses the shared hexagonal engine. See the
[redesign and migration guide](docs/redesign.md) for application boundaries,
generated sessions, recoverable evidence and rollback. Python and HTTP call the
same pipeline services; PostgreSQL, MinIO, Redis, Redpanda, Prometheus and Grafana
remain the default server stack. A persistent capture spool is mounted separately
from completed evidence storage.

```python
from inferops import Engine

async def run_project():
    async with Engine.from_project("projects/driver_behaviour", runtime="cpu") as engine:
        summary = await engine.run("driver_behaviour", "clip.mp4")
        print(summary)
```

This embedded example uses in-memory persistence and evidence disabled. The
[Python guide](docs/python-library.md#shared-pipeline-engine) shows explicit
durable adapters and processing sessions. Server startup enables incident
capture and uses PostgreSQL/MinIO by default. There is one execution engine.
Rollback uses a designated previous release artifact; preserve database, object
and spool volumes when changing releases.
The [verification report](docs/cleanup-verification.md) records recovery and
rollback checks plus evidence-enabled CPU measurements. Capacity defaults are
provisional; production cutover remains gated on deployment-host benchmarks.


InferOps runs a project's models, pipelines, rules, alerts and evidence storage.
This directory is the project root: setup, development and Docker commands run
here. The `inferops/` child is the Python package, including the web interface.

## Hosting server requirements

The following are **planning estimates**, not benchmarked minimums or guarantees
of real-time performance. They assume one Linux server hosts the API, inference,
Redpanda, Redis, PostgreSQL, MinIO, Prometheus and Grafana. The default driver
project runs RF-DETR at 576 × 576 and FaceMesh at 256 × 256. Size other projects
for their own models, input resolution, sampled FPS and concurrent jobs.

### Hardware sizing

| Deployment | CPU | System RAM | GPU | Free SSD space at setup |
| --- | --- | --- | --- | --- |
| CPU evaluation, short clips or occasional images | 4 vCPU | 8 GB | None | 50 GB |
| CPU hosting starting point | 8 vCPU | 16–32 GB | None | 100 GB |
| CUDA hosting starting point | 8 vCPU | 32 GB | Compatible NVIDIA GPU; 8 GB VRAM starting point, 16 GB preferred | 100 GB |
| Triton / DeepStream hosting starting point | 8–16 vCPU | 32–64 GB | Compatible NVIDIA GPU; 16 GB VRAM or more | 150 GB |

CPU mode is useful for evaluation and workloads that tolerate slower inference.
For sustained video processing, start by evaluating CUDA; choose Triton for
separate model serving. DeepStream also starts Triton. The shared engine uses
incremental PyAV decoding in every runtime mode. The shipped Triton configuration loads two RF-DETR GPU
instances and one FaceMesh instance on GPU 0, with TensorRT workspace limits of
2 GiB and 1 GiB respectively. Leave additional VRAM for weights, activations
and engine building; the workspace limits are not total memory requirements.

**The shared engine decodes video incrementally.** Its memory depends on bounded
processing/evidence buffers rather than the full clip duration. Benchmark the
actual models, concurrency and evidence workload when sizing the server.

The SSD estimates cover an initial allowance for images, build cache, weights
and service data. Add capacity for uploaded media, evidence retention, database
growth and backups. Keep Docker's data directory and the writable project/model
repository on persistent storage. Named volumes survive `docker compose down`;
`docker compose down -v` removes them. Back up volumes and project configuration
and weights separately, and verify restoration.

### Operating system and software

- **Host:** use a Linux x86_64 server as the deployment baseline. Ubuntu 22.04
  LTS is a practical common starting point for the pinned GPU images; Ubuntu
  24.04 LTS is another option to evaluate for CPU/CUDA. ARM/Jetson deployments
  require separate image, wheel and runtime compatibility validation.
- **Containers:** install Docker Engine and **Docker Compose 2.20.3 or newer**.
  The root Compose file uses
  [Compose `include`](https://docs.docker.com/compose/how-tos/multiple-compose-files/include/).
  The deployment account needs permission to run Docker and write the selected
  project directory so model preparation can save weights.
- **Python:** the package requires Python **3.10+**; use **3.11** for a native
  server environment to match the CPU/CUDA API image. Docker supplies Python
  and OS dependencies, including FFmpeg; host Python is only needed for native
  setup or the Python launcher. Native setup also needs `venv`, `pip`, `make`
  for Make targets, and system libraries required by the selected video runtime.
- **Dependencies:** [pyproject.toml](pyproject.toml) is the source of truth.
  For a native CPU API/video server, install `pip install -e '.[server,cpu,video]'`;
  use `.[server,cuda,video]` for CUDA. Choose one ONNX Runtime wheel per
  environment. Docker installs the server, runtime and video dependencies.
  The UI needs no Node.js or frontend build.
- **Network:** allow outbound HTTPS during builds and model preparation to
  package indexes, image registries and configured artifact sources. The driver
  project's weights are downloaded from Google Drive and Wasabi; model files
  are ignored by Git. Provide `HF_TOKEN` when a selected project uses gated
  Hugging Face artifacts. Allow access to configured camera/source endpoints.

### GPU runtime compatibility

GPU modes need a working host NVIDIA driver, GPU access for the container, and
the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The helper `make setup-container-runtime` configures the toolkit on Ubuntu/Debian
and restarts Docker; it does **not** install a GPU driver.

| Mode | Runtime supplied by this checkout | Compatibility to check on the server |
| --- | --- | --- |
| `cpu` / `auto` | ONNX Runtime CPU | No NVIDIA software required; `auto` selects CPU |
| `cuda` | `onnxruntime-gpu[cuda,cudnn]>=1.21,<1.27` | CUDA 12 and cuDNN 9 library compatibility, supported GPU and matching driver; check the resolved wheel against the [ONNX Runtime requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements) |
| `triton` | `nvcr.io/nvidia/tritonserver:24.04-py3` | CUDA 12.4-based image; its [release notes](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2470/release-notes/rel-24-04.html) specify driver 545+ with documented data-center exceptions, and compute capability 6.0+ |
| `deepstream` | `nvcr.io/nvidia/deepstream:7.1-triton-multiarch` plus Triton 24.04 | Check both images; [DeepStream 7.1's matrix](https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_Installation.html) lists Ubuntu 22.04, CUDA 12.6, and validated drivers 535.183.06 for data-center GPUs / 560.35.03 for RTX GPUs; the GPU also needs hardware decoding support for the input codec |

The CUDA extra supplies user-space CUDA/cuDNN libraries through pip; GPU images
supply their runtime libraries. A host CUDA compiler toolkit is not required
for these container workflows. Driver versions above describe the pinned
releases, not a universal driver recommendation: choose a maintained driver
compatible with the GPU and every selected runtime. Newer GPU generations may
need newer images. Validate compatibility before provisioning the server.
`tensorrt` is a Python library runtime option; it is not a shipped Compose mode.

### Server access and deployment checks

The supplied Compose stack publishes ports on all host interfaces and uses
development credentials. Before exposing it to remote users, change MinIO and
Grafana credentials in `.env`, replace PostgreSQL's hard-coded development
password in the deployment configuration and matching `INFEROPS_PG_URL`, and
restrict access to internal services. The workspace/API has no built-in login;
put it behind an authenticated HTTPS reverse proxy or private access gateway
with WebSocket support. Grafana enables anonymous viewer access by default.

Expose the application through the gateway (normally HTTPS 443). Keep PostgreSQL
5432, Redis 6379, Kafka 19092 and Triton 8000–8002 private. Restrict MinIO
9000/9001, Grafana 3000 and Prometheus 9090 to trusted users/networks. Configure
the browser-facing `INFEROPS_*_PUBLIC_URL` values for remote service links and
evidence access; see [interfaces and ports](docs/interfaces.md) for all settings.
Docker-published ports require Docker-aware firewall or upstream network rules;
see [Docker's firewall guidance](https://docs.docker.com/engine/network/packet-filtering-firewalls/).

From this directory, after selecting the project/runtime in `.env`:

```bash
docker version
docker compose version
docker compose config --quiet
# GPU modes only: check the host driver, then verify container GPU access.
# nvidia-smi
# docker run --rm --gpus all ubuntu:22.04 nvidia-smi
docker compose up -d --build
docker compose ps
docker compose logs prepare api
curl --fail http://localhost:8080/healthz
curl --fail http://localhost:8080/readyz
```

Use your configured API port if it differs from 8080. Startup prepares and
validates real weights; missing artifacts or incompatible runtimes block it.
Health/readiness checks verify startup, not capacity. Run representative images
and videos at the intended concurrency, confirm GPU execution in accelerated
modes, and monitor CPU, RAM, VRAM, inference latency and disk growth before
settling on server sizing. The [verification scenarios](tests/scenarios/README.md)
provide examples for exercising the running application.

## Start the complete project

Run from the Git repository root, which contains this README and `compose.yaml`.
With Docker Compose 2.20.3 or newer:

```bash
cp .env.example .env             # first setup only; retain an existing .env
docker compose up -d --build
```

Open **[InferOps](http://localhost:8080/ui/)**. It provides pipeline execution,
live previews, activity, configuration and links to the original service UIs:

| Interface | Default address | Development login |
| --- | --- | --- |
| InferOps project workspace | [localhost:8080/ui/](http://localhost:8080/ui/) | None |
| MinIO original Console | [localhost:9001](http://localhost:9001/) | `minio` / `minio12345` |
| Grafana, InferOps Overview | [localhost:3000](http://localhost:3000/d/inferops-overview/inferops-overview) | Viewer access; `admin` / `admin` to edit |
| Prometheus | [localhost:9090](http://localhost:9090/graph) | None |
| Interactive API docs | [localhost:8080/docs](http://localhost:8080/docs) | None |
| All pages, ports and routes | [localhost:8080/ui/guide](http://localhost:8080/ui/guide) | None |

Grafana and Prometheus start by default. The workspace opens their own
interfaces in new tabs. `/frontend/` remains a compatibility address for
`/ui/`; the separate frontend directory and hand-built MinIO evidence page
have been removed. HTML and assets are served by FastAPI and included in
installed Python packages. No frontend server, npm setup or UI build is required.

The full MinIO admin console requires an older official image: newer upstream
releases removed that console. The version choice, existing-storage migration,
remote access and troubleshooting are covered in the
[interfaces and ports guide](docs/interfaces.md). This is a development stack;
its example credentials and public MinIO metrics are configured for local use.

## Select the project and runtime

Edit `.env`:

```dotenv
INFEROPS_PROJECT_ROOT=./projects/driver_behaviour
INFEROPS_EXECUTION_MODE=cpu
INFEROPS_API_PORT=8080
```

`cpu`, `cuda`, `triton` and `deepstream` select the matching Compose overlay;
`auto` is a CPU alias. GPU modes require configured NVIDIA drivers and
container runtime. `make setup-container-runtime` installs the container
runtime on Ubuntu/Debian with sudo authentication.

Docker validates the selected project, prepares required real weights, and
waits for storage before starting the API. Missing or invalid weights stop
startup. An empty project opens the workspace without inference. Runtime
variants count as one logical model in the UI.

```bash
docker compose logs prepare api
docker compose ps
```

After changing engine code or UI assets, use `docker compose up -d --build`.
For artifact source or tensor changes, prepare and recreate:

```bash
docker compose run --rm --no-deps prepare inferops project prepare --force
docker compose up -d --build --force-recreate api
```

In Triton or DeepStream modes, recreate `triton api` together. Named volumes
retain databases, objects, broker state and cache across `docker compose down`.
Project weights remain in `projects/<name>/model_repository/`.

## Project layout

```text
README.md, pyproject.toml       Project documentation and Python package metadata
.env.example, .env             Example and local configuration
Makefile, compose.yaml          Development commands and Compose entry point
.dockerignore                  Build context exclusions
inferops/                      Importable Python engine
  __init__.py, public.py        Stable Engine, Model and InferenceSession API
  domain/                      Models, frames, tracking, fusion and incident/evidence policies
  application/                 Execution, sessions, model pool, incidents, evidence, jobs and revisions
  ports/                       Inference, media, persistence, delivery and extension contracts
  adapters/inbound/            HTTP/WebSocket + packaged UI, CLI, source and broker workers
  adapters/outbound/           Inference/plugins, stages, media, projects, storage and observability
  bootstrap.py                 Explicit resource construction
  sdk/                         Remote HTTP client
  testing/                     Test configurations and tiny model generator
projects/                      Self-contained projects, adapters, rules and weights
configs/                       Shared platform defaults
deploy/                        Dockerfiles, runtime overlays, Grafana and Prometheus
scripts/                       Setup, runtime launch and cache cleanup
tests/                         Engine tests and browser/video scenarios
docs/                          Interfaces, API reference, project contracts and validation
research/                      Training notebooks and sample media
```

This directory is also the Git repository root: `.git/`, `.gitignore`, this
README, `pyproject.toml` and launch configuration all belong here. The child
`inferops/` directory is the importable Python package. `.venv/` stores installed
dependencies; `.run/` stores runtime reports and verification output. Pytest and
Ruff caches live under `.run/cache/`. Test-only models use process-owned system
temporary directories and do not create a root `model_repository/`. Real project
weights remain under `projects/<project>/model_repository/`.

`inferops.egg-info/` is generated installation metadata from the editable Python
install; it contains package information and entry points, not another source
implementation. Build output and tool caches are generated too and can be
recreated.

### Generated files and cleanup

`inferops/` is the engine's source package. Python packaging creates a temporary
copy at `build/lib/inferops/` when building or installing a wheel; edit the source
package, since the build copy can become stale. `dist/` contains built archives
and `*.egg-info/` contains generated packaging metadata. These can be recreated
from `pyproject.toml` and are ignored by Git and the Docker build context.

From this project directory:

```bash
make clean CLEAN_ARGS=--dry-run   # list generated directories to remove
make clean                      # remove caches and packaging output
```

Cleanup covers caches throughout this repository and packaging output in
project adapters. It preserves `.venv/`, `.env`, project weights, runtime data,
backups and verification reports in `.run/`. Tests, research sample clips and
registered pipeline stages/plugins support development and project extensions.

## Development and new projects

```bash
make setup PROJECT=driver_behaviour
make run PROJECT=driver_behaviour MODE=cpu
make new-project NAME=ppe_site
make new-model PROJECT=ppe_site NAME=helmet_detector FORMAT=yolo LABELS=person,helmet
make test-all
```

Native `make run` starts storage, Grafana and Prometheus in Docker, then serves
the Python API in the foreground. Prometheus targets that host API. Stop the
native API with Ctrl+C or `make stop` before switching to containers:

```bash
make run-container PROJECT=driver_behaviour MODE=cuda RUN_ARGS=--build
```

The launcher handles its own native API occupying the requested port. It does
not stop unrelated processes. `API_PORT=<port>` chooses another native/API
host port. Custom native storage ports also require matching service endpoints.

| Configure | Location in `projects/<project>/` |
| --- | --- |
| Identity, enabled pipelines, prewarm and service overrides | `project.yaml` |
| Weights, tensors, labels, adapters and runtime variants | `models/<model>/model.yaml` |
| Sources, stage ordering, tracking and evidence | `configs/pipelines/*.yaml` |
| Temporal signals and rules | `configs/fusion/*.yaml` |
| Severity, cooldowns, dedupe and actions | `configs/alerts/*.yaml` |

Platform defaults come from `configs/platform.yaml`; project settings and
environment variables can override them. The UI configuration editor saves
and validates YAML. Live sections can reload immediately; startup, sources,
tensors and platform settings may require preparation and restart.

See [interfaces and ports](docs/interfaces.md), [API reference](docs/api.md),
[project configuration](docs/project-blocks.md), [project examples](projects/README.md)
and [verification scenarios](tests/scenarios/README.md).
