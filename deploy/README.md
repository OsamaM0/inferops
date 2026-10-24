# Deploying and operating InferOps

The default deployment runs one API/engine owner, PostgreSQL and MediaMTX,
with persistent disk for completed evidence and a separate recovery spool.
MinIO/S3 (`storage`), Redpanda (`broker`) and Prometheus/Grafana
(`monitoring`) are optional. MediaMTX supplies shared RTSP paths, WebRTC/HLS,
native recording and playback; see the [media routing guide](../docs/media-router.md).
CPU/CUDA execute models locally; Triton mode adds a shared model server.

Read the [documentation index](../docs/README.md), [architecture](../docs/architecture.md),
[full lifecycle](../docs/lifecycle.md) and [broker guide](../docs/messaging.md).
Hardware/runtime prerequisites and remote-access configuration are in the
[root README](../README.md#hosting-server-requirements).

## Configuration and startup

The Docker-only launcher handles first setup, checks and readiness:

```bash
./run.sh                         # or: make start
./run.sh --mode cpu --port 8082   # optional one-launch overrides
./run.sh status
./run.sh logs
./run.sh stop
```

It creates `.env` only when absent and preserves existing settings. The commands
below remain available for manual Compose operation.

Run from the repository root, containing `README.md`, `pyproject.toml` and the
root `compose.yaml`. Preserve an existing `.env`:

```bash
cp .env.example .env             # first setup only
```

Select the project, runtime and API port under `runtime` in
[`configs/inferops.yaml`](../configs/inferops.yaml). Docker images/ports, optional
services, media, evidence, source defaults and project parameters also live
there. Keep secrets and essential host overrides in `.env`. See
[central configuration](../docs/configuration.md).

```bash
./run.sh configure
```

Then validate and start:

```bash
docker compose config --quiet
docker compose up -d --build
docker compose ps
docker compose logs prepare api
curl --fail http://localhost:8080/healthz
curl --fail http://localhost:8080/readyz
```

Open [the workspace](http://localhost:8080/ui/) or
[interactive API docs](http://localhost:8080/docs). Health/readiness establish
startup, not throughput or full dependency health. Check `/v1/system/status`,
Prometheus targets and representative inference/evidence as well.

The root [Compose entry point](../compose.yaml) includes the
[base stack](compose.yaml) and the generated runtime selection using
[Compose include](https://docs.docker.com/compose/how-tos/multiple-compose-files/include/)
(Docker Compose 2.20.3+). `project_directory: .` resolves mounts/build context
from the repository root. Run the root entry point rather than invoking the
base file alone and losing its runtime/project context.

## Startup troubleshooting

`socket.gaierror: [Errno -3] Temporary failure in name resolution` during
`SqlRepository.start()` means the API could not resolve the database hostname.
A healthy PostgreSQL container does not establish that the API can reach it.
Containers discover `postgres` through their shared Compose network. In the
reported failure, `docker inspect` showed the API had no network attachment,
while PostgreSQL was attached to `inferops_default`.

```bash
./run.sh check
./run.sh repair                  # recreate only the API, keep all data volumes
```

Ordinary `./run.sh` startup also detects and recreates an API with no network.
For manual recovery, use `docker compose up -d --build --force-recreate api`.
If a custom `INFEROPS_DATABASE_URL` or legacy `INFEROPS_PG_URL` is configured,
check its hostname too: the bundled Docker database is `postgres:5432`;
a native API uses `127.0.0.1` and the published `INFEROPS_PG_PORT`.
Do not replace the Docker database hostname with `localhost` inside the API.

If Docker reports `port is already allocated`, another service owns that host
port. The launcher identifies Docker port owners and stops with a clear error.
Choose a free port without stopping the other service:

```bash
./run.sh --port 8082
# To persist the choice, set runtime.api_port: 8082 in configs/inferops.yaml.
```

`./run.sh` waits up to 600 seconds for service readiness and prints recent API,
preparation and database logs on failure. Check `./run.sh logs` if model
downloads, invalid weights or runtime initialization prevent startup. Docker
must be running and accessible to your user; `docker info` checks access.

## What starts and where

| Service | Work | Startup gate |
| --- | --- | --- |
| `prepare` | Validate project and prepare real model artifacts in the project mount | Must complete before API and optional Triton |
| `minio-init` | Optional `storage` profile: wait for MinIO readiness | Must complete before API when enabled |
| `api` | FastAPI/UI, shared engine, sessions, jobs, evidence workers and outbox relay | Successful preparation; health/readiness for explicitly enabled dependencies; Triton when selected |
| `postgres` | Authoritative application records and delivery scheduling | Native PostgreSQL health check |
| `minio` | Optional `storage` profile: S3 artifacts and original console | Container health/readiness |
| `redpanda` | Kafka-compatible distribution and optional worker input/output | `rpk cluster info` |
| `prometheus` | Scrape API, broker, MinIO and optional Triton metrics | Starts independently of API processing |
| `grafana` | Provisioned dashboards using Prometheus | Depends on Prometheus |
| `triton` | Optional separate model serving | Preparation plus Triton readiness |
| `mediamtx` | Default RTSP relay, WebRTC/HLS, recording and playback | Starts by default; authenticated API checked in Services & runtime |

The broker and pipeline worker entry points are not separate services in this
Compose stack. Evidence workers and the relay run inside the API process.
The default stack starts `prepare`, `postgres`, `mediamtx` and `api`.
Evidence uses `FilesystemStorage` through the existing `ObjectStore` port, under
`/var/lib/inferops/data/evidence` on the named `engine-data` volume. Native
startup uses `$INFEROPS_DATA_PATH/evidence` (default `.run/server/evidence`).

To opt into MinIO and select the S3 adapter with the launcher:

```bash
make run-container MODE=cpu RUN_ARGS='--profiles storage'
# Native API alternative:
make run MODE=cpu RUN_ARGS='--profiles storage'
```

For direct Compose, select both the adapter and service profile:

```bash
# Set storage.kind: s3 and runtime.services: [storage] in configs/inferops.yaml.
./run.sh
```

For repeated use, set `storage.kind: s3` and add `storage` to
`runtime.services` in `configs/inferops.yaml`. An external S3 endpoint needs the adapter settings
without starting local MinIO. Install the `s3` extra for a minimal native install.
The static-workers overlay explicitly uses shared S3 storage.

Broker and monitoring profiles require matching adapter settings. The retired
local runtime data has been deleted. First startup creates empty PostgreSQL,
evidence, recording and spool storage; there is no SQLite import step. Preserve
these new volumes across subsequent ordinary restarts.

Combined mode uses **one active application owner per database/spool bundle**. Do not scale this
API with multiple Uvicorn workers or run another default server worker against
the same bundle: startup recovery fences prior owners. The worker command and
ownership implications are documented in [messaging](../docs/messaging.md#run-a-worker-as-the-application-owner).

For fixed camera shards, the optional [workers overlay](compose/workers.yaml)
adds PostgreSQL leases/fenced writes, separate persistent spools, authenticated
control routing and a restart watchdog. Follow [static workers](../docs/static-workers.md)
for the manifest, migration, retention and supported layout. This keeps one API
owner and uses shared Triton for GPU execution.

## Runtime overlays and images

| Mode | Overlay | Model execution |
| --- | --- | --- |
| `cpu` / `auto` | `compose/cpu.yaml` / `compose/auto.yaml` | ONNX Runtime CPU in API |
| `cuda` | `compose/cuda.yaml` | GPU ONNX Runtime wheel and GPU access in API |
| `triton` | `compose/triton.yaml` | API client calls Triton; project model repository mounted read-only there |

`docker/api.Dockerfile` packages the Python engine and built-in HTML/CSS/JS.
Projects and weights arrive through `/project`, not the engine image. A changed
engine/UI needs a rebuild; a changed weight/tensor definition needs preparation
and potentially model-server recreation. No frontend build server is required.
`tensorrt` is a Python library extra, not a shipped Compose overlay.

## CI/CD and Docker Hub

The [GitHub Actions workflow](../.github/workflows/ci.yaml) keeps the Python
3.10/3.12 tests, architecture checks, Ruff and PostgreSQL/MinIO integration
checks. After those pass, it builds the CPU, CUDA and Triton client images
from `deploy/docker/api.Dockerfile` for `linux/amd64`. Each image must pass
CLI, inference-client import and packaged-UI/config smoke tests before upload.
GPU inference and end-to-end startup still require the deployment's hardware,
project assets and services.

The target is **`osamam0/inferops-driverbeh`**. Docker repository names must be
lowercase, so the requested `inferops-DriverBeh` is normalized to `inferops-driverbeh`.

One-time setup:

1. Create the `inferops-driverbeh` repository under `osamam0` on Docker Hub.
2. Create a Docker Hub personal access token with **Read & Write** access.
3. In GitHub **Settings → Secrets and variables → Actions**, add the repository
   secret **`DOCKERHUB_TOKEN`** with that token. Keep it out of `.env` and Git.
4. The login defaults to `osamam0`. Optionally add the Actions variable
   **`DOCKERHUB_USERNAME`** to use a different login, and **`DOCKERHUB_IMAGE`** to
   override the full image name (for example, `your-organization/inferops`).
5. Push the workflow to GitHub. A push to the repository's default branch runs
   validation and publication. **Actions → CI and Docker Hub → Run workflow**
   also publishes when run against the default branch.

| Trigger | Published tags |
| --- | --- |
| Default-branch push or manual run | `latest` (CPU), `cpu`, `cuda`, `triton`, plus `sha-<full-commit>-<runtime>` |
| Version tag such as `v1.2.3` | `1.2.3` (CPU), `1.2.3-cpu`, `1.2.3-cuda`, `1.2.3-triton`, plus commit tags |
| Pull request, feature-branch push or manual run on another branch | Build and smoke tests only; no login or upload |

To publish a release after committing the intended changes:

```bash
git tag v1.2.3
git push origin v1.2.3
```

Version tags do not move the default-branch aliases; prerelease tags such as
`v1.2.3-rc.1` get their own version tags. Published tags and the image digest
appear in each Docker job's summary. Missing registry credentials fail the
publication job with a setup message; pull requests need no Docker Hub secret.
The build/test/push sequence follows Docker's
[test-before-push workflow](https://docs.docker.com/build/ci/github-actions/test-before-push/).

To deploy a published image, set `runtime.mode: cpu` and
`docker.image: osamam0/inferops-driverbeh:cpu` in `configs/inferops.yaml`.
`INFEROPS_IMAGE` remains available as an essential host override in `.env`.

Then run from the repository root:

```bash
./run.sh configure
docker compose pull api prepare
docker compose up -d --no-build --wait --wait-timeout 600
# Or use the Docker-only launcher after pulling: ./run.sh --no-build
```

For CUDA or Triton, change both the mode and the image tag to `cuda` or
`triton`. Use a version tag or image digest for a fixed deployment and keep
the selected project mounted at `/project`; projects and weights are not
baked into these images. The Triton client image uses the separate upstream
Triton server container already configured in Compose. For a private Docker
Hub repository, run `docker login` on the deployment host before pulling.
When `INFEROPS_IMAGE` is unset, Compose uses the existing local image names
and build definitions. CI publishes images; deployment to a host uses the
commands above.

## Persistent data and mounts

For a smaller image, select service clients separately from the inference runtime:

```bash
docker build -f deploy/docker/api.Dockerfile --build-arg INFERENCE_EXTRAS=cpu \
  --build-arg SERVER_EXTRAS=http,artifacts,video -t inferops-engine:minimal .
```

The HTTP extra includes PostgreSQL access; add `s3` for optional S3 evidence storage,
`broker` for Kafka, and `tracing` for an OTLP exporter. The default `server,video` build retains the full compatible
client bundle; MinIO, broker, monitoring and Triton containers run only when selected. Choose one
inference extra (`cpu`, `cuda` or `triton`) per environment. The Triton image needs
the client and uses a separate GPU model server.

| Volume/mount | Contents | Recovery importance |
| --- | --- | --- |
| `engine-data` → `/var/lib/inferops/data` | Default snapshots, annotated clips and artifact manifests under `evidence/` | Completed evidence bytes; retain across API replacement |
| `media-recordings` | Original MediaMTX recording segments | Default 24-hour retention; separate from incident evidence |
| `source-data` | Uploads and replaceable preview cache | Retain uploaded media |
| `postgres-data` | Sources, run/result history, sessions, incidents, evidence metadata, jobs, inbox/outbox | Authoritative records and retry identities |
| `minio-upstream-data` | Optional S3 snapshots, clips and artifact manifests | Completed evidence bytes when S3 is selected |
| `evidence-spool` → `/var/lib/inferops/spool` | Checksummed capture generations and retained encoded artifacts | Pending/incomplete capture recovery; do not treat it as disposable cache |
| `redpanda-data` | Broker logs and consumer progress | Retained messages and replay positions |
| `grafana-data` | Grafana state | Dashboard/user changes |
| `hf-cache` / `triton-cache` | Preparation/model-serving caches | Rebuildable; downloads/build time may be required |
| Selected host project → `/project` | Configuration, plugins and `model_repository/` weights | Back up separately from Docker volumes |
| `configs/` and monitoring bind mounts | Platform defaults, scrape/provisioning/dashboard configuration | Reproducible deployment configuration |

Named volumes survive `docker compose down`. `docker compose down -v` deletes
them and is not a routine cleanup or upgrade step. Back up SQL, evidence disk (or S3) and spool
with a consistent operational plan and verify restoration; also preserve project
configuration, required weights and deployment settings. Prometheus currently
has no named data volume in this Compose file, so do not rely on its local
history surviving container replacement.

## Run one complete lifecycle

The upload endpoint uses the same pipeline services as Python:

```bash
curl --fail -X POST http://localhost:8080/v1/run/driver_behaviour/upload \
  -F video=@research/driver_training/test.mp4 -F stream_id=documentation-check
curl --fail 'http://localhost:8080/v1/alerts?limit=20'
curl --fail 'http://localhost:8080/v1/evidence/records?limit=20'
```

Keep the returned generated session ID to query only that run with
`/v1/evidence/records?session_id=<session_id>`. Inspect coverage/errors as well as
state, then play the snapshot/clip through each record's asset endpoint. Finite
source boundaries can legitimately produce `Incomplete` captures. A successful
upload is not proof of complete context, broker delivery or live capacity.

For continued image sessions, cameras and browser capture, see
[API examples](../docs/api.md#stateful-image-processing). For Python with the same
durable bundle, see the [library guide](../docs/python-library.md).

## Observe and diagnose

| Signal | Meaning |
| --- | --- |
| `/v1/system/status`, service logs | Dependency/runtime problems |
| API latency and live drop counts | Processing pressure and response behavior |
| Evidence states, active/queued captures, ring/spool bytes | Capture pressure and incomplete/failed outcomes |
| Pending outbox and oldest age | SQL delivery backlog |
| Broker group lag | Input/external consumer progress; separate from outbox age |
| MinIO/host disk growth | Completed artifact retention and local capacity |
| Prometheus target state | Whether measurements are being scraped |

Prometheus scrapes configured API, MinIO, Redpanda and Triton targets. No
PostgreSQL/GPU exporter service is shipped. The optional Triton target can
be down in CPU mode. Grafana visualizes collected metrics; it is not the record
of accepted incidents. See [interfaces and ports](../docs/interfaces.md) for
addresses/credentials and [messaging diagnostics](../docs/messaging.md#failure-diagnosis-and-operational-signals).

Memory, capture workers, queues, spool quota/reserve, sample rate, pre/post
windows and timeouts are configurable provisional settings. Benchmark the
intended concurrent live workload **with evidence enabled** before choosing
production values. Record p50/p95/p99 reply latency, drops, evidence completion,
memory, spool growth and outbox lag.

## Update, stop and recover

After an engine/UI change:

```bash
docker compose up -d --build
```

For an artifact/tensor change:

```bash
docker compose run --rm --no-deps prepare inferops project prepare --force
docker compose up -d --build --force-recreate api
```

In Triton modes recreate `triton api` together. Quiesce admission and
active work before a release switch. Graceful shutdown finalizes sessions,
drains capture within its configured budget and leaves unfinished durable work
recoverable. Compose's stop grace must cover the chosen application shutdown
budget; the current base file does not set a custom `stop_grace_period`.

```bash
docker compose stop -t 60 api
docker compose up -d api
```

The 60 seconds is an example operator budget, not a guaranteed drain time.
Recovery can reuse committed capture material and matching uploaded objects;
it cannot recreate missing source frames. Query explicit states/reasons after
restart. Follow [upgrade/rollback contracts](../docs/lifecycle.md#upgrade-and-rollback)
for a supported release rollback; preserve schemas, records and volumes.

The supplied ports and credentials target local development. Before remote
access, apply the [documented server access configuration](../README.md#server-access-and-deployment-checks),
including authenticated HTTPS access and private infrastructure endpoints.
