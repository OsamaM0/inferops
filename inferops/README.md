# InferOps

InferOps is a model-agnostic inference platform that loads models, pipelines,
fusion rules, and alerts from YAML. It can run as a single-process CPU service
for local development or as a broker-backed stack with Triton, storage, and
observability services.

At the core is a typed pipeline DAG composing:

**source → decode → preprocess → infer → postprocess → fuse → alert → sinks**

## What you get

- YAML-driven model registry, pipeline specs, fusion rules, and alert rules
- ONNX Runtime and Triton inference backends
- In-process and Redpanda transports behind the same abstraction
- FastAPI, WebSocket fan-out, and Prometheus metrics
- Worker entrypoints for source-driven and topic-driven pipelines
- Evidence browsing and event/alert storage integrations

## Quick start (local, CPU-only)

Prerequisites:

- Python 3.10+
- `make`
- Optional: `jq` for prettier JSON output from `curl`

Run all `make` commands from the repository root; the root `Makefile` is the only Makefile entrypoint.

Level 1 sets up InferOps itself. This installs dependencies, validates the
core platform config, installs the native `make serve` prerequisites on
Debian/Ubuntu Linux, and starts the API/dashboard with no active domain:

```bash
python -m venv inferops/.venv
source inferops/.venv/bin/activate
make setup
make validate
make serve-mock
```

Level 2 selects one domain pack for the process. The bundled driver behaviour
domain includes its own model specs, pipelines, fusion rules, alert rules,
plugins, and model asset setup:

```bash
make setup DOMAIN=driver_behaviour
make validate DOMAIN=driver_behaviour
make test
make serve-mock DOMAIN=driver_behaviour
```

Open `http://localhost:8080/docs` after the server starts. The dashboard is available at `http://localhost:8090/dashboard.html`.

Notes:

- `make setup` installs Python dependencies, optional model assets, and config validation. On Debian/Ubuntu Linux it also bootstraps native `make serve` prerequisites by downloading `minio` and `mc` into `.tools/bin` and installing `redpanda`/`rpk` from the official APT repository.
- Use `SKIP_RUNTIME=1 make setup` if you only want the Python environment and plan to use `make serve-mock` or `make up`.
- `make setup-runtime` reruns only the native runtime bootstrap when you need to repair or upgrade the local `make serve` prerequisites.
- `make serve` is the native broker-backed path. After `make setup`, it expects local `minio` plus `rpk`/`redpanda` binaries to be available either from `.tools/bin` or the system PATH.
- Core InferOps starts without driver models or driver pipelines unless you pass `DOMAIN=driver_behaviour` or set `INFEROPS_ACTIVE_DOMAIN=driver_behaviour`.
- The checked-in local profile in `configs/platform.yaml` sets `transport.kind: redpanda`; use `make serve-mock DOMAIN=driver_behaviour` or set `INFEROPS_TRANSPORT_KIND=inproc` for the lightweight local path.
- If your shell does not resolve the `inferops` console script, use `python -m inferops.cli ...` instead.
- If you need Triton-backed model specs, install the extra client dependency with `pip install -e ".[dev,triton]"`.
- If you need PyAV-based video stages, install `pip install -e ".[dev,video]"`.

### Smoke-test the API

```bash
curl http://localhost:8080/healthz
curl http://localhost:8080/v1/models
curl -F 'image=@some.jpg' -F 'stream_id=demo' \
  http://localhost:8080/v1/infer/driver_behaviour
```

The bundled `driver_behaviour` pipeline runs the RF-DETR driver model when
available and falls back to the synthetic detector in local development, then
passes results through fusion, alerts, and evidence storage.

## Configuration model

InferOps merges configuration in layers:

1. Code defaults from `inferops/core/config.py`
2. Platform defaults from `configs/platform.yaml`
3. One active domain selected by `INFEROPS_ACTIVE_DOMAIN` or `INFEROPS_DOMAIN_ROOT`
4. YAML specs in that active domain's `configs/models`, `configs/pipelines`, `configs/fusion`, and `configs/alerts`
5. Environment variables such as `INFEROPS_*`

Common overrides:

- `INFEROPS_CONFIGS_DIR`: alternate config root, default `configs`
- `INFEROPS_ACTIVE_DOMAIN`: active domain name, for example `driver_behaviour`
- `INFEROPS_DOMAIN_ROOT`: explicit domain root, for example `examples/driver_behaviour`
- `INFEROPS_MODEL_REPOSITORY`: alternate model artifact root, default `model_repository`
- `INFEROPS_TRANSPORT_KIND`: `inproc` or `redpanda`
- `INFEROPS_TRANSPORT_BOOTSTRAP`: broker bootstrap, default `localhost:19092`
- `INFEROPS_TRITON_URL` and `INFEROPS_TRITON_HTTP_URL`: Triton gRPC and HTTP endpoints
- `INFEROPS_REDIS_URL`, `INFEROPS_PG_URL`, and `INFEROPS_S3_ENDPOINT`: backing service connections
- `INFEROPS_API_HOST`, `INFEROPS_API_PORT`, and `INFEROPS_API_CORS_ORIGINS`: API server settings

## Common CLI workflows

```bash
inferops validate --domain driver_behaviour
inferops preflight --domain driver_behaviour
INFEROPS_ACTIVE_DOMAIN=driver_behaviour inferops models list
INFEROPS_ACTIVE_DOMAIN=driver_behaviour inferops models show rfdetr_driver_synth
INFEROPS_ACTIVE_DOMAIN=driver_behaviour inferops pipelines list
INFEROPS_ACTIVE_DOMAIN=driver_behaviour inferops pipelines run driver_behaviour path/to/image.jpg
INFEROPS_ACTIVE_DOMAIN=driver_behaviour inferops worker pipeline driver_behaviour
INFEROPS_ACTIVE_DOMAIN=driver_behaviour inferops worker topic driver_video \
  --input-topic inferops.frames \
  --output-topic inferops.results
```

`inferops validate --domain <name>` checks the selected domain's YAML contracts.
`inferops preflight --domain <name>` also checks backend dependencies, plugin
imports, stage references, and local model artifacts. `inferops pipelines run`
is the fastest local sanity check when you want to execute a pipeline directly
from the CLI with an image file.

## HTTP API

Health and observability:

- `GET /`
- `GET /healthz`
- `GET /readyz`
- `GET /metrics`
- `GET /v1/system/status`

Registry and pipeline management:

- `GET /v1/models`
- `GET /v1/models/{name}`
- `POST /v1/models/reload`
- `GET /v1/pipelines`
- `GET /v1/pipelines/{name}`
- `POST /v1/pipelines/reload`

Execution:

- `POST /v1/infer/{pipeline}` for multipart image uploads
- `POST /v1/run/{pipeline}` for JSON payloads such as `{"video_path": "/path/to/video.mp4"}`
- `POST /v1/run/{pipeline}/upload` for multipart video uploads
- `POST /v1/jobs/{pipeline}`
- `GET /v1/jobs`
- `GET /v1/jobs/{job_id}`

Streaming, history, and evidence:

- `POST /v1/streams?pipeline=<name>`
- `GET /v1/streams`
- `DELETE /v1/streams/{stream_id}`
- `WS /v1/ws/{channel}`
- `GET /v1/events`
- `GET /v1/alerts`
- `GET /v1/evidence`
- `GET /v1/evidence/clips`

## Full local stack with Docker Compose

Use Docker Compose when you want the broker-backed stack with Redpanda,
Postgres, MinIO, Prometheus, and Grafana.

```bash
make setup DOMAIN=driver_behaviour
make up
```

Important details:

- The `api` container is configured for `INFEROPS_TRANSPORT_KIND=redpanda`.
- The compose stack exposes API `8080`, Redpanda `19092`, MinIO `9000`, MinIO console `9001`, Prometheus `9090`, and Grafana `3000`.
- Stop and remove volumes with `make down`.

## Project layout

```
inferops/                # Python package: registry, pipeline runtime, API, workers
configs/                 # Core platform defaults
examples/                # Optional domain packs with configs, plugins, and setup assets
model_repository/        # Triton/ONNX model artifacts
scripts/                 # Model download/build helpers and demos
tests/                   # Smoke, contract, and pipeline tests
deploy/docker/           # Container build assets
docker-compose.yml       # Local multi-service stack
```

## Development

```bash
make validate DOMAIN=driver_behaviour
make test
make lint
```

For a narrow API smoke test from the repository root, run `inferops/.venv/bin/python -m pytest -q inferops/tests/test_api_smoke.py`.
