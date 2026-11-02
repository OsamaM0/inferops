# Development

Set up a native environment, add projects and models, run the tests, and find your way around the repository.

## Commands

```bash
make setup PROJECT=driver_behaviour          # venv, dependencies and weights
make run PROJECT=driver_behaviour MODE=cpu   # native API; services in Docker
make test-all
```

| Command | What it does |
| --- | --- |
| `make new-project NAME=ppe_site` | Create a project folder |
| `make new-model PROJECT=ppe_site NAME=helmet FORMAT=yolo LABELS=person,helmet` | Add a model to a project |
| `make validate` / `make preflight MODE=cuda` | Check configuration, plugins and weights |
| `make run-container PROJECT=driver_behaviour MODE=cuda RUN_ARGS=--build` | Run the API in Docker instead |
| `make stop` | Stop the native API |
| `make clean` | Remove caches and build output (`CLEAN_ARGS=--dry-run` to preview) |
| `make help` | List all commands |

Native `make run` starts storage, Grafana and Prometheus in Docker, then serves the Python API in the foreground, and Prometheus scrapes that host API. Stop the native API with Ctrl+C or `make stop` before switching to containers.

The launcher only stops its own native API when it holds the requested port. It never stops unrelated processes. `API_PORT=<port>` selects another host port, and custom native storage ports need matching service endpoints.

Settings live in [`configs/inferops.yaml`](../configs/inferops.yaml); the dashboard editors save to the same file. Live changes can restart managed streams with the saved revision. Artifact, tensor and startup settings may need preparation and a restart. Registered sources keep their saved per-stream values, and the YAML supplies defaults for new streams. Precedence and migration: [central configuration](configuration.md).

More: [interfaces and ports](interfaces.md), [API reference](api.md), [project configuration](project-blocks.md), [project examples](../projects/README.md) and [verification scenarios](../tests/scenarios/README.md).

## PostgreSQL-backed tests

Use a dedicated test instance. The test helpers create and remove their own databases:

```bash
docker compose -f deploy/testing/compose.postgres.yaml up -d --wait
export INFEROPS_TEST_PG_URL=postgresql://postgres:inferops-test@127.0.0.1:55432/inferops_test
make test
docker compose -f deploy/testing/compose.postgres.yaml down   # remove the test service
```

## Repository layout

```text
configs/inferops.yaml   Central workspace configuration
.env.example, .env      Secrets and essential host overrides
run.sh, compose.yaml    Docker launcher and Compose entry point
runpod-start.sh         Native single-Pod launcher (RunPod)
Makefile                Development commands
inferops/               Python engine and packaged web UI
  domain/               Models, frames, tracking, fusion, incident and evidence policies
  application/          Execution, sessions, model pool, incidents, evidence, jobs
  ports/                Inference, media, persistence and extension contracts
  adapters/inbound/     HTTP/WebSocket + UI, CLI, source and broker workers
  adapters/outbound/    Inference backends, stages, media, projects, storage, observability
  bootstrap.py          Explicit resource construction
  sdk/                  Remote HTTP client
projects/               Self-contained projects: models, rules, plugins, weights
deploy/                 Dockerfiles, Compose overlays, RunPod, observability
scripts/                Setup, launch, benchmarks and verification
tests/                  Engine tests and browser/video scenarios
docs/                   Guides and references
research/               Training notebooks and sample media
```

The repository root holds `.git/`, `.gitignore`, the README, `pyproject.toml` and the launch configuration. The child
`inferops/` directory is the importable Python package. `.venv/` stores installed
dependencies; `.run/` stores runtime reports and verification output. Pytest and
Ruff caches live under `.run/cache/`. Test-only models use process-owned system
temporary directories and do not create a root `model_repository/`. Real model
weights remain under `models/model_repository/`.

`inferops.egg-info/` is generated installation metadata from the editable Python
install; it contains package information and entry points, not another source
implementation. Build output and tool caches are generated too and can be
recreated.

## Generated files and cleanup

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
