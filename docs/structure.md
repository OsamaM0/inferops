# Consolidated InferOps structure

For how these modules collaborate, read [Hexagonal architecture](architecture.md),
the [full lifecycle](lifecycle.md) and [Redpanda messaging](messaging.md).
The [documentation index](README.md) provides the complete reading path.

The package now has six top-level directories: `domain`, `application`, `ports`,
`adapters`, `sdk` and `testing`. Its top-level Python files are `__init__.py`,
`public.py` and `bootstrap.py`. The former `core`, `registry`, `pipeline`,
`fusion`, `alerts`, `plugins`, `runtimes`, `api`, `cli`, `workers`, `obs`,
`compat` directories and `library.py` are no longer alternate module homes.

```text
inferops/
├── __init__.py                   Stable public exports
├── public.py                     Engine, Model and InferenceSession handles
├── bootstrap.py                  Explicit construction and resource wiring
├── domain/
│   ├── models.py                 Model contracts and runtime-mode normalization
│   ├── types.py                  Predictions, frames and stage data
│   ├── pipeline.py               Pipeline configuration contracts
│   ├── processing.py             Media frame and run-summary contracts
│   ├── tracking/                 Existing tracking algorithms/state
│   ├── fusion/                   Existing signals, windows, rules/state machines
│   ├── incidents.py              Severity, dedupe and cooldown decisions
│   ├── incident_spec.py          Incident policy schemas
│   ├── evidence.py               Evidence policy and capture states
│   └── errors.py                 Shared application/domain errors
├── application/
│   ├── projects.py               Activate compiled revisions, retain old resources
│   ├── models.py                 Shared backend pool, raw and image inference
│   ├── execution.py              Ordered stages and independent model groups
│   ├── sessions.py               Identity, timelines, serialization and temporal state
│   ├── incidents.py              Acceptance transaction, evidence requests, outbox
│   ├── evidence.py               Capture, recovery, upload, retry and deletion
│   ├── jobs.py                   Durable job lifecycle
│   ├── delivery.py               Outbox relay
│   └── engine.py                 Resource lifecycle and session admission
├── ports/                        Inference, media, persistence, projects,
│                                messaging/storage and extension contracts
├── adapters/
│   ├── inbound/
│   │   ├── http/                 FastAPI, WebSockets, packaged UI and configuration
│   │   ├── cli/                  Local command entry point
│   │   ├── workers/              Source and broker entry points
│   │   └── sources.py            Incremental files/cameras and bounded live input
│   └── outbound/
│       ├── inference/            ONNX Runtime, TensorRT EP, Triton and model plugins
│       ├── stages/               Existing processing/extension adapters and factory
│       ├── media/                Encoding, annotations, spool and DeepStream helper
│       ├── persistence/          PostgreSQL, SQLite and memory repositories
│       ├── storage/              S3/MinIO and filesystem objects
│       ├── messaging/            Redpanda/in-process transport and delivery
│       ├── projects/             YAML, manifests, artifacts, packages and setup
│       └── observability/        Metrics, logging and tracing
├── sdk/                          Remote HTTP client
└── testing/                      Test configuration, fixtures and tiny models
```

Names inside a layer describe distinct responsibilities. The model backends and
tracking/fusion algorithms retain their implementations. The change eliminates
parallel top-level homes and duplicate local inference lifecycle/prediction code.
Ordinary helpers are modules; the ports represent actual extension or external
capabilities.

## Public API and canonical imports

Continue importing `Engine`, `Model`, `InferenceSession`, `ModelSpec` and `Result`
from `inferops`. Both engine predictions and standalone handles now use
`application.models.ModelPool`. Project model handles share a backend even when
one uses the logical name and another the explicit variant name. Closing waits
for active inference before tearing down a backend.

Update internal imports and YAML plugin references in external project packages:

| Previous path | Canonical path |
| --- | --- |
| `inferops.core.types` / `core.errors` | `inferops.domain.types` / `domain.errors` |
| `inferops.registry.model_spec` | `inferops.domain.models` |
| `inferops.pipeline.spec` | `inferops.domain.pipeline` |
| `inferops.core.tracking` | `inferops.domain.tracking` |
| `inferops.fusion` | `inferops.domain.fusion` |
| `inferops.alerts.spec` | `inferops.domain.incident_spec` |
| `inferops.runtimes.base` | `inferops.ports.inference` |
| `inferops.plugins.base` / `alerts.sinks` | `inferops.ports.extensions` |
| `inferops.runtimes` / `plugins.builtin` | `inferops.adapters.outbound.inference` / `inference.plugins` |
| `inferops.pipeline.stage` / `pipeline.stages` | `inferops.adapters.outbound.stages.base` / `stages` |
| Project/configuration helpers in `core` and `registry` | `inferops.adapters.outbound.projects` |
| `inferops.api.main:app` | `inferops.adapters.inbound.http.main:app` |
| `python -m inferops.cli` | `python -m inferops.adapters.inbound.cli` |

The `inferops` CLI command and HTTP/SDK contracts stay the same. Packaging,
Docker/Make/launcher commands, project imports/configuration and documentation
use the new locations. No old internal-path shim trees were recreated. Release
rollback restores the matching project configuration/imports with its designated
wheel; durable database and object formats remain unchanged.

## Boundary and regression verification

Domain imports only domain code. Application imports domain/application/ports.
Ports import domain/ports. The boundary test no longer exempts old modules and
also enforces the six-directory package layout. Importing the public library
creates no server clients and reads no deployment settings.

Reports are under `.run/verification/structure-*`: core and driver test XML,
PostgreSQL/MinIO recovery/rollback tests, wheel/import checks, CLI/project validation
and full CPU/CUDA video execution with evidence capture enabled. See
[the architecture gallery](architecture/index.html), [current repository tree](architecture/repository-tree.md)
and [verification details](cleanup-verification.md).
