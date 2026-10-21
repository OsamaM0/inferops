# InferOps package structure

The [unified dashboard guide](unified-dashboard.md#structure) maps the new source registry, capacity, UI, Triton qualification and optional deployment components. The existing domain/application/ports/adapters boundaries remain the implementation structure.

For how these modules collaborate, read [Hexagonal architecture](architecture.md),
the [full lifecycle](lifecycle.md) and [Redpanda messaging](messaging.md).
The [documentation index](README.md) provides the complete reading path.

The package has six top-level directories: `domain`, `application`, `ports`,
`adapters`, `sdk` and `testing`. Its top-level Python files are `__init__.py`,
`public.py` and `bootstrap.py`.

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
│       ├── media/                MediaMTX, encoding, annotations and spool
│       ├── persistence/          PostgreSQL and memory repositories
│       ├── storage/              S3/MinIO and filesystem objects
│       ├── messaging/            Redpanda/in-process transport and delivery
│       ├── projects/             YAML, manifests, artifacts, packages and setup
│       └── observability/        Metrics, logging and tracing
├── sdk/                          Remote HTTP client
└── testing/                      Test configuration, fixtures and tiny models
```

## Public API and canonical imports

Continue importing `Engine`, `Model`, `InferenceSession`, `ModelSpec` and `Result`
from `inferops`. Both engine predictions and standalone handles now use
`application.models.ModelPool`. Project model handles share a backend even when
one uses the logical name and another the explicit variant name. Closing waits
for active inference before tearing down a backend.

The CLI entry point is `inferops`; the server entry point is
`inferops.adapters.inbound.http.main:app`. Project model plugins live under
`inferops.adapters.outbound.inference.plugins`, with extension contracts in
`inferops.ports.extensions`. See [adapter authoring](adapters.md).

## Dependency boundaries

Domain imports only domain code. Application imports domain/application/ports.
Ports import domain/ports. The [boundary tests](../tests/test_hexagonal_boundaries.py)
enforce these rules and the six-directory package layout. Importing the public
library creates no server clients and reads no deployment settings.
