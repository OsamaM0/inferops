# Plan: Reusable Model-Agnostic ONNX Inference Platform ("InferOps")

> **Sequence diagram:** [`docs/sequence.puml`](docs/sequence.puml) — full PlantUML covering all 11 flows (main pipeline + every alternative). Render with PlantUML CLI, VS Code PlantUML extension, or paste into [planttext.com](https://www.planttext.com/).

---

## As-Built Architecture Summary (May 2026)

The platform is **fully implemented** and running. Below is a concise map of what is live, followed by the sequence diagram reference.

### Services (native stack — `make serve`)

| Service | Port(s) | Role |
|---|---|---|
| FastAPI API | :8080 | REST + WebSocket entry-point |
| Dashboard | :8090 | Static HTML + WS-fed live view |
| Triton Inference Server | :8000 HTTP / :8001 gRPC / :8002 metrics | GPU model serving (rfdetr_driver_behaviour_onnx) |
| MinIO | :9000 API / :9001 console | S3-compatible evidence store (JSON + JPEG + MP4) |
| Redpanda | :19092 Kafka / :19644 admin | Event transport (fusion events, alert topic, Kafka source/sink) |
| PostgreSQL | :5432 | Events, jobs, alerts tables |
| Prometheus + Grafana | scrape :8080/metrics | Pipeline latency, infer rate, alert rate |

### Active Pipelines

| Pipeline | Source | Stages | Sink | Use-case |
|---|---|---|---|---|
| **`driver_behaviour`** (main) | `source_rest` (REST upload) | decode_any → infer → fuse → alert | `sink_s3` (evidence_mode=both) | Single-image or video-file POST |
| **`driver_video`** | `source_rest` | decode_any → infer → fuse → alert | `sink_ws` (live.{stream_id}) | Real-time WS delivery, no S3 write |
| **Live RTSP stream** | `source_rtsp` (auto-swapped by /v1/streams) | same as driver_behaviour | sink_s3 or sink_ws | Continuous camera feed |
| **Live MJPEG stream** | `source_mjpeg` | same | same | HTTP MJPEG camera |
| **Kafka-sourced** | `source_kafka` (topic: ingest.frames) | decode → infer → fuse → alert | `sink_kafka` (inferops.events.driver) | Event-driven ingestion |

### Inference Backend Fallback Chain

```
rfdetr_driver_triton  (Triton gRPC, KIND_GPU)
    ↓  BackendError
rfdetr_driver_synth   backend=onnxruntime  (CPU/CUDA, asyncio thread)
    ↓  BackendError
rfdetr_driver_synth   backend=synthetic   (deterministic fake, no weights)
```

### Fusion Engine — `driver_fusion`

8 signals extracted per frame → rolling-time / EWMA windows → 7 rules → 6-state FSM → publish `inferops.events.driver` on Redpanda.

| Signal | Window | Derived | Rule threshold |
|---|---|---|---|
| `phone_score` | rolling_time 5 s | `phone = fraction_above(0.5)` | phone ≥ 0.4 → `phone_use` [medium] |
| `seatbelt_score` | rolling_time 5 s | `seatbelt = latest()` | seatbelt < 0.4 → `no_seatbelt` [medium] |
| `smoking_score` | rolling_time 5 s | `smoking = fraction_above(0.2)` | smoking ≥ 0.2 → `smoking` [medium] |
| `eating_score` | rolling_time 5 s | `eating = fraction_above(0.5)` | eating ≥ 0.5 → `eating` [low] |
| `drowsy_ear/eye` | EWMA α=0.3 | `drowsy_score = max(ear,eye)` | drowsy_score ≥ 0.5 → `drowsiness` [**high**] |
| `distracted_score` | EWMA α=0.3 | `distracted = latest()` | distracted ≥ 0.5 → `distraction` [low] |
| `hand_on_wheel` | rolling_time 5 s | `hand_off_wheel = 1 − fraction_above(0.5)` | hand_off_wheel ≥ 0.6 → `hands_off_wheel` [medium] |

**FSM states:** `NORMAL → DISTRACTED / DROWSY / UNSAFE → CRITICAL → RECOVERED → NORMAL`

### Alert Engine — `driver_alerts`

Dedupe strategy: `by_key_template "{type}:{entity_id}"` · window 60 s · cooldown 30 s.

Active sinks: **WebSocket** (`alerts.{entity_id}`) · **PostgreSQL** · **stdout**.
Available but optional: `webhook`, `kafka` (topic `inferops.alerts`), `slack` (stub), `plugin` (custom class).

### Transport Modes

| Mode | Config | Mechanism |
|---|---|---|
| **Redpanda** (default) | `INFEROPS_TRANSPORT_KIND=redpanda` | `aiokafka` AIOKafkaProducer/Consumer; max 64 KiB/msg |
| **InProc** (dev/test) | `INFEROPS_TRANSPORT_KIND=inproc` | `asyncio.Queue` (10 000 msg), no broker needed |

### Evidence Storage Modes (`sink_s3`)

| `evidence_mode` | What is stored |
|---|---|
| `json` | Detection metadata JSON only |
| `image` | JSON + annotated JPEG per alert frame |
| `video` | JSON + MP4 clip (pre+post frames around alert) |
| `both` (default) | JSON + JPEG + MP4 |

---

## TL;DR
Extract the generic plumbing already present in `DriverBehaviourDet` (Kafka bus, Redis frame cache, Triton client, ingest workers, evidence clipper, observability) into a new standalone monorepo called **InferOps**. Add two missing pieces it lacks today: (1) a **Model Registry** with declarative ONNX model specs, and (2) a **Pipeline Engine** that composes typed stages (source → decode → preprocess → infer → postprocess → aggregate → route → sink) from YAML. Driver-behaviour becomes one example app under `examples/driver-behaviour/` consisting solely of YAML configs + small Python plugins, with zero code in the platform core.

---

## 1. Executive Summary

**InferOps** is a config-driven inference platform where a developer onboards a new model by:
1. Dropping an ONNX file into `model_repository/<model>/1/model.onnx`
2. Writing `models/<model>.yaml` (inputs, outputs, classes, pre/post plugin refs)
3. Writing `pipelines/<task>.yaml` (stages, sources, sinks)
4. `inferops up` — the API, workers, and pipeline runners boot.

Core is task-agnostic: object detection, classification, segmentation, OCR, face analysis, video analytics, embedding extraction all map to the same `Stage` interface. Hot path is async Python + Triton (GPU) for inference; ONNX Runtime is the fallback for local/CPU dev. Events flow on Kafka; binary frames stay in Redis/S3 referenced by URI.

## 2. Architecture Decisions & Reasoning

| Decision | Choice | Reasoning |
|---|---|---|
| Inference backend | **Triton primary, onnxruntime fallback**, TRT path reserved | Triton already wired; gives dynamic batching, multi-instance, metrics. ORT for laptops/CI. |
| Transport | **Redpanda everywhere (dev + prod)** behind a `Transport` abstraction | Single broker tech to operate; Kafka-protocol compatible so we're not locked in. Pipeline/business code talks to `Transport`, never to `aiokafka` directly. |
| Binary payloads | **Redis (hot) + S3 (cold), refs on bus** | Keep transport messages small; `frame_ref` URIs already used. Hard rule: no raw frames on Redpanda. |
| Config | **Pydantic v2 + YAML, layered (platform → env → model → pipeline → tenant)** | Strong typing, JSON Schema export, env override for ops. |
| Pipeline engine | **DAG of typed `Stage` objects, async queues between** | Avoid coupling stages via Kafka unless cross-process. Same abstraction runs in-proc or distributed. |
| Extension model | **Python entry-points + import paths in YAML** | No registry server; plugins discovered via `pyproject.toml` or `module:Class` strings. |
| API | **FastAPI** | Already present; uvicorn workers; OpenAPI for SDK gen. |
| Storage | **Postgres for control plane, S3 for artifacts** | Already wired. Domain event tables become generic `events(payload JSONB)`. |
| Deployment | **Docker Compose (dev) → Helm (prod, GPU node selectors)** | Already wired; refactor for parametric service set. |
| Observability | **structlog + Prometheus + OpenTelemetry → OTel Collector** | Already wired. |
| Concurrency | **asyncio in workers, threadpool for CPU codec/preproc** | Avoid blocking the loop with JPEG/PyAV. |
| Language | **Python 3.11** | Match existing repo; sufficient for the I/O-bound hot path. |

**Explicitly rejected:**
- Ray Serve / KServe — heavy, opinionated, hides Triton.
- Hardcoded "detection" types in core — every task plugs in via stage config.
- Embedding business logic (state machines, alert types) into the platform — those become *user pipelines/plugins*.

## 3. Repository Structure (new repo `inferops/`)

```
inferops/
├── pyproject.toml
├── README.md
├── docker-compose.yml                 # dev: redpanda, redis, postgres, minio, triton, otel
├── docker-compose.gpu.yml             # override: triton GPU image, nvidia runtime
├── Makefile
├── .env.example
│
├── inferops/                          # the platform package (pip-installable)
│   ├── core/
│   │   ├── config.py                  # layered Pydantic settings loader
│   │   ├── types.py                   # Frame, Tensor, Detection, Classification, Embedding, Result
│   │   ├── errors.py
│   │   └── ids.py                     # correlation/request IDs
│   ├── registry/
│   │   ├── model_spec.py              # ModelSpec pydantic model
│   │   ├── loader.py                  # load+validate models/*.yaml
│   │   └── store.py                   # in-memory registry, hot-reload
│   ├── runtimes/
│   │   ├── base.py                    # ABC InferenceBackend
│   │   ├── onnxruntime_backend.py
│   │   ├── triton_backend.py
│   │   └── tensorrt_backend.py        # stub for future
│   ├── pipeline/
│   │   ├── stage.py                   # Stage ABC, StageContext, StageResult
│   │   ├── graph.py                   # DAG builder + topological scheduler
│   │   ├── runner.py                  # async pipeline executor (in-proc & worker modes)
│   │   ├── spec.py                    # PipelineSpec pydantic model
│   │   └── stages/                    # built-in stages
│   │       ├── source_rest.py
│   │       ├── source_rtsp.py
│   │       ├── source_mjpeg.py
│   │       ├── source_kafka.py        # uses Transport, named "kafka" for familiarity
│   │       ├── source_folder.py
│   │       ├── decode_image.py
│   │       ├── decode_video.py
│   │       ├── preprocess_generic.py  # resize+normalize+layout from config
│   │       ├── infer.py               # calls registry.backend.run()
│   │       ├── postprocess_passthrough.py
│   │       ├── aggregate_window.py
│   │       ├── fuse.py                # invokes inferops.fusion.engine.FusionEngine
│   │       ├── alert.py               # invokes inferops.alerts.engine.AlertEngine
│   │       ├── route_by.py
│   │       ├── sink_rest.py
│   │       ├── sink_ws.py
│   │       ├── sink_kafka.py
│   │       ├── sink_postgres.py
│   │       ├── sink_s3.py
│   │       ├── sink_webhook.py
│   │       └── sink_file.py
│   ├── adapters/                      # I/O subsystems used by stages
│   │   ├── transport/                 # broker-agnostic transport layer
│   │   │   ├── base.py                # Transport, Producer, Consumer, Message ABCs
│   │   │   ├── redpanda.py            # concrete impl (aiokafka under the hood)
│   │   │   └── inproc.py              # in-process async queue impl for tests/dev
│   │   ├── redis_client.py            # lifted from packages/common/redis
│   │   ├── frame_cache.py             # lifted from packages/common/frame_cache
│   │   ├── s3_client.py               # lifted from packages/storage/s3
│   │   └── postgres_dao.py            # generic events/jobs/models tables
│   ├── fusion/                        # GENERIC signal fusion engine
│   │   ├── engine.py                  # FusionEngine: ingests typed signals, runs windows + rules
│   │   ├── window.py                  # rolling windows, EWMA, fraction_above, mean, latest, derivative
│   │   ├── signals.py                 # Signal, SignalSource, SignalRegistry
│   │   ├── rules.py                   # Rule, RuleSet, expression evaluator (safe subset)
│   │   ├── state_machine.py           # generic FSM: states, transitions, hysteresis, dwell
│   │   └── spec.py                    # FusionSpec / WindowSpec / RuleSpec / StateMachineSpec pydantic
│   ├── alerts/                        # GENERIC alerting engine
│   │   ├── engine.py                  # AlertEngine: dedupe, cooldown, severity mapping, fan-out
│   │   ├── severity.py                # SeverityMap (configurable levels + colors + escalation)
│   │   ├── dedupe.py                  # dedupe_key strategies (by type, by entity, by window)
│   │   ├── cooldown.py                # per-key + per-type + global cooldown
│   │   ├── sinks/                     # alert delivery sinks
│   │   │   ├── base.py                # AlertSink ABC
│   │   │   ├── webhook.py
│   │   │   ├── email.py               # stub
│   │   │   ├── slack.py               # stub
│   │   │   ├── kafka.py               # via Transport
│   │   │   ├── postgres.py
│   │   │   ├── ws.py
│   │   │   └── stdout.py
│   │   └── spec.py                    # AlertSpec / SinkSpec pydantic
│   ├── plugins/
│   │   ├── base.py                    # Preprocessor, Postprocessor, Aggregator ABCs
│   │   └── builtin/
│   │       ├── image_letterbox.py
│   │       ├── imagenet_normalize.py
│   │       ├── softmax_topk.py
│   │       ├── detr_decode.py
│   │       ├── yolo_decode.py
│   │       └── ocr_ctc_decode.py
│   ├── api/
│   │   ├── main.py                    # FastAPI app factory
│   │   ├── deps.py
│   │   └── routes/
│   │       ├── health.py
│   │       ├── models.py              # GET/POST /v1/models  (registry CRUD)
│   │       ├── pipelines.py           # GET/POST /v1/pipelines
│   │       ├── infer.py               # POST /v1/infer/{pipeline}  (sync)
│   │       ├── jobs.py                # POST /v1/jobs/{pipeline} async
│   │       ├── streams.py             # POST /v1/streams  start RTSP/MJPEG
│   │       └── ws_live.py             # WS /v1/ws/{stream_id}
│   ├── workers/                       # long-running services
│   │   ├── ingest_worker.py           # generic source→bus
│   │   ├── inference_worker.py        # bus→infer→bus
│   │   ├── pipeline_worker.py         # runs a full pipeline by name
│   │   └── sink_worker.py             # bus→sink
│   ├── obs/
│   │   ├── logging.py
│   │   ├── metrics.py
│   │   └── tracing.py
│   ├── cli/
│   │   └── __main__.py                # `inferops` CLI (typer)
│   └── sdk/
│       └── client.py                  # Python client for the REST API
│
├── model_repository/                  # Triton-compatible layout
│   └── .gitkeep
│
├── configs/
│   ├── platform.yaml                  # global defaults (incl. transport block)
│   ├── models/                        # *.yaml ModelSpec files
│   ├── pipelines/                     # *.yaml PipelineSpec files
│   ├── fusion/                        # *.yaml FusionSpec files (windows + rules + FSM)
│   └── alerts/                        # *.yaml AlertSpec files (severity, dedupe, sinks)
│
├── deploy/
│   ├── docker/
│   │   ├── api.Dockerfile
│   │   ├── worker.Dockerfile
│   │   └── triton.Dockerfile
│   └── helm/inferops/                 # parametric chart (services from values)
│
├── observability/                     # otel-collector, prometheus, grafana (lifted)
├── examples/
│   ├── classification-resnet50/
│   ├── detection-yolov8/
│   └── driver-behaviour/              # ← current project re-implemented as config + plugins
│       ├── README.md
│       ├── configs/
│       │   ├── models/rfdetr.yaml
│       │   ├── models/face_mesh.yaml
│       │   ├── pipelines/driver_video.yaml
│       │   └── tenants/default.yaml   # thresholds, alert mapping
│       └── plugins/
│           ├── postprocess_rfdetr.py
│           ├── face_features.py        # EAR + head pose (lifted from face_worker)
│           ├── behavior_state_machine.py
│           └── behavior_aggregator.py
├── docs/
│   ├── architecture.md
│   ├── onboarding.md
│   ├── model_spec.md
│   ├── pipeline_spec.md
│   └── plugins.md
└── tests/
    ├── unit/
    ├── contract/                      # validates every YAML in configs/ + examples/
    ├── integration/                   # docker-compose up + end-to-end one image
    └── benchmark/
```

## 4. Core Abstractions & Interfaces

All interfaces are abstract base classes living in `inferops/`:

- `InferenceBackend` (`runtimes/base.py`)
  - `load(spec: ModelSpec) -> None`
  - `async infer(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]`
  - `async warmup() -> None`
  - `close() -> None`
- `Preprocessor` (`plugins/base.py`) — `async __call__(frame: Frame, spec: ModelSpec) -> dict[str, np.ndarray]`
- `Postprocessor` — `async __call__(raw: dict[str, np.ndarray], frame: Frame, spec: ModelSpec) -> Result`
- `Aggregator` — `async __call__(results: Sequence[Result], ctx: StageContext) -> Optional[Event]`
- `Stage` (`pipeline/stage.py`)
  - `name: str`, `inputs: list[str]`, `outputs: list[str]`
  - `async setup(ctx: StageContext) -> None`
  - `async process(item: StageItem) -> Optional[StageItem] | AsyncIterator[StageItem]`
  - `async teardown() -> None`
- `SourceStage` / `SinkStage` — marker subclasses with `async run()` (push) instead of pull.
- `Result` is a tagged union: `Detections | Classifications | Embedding | Segmentation | OCRText | Raw`.

### 4.a Transport abstraction (Redpanda-backed)

```
class Message(BaseModel):
    topic: str
    key: bytes | None
    value: bytes                       # serialized envelope (orjson of StageItem)
    headers: dict[str, bytes]          # carries trace context + content_type + schema_version
    partition: int | None = None
    offset: int | None = None
    ts_ms: int | None = None

class Producer(Protocol):
    async def send(self, topic: str, value: bytes, *, key: bytes | None = None,
                   headers: dict[str, bytes] | None = None) -> None: ...
    async def flush(self) -> None: ...
    async def close(self) -> None: ...

class Consumer(Protocol):
    async def subscribe(self, topics: list[str], group: str) -> None: ...
    def stream(self) -> AsyncIterator[Message]: ...
    async def commit(self, msg: Message) -> None: ...
    async def close(self) -> None: ...

class Transport(Protocol):
    async def producer(self, name: str) -> Producer: ...
    async def consumer(self, name: str) -> Consumer: ...
    async def admin_ensure_topic(self, topic: str, *, partitions: int, retention_ms: int) -> None: ...
    async def health(self) -> bool: ...
```

- `RedpandaTransport` is the production implementation (Kafka wire protocol via `aiokafka`).
- `InProcTransport` is an in-process implementation backed by `asyncio.Queue`s, used by tests and the in-proc pipeline mode. Same `Transport` contract.
- Pipeline stages, fusion engine, alert engine, and workers depend **only** on the `Transport` ABC. They never import `aiokafka` or any Redpanda-specific symbol.
- The Transport layer is responsible for: serialization framing, header injection (W3C traceparent, `x-schema-version`, `x-content-type`), idempotent producer config, consumer-group naming convention, and DLQ topic routing.
- **Binary rule:** payloads larger than `transport.max_value_bytes` (default 64 KiB) MUST be persisted to Redis or S3 by the producing stage; only the reference URI is published. The Transport layer rejects oversize values with a clear error.
- Topic naming convention: `inferops.<pipeline>.<edge>` for pipeline edges; `inferops.events.<domain>` for fusion outputs; `inferops.alerts.<channel>` for alert outputs; `inferops.dlq.<topic>` for dead-letter.

### 4.b Fusion engine (generic, configurable)

The fusion engine consumes typed **signals**, maintains **windows** per entity (e.g. `stream_id`), evaluates **rules**, drives a **state machine**, and emits **events**. No domain semantics in core.

```
class Signal(BaseModel):
    name: str                          # e.g. "phone_score", "ear", "speed_mps"
    entity_id: str                     # window key (stream_id, vehicle_id, ...)
    ts_ms: int
    value: float | bool
    attrs: dict[str, Any] = {}

class WindowSpec(BaseModel):
    kind: Literal["rolling_time", "rolling_count", "ewma"]
    duration_ms: int | None = None
    count: int | None = None
    alpha: float | None = None         # ewma
    reducers: list[Literal["latest","mean","max","min","fraction_above","derivative","slope"]]
    reducer_params: dict[str, Any] = {}  # e.g. {"fraction_above": {"threshold": 0.4}}

class RuleSpec(BaseModel):
    name: str
    when: str                          # safe expression DSL: "phone.fraction_above(0.4) > 0.5"
    emit: dict[str, Any] | None = None # signal/event payload to publish
    cooldown_ms: int = 0
    severity: str | None = None        # resolved by AlertEngine SeverityMap

class TransitionSpec(BaseModel):
    from_state: str
    to_state: str
    when: str                          # expression over current signals + state dwell time
    min_dwell_ms: int = 0              # hysteresis

class StateMachineSpec(BaseModel):
    initial: str
    states: list[str]
    transitions: list[TransitionSpec]
    on_enter: dict[str, list[str]] = {}  # state -> list of action names (emit_event, raise_alert)
    on_exit:  dict[str, list[str]] = {}

class FusionSpec(BaseModel):
    name: str
    entity_key: str = "stream_id"
    signals: list[str]                 # signals this fusion subscribes to
    windows: dict[str, WindowSpec]     # signal_name -> window
    derived: dict[str, str] = {}       # name -> expression (e.g. "drowsy = ear.mean() < 0.21")
    rules: list[RuleSpec] = []
    state_machine: StateMachineSpec | None = None
    outputs:
        events_topic: str              # via Transport
        alerts_topic: str | None = None
```

- **Expression DSL**: safe subset (no `__`, no `import`, no attribute access beyond whitelisted methods). Implemented with `asteval` or hand-rolled AST walker over the signals/windows namespace.
- **Plugins**: when YAML isn't expressive enough, a rule can reference `evaluator: "module:Class"` implementing `Evaluator.evaluate(ctx) -> EvalResult`. Same for custom reducers.
- **Stateless core**: window state lives in Redis (key: `fusion:{name}:{entity_id}:{signal}`) so fusion workers can be scaled and restarted.
- Built-in stage `fuse` wraps `FusionEngine` so a pipeline YAML can pull fusion into the DAG; alternatively a dedicated `fusion_worker` consumes upstream topics directly.

### 4.c Alert engine (generic, configurable)

Consumes alert candidates from the fusion engine (or any stage), applies severity mapping, dedupe and cooldown, then fans out to sinks. Nothing in here mentions "phone", "drowsiness", "vehicle", etc.

```
class AlertCandidate(BaseModel):
    type: str                          # free-form, defined by config
    entity_id: str
    ts_ms: int
    payload: dict[str, Any]
    severity_hint: str | None = None
    dedupe_key: str | None = None      # if None, computed from DedupeSpec

class SeverityLevel(BaseModel):
    name: str                          # "info"|"low"|"medium"|"high"|"critical" (configurable)
    rank: int                          # 0..N
    color: str | None = None
    notify_channels: list[str] = []    # sink names

class SeverityMapSpec(BaseModel):
    levels: list[SeverityLevel]
    mapping: dict[str, str] = {}       # alert.type -> level name
    rules: list[RuleSpec] = []         # expression-based dynamic severity

class DedupeSpec(BaseModel):
    strategy: Literal["by_type_entity", "by_payload_hash", "by_key_template", "plugin"]
    key_template: str | None = None    # e.g. "{type}:{entity_id}:{payload.zone}"
    plugin: str | None = None
    window_ms: int = 60_000

class CooldownSpec(BaseModel):
    per_dedupe_key_ms: int = 30_000
    per_type_ms: int = 0
    global_ms: int = 0

class SinkSpec(BaseModel):
    name: str
    kind: Literal["webhook","email","slack","kafka","postgres","ws","stdout","plugin"]
    params: dict[str, Any] = {}
    min_severity: str | None = None    # gate per sink

class AlertSpec(BaseModel):
    name: str
    severity: SeverityMapSpec
    dedupe: DedupeSpec
    cooldown: CooldownSpec
    sinks: list[SinkSpec]
    persistence:
        enabled: bool = true
        table: str = "alerts"
```

- Dedupe + cooldown state stored in Redis (`alerts:dedupe:{key}` with TTL, `alerts:cooldown:{type}` etc.).
- Each sink runs in an isolated task; one sink failure cannot block the others (bounded retry + DLQ).
- Sinks are plugin-pluggable: `kind: plugin` + `params.target: "module:Class"`.
- Persistence layer writes the resolved `Alert` row (generic schema: `id, type, entity_id, severity, payload JSONB, dedupe_key, status, ts`) — same table works for every domain.

### 4.d Example: driver-behaviour expressed in these primitives

`examples/driver-behaviour/configs/fusion/driver.yaml` (excerpt):
```
name: driver_fusion
entity_key: stream_id
signals: [phone_score, seatbelt_score, smoking_score, eating_score, ear, yaw_deg, pitch_deg, hand_on_wheel]
windows:
  phone_score:    { kind: rolling_time, duration_ms: 5000, reducers: [fraction_above], reducer_params: { fraction_above: { threshold: 0.4 } } }
  ear:            { kind: ewma, alpha: 0.3, reducers: [latest, mean] }
  yaw_deg:        { kind: rolling_time, duration_ms: 2000, reducers: [mean] }
derived:
  drowsy:  "ear.mean() < 0.21"
  distracted: "abs(yaw_deg.mean()) > 30 or abs(pitch_deg.mean()) > 20"
state_machine:
  initial: NORMAL
  states: [NORMAL, DISTRACTED, DROWSY, UNSAFE, CRITICAL, RECOVERED]
  transitions:
    - { from_state: NORMAL,     to_state: DISTRACTED, when: "distracted",                                min_dwell_ms: 1500 }
    - { from_state: NORMAL,     to_state: DROWSY,     when: "drowsy",                                    min_dwell_ms: 1500 }
    - { from_state: DROWSY,     to_state: CRITICAL,   when: "drowsy and phone_score.fraction_above(0.4) > 0.5", min_dwell_ms: 1000 }
rules:
  - { name: phone_use,    when: "phone_score.fraction_above(0.4) > 0.5",      severity: high,     cooldown_ms: 30000, emit: { type: PHONE_USE } }
  - { name: no_seatbelt,  when: "seatbelt_score.fraction_above(0.4) > 0.6",   severity: medium,   cooldown_ms: 60000, emit: { type: NO_SEATBELT } }
outputs:
  events_topic: inferops.events.driver
  alerts_topic: inferops.alerts.driver
```

`examples/driver-behaviour/configs/alerts/driver.yaml` (excerpt):
```
name: driver_alerts
severity:
  levels:
    - { name: low,      rank: 1, color: "#9cf" }
    - { name: medium,   rank: 2, color: "#fc6", notify_channels: [webhook] }
    - { name: high,     rank: 3, color: "#f63", notify_channels: [webhook, ws] }
    - { name: critical, rank: 4, color: "#f00", notify_channels: [webhook, ws, slack] }
  mapping:
    PHONE_USE: high
    NO_SEATBELT: medium
    DROWSINESS: critical
dedupe:
  strategy: by_key_template
  key_template: "{type}:{entity_id}"
  window_ms: 60000
cooldown:
  per_dedupe_key_ms: 30000
sinks:
  - { name: webhook, kind: webhook, params: { url: "${ALERT_WEBHOOK_URL}" }, min_severity: medium }
  - { name: ws,      kind: ws,      params: { channel: "alerts.{entity_id}" } }
  - { name: slack,   kind: slack,   params: { webhook_url: "${SLACK_URL}" }, min_severity: critical }
  - { name: pg,      kind: postgres, params: { table: alerts } }
persistence: { enabled: true, table: alerts }
```

The driver-behaviour example contributes **zero Python code** for fusion/alerting — only YAML plus a tiny postprocess plugin that emits the named signals.

The DAG runner connects stages with bounded `asyncio.Queue`s; back-pressure is built in. Cross-process pipelines use Kafka topics as queues with the same `StageItem` envelope.

## 5. Configuration Schema Design

Five layers, merged in order (later overrides earlier):

1. `inferops/core/config.py` defaults
2. `configs/platform.yaml`
3. Environment variables (`INFEROPS_*`)
4. `configs/models/*.yaml` + `configs/pipelines/*.yaml` (selected by request)
5. `configs/tenants/<id>.yaml` (per-request override)

### ModelSpec (Pydantic)
```
name, version, task (enum: classification|detection|segmentation|ocr|embedding|landmarks|custom),
backend (onnxruntime|triton|tensorrt),
artifact: { path | triton_model_name | url, sha256 },
inputs: [ { name, shape (with -1 for dynamic), dtype, layout (NCHW|NHWC), color (RGB|BGR|GRAY) } ],
outputs: [ { name, role (logits|boxes|scores|embedding|mask|text), dtype } ],
labels: [str] | { from_file: path },
preprocess: { plugin: "module:Class", params: {...} },
postprocess: { plugin: "module:Class", params: { conf_threshold, nms_iou, top_k, ... } },
batching: { enabled, max_batch_size, max_delay_ms },
device: { preferred (gpu|cpu), gpu_id },
warmup: { iterations, dummy_input_seed }
```

### PipelineSpec
```
name, version,
source: { stage: "source_rtsp", params: { url_from: request|env|config } },
stages: [ { name, stage: "decode_image", params: {} }, { name, stage: "infer", params: { model: "rfdetr" } }, ... ],
edges: [ ["source", "decode"], ["decode", "preprocess"], ... ],
sinks: [ { stage: "sink_kafka", params: { topic: "events.behavior" } } ],
limits: { max_concurrency, queue_size, timeout_ms },
observability: { tags: { app: driver-behaviour } }
```

## 6. Example YAML Configs

### 6.1 `configs/models/resnet50.yaml` (classification)
```
name: resnet50
version: 1
task: classification
backend: onnxruntime
artifact: { path: model_repository/resnet50/1/model.onnx }
inputs:
  - { name: input, shape: [-1, 3, 224, 224], dtype: float32, layout: NCHW, color: RGB }
outputs:
  - { name: logits, role: logits, dtype: float32 }
labels: { from_file: configs/labels/imagenet1k.txt }
preprocess:
  plugin: inferops.plugins.builtin.imagenet_normalize:ImageNetNormalize
  params: { resize: 256, crop: 224 }
postprocess:
  plugin: inferops.plugins.builtin.softmax_topk:SoftmaxTopK
  params: { top_k: 5 }
batching: { enabled: true, max_batch_size: 32, max_delay_ms: 5 }
device: { preferred: gpu }
warmup: { iterations: 2 }
```

### 6.2 `configs/models/yolov8.yaml` (detection)
```
name: yolov8n
version: 1
task: detection
backend: triton
artifact: { triton_model_name: yolov8n_onnx }
inputs:
  - { name: images, shape: [-1, 3, 640, 640], dtype: float32, layout: NCHW, color: RGB }
outputs:
  - { name: output0, role: boxes, dtype: float32 }
labels: { from_file: configs/labels/coco.txt }
preprocess:
  plugin: inferops.plugins.builtin.image_letterbox:Letterbox
  params: { size: 640, pad_value: 114 }
postprocess:
  plugin: inferops.plugins.builtin.yolo_decode:YoloDecode
  params: { conf_threshold: 0.25, nms_iou: 0.45 }
batching: { enabled: true, max_batch_size: 16, max_delay_ms: 8 }
```

### 6.3 `configs/pipelines/video_detection.yaml`
```
name: video_detection
version: 1
source: { stage: source_rtsp, params: { adaptive_fps: { base: 5, escalated: 15 } } }
stages:
  - { name: decode,    stage: decode_video,        params: { encoding: jpeg } }
  - { name: cache,     stage: sink_redis_frame,    params: { ttl_s: 30 } }
  - { name: infer,     stage: infer,               params: { model: yolov8n } }
  - { name: window,    stage: aggregate_window,    params: { window_ms: 5000, ewma_alpha: 0.3 } }
sinks:
  - { stage: sink_kafka,    params: { topic: events.detections } }
  - { stage: sink_ws,       params: { channel: live.{stream_id} } }
  - { stage: sink_postgres, params: { table: events } }
limits: { max_concurrency: 4, queue_size: 32, timeout_ms: 1000 }
```

### 6.4 `examples/driver-behaviour/configs/pipelines/driver_video.yaml`
Same shape, with extra stages referencing the example's plugins (`behavior_aggregator`, `behavior_state_machine`) and adding evidence/alert sinks.

## 7. Runtime Strategy

| Mode | Inference backend | Compose file | Notes |
|---|---|---|---|
| Local CPU dev | `onnxruntime` (CPUExecutionProvider) | `docker-compose.yml` | Triton optional; smallest footprint; supports IDE debugging. |
| Local single-GPU | `triton` CPU image OR `tritonserver:24.xx-py3` with `--gpus all` | `docker-compose.yml` + `docker-compose.gpu.yml` | Mount `model_repository/`, dynamic batching enabled. |
| Production cluster | `triton` GPU image on GPU node pool; workers on CPU pool | `deploy/helm/inferops/` | `values.yaml` declares services; GPU `nodeSelector`/`tolerations`; HPA on Kafka lag + GPU util. |
| CI smoke | `onnxruntime` + tiny test ONNX (1 conv) | dedicated compose with mocks | Runs end-to-end in <30s. |

`scripts/check_env.sh` validates: NVIDIA driver, CUDA runtime, `nvidia-container-toolkit`, GPU memory, Docker, free ports, ONNX model presence.

## 8. API Design

All routes under `/v1`:

| Method | Path | Purpose |
|---|---|---|
| GET | `/healthz`, `/readyz` | liveness/readiness |
| GET | `/models` | list registered models |
| GET | `/models/{name}` | model spec + status |
| POST | `/models/reload` | hot-reload registry from disk |
| GET | `/pipelines` | list pipelines |
| GET | `/pipelines/{name}` | spec |
| POST | `/infer/{pipeline}` | synchronous inference (multipart image/JSON tensor); returns Result |
| POST | `/jobs/{pipeline}` | async; returns job_id, results on Kafka/WS |
| GET | `/jobs/{id}` | poll job status/result |
| POST | `/streams` | start a live stream pipeline (RTSP/MJPEG) |
| DELETE | `/streams/{id}` | stop |
| WS | `/ws/{stream_id}` | live results (frame_ref + annotations) |

OpenAPI auto-generated; Python SDK in `inferops/sdk/client.py` exposes `Client.infer(pipeline, image)`, `Client.start_stream(...)`.

CLI (`inferops` via typer):
- `inferops init` — scaffold project
- `inferops validate` — JSON-Schema-validate every config
- `inferops models list|show|reload`
- `inferops pipelines list|show|run <name> <input>`
- `inferops bench <pipeline> --duration 60s`
- `inferops up|down` — wraps compose

## 9. Worker / Service Design

Workers are thin: each loads a `PipelineSpec` slice and a Kafka consumer group. Same Python image, different entrypoints:

- `ingest_worker` — runs Source stages → Kafka.
- `inference_worker` — runs Preprocess+Infer+Postprocess for a (model, topic) pair. Horizontally scaled per model.
- `pipeline_worker` — runs a full DAG in-proc; ideal when no inter-stage Kafka topic is needed.
- `sink_worker` — runs Sink stages from a topic. One per sink class for isolation.
- `api` — FastAPI; synchronous infer path uses in-proc pipeline runner (no Kafka round-trip).

Scaling: HPA on Kafka consumer-group lag (KEDA) for ingest/sink, GPU util for inference workers.

## 10. Observability Plan

- **Logging:** `structlog` JSON; every record carries `trace_id`, `span_id`, `pipeline`, `model`, `stream_id`, `request_id`.
- **Metrics (Prometheus):**
  - `inferops_stage_latency_seconds{pipeline,stage}` (histogram)
  - `inferops_infer_latency_seconds{model}` (histogram)
  - `inferops_infer_batch_size{model}` (histogram)
  - `inferops_queue_depth{pipeline,edge}`
  - `inferops_kafka_lag{topic,group}`
  - `inferops_gpu_util{gpu_id}` (DCGM exporter)
  - `inferops_errors_total{stage,kind}`
- **Tracing:** OpenTelemetry spans per stage; Kafka producer injects W3C traceparent into headers; consumer continues span.
- **Dashboards:** Grafana `inferops-overview` (throughput, latency p50/p95/p99, error rate, GPU), `inferops-pipeline/<name>` (per-stage), `inferops-model/<name>` (per-model).

## 11. Testing Strategy

- **Unit** — pure functions: config merger, DAG topo-sort, every builtin plugin with golden tensors.
- **Contract** — load every YAML in `configs/` and `examples/*/configs/`, assert schema validates, assert all referenced plugins import.
- **Integration** — `docker-compose up`; POST a fixture image to `/v1/infer/classification_resnet50`, assert top-1 label. RTSP smoke against `mediamtx`.
- **Benchmark** — `inferops bench` produces JSON: throughput (FPS), latency percentiles, GPU util.
- **Parity** — `triton/scripts/verify_parity.py` style: ORT vs Triton output max-abs-diff under tolerance.
- **CI** — GitHub Actions matrix (py3.11, CPU-only), runs unit + contract + ORT integration on every PR; nightly GPU job runs Triton integration + bench.

## 12. Implementation Plan — Phases

### Phase 0 — Scaffold (foundation)
- Create new repo `inferops/` (separate from current workspace, or as `platform/` subtree — see Decisions).
- `pyproject.toml`, `Makefile`, `.env.example`, base CI.
- Build `inferops/adapters/transport/` (`base.py` + `redpanda.py` + `inproc.py`) from scratch, with `aiokafka` as Redpanda's wire driver. Port message envelope, header conventions, DLQ helpers from `packages/common/kafka/bus.py` but expose **only** the `Transport`/`Producer`/`Consumer` interfaces to callers.
- Lift `packages/common/{obs,redis,frame_cache}` → `inferops/adapters/` and `inferops/obs/` with zero domain references.
- Build `inferops/core/{config,types,errors,ids}.py`.

### Phase 1 — Model Registry + Runtimes (parallel with Phase 2)
- Implement `ModelSpec` Pydantic model + loader + in-memory store + hot-reload.
- Implement `OnnxRuntimeBackend` (CPU+CUDA EP), `TritonBackend` (lift from `triton_client.py`), TensorRT stub.
- Plugin loader resolving `"module:Class"` strings.
- Unit + contract tests.

### Phase 2 — Pipeline Engine (parallel with Phase 1)
- `Stage` ABC + `StageItem` envelope + `StageContext`.
- DAG builder, topo-sort, async runner with bounded queues + back-pressure + cancellation.
- Built-in stages: `source_rest`, `decode_image`, `preprocess_generic`, `infer`, `postprocess_passthrough`, `sink_rest`. Just enough for synchronous image-in / JSON-out.
- Built-in plugins: `imagenet_normalize`, `softmax_topk`, `image_letterbox`.
- Unit tests for DAG; integration test: classify a JPEG via `onnxruntime`.

### Phase 3 — API + CLI
- FastAPI app factory, `health`, `models`, `pipelines`, `infer` routes.
- `inferops` CLI with `validate`, `models`, `pipelines`, `up`, `bench`.
- SDK client.
- WebSocket route + `sink_ws`.

### Phase 4 — Streaming + Distributed
- Lift ingest workers (`rtsp_worker`, `mjpeg_worker`, `file_injector`, `ring_buffer`, `stream_lease`) into `inferops/pipeline/stages/source_*.py`, talking through `Transport` only.
- Lift evidence clipper into a built-in stage `sink_clip_evidence`.
- Distributed mode: `inference_worker` reads/writes Redpanda topics defined by pipeline edges, using `Transport`.
- `sink_kafka` (Transport-backed), `sink_postgres`, `sink_s3`, `sink_webhook`.
- Postgres DAO refactored to generic `events(id, pipeline, payload JSONB, ts)` + `jobs` + `models` + `alerts` tables.
- Enforce binary-payload rule: producer wrapper rejects values > `max_value_bytes`; large blobs auto-staged to Redis/S3 by helper.

### Phase 5 — Fusion & Alert engines (generic)
- Implement `inferops/fusion/`: `window.py` (lift math from `packages/fusion/temporal_window.py`), `signals.py`, `rules.py` with safe-expression evaluator, generic `state_machine.py` (lift core FSM mechanics from `packages/fusion/state_machine.py`, drop the driver enum), `engine.py`, `spec.py`.
- Implement `inferops/alerts/`: `severity.py`, `dedupe.py`, `cooldown.py`, `engine.py`, `spec.py`, plus `sinks/{base,webhook,kafka,postgres,ws,stdout}.py` (lift from `packages/alerts/sinks.py`); `email`/`slack` as stubs.
- Built-in stages `fuse` and `alert` so YAML pipelines can wire them inline; standalone `fusion_worker` and `alert_worker` for distributed deployments.
- Contract tests: load every `configs/fusion/*.yaml` and `configs/alerts/*.yaml`; unit tests for window math, FSM, dedupe, cooldown, severity mapping, safe-expression evaluator (must reject `__`, `import`, attribute escapes).
- Redis-backed window/dedupe/cooldown state so engines are stateless and horizontally scalable.

### Phase 6 — Examples
- `examples/classification-resnet50/` (smallest possible, no fusion/alerts).
- `examples/detection-yolov8/` (fuse zone-occupancy signals + alert on intrusion — demonstrates fusion/alerts without driver context).
- `examples/driver-behaviour/`: port current behavior to YAML configs (`configs/fusion/driver.yaml`, `configs/alerts/driver.yaml`) plus minimal plugins (`postprocess_rfdetr.py`, `face_features.py`). **Delete** original `packages/fusion` and `packages/alerts` once parity passes.

### Phase 7 — Deployment
- Parametric Helm chart `deploy/helm/inferops/` where `values.yaml` enumerates services (api, inference-worker per model, ingest-worker, fusion-worker, alert-worker, sink-worker); replicas, resources, node selectors, GPU per service.
- Redpanda chart as dependency (or external).
- KEDA ScaledObject for Redpanda consumer-group-lag autoscaling.
- `deploy/docker/*.Dockerfile` for api, worker, triton (with model_repository volume).

### Phase 8 — Hardening
- Benchmark harness + nightly CI.
- Hot-reload model registry without restart.
- Auth on API (API keys + per-tenant scopes).
- Rate-limit & quotas per tenant.
- TensorRT backend (real, not stub).
- Documentation pass (`docs/architecture.md`, `docs/onboarding.md`).

## 13. MVP vs Future

**MVP (Phases 0–3 + minimal Phase 5):**
- Single-process API
- `onnxruntime` backend + `triton` backend
- Sync `/v1/infer/{pipeline}` for image inputs
- ResNet50 classification example end-to-end
- Pipeline YAML validation + CLI
- Docker Compose dev stack
- Unit + contract tests + one integration test

**Future (Phases 4–7):**
- Streaming (RTSP/MJPEG) via distributed Kafka workers
- Driver-behaviour example with state machine + evidence
- Helm + KEDA autoscaling
- TensorRT backend
- Multi-tenancy + auth + quotas
- Hot-reload + model A/B
- gRPC API in addition to REST

## 14. Risks & Tradeoffs

| Risk | Mitigation |
|---|---|
| Pipeline-as-config becomes a bespoke DSL nobody understands | Keep stage vocabulary tiny (~20 stages), document every stage, ship 3 examples, validate with JSON Schema in CI. |
| Plugin loader runs arbitrary code from YAML | Plugins must be importable from installed packages; CLI emits warning when loading from outside the venv; sign tenant configs in prod. |
| Async + GPU underutilization | Inference stage batches via dynamic batching window (Triton native; ORT-side `BatchedRunner`); benchmarks gate releases. |
| Lifting code from `DriverBehaviourDet` while it's still running | Do extraction in new repo; copy, don't move, until the example passes parity tests; then delete source. |
| Postgres generic event table loses query ergonomics | Add optional generated columns (`payload->>'severity'`) + per-pipeline views. |
| Backwards compatibility of YAML specs | Version every spec (`version: 1`); add `inferops migrate` for schema bumps. |
| Two backends to maintain (ORT + Triton) | Cover both with the same contract tests; parity check script. |
| Kafka required for streaming dev | Provide in-proc pipeline mode (`pipeline_worker`) that needs no Kafka. |

## 15. Starter Code Skeleton

The skeleton covers every file in §3 with empty `__init__.py` + stubs for the abstract classes and one working classification example. Concretely, the implementation handoff will produce:

- All `__init__.py` files + module docstrings
- `inferops/core/{config,types,errors,ids}.py` — concrete (not stubs)
- `inferops/registry/{model_spec,loader,store}.py` — concrete
- `inferops/runtimes/{base,onnxruntime_backend,triton_backend}.py` — concrete; `tensorrt_backend.py` raises `NotImplementedError`
- `inferops/pipeline/{stage,graph,runner,spec}.py` — concrete
- `inferops/pipeline/stages/{source_rest,decode_image,preprocess_generic,infer,postprocess_passthrough,sink_rest,sink_ws,sink_kafka}.py` — concrete
- Remaining stages — stubs raising `NotImplementedError` with TODO comment referencing phase
- `inferops/plugins/base.py` + `imagenet_normalize.py` + `softmax_topk.py` + `image_letterbox.py` — concrete
- Other plugins — stubs
- `inferops/api/main.py` + `routes/{health,models,pipelines,infer}.py` — concrete
- Other routes — stubs
- `inferops/workers/*.py` — stubs with `if __name__ == "__main__"` entrypoints
- `inferops/cli/__main__.py` — concrete for `init`, `validate`, `up`, `infer`
- `configs/platform.yaml`, `configs/models/resnet50.yaml`, `configs/pipelines/classify_image.yaml` — concrete
- `docker-compose.yml`, `Makefile`, `pyproject.toml`, `.env.example` — concrete
- `examples/classification-resnet50/README.md` — concrete onboarding walkthrough
- `examples/driver-behaviour/` — scaffolded (configs + plugin skeletons referencing Phase 5)

## 16. Example Interface Code (signatures only, in plan)

These are the contracts the skeleton will implement. Full bodies belong to the implementation handoff.

- `class InferenceBackend(ABC)` with `load`, `infer`, `warmup`, `close`
- `class Preprocessor(ABC)` with `async __call__(frame, spec) -> dict[str, ndarray]`
- `class Postprocessor(ABC)` with `async __call__(raw, frame, spec) -> Result`
- `class Stage(ABC)` with `name`, `setup`, `process`, `teardown` + `SourceStage`/`SinkStage` markers
- `class ModelSpec(BaseModel)`, `class PipelineSpec(BaseModel)` — Pydantic v2 with `model_validate`/`model_dump`
- `class Frame(BaseModel)` (id, ts, w, h, encoding, data | ref)
- `class Result` tagged union via `Annotated[Union[...], Field(discriminator="kind")]`

## 17. Docker Compose (dev) Services

| Service | Purpose | Notes |
|---|---|---|
| `redpanda` | Kafka API | dev broker |
| `redis` | hot state / frame cache | |
| `postgres` | control plane | |
| `minio` | S3-compatible artifacts | |
| `triton` | inference server | CPU image by default; `docker-compose.gpu.yml` switches to GPU |
| `api` | FastAPI | depends_on: postgres, redis, redpanda |
| `pipeline-worker` | runs example pipeline | scale=1 in dev |
| `mediamtx` | synthetic RTSP source | for streaming examples |
| `otel-collector` | OTel ingestion | exports to Tempo/Prom/Loki if present |
| `prometheus` | metrics | |
| `grafana` | dashboards | provisioned |

## 18. Developer Commands

```
make dev               # docker compose up -d (CPU)
make dev-gpu           # compose up with GPU override
make logs s=api        # tail a service
inferops init my-proj  # scaffold
inferops validate      # JSON-schema validate all configs
inferops models list
inferops pipelines run classify_image --input ./cat.jpg
inferops bench classify_image --duration 30s
make test              # pytest -m "not gpu and not integration"
make test-integration  # compose up + integration suite
make fmt lint
```

## 19. Onboarding Guide Outline

1. Clone repo, `cp .env.example .env`, `make dev`.
2. Drop ONNX into `model_repository/<name>/1/model.onnx`.
3. Write `configs/models/<name>.yaml` from the template.
4. Write or reuse `configs/pipelines/<task>.yaml`.
5. `inferops validate` then `inferops pipelines run <task> --input <sample>`.
6. Expose via API: `curl -F image=@cat.jpg http://localhost:8080/v1/infer/<task>`.
7. To go distributed: `docker compose up -d inference-worker`, point pipeline at Kafka topics.
8. To deploy: edit `deploy/helm/inferops/values.yaml`, `helm install inferops ./deploy/helm/inferops`.

---

## 4.e Partition-key strategy & message ordering (concrete)

Goal: ordering is preserved *per logical processing unit*, never globally. Redpanda guarantees order **within a partition**, so every topic chooses a key that maps the unit-of-order onto a single partition.

### Key by `stream_id`
Used when ordering frames or per-source events matters.
- Topics: `inferops.<pipeline>.frames.raw`, `inferops.<pipeline>.decoded`, `inferops.<pipeline>.preproc`, `inferops.<pipeline>.infer.<model>`, `inferops.events.<domain>` (when entity = stream).
- `key = stream_id.encode("utf-8")`. Same stream → same partition → strict order of frames and their derived results.
- Partition count = `max(streams_per_tenant × tenants × headroom, 12)`; resize via repartition tool, never live (loses order).

### Key by `job_id`
Used for one-shot async inference jobs (`POST /v1/jobs/{pipeline}`) where each job is independent but its lifecycle messages must stay together.
- Topics: `inferops.jobs.requests`, `inferops.jobs.progress`, `inferops.jobs.results`, `inferops.jobs.errors`.
- `key = job_id.encode()`. Guarantees a single worker processes the job to completion and its progress events arrive in order at the API.

### Key by `entity_id`
Used by fusion and alerts. `entity_id` is the fusion window key (often `stream_id`, but configurable per `FusionSpec.entity_key` — could be `vehicle_id`, `camera_zone_id`, `user_id`).
- Topics: `inferops.events.<domain>`, `inferops.alerts.<channel>`, `inferops.fusion.<name>.signals`, `inferops.fusion.<name>.state`.
- `key = entity_id.encode()`. Guarantees one fusion-worker instance owns each entity's window/FSM state in steady state.

### Key = `None` (round-robin)
Used only when ordering is irrelevant and we want max throughput.
- Topics: `inferops.dlq.*`, `inferops.evidence.uploads`, `inferops.audit.events`, `inferops.metrics.bench`.
- Producer omits key; broker round-robins.

### Strict-ordering vs. relaxed-ordering topics

| Topic | Key | Ordering |
|---|---|---|
| `inferops.<p>.frames.raw` | stream_id | STRICT |
| `inferops.<p>.decoded` | stream_id | STRICT |
| `inferops.<p>.infer.<model>` | stream_id | STRICT |
| `inferops.events.<domain>` | entity_id | STRICT |
| `inferops.fusion.<name>.signals` | entity_id | STRICT |
| `inferops.alerts.<channel>` | entity_id | RELAXED (dedupe is content-based) |
| `inferops.jobs.*` | job_id | STRICT per job |
| `inferops.evidence.uploads` | none | RELAXED |
| `inferops.dlq.*` | none | RELAXED |
| `inferops.audit.*` | none | RELAXED |

### Replay semantics
- Replay re-reads from an earlier offset on a strict-order topic. Order **within a partition is preserved**, so stateful consumers (fusion, alerts) reset their per-entity state when replay is detected (`replay=true` header) and rebuild from the rewound stream.
- For at-most-once sinks (webhook), replay is gated behind an explicit `--allow-resend` flag.
- For idempotent sinks (postgres, s3), replay is safe by content key (`alert_id`, `event_id` UUIDv7 derived from `entity_id+ts+payload_hash`).
- Cross-partition ordering is **never** restored by replay (e.g. you cannot reorder events across two `stream_id`s — they were never ordered to begin with).

### Multiple workers consuming the same stage
- All workers for a stage share one **consumer group** `inferops.<pipeline>.<stage>`. Redpanda assigns partitions to group members; each partition is consumed by exactly one member at a time.
- Therefore: parallelism per stage = `min(#workers, #partitions)`. To scale beyond N workers, increase partitions first.
- Sticky-partition assignor configured (`partition.assignment.strategy=sticky`) to minimize state-shuffle on rebalance — important for fusion workers whose in-memory window cache must be rebuilt when a partition moves.
- On rebalance, a fusion worker `LOSES` partitions: it drops in-mem state for those entities; the new owner reloads window/FSM state from Redis (see §4.f).
- DLQ topics use a separate consumer group `inferops.dlq.replay` so replay never starves live consumers.

---

## 4.f Redis-backed fusion state — full spec

State lives in Redis so fusion workers are stateless and can rebalance freely. In-memory holds only a write-through cache.

### Key schema (all binary-safe, lowercase, colon-delimited)
```
fusion:{fusion_name}:win:{entity_id}:{signal}   # sorted set, ZSET: score=ts_ms, member=value-or-payload
fusion:{fusion_name}:ewma:{entity_id}:{signal}  # string (struct: f64 value, u64 ts_ms)
fusion:{fusion_name}:fsm:{entity_id}            # hash: { state, since_ms, last_eval_ms, version }
fusion:{fusion_name}:rule:{entity_id}:{rule}    # string: last_fire_ms (TTL = cooldown_ms)
fusion:{fusion_name}:lock:{entity_id}           # string (Redlock-style fencing token), TTL = lock_ms
fusion:{fusion_name}:meta:{entity_id}           # hash: { last_seen_ms, partition, owner }
fusion:{fusion_name}:idx:active                 # sorted set: score=last_seen_ms, member=entity_id (sweeper)
```

### TTL rules
| Key family | TTL | Refreshed by |
|---|---|---|
| `win:*` | `max(window_duration_ms × 2, 60s)` | every `ZADD` |
| `ewma:*` | `idle_eviction_ms` (default 1h) | every update |
| `fsm:*` | `idle_eviction_ms` | every transition or eval |
| `rule:*` | `cooldown_ms` | on fire only |
| `lock:*` | `lock_ms` (default 5s) | manual `EXPIRE` while held |
| `meta:*` | `idle_eviction_ms` | every event |
| `idx:active` | none | sweeper trims |

### Atomic update strategy
All state mutations are single-key Lua scripts:
- `fusion_window_push.lua`: `ZADD`, `ZREMRANGEBYSCORE` (prune outside window), `EXPIRE`, compute reducers, return reducer values.
- `fusion_ewma_update.lua`: read prior, compute `new = α·sample + (1-α)·prior`, write, `EXPIRE`.
- `fusion_fsm_transition.lua`: `HGET state`, CAS check on `version`, `HSET state/since/last_eval`, `HINCRBY version`. Returns prior state for trace.
- `alerts_dedupe_cooldown.lua`: `SETNX key 1 EX cooldown`. Returns `1` if alert may fire.

Atomicity is per-key. Multi-key transactions are avoided to keep scripts simple and Redis Cluster-safe (when we go there).

### Concurrency: multiple workers on the same key
- Partition-key strategy already routes one `entity_id` to one worker in steady state. Cross-worker contention happens only during rebalance windows.
- Each fusion worker takes a short fencing lease `fusion:{name}:lock:{entity_id}` with a fencing token (monotonic counter from Redis `INCR`). Scripts compare the caller's token against `lock.token`; stale writers are rejected.
- On rejection, the worker drops the message (it was assigned to another partition mid-flight) and the broker rebalance corrects ownership on next poll.

### Recovery after worker restart
1. Worker reconnects to Redpanda, group rebalance assigns partitions.
2. For each new partition, worker lists `entity_id`s by sweeping `fusion:{name}:idx:active` (no full key scan).
3. For each entity, worker lazily warms in-memory cache on first message — does NOT pre-warm (rebalance happens often; pre-warming wastes work).
4. FSM `version` field detects split-brain writes from a slower predecessor; predecessor's late writes are rejected by Lua CAS.

### Redis unavailable
- All Lua-scripted state ops use `tenacity` with capped exponential backoff (5 tries, max 2s).
- After 5 failures, worker enters **degraded mode**: emits `fusion_degraded` metric, *pauses* consuming the affected partition (`consumer.pause(partition)`), continues to consume non-fusion partitions.
- Resume conditions: a successful `PING` health probe, then `consumer.resume(partition)`. No state is invented; we trade availability for correctness.
- Alerts engine has its own fallback: if Redis dedupe is down, alerts go through a small in-memory bloom filter as best-effort dedupe and are tagged `dedupe_degraded=true`.

### Cleanup of expired / inactive streams
- A `fusion_sweeper` cron task (in-process, 60s interval) does `ZRANGEBYSCORE idx:active 0 (now - idle_eviction_ms)` and `DEL`s all sub-keys for the returned entities, then `ZREM`s them from the index.
- TTL provides a backstop: if the sweeper dies, keys still expire individually.

### In-memory vs Redis split
| State | In-memory only | Redis (canonical) |
|---|---|---|
| Active consumer group assignment | yes | no |
| Hot LRU cache of recent window reducers (1k entries, 1s TTL) | yes | no |
| Window samples | mirrored | YES |
| EWMA value | mirrored | YES |
| FSM current state | mirrored | YES |
| Rule cooldown timestamps | no | YES |
| Alert dedupe set | no | YES |
| Metrics counters | yes (prom) | no |
| Tracing spans | yes (otel) | no |

### Tradeoffs
- **Latency tax**: every window push is one Lua roundtrip (~0.5–2 ms LAN). Acceptable for ≤5 kHz signal rate per worker; above that, batch with `MULTI`.
- **No multi-key transactions**: simpler & cluster-safe, but two related counters can briefly disagree (e.g. FSM transitioned, rule cooldown not yet set). Window of inconsistency < 5 ms; rule re-eval on next sample corrects it.
- **Sweeper not exact**: an entity that sees one event per `idle_eviction_ms ± epsilon` may evict mid-life. Mitigation: configurable `min_lifetime_ms`.
- **Redis is a SPOF for fusion correctness**: production must run Redis with replication + Sentinel/Cluster; benchmarks show < 0.1 ms p99 add for read replicas.

---

## 4.g Fusion rules DSL — strict bounded grammar

The DSL is a **pure expression language**. It is not a programming language. It returns a single value (`bool`, `float`, or `int`).

### Grammar (EBNF)
```
expr      = or_expr
or_expr   = and_expr ( "or" and_expr )*
and_expr  = not_expr ( "and" not_expr )*
not_expr  = "not" not_expr | cmp_expr
cmp_expr  = add_expr ( ("==" | "!=" | "<" | "<=" | ">" | ">=") add_expr )?
add_expr  = mul_expr ( ("+" | "-") mul_expr )*
mul_expr  = unary    ( ("*" | "/" | "%") unary )*
unary     = ("-" | "+") unary | atom
atom      = NUMBER | BOOL | name_chain | func_call | "(" expr ")"
name_chain = NAME ( "." METHOD "(" args? ")" )?     # ONE method call max
func_call = FUNC "(" args? ")"
args      = expr ( "," expr )*
NAME      = signal name from FusionSpec.signals + derived
METHOD    = whitelisted reducer name
FUNC      = whitelisted function name
NUMBER    = int | float (decimal, no scientific notation in v1)
BOOL      = "true" | "false"
```

### Allowed operators
- Boolean: `and`, `or`, `not`
- Comparison: `==`, `!=`, `<`, `<=`, `>`, `>=`
- Arithmetic: `+`, `-`, `*`, `/`, `%`, unary `-`/`+`
- Parentheses for grouping

### Whitelisted functions (top-level)
```
abs(x), min(x, y, ...), max(x, y, ...), clamp(x, lo, hi),
round(x), floor(x), ceil(x), sqrt(x),
since_ms(state_name)        # ms since current state entered (FSM context)
prev_state()                # string of previous state
now_ms()                    # current eval time
in_window("HH:MM-HH:MM", tz="UTC")  # time-of-day gate
```
No others. No `pow`, `**`, `exp`, `log` in v1 (add via plugin).

### Whitelisted reducer methods on signals
`latest()`, `mean()`, `min()`, `max()`, `count()`, `sum()`, `stddev()`, `fraction_above(thr)`, `fraction_below(thr)`, `derivative()`, `slope()`, `last_ms()` (ms since last sample). Reducer must be declared in `WindowSpec.reducers` to be callable — referencing an undeclared reducer is a validation error.

### Prohibited
- Attribute access beyond the one whitelisted method call (no `x.y.z`)
- Subscripting (`x[0]`), slicing
- Calls on non-NAME / non-FUNC nodes (no `(lambda)(...)`)
- `import`, `__*__`, generator expressions, comprehensions
- Walrus `:=`, ternary `a if b else c` (use `max`/`min` or a rule split)
- String literals **except** inside `in_window(...)`
- f-strings, format specs
- Any identifier starting with `_`
- Arbitrary Python — period.

### Validation pipeline (fail-fast)
1. **Lex/parse**: Python `ast.parse(expr, mode="eval")`. Syntax error → reject.
2. **AST walk** with allowlist visitor:
   - Allowed node types: `Expression`, `BoolOp`, `BinOp`, `UnaryOp`, `Compare`, `Name`, `Constant` (bool/int/float, or str inside `in_window`), `Call`, `Attribute` (only when child of `Call` and parent is a `Name` resolving to a declared signal), `Load`.
   - Any other node → reject with line/col.
3. **Name resolution**: every `Name` must be in (`signals` ∪ `derived` ∪ `whitelisted FUNCs`). Unknown → reject with suggestion (Levenshtein).
4. **Method resolution**: `signal.method()` — method must be in reducer whitelist AND declared in that signal's `WindowSpec.reducers`. Unknown → reject.
5. **Type sketch**: each subexpression annotated `bool|number|unknown`. `RuleSpec.when` must resolve to `bool`. `derived` may resolve to `bool` or `number`. `StateMachineSpec.transitions[*].when` must resolve to `bool`.
6. **Constant fold + dead-code detection**: `false and X` warns; `1==1` warns.
7. All validation runs at config-load time (`inferops validate`). CI gate fails the build on warnings if `strict: true`.

### Evaluation
- Compiled once to a small bytecode tree (visitor builds a closure tree). No `eval()`, no `exec()`. ~50× faster than `asteval` for our tree sizes.
- Per-eval timeout: 5 ms (configurable). Exceeded → eval returns `None`, rule skipped, metric `fusion_eval_timeout_total{rule}` incremented.
- Reducer results memoized per (entity, signal, eval_tick) — same reducer called twice in one tick costs one Lua call.

### When a plugin is required instead of DSL
| Need | Why DSL can't | Use plugin |
|---|---|---|
| Calling an external service / DB lookup | I/O forbidden | `Evaluator` plugin returning `bool` |
| Multi-signal correlation requiring nested method chains | One-method-call limit | `DerivedSignal` plugin |
| Custom reducer (e.g. percentile, FFT energy) | Not in whitelist | `Reducer` plugin registered in `inferops.plugins.builtin.reducers:Percentile` |
| Heavyweight ML (a classifier over windowed features) | Not pure-expression | `Evaluator` plugin invoking a model |
| State-machine action that calls a webhook | Side effects forbidden | `Action` plugin |

Plugins are loaded by `module:Class` strings, validated to subclass the right ABC, and listed in `inferops models|fusion list` output for auditability. Plugin authors get only the same read-only signal/window context as the DSL — they cannot mutate engine state.

---

## 4.h Alerts subsystem — execution metadata & lifecycle

### Lifecycle states
```
proposed   → emitted   → delivering ⇄ delivered
                    ↘         ↘
                  suppressed  failed
                                ↘
                              resolved (manual or auto)
```
- `proposed`: candidate produced by fusion/rule.
- `emitted`: passed dedupe + cooldown + severity mapping; persisted; about to fan out.
- `suppressed`: blocked by dedupe or cooldown; persisted with reason.
- `delivering`: at least one sink attempt in flight.
- `delivered`: all required sinks (those whose `min_severity` matched) returned success.
- `failed`: max retries exhausted on at least one required sink (overall status is `partial` if some sinks succeeded).
- `resolved`: terminating-condition rule fired (e.g. driver returned to NORMAL for 60s) or operator API call.

### Identity & dedupe
- `alert_id`: UUIDv7 (time-ordered), generated when the candidate is `proposed`.
- `dedupe_key`: resolved per `DedupeSpec.strategy`; stored verbatim. Within `DedupeSpec.window_ms` an incoming candidate with the same key is `suppressed` and a back-reference (`suppressed_by_alert_id`) is recorded.
- `correlation_id`: copied from the originating message header (W3C traceparent's trace-id), enables joining with frames/inference spans.

### Per-sink delivery record
Each sink attempt is its own row:
- `attempt_id` (UUIDv7), `alert_id` (FK), `sink_name`, `attempt_no`
- `status` ∈ `pending|in_flight|success|retrying|dropped|failed`
- `started_at`, `finished_at`
- `latency_ms`
- `http_status`, `error_class`, `error_message`, `error_payload_hash`
- `request_id` (sink-side, e.g. webhook returns `X-Request-Id`)
- `next_retry_at` (when `retrying`)
- Retry policy per sink: exponential backoff `min(base × 2^n, cap)` with jitter; max attempts configurable per sink; after exhaustion → DLQ topic `inferops.alerts.dlq` with full envelope for manual replay.

### Timestamps on the alert row
| Field | Meaning |
|---|---|
| `proposed_at` | candidate created |
| `emitted_at` | passed gating, persisted |
| `first_delivery_at` | first sink attempt started |
| `last_delivery_at` | last sink attempt finished |
| `delivered_at` | overall status reached `delivered` |
| `failed_at` | overall status reached `failed` |
| `resolved_at` | resolved (manual or auto) |
| `suppressed_at` | suppressed by dedupe/cooldown |

### Evidence linkage
- `evidence_refs` JSONB array: `[{kind: "video", uri: "s3://...", ts_start_ms, ts_end_ms, sha256}, {kind: "frame", uri: "redis://...", ts_ms}, {kind: "log", uri: "..."} ]`.
- Evidence is attached *asynchronously* by the `evidence_clipper` stage and updates the alert row by `alert_id`. Alert may transition to `delivered` before evidence is ready; sinks that require evidence (e.g. email with attachment) wait or poll via `evidence_ready` event keyed by `alert_id`.

### Auditability & traceability
- Append-only `alert_events` table records every state transition with `actor` (`system|user:<id>|sink:<name>`), `from_status`, `to_status`, `reason`, `meta JSONB`, `ts`.
- Every alert and attempt row carries `trace_id`/`span_id`.
- Operator API: `GET /v1/alerts/{alert_id}` returns alert + attempts + events + evidence in one document.
- Compliance export: `inferops alerts export --since … --format jsonl` streams alerts + attempts + events for archival.

### Tables (generic, domain-agnostic)
```
alerts (
  alert_id UUID PK,
  pipeline TEXT, entity_id TEXT, type TEXT,
  severity TEXT, severity_rank SMALLINT,
  status TEXT,                        -- lifecycle state
  dedupe_key TEXT, suppressed_by_alert_id UUID NULL,
  correlation_id TEXT, trace_id TEXT,
  payload JSONB,                      -- domain-specific body (opaque)
  evidence_refs JSONB DEFAULT '[]',
  proposed_at, emitted_at, first_delivery_at, last_delivery_at,
  delivered_at, failed_at, resolved_at, suppressed_at TIMESTAMPTZ NULL,
  meta JSONB                          -- everything else
)
INDEX alerts_entity_ts (pipeline, entity_id, proposed_at DESC)
INDEX alerts_status   (status, severity_rank DESC, proposed_at DESC)
INDEX alerts_dedupe   (dedupe_key) WHERE suppressed_at IS NULL

alert_attempts (
  attempt_id UUID PK,
  alert_id UUID FK,
  sink_name TEXT, attempt_no INT,
  status TEXT, http_status INT NULL,
  error_class TEXT NULL, error_message TEXT NULL, error_payload_hash TEXT NULL,
  request_id TEXT NULL,
  started_at, finished_at TIMESTAMPTZ,
  latency_ms INT, next_retry_at TIMESTAMPTZ NULL,
  trace_id TEXT, meta JSONB
)
INDEX alert_attempts_alert (alert_id)
INDEX alert_attempts_retry (next_retry_at) WHERE status = 'retrying'

alert_events (
  event_id BIGSERIAL PK,
  alert_id UUID FK,
  ts TIMESTAMPTZ,
  actor TEXT,                         -- system | user:<id> | sink:<name>
  from_status TEXT, to_status TEXT,
  reason TEXT, meta JSONB,
  trace_id TEXT
)
INDEX alert_events_alert_ts (alert_id, ts)
```

Yes — three tables. The generic `alerts` row carries the body and current status; `alert_attempts` is the per-sink ledger; `alert_events` is the immutable audit log. The driver-behaviour example needs no extra tables — its domain lives in `payload` and `type`.

---

## Decisions to Confirm with User
1. **New repo vs. in-place extraction.** Recommend new sibling repo `inferops/` so the current driver project keeps running. Could also live as `platform/` directory in this workspace.
2. **Driver-behaviour fate.** Recommend porting it to `examples/driver-behaviour/` and deprecating `packages/fusion`+`packages/alerts` once parity is proven.
3. **In-proc vs. Kafka-only.** Recommend supporting both (in-proc for sync API, Kafka for streaming) — adds some complexity but huge DX win.
4. **TensorRT in MVP?** Recommend stub-only in MVP; full impl in Phase 7.
5. **Language.** Python only initially; Rust/Go workers later if needed for ingest throughput.

## Relevant Files (in current workspace, for reference / lift sources)
- packages/common/kafka/bus.py — extract envelope/header logic into `inferops/adapters/transport/redpanda.py`; do NOT expose `aiokafka` outside that module
- packages/common/redis/client.py — lift into `inferops/adapters/redis_client.py`
- packages/common/frame_cache/cache.py — lift into `inferops/adapters/frame_cache.py`
- packages/common/obs/{logging,metrics,tracing}.py — lift into `inferops/obs/`
- packages/inference/triton_client.py — becomes `inferops/runtimes/triton_backend.py`
- packages/inference/preprocess.py — split into `imagenet_normalize` + `image_letterbox` plugins
- packages/inference/postprocess.py — becomes example plugin `examples/driver-behaviour/plugins/postprocess_rfdetr.py`
- packages/ingest/{rtsp_worker,mjpeg_worker,file_injector,ring_buffer,stream_lease}.py — become `inferops/pipeline/stages/source_*.py`
- packages/evidence/clipper.py — becomes built-in `sink_clip_evidence` stage
- packages/fusion/temporal_window.py — math lifts into `inferops/fusion/window.py` (generic)
- packages/fusion/state_machine.py — mechanics lift into `inferops/fusion/state_machine.py` (drop driver enum; states come from config)
- packages/fusion/fusion_service.py — replaced by generic `inferops/fusion/engine.py` + driver YAML in example
- packages/alerts/alert_service.py — replaced by generic `inferops/alerts/engine.py`
- packages/alerts/sinks.py — `AlertSink` ABC + concrete impls lift into `inferops/alerts/sinks/`
- packages/storage/postgres/models.py — refactored to generic `events`/`jobs`/`models`/`alerts` tables
- packages/api/* — port routes, drop domain-specific (`tenants`, `evidence`) into examples or generalize
- triton/model_repository/ — keep layout, becomes `model_repository/`
- observability/ — lift wholesale
- docker-compose.dev.yml — becomes `docker-compose.yml` parameterized (Redpanda by default)
- deploy/helm/driver-analytics/ — template for `deploy/helm/inferops/`

## Verification
1. `inferops validate` succeeds on all three example configs.
2. `pytest tests/unit tests/contract` green on Python 3.11 (CPU).
3. `make dev && curl -F image=@tests/fixtures/cat.jpg :8080/v1/infer/classify_image` returns top-5 with "tabby" in top-3.
4. `make dev && inferops pipelines run video_detection --rtsp rtsp://mediamtx:8554/test` produces detections on Kafka topic `events.detections` for ≥ 30 s without errors.
5. **Verification (new)** — fusion safe-expression evaluator rejects `__import__`, attribute escapes, and arbitrary calls; cooldown/dedupe deterministic under replay.
6. `examples/driver-behaviour` parity test: same input video produces alerts within ±1 frame of legacy pipeline.
7. Grafana `inferops-overview` dashboard shows non-zero throughput & latency p95 < 200 ms for classification on CPU dev box.
8. Helm `helm template deploy/helm/inferops/` renders without errors for 1-GPU and 4-GPU `values.yaml`.
9. **Transport boundary test**: `grep -R "aiokafka\|redpanda" inferops/` must match only inside `inferops/adapters/transport/`; CI fails otherwise.
