# InferOps documentation

InferOps is an inference library and a server application built around the same
processing services. A project supplies its models, processing stages and rules;
the engine owns execution, isolated sessions, incident acceptance and evidence.

## Start here

| Your goal | Read |
| --- | --- |
| Understand the design and why each layer exists | [Architecture and Hexagonal design](architecture.md) |
| Follow startup, a frame, an incident and shutdown | [Full application lifecycle](lifecycle.md) |
| Understand Redpanda, topics, outbox, inbox and retries | [Messaging and Redpanda](messaging.md) |
| Find a module or decide where new code belongs | [Package structure](structure.md) and [complete file inventory](architecture/repository-tree.md) |
| Explore the drawings | [Offline architecture gallery](architecture/index.html) |
| Install and call the Python library | [Python guide](python-library.md) |
| Start and operate the server | [Deployment guide](../deploy/README.md) and [server requirements](../README.md#hosting-server-requirements) |
| Call HTTP or WebSocket endpoints | [API reference](api.md) and the running server's `/docs` |
| Create a project or an adapter | [Project configuration](project-blocks.md) and [adapter authoring](adapters.md) |
| Review recovery, migration and release evidence | [Redesign contracts](redesign.md) and [verification record](cleanup-verification.md) |

For a first reading, follow **architecture → lifecycle → messaging → deployment**.
These guides describe the checked-in implementation. Operational limits are
provisional; a successful offline video test does not establish live capacity.

## The complete path

```text
Project configuration + model artifacts
  → explicit bootstrap and recovery
  → Python / HTTP / WebSocket / CLI / worker input
  → generated session and media timeline
  → model inference → tracking → fusion → incident policy
  → PostgreSQL acceptance transaction
      ├─ evidence request → persistent spool → MinIO artifacts
      └─ outbox → Redpanda / configured delivery targets
  → status queries, live notifications and operational metrics
  → session close → bounded drain → resource shutdown
```

Model-only prediction uses the shared model pool without opening a processing
session or running tracking, fusion, incidents or evidence. Local library use
does not require the server stack. See [Python execution modes](lifecycle.md#choose-an-entry-point).

## Vocabulary

| Term | Meaning in this project |
| --- | --- |
| Project | A product pack containing model definitions, plugins, pipelines and rules, such as `projects/driver_behaviour/` |
| Model | A logical prediction capability; runtime variants can implement the same logical model |
| Pipeline | An ordered processing definition with explicit independent model groups and output sinks |
| Engine | One explicitly constructed owner of model resources, services and active processing sessions |
| Processing session | An isolated run with a generated identity, owner generation, pinned project revision and mutable temporal state |
| `InferenceSession` | The public collection of model handles; it differs from a pipeline processing session |
| Stream name | External descriptive metadata; repeating it does not continue a session |
| Timeline | A continuous media timestamp segment within a session |
| Fusion event | A temporal-rule or state-machine output, before incident policy accepts or suppresses it |
| Incident / alert | An accepted policy decision; the API and SQL projection retain the name `alert` |
| Evidence | The incident's snapshot, clip, artifact manifest and durable completeness/status record |
| Spool | Recoverable local capture material while encoding/upload is pending or retained for inspection |
| Outbox / inbox | SQL records for retryable delivery and deduplication of durable consumer effects |
| Broker offset | A record's position within one topic partition; neither a media timestamp nor a session identity |

## Maintaining these guides

When changing execution, update the relevant source-linked lifecycle section.
When changing topics or wire shapes, update the messaging contract and examples.
When changing deployment wiring, update the service/volume tables. Diagram
sources and rendered SVGs live together in `docs/architecture/`; see
[diagram maintenance](architecture/README.md). Links are relative so the
repository can be read locally or through a Git hosting service.
