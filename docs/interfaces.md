# Frontends, pages and ports

All commands below run from the project root, the directory containing this
README's sibling `../compose.yaml` and `../pyproject.toml`. This is also the Git
repository root; its `inferops/` child contains the Python package.

```bash
cp .env.example .env            # first setup only
docker compose up -d --build
docker compose ps
```

Persistent disk is the default evidence store. MinIO uses the optional `storage`
profile; Grafana and Prometheus use `monitoring`. Enable MinIO and its adapter
with `make run-container MODE=cpu RUN_ARGS='--profiles storage'`, or
`INFEROPS_STORAGE_KIND=s3 docker compose --profile storage up -d --build`.
See [deployment storage settings](../deploy/README.md#what-starts-and-where).

## Web interfaces and pages

| Page | Default URL | What it does |
| --- | --- | --- |
| InferOps workspace | http://localhost:8080/ui/ | Project overview, pipeline inputs, previews, activity, YAML editing and original service links |
| Browser landing page | http://localhost:8080/ | Redirects browser HTML requests to the workspace; API clients receive discovery JSON |
| Compatibility workspace | http://localhost:8080/frontend/ | Same packaged workspace; no standalone frontend folder |
| Built-in interface guide | http://localhost:8080/ui/guide | Integrated workflow guide and configured ports; API routes are in the console API explorer |
| MinIO Console | http://localhost:9001/ | Original bucket/object browser and storage administration |
| Grafana Overview | http://localhost:3000/d/inferops-overview/inferops-overview | Provisioned InferOps metrics dashboard |
| Grafana dashboards | http://localhost:3000/dashboards | Browse and create dashboards |
| Grafana Explore | http://localhost:3000/explore | Query the provisioned Prometheus datasource |
| Prometheus queries | http://localhost:9090/graph | Query and graph metrics |
| Prometheus targets | http://localhost:9090/targets | Inspect scrape status and errors |
| Prometheus alerts | http://localhost:9090/alerts | Inspect configured alerting rules; no rules are shipped by default |
| Swagger | http://localhost:8080/docs | Interactive API requests and schemas |
| ReDoc | http://localhost:8080/redoc | Readable API documentation |
| OpenAPI | http://localhost:8080/openapi.json | JSON schema for client generation |
| Health / readiness | http://localhost:8080/healthz and /readyz | API liveness and startup readiness |
| API metrics | http://localhost:8080/metrics | Prometheus text output |
| Service discovery | http://localhost:8080/v1/interfaces | Browser-facing service and page URLs as JSON |
| Evidence compatibility link | http://localhost:8080/v1/evidence | Redirects to the original MinIO Console |

There is no custom MinIO or Grafana replacement. PostgreSQL and optional Redpanda
and Triton provide protocol/API endpoints; this project does not ship separate
web frontends for them.

Service cards open the native interfaces in new tabs. This preserves their
login, downloads and navigation. The links resolve from the browser's API
hostname, including remote hosts, rather than exposing Docker names such as
`minio` or `grafana`. The API UI remains on the same origin as its endpoints.

## Use the InferOps workspace

Use the top navigation to jump between Services, Overview, Execution, Activity
and Configuration. The API connection defaults to this page's origin. The
service cards show health from the API; unavailable services remain clickable
so their own page can be inspected. The overview separates configured logical
models from loaded models and shows the actual execution mode and devices.

| Input mode | Operation | Evidence behavior |
| --- | --- | --- |
| Camera | Capture a frame or run an analysis loop using browser camera access | Runs the configured infer pipeline |
| Image | Upload an image and inspect its detections and result | Runs the configured infer pipeline |
| Video: Stream and analyze | Play locally and send sampled frames over the live WebSocket | Preview path; skips durable evidence upload |
| Video: Analyze all frames and save evidence | Upload the full video and wait for processing to finish | Saves configured JSON, snapshots and triggered clips |
| RTSP | Start a backend stream and inspect its status and activity | Follows the selected pipeline's sink configuration |

Camera access requires localhost or HTTPS and a browser permission grant.
Registered RTSP sources support native WebRTC/HLS playback through the default
[MediaMTX service](media-router.md), alongside the annotated JPEG wall.
The legacy RTSP playground shows stream status and events. Use a unique stream
ID for independent runs.
Activity includes persisted alerts and fusion events. The YAML editor validates
before saving; model, pipeline, fusion and alert sections can reload live.
Project/platform changes require restart, and weight/source/tensor changes
require preparation and restart.

For the shipped example, select Driver Behaviour, choose Video and
`research/driver_training/test.mp4`, then click **Analyze all frames and save
evidence**. Open Activity & evidence to view saved snapshots and play clips.
When S3 is selected, MinIO Console can also browse the configured bucket's
`evidence/<evidence-id>/` objects. Configuration controls which snapshots
and alert clips are created; not every video necessarily emits an alert.

## Original service interfaces

MinIO uses the original upstream binary from official GitHub release
`RELEASE.2025-04-22T22-12-26Z`, which includes the full community admin console.
The old Docker Hub and Quay image tags are no longer available, so
`deploy/docker/minio.Dockerfile` downloads that upstream binary, verifies its
published SHA-256, and packages it as `inferops-minio:console`. It supports
amd64 and arm64; no MinIO server or console source is changed. Upstream's [May 2025 breaking release](https://github.com/minio/minio/releases/tag/RELEASE.2025-05-24T17-08-30Z)
removed the embedded admin UI in favor of an object browser. This deliberately
older version restores the requested interface, but does not include later
server/security fixes. The Dockerfile pins the version explicitly; changing
it changes which upstream console is available. The upstream repository is
archived, so this pin should be reconsidered before production use.

The default MinIO login is `minio` / `minio12345`, overridden by
`INFEROPS_S3_ACCESS_KEY` and `INFEROPS_S3_SECRET_KEY`. Grafana allows anonymous
viewing of the provisioned overview; use `admin` / `admin` for editing, with
`INFEROPS_GRAFANA_USER` and `INFEROPS_GRAFANA_PASSWORD` for first initialization.
Changing those variables does not reset an already-initialized Grafana
volume's password; use Grafana's admin CLI to reset it if needed.

Grafana's Prometheus datasource has stable UID `prometheus`. Provisioning
lives in `deploy/observability/grafana/`; metrics are scraped every five seconds.
Container mode scrapes `api:8080`. Native mode generates a config in `.run/`
that scrapes `host.docker.internal:<API_PORT>`. Triton targets are only healthy
when a Triton runtime is started. The overview uses engine metrics
for CPU and GPU inference; no separate GPU exporter is required for its panels.

Optional MinIO uses the named volume `inferops_minio-upstream-data`. Retired
local runtime volumes and backups were discarded; new storage starts empty.
Default evidence uses the separate `engine-data` volume. Named volumes survive
ordinary restarts and `docker compose down`; `docker compose down -v` deletes them.

## Published ports

Ports below are host defaults for enabled services. Docker uses the internal
addresses shown even when a host port changes. The ports in this table use TCP;
MediaMTX's additional TCP and UDP ports are listed in the [media guide](media-router.md).

| Service | Host | Container / internal address | Protocol / purpose | `.env` override |
| --- | --- | --- | --- | --- |
| InferOps API/UI | 8080 | `api:8080` | HTTP, REST, WebSocket, docs, metrics | `INFEROPS_API_PORT` |
| MinIO S3 | 9000 | `minio:9000` | S3-compatible API; not the console | `INFEROPS_S3_PORT` |
| MinIO Console | 9001 | `minio:9001` | Original web UI | `INFEROPS_MINIO_CONSOLE_PORT` |
| Grafana | 3000 | `grafana:3000` | Web UI and HTTP API | `INFEROPS_GRAFANA_PORT` |
| Prometheus | 9090 | `prometheus:9090` | Web UI and HTTP API | `INFEROPS_PROMETHEUS_PORT` |
| PostgreSQL | 5432 | `postgres:5432` | PostgreSQL protocol | `INFEROPS_PG_PORT` |
| Redpanda external Kafka | 19092 | `redpanda:19092` | Kafka; containers use `redpanda:9092` | `INFEROPS_KAFKA_PORT` |
| Triton HTTP | 8000 | `triton:8000` | Inference HTTP API; only triton modes | `INFEROPS_TRITON_HTTP_PORT` |
| Triton gRPC | 8001 | `triton:8001` | Inference gRPC; only triton modes | `INFEROPS_TRITON_GRPC_PORT` |
| Triton metrics | 8002 | `triton:8002` | Prometheus metrics; only triton modes | `INFEROPS_TRITON_METRICS_PORT` |

Redpanda's container-only ports are `9092` (Kafka), `9644` (admin API and
`/public_metrics`), `8081` (Schema Registry), `8082` (Pandaproxy), and `33145`
(broker RPC). They are not published to the host. `prepare` and `minio-init`
are one-shot jobs without published ports. Compose creates no additional
frontend ports. PostgreSQL management requires a compatible client. Storage,
broker and monitoring endpoints exist only when their optional profiles run.

For native mode, changing a host port also requires its matching engine
endpoint: `INFEROPS_S3_ENDPOINT`, `INFEROPS_PG_URL`,
`INFEROPS_TRITON_URL`, `INFEROPS_TRITON_HTTP_URL`, or transport bootstrap.
Native monitoring health checks use `INFEROPS_GRAFANA_INTERNAL_URL` and
`INFEROPS_PROMETHEUS_INTERNAL_URL` if their ports change. Docker supplies its
own internal endpoints. Kafka remote clients also need `INFEROPS_KAFKA_HOST`.

## Remote access and reverse proxies

Change service ports in `.env`, then recreate the affected containers:

```bash
docker compose up -d
```

For HTTPS or port-based proxy hosts, set complete browser-facing URLs:

```dotenv
INFEROPS_MINIO_CONSOLE_PUBLIC_URL=https://storage.example.com
INFEROPS_GRAFANA_PUBLIC_URL=https://metrics.example.com
INFEROPS_PROMETHEUS_PUBLIC_URL=https://prometheus.example.com
INFEROPS_S3_PUBLIC_URL=https://s3.example.com
```

The first three values control service links. The S3 public URL controls
presigned evidence links; it must point to the S3 API, not port 9001. URL
values must use HTTP(S) and contain no credentials. Configure each service's
proxy/root path and WebSocket forwarding when using subpaths. The application
UI uses `/ui/static/` assets and `/v1/` endpoints; forward both to the API.

## Troubleshooting

| Symptom | Check / action |
| --- | --- |
| Grafana does not open | `docker compose ps grafana prometheus`; start them with `docker compose up -d grafana prometheus` |
| Grafana is empty | Check http://localhost:9090/targets and the datasource; generate inference traffic; optional Triton panels need their runtime |
| MinIO login works but admin menus are missing | Check the release in `deploy/docker/minio.Dockerfile`; later releases removed those menus |
| MinIO does not open | Use port 9001; run `docker compose logs minio`; check the published console port |
| Workspace cannot connect | Inspect the API base, `docker compose logs api prepare`, and `/healthz`; missing real weights block startup |
| Link uses the wrong remote port/hostname | Set that service's `INFEROPS_*_PUBLIC_URL` and recreate the API |
| Browser camera unavailable | Use HTTPS or localhost and allow camera access |
| Port already occupied | Stop the owning process or change that service's published port; `make stop` stops InferOps' registered native API |
| Code changes not visible | `docker compose up -d --build`; hard-refresh the browser |

See [API contracts](api.md), [project configuration](project-blocks.md) and
[verification scenarios](../tests/scenarios/README.md).

## Screenshots

[InferOps workspace](images/workspace-desktop.png) ·
[Mobile workspace](images/workspace-mobile.png) ·
[Pipeline execution](images/workspace-execution.png) ·
[Original MinIO Console](images/minio-original-console.png) ·
[Original Grafana dashboard](images/grafana-original-dashboard.png) ·
[Original Prometheus UI](images/prometheus-original-ui.png)

These captures show the API-packaged workspace and original service UIs.
