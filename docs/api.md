# HTTP and WebSocket API reference

The API and UI share http://localhost:8080 by default. All `/v1/` routes are
provided by the Python engine. [Swagger](http://localhost:8080/docs) contains
the authoritative request/response schemas for the running version;
[ReDoc](http://localhost:8080/redoc) and [OpenAPI JSON](http://localhost:8080/openapi.json)
are also available. [The built-in guide](http://localhost:8080/ui/guide) lists live routes.

## HTTP endpoints

| Method | Route | Parameters / request body | Purpose |
| --- | --- | --- | --- |
| GET | `/healthz` | — | Healthz |
| GET | `/readyz` | — | Readyz |
| GET | `/metrics` | — | Metrics |
| GET | `/v1/configs` | — | List Configs |
| GET | `/v1/configs/{section}/{name}` | — | Get Config |
| PUT | `/v1/configs/{section}/{name}` | application/json: `content`, `reload` | Update Config |
| POST | `/v1/configs/reload` | `section` (query) | Reload Configs |
| GET | `/v1/models` | `view` (query) | List Models |
| GET | `/v1/models/{name}` | — | Get Model |
| POST | `/v1/models/reload` | — | Reload Models |
| GET | `/v1/pipelines` | — | List Pipelines |
| GET | `/v1/pipelines/{name}` | — | Get Pipeline |
| POST | `/v1/pipelines/reload` | — | Reload Pipelines |
| POST | `/v1/jobs/{pipeline}` | application/json body | Start Job |
| GET | `/v1/jobs` | `limit` (query) | Get Jobs |
| GET | `/v1/jobs/{job_id}` | — | Get Job Status |
| POST | `/v1/infer/{pipeline}` | multipart/form-data: `image`, `stream_id`, `model_override` | Infer |
| POST | `/v1/run/{pipeline}` | application/json body | Run a pipeline once with a JSON payload (e.g. {'video_path': '/...mp4'}). |
| POST | `/v1/run/{pipeline}/upload` | multipart/form-data: `video`, `stream_id` | Upload a video file, run it through a pipeline once, and return a summary. |
| POST | `/v1/streams` | `pipeline` (query); `rtsp_url` (query); `mjpeg_url` (query); `stream_id` (query); `fps` (query) | Start a long-running pipeline stream. |
| GET | `/v1/streams` | — | List Streams |
| DELETE | `/v1/streams/{stream_id}` | — | Stop Stream |
| GET | `/v1/system/status` | — | System Status |
| GET | `/v1/events` | `pipeline` (query); `limit` (query) | Get Events |
| GET | `/v1/alerts` | `limit` (query) | Get Alerts |
| GET | `/v1/evidence/file/{bucket}/{key}` | — | Proxy a file from MinIO with HTTP Range support (required for HTML5 video). |
| GET | `/v1/evidence/clips` | `limit` (query) | List alert clips with proxy download URLs. |
| GET | `/v1/evidence` | — | Open the original MinIO Console; object listing and file APIs remain available. |
| GET | `/v1/interfaces` | — | Interfaces |
| GET | `/v1/interfaces/{name}` | — | Open Interface |
| GET | `/` | — | Root |

## WebSockets

| Path | Client input | Server output |
| --- | --- | --- |
| `/v1/live/{pipeline}?stream_id=<id>&model=<optional>` | JPEG binary frames from camera/video | Inference results, overlay boxes, signals, model and timing metadata |
| `/v1/ws/{channel}` | Connect to a pipeline sink channel | Binary JSON envelopes broadcast by `sink_ws` |

The workspace decodes text and binary socket responses. Channel names are
configured by pipeline sinks; alerts typically use `alerts.<stream_id>`.
Use `ws://` for HTTP and `wss://` for HTTPS. WebSocket contracts are not
represented in OpenAPI; see `inferops/api/routes/live.py` and `ws_live.py`.

## Request examples

Run these commands from any terminal while the API is available:

```bash
curl http://localhost:8080/v1/interfaces
curl http://localhost:8080/v1/system/status
curl "http://localhost:8080/v1/models?view=logical"
curl http://localhost:8080/v1/pipelines
curl "http://localhost:8080/v1/alerts?limit=20"

# Analyze one image through a selected pipeline.
curl -X POST http://localhost:8080/v1/infer/driver_behaviour \
  -F image=@photo.jpg -F stream_id=example-image

# Process every video frame and wait for configured evidence to be stored.
curl -X POST http://localhost:8080/v1/run/driver_behaviour/upload \
  -F video=@clip.mp4 -F stream_id=example-video

# Read configuration before editing it.
curl http://localhost:8080/v1/configs/pipelines/driver_behaviour
```

`PUT /v1/configs/{section}/{name}` accepts `{"content":"<YAML>","reload":true}`.
Sections are `project`, `platform`, `models`, `pipelines`, `fusion` and `alerts`.
Saving validates YAML and references. Live reload applies to models, pipelines,
fusion and alerts; project/platform and artifact changes may need restart or
preparation. `POST /v1/configs/reload?section=all` reloads live sections.

`POST /v1/jobs/{pipeline}` creates a background run; inspect its status with
`GET /v1/jobs/{job_id}`. `POST /v1/run/{pipeline}` runs a JSON payload once;
server-local paths refer to the API host/container, so browser uploads should
use `/run/{pipeline}/upload`. Multipart image/video field names are `image`
and `video` respectively. A unique `stream_id` keeps run state isolated.

`POST /v1/streams` accepts the query parameters shown in Swagger (`pipeline`,
`rtsp_url` or `mjpeg_url`, `stream_id`, `fps`). Stop it with `DELETE /v1/streams/{stream_id}`.

## Evidence and errors

`GET /v1/evidence/clips` lists saved clips and their API proxy URLs.
`GET /v1/evidence/file/{bucket}/{key}` supports HTTP byte ranges for video
playback. Object keys may contain slashes. `/v1/evidence` redirects to the
original MinIO Console instead of rendering a replacement storage UI.

Unknown resources return 404, invalid request data returns 422, and backend
or pipeline runtime failures return 503 with a JSON `detail` message. Model
startup errors can stop the API before it becomes healthy. `/readyz` reports
API startup readiness; it is not a complete health check of external services.
Use `/v1/system/status` and Prometheus targets to inspect those dependencies.

## Other service APIs

| Service | Default host endpoint | Primary API |
| --- | --- | --- |
| MinIO | `http://localhost:9000` | S3-compatible operations; `/minio/health/live`, `/minio/health/ready`; metrics at `/minio/v2/metrics/cluster` |
| Grafana | `http://localhost:3000` | `/api/health`, `/api/search`, `/api/dashboards/uid/inferops-overview`; writes require appropriate login/permissions |
| Prometheus | `http://localhost:9090` | `/api/v1/query`, `/api/v1/query_range`, `/api/v1/targets`, `/-/ready`, `/metrics` |
| Triton | `http://localhost:8000` | `/v2/health/live`, `/v2/health/ready`, `/v2/models/<model>`, `/v2/models/<model>/infer`; gRPC on 8001 and metrics on 8002 |
| Redpanda | Internal `http://redpanda:9644` | Admin API and `/public_metrics`; Kafka protocol on host port 19092 |

See [interfaces, pages and all ports](interfaces.md) for startup, service
credentials, remote URLs and the difference between host and container ports.
PostgreSQL and Redis expose their native protocols rather than HTTP APIs.
