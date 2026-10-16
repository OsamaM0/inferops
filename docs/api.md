# HTTP and WebSocket API reference

Read the [full lifecycle](lifecycle.md) for session, incident and evidence semantics,
and [messaging](messaging.md) for durable delivery and notification guarantees.
The [documentation index](README.md) links the architecture and deployment guides.

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
| POST | `/v1/jobs/{job_id}/cancel` | — | Cancel a queued/running job |
| POST | `/v1/jobs/{job_id}/retry` | — | Create a linked retry attempt |
| POST | `/v1/sessions` | JSON: `pipeline`, optional `stream_id` | Create an isolated processing session and capability |
| DELETE | `/v1/sessions/{session_id}` | `X-Session-Capability` header | Close an owned session |
| POST | `/v1/infer/{pipeline}` | multipart: `image`, `stream_id`, `model_override`, optional `session_id`, `media_time_ns`, `seq`; capability header for continuation | Infer |
| POST | `/v1/run/{pipeline}` | application/json body | Run a pipeline once with a JSON payload (e.g. {'video_path': '/...mp4'}). |
| POST | `/v1/run/{pipeline}/upload` | multipart/form-data: `video`, `stream_id` | Upload a video file, run it through a pipeline once, and return a summary. |
| POST | `/v1/streams` | `pipeline` (query); `rtsp_url` (query); `mjpeg_url` (query); `stream_id` (query); `fps` (query) | Start a long-running pipeline stream. |
| GET | `/v1/streams` | — | List Streams |
| DELETE | `/v1/streams/{session_id}` | `X-Session-Capability` header | Stop an owned camera stream |
| GET | `/v1/system/status` | — | System Status |
| GET | `/v1/events` | `pipeline` (query); `limit` (query) | Get Events |
| GET | `/v1/alerts` | `limit` (query) | Get Alerts |
| GET | `/v1/evidence/file/{bucket}/{key}` | — | Proxy a file from MinIO with HTTP Range support (required for HTML5 video). |
| GET | `/v1/evidence/clips` | `limit` (query) | List alert clips with proxy download URLs. |
| GET | `/v1/evidence` | — | Open the original MinIO Console; object listing and file APIs remain available. |
| GET | `/v1/evidence/records` | `session_id`, `limit` (query) | List durable capture status records |
| GET | `/v1/evidence/records/{evidence_id}` | — | Inspect capture state, errors, coverage and artifacts |
| POST | `/v1/evidence/records/{evidence_id}/retry` | — | Retry using retained capture material |
| DELETE | `/v1/evidence/records/{evidence_id}` | — | Manually remove artifacts and retain an audit tombstone |
| GET | `/v1/evidence/records/{evidence_id}/assets/{kind}` | `kind`: `snapshot` or `clip`; HTTP Range | Read an artifact through a storage-neutral endpoint |
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
represented in OpenAPI; the implementation is
`inferops/adapters/inbound/http/app.py`. Python and HTTP use the same session
services. No alternate legacy route implementation is shipped.

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
and `video` respectively. The server generates session identities; `stream_id`
is descriptive metadata and may repeat across isolated runs.

`POST /v1/streams` accepts the query parameters shown in Swagger (`pipeline`,
`rtsp_url` or `mjpeg_url`, `stream_id`, `fps`). Without a camera URL it uses the
pipeline's configured autonomous source (folder, camera or broker); `source_rest`
needs an explicit camera URL here. Its response includes a generated
`session_id` and capability. Stop it with `DELETE /v1/streams/{session_id}` and
`X-Session-Capability`.

## Stateful image processing

Repeated image calls continue temporal state only through an explicit handle.
An image call without `session_id` creates and closes an isolated one-shot
session, even when another call uses the same external `stream_id`.

```bash
curl -X POST http://localhost:8080/v1/sessions \
  -H 'Content-Type: application/json' \
  -d '{"pipeline":"driver_behaviour","stream_id":"camera-1"}'
```

The response contains `session_id`, `capability` and `timeline_id`. Copy its
values into `SESSION_ID` and `CAPABILITY`, then process captures in sequence:

```bash
curl -X POST http://localhost:8080/v1/infer/driver_behaviour \
  -H "X-Session-Capability: $CAPABILITY" \
  -F image=@frame.jpg -F "session_id=$SESSION_ID" \
  -F media_time_ns=0 -F seq=0

curl -X DELETE "http://localhost:8080/v1/sessions/$SESSION_ID" \
  -H "X-Session-Capability: $CAPABILITY"
```

Use capture-relative media time in nanoseconds. Equal timestamps are valid with
increasing sequence numbers. A backwards timestamp starts a fresh timeline and
resets temporal state; evidence cannot combine those timelines. If timestamps
are absent, the server marks its capture-interval estimate explicitly. Wall-clock
audit timestamps do not drive fusion or evidence windows. HTTP sessions expire
after five idle minutes and cannot resume algorithm state after process restart.
A missing/incorrect capability returns 403; selecting another pipeline for an
existing session returns 409.

Each live browser connection owns a separate session. Before a binary JPEG, the
client may send JSON with `media_time_ns`, `client_ts_ms` and `seq` for that next
frame. The server assigns its own sequence for accepted decoded frames and
returns session/timeline IDs and drop counts. Live processing keeps one pending
frame, observing input for evidence before replacing stale inference work.

Evidence notifications are available through `/v1/ws/evidence.<session_id>`;
incident notifications use `/v1/ws/alerts.<session_id>`. Notifications can be
missed, so query records to recover authoritative status.

## Evidence and errors

`GET /v1/evidence/clips` lists saved clips and their API proxy URLs.
`GET /v1/evidence/file/{bucket}/{key}` supports HTTP byte ranges for video
playback. Object keys may contain slashes. `/v1/evidence` redirects to the
original MinIO Console instead of rendering a replacement storage UI.

The default server captures a snapshot and clip for every durably accepted
incident. `/v1/evidence/records` exposes `Pending`, `Saved`, `Incomplete` or
`Failed` along with requested/actual media intervals, missing context, attempts,
reason codes and retry availability. A short one-shot input commonly produces
`Incomplete` evidence because a complete pre/post window is unavailable. Manual
retry uses retained spool material; it cannot create missing camera footage.
Completed evidence remains until manual deletion. Storage/queue overload is a
visible status outcome rather than silently losing an accepted incident.

The `/records/.../assets/...` route reads through the configured object storage
adapter. Existing MinIO file and clip routes remain compatibility entry points.
The [redesign guide](redesign.md) explains spool durability, outbox delivery and
rollback through a designated previous release artifact.

Unknown resources return 404, malformed/invalid configuration returns 400,
request-schema validation returns 422, ownership errors return 403, and backend
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
