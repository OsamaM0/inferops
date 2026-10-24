# MediaMTX routing, recording and playback

MediaMTX 1.21.2 is the default media service. InferOps retains source identity,
processing sessions, inference, annotations, incident decisions and evidence.

```text
RTSP camera → shared/on-demand MediaMTX path
                 ├→ PyAV → AI → annotated evidence spool → persistent disk (optional S3)
                 ├→ WebRTC/HLS → dashboard Live video
                 └→ original fMP4 recording → dashboard Recordings
PostgreSQL ← source catalog, run/result history, incidents and evidence state
```

## Default features

| Feature | Behavior |
| --- | --- |
| RTSP relay/fan-out | One path shares one upstream camera connection among inference and viewers |
| On-demand input | First reader opens the camera; last reader starts a five-second close timer |
| WebRTC | Default browser live player; signaling TCP 8889, media UDP 8189 |
| HLS | Alternate browser player on TCP 8888 |
| Recording | Original compressed video, fMP4, one-second parts and one-minute segments |
| Retention | Completed original recording segments expire after 24 hours by default |
| Playback | Lists recorded timespans and serves MP4 downloads/video on TCP 9996 |
| Control API | Authenticated path management on loopback TCP 9997 |
| Reconciliation | Each owner restores its registered paths every ten seconds after router restart |
| Metrics | Internal TCP 9998; included in the optional Prometheus scrape configuration |
| Isolation | Managed paths use `inferops/<owner-scope>/<camera-id>` |

Native recording runs while a stream is being read. Inference or an open live
viewer supplies that reader; this is not an always-on recorder for stopped cameras.
A segment must finalize before it is available for playback. Closing HLS playback
can leave a short server-side cleanup period before the upstream disconnects.
Offline placeholder video stays disabled so it cannot become AI input.

## Start

```bash
# All default services, with a native API:
make run MODE=cpu
# Or a container API:
make run-container MODE=cpu
# Direct Docker Compose:
docker compose up -d --build
```

For the native source-control launcher:

```bash
docker compose up -d postgres mediamtx
.venv/bin/python scripts/serve_local.py --runtime cuda
```

Use one application owner for a database/spool bundle. Start with empty
PostgreSQL, evidence disk, recording and spool storage. No SQLite records or historical
evidence are imported.

## Configuration

`INFEROPS_MEDIA_ROUTER=mediamtx` is the default. An explicit `direct` selection
is available for embedded/testing workflows and bypasses the router.

| Setting | Default / purpose |
| --- | --- |
| `INFEROPS_MEDIAMTX_API_URL` | Native `http://127.0.0.1:9997`; Docker `http://mediamtx:9997` |
| `INFEROPS_MEDIAMTX_RTSP_URL` | Native `rtsp://127.0.0.1:8554`; Docker `rtsp://mediamtx:8554` |
| `INFEROPS_MEDIAMTX_PLAYBACK_URL` | Native `http://127.0.0.1:9996`; Docker `http://mediamtx:9996` |
| `INFEROPS_MEDIAMTX_RECORD` | `true`; set `false` to disable recording on managed paths |
| `INFEROPS_MEDIAMTX_RECORD_RETENTION` | `24h`; MediaMTX duration |
| `INFEROPS_MEDIAMTX_RECORD_SEGMENT` | `1m`; minimum segment duration |
| `INFEROPS_MEDIAMTX_API_USER/PASSWORD` | Matching Control API credentials on API/workers/router |
| `INFEROPS_MEDIAMTX_BIND_ADDRESS` | `127.0.0.1`; set `0.0.0.0` for LAN access |
| `INFEROPS_MEDIAMTX_PUBLIC_HOST` | `127.0.0.1`; set the server LAN hostname/IP for WebRTC |
| `INFEROPS_MEDIAMTX_WEBRTC_PUBLIC_URL` | Optional browser-facing WebRTC base, including a proxy prefix |
| `INFEROPS_MEDIAMTX_HLS_PUBLIC_URL` | Optional browser-facing HLS base, including a proxy prefix |
| `INFEROPS_MEDIAMTX_PLAYBACK_PUBLIC_URL` | Optional browser-facing playback base, including a proxy prefix |

Each published port can be changed using its corresponding `..._PORT` setting.
Recording data uses the persistent `media-recordings` volume, separate from incident evidence
and the capture spool. Disk must be sized for the configured camera bitrates and
retention. Time retention does not impose a total byte quota.

After changing configuration or retention settings, recreate the media container:

```bash
docker compose up -d --force-recreate mediamtx
```

## Dashboard and API

Register an RTSP source through Sources. **Live video** opens MediaMTX's native
player through the API's same-origin proxy and defaults to HLS. Player HTML,
JavaScript, playlists and segments use the application URL, including HTTPS
hosted proxies where ports 8888/8889 are not publicly exposed. WebRTC signaling
is also proxied, but its media still requires reachable ICE ports or TURN.
**Open player** retains the direct public URL for separately exposed media services.
**Recordings** lists the most recent finalized original-video
timespans and plays the selected entry. Both are unannotated camera video.
Annotated previews and durable incident evidence remain separate.

`GET /v1/cameras/{camera_id}/live` returns browser-facing player URLs.
The `hls_proxy_url` and `webrtc_proxy_url` fields are relative to the API origin.
If media uses a different internal host or port, configure
`INFEROPS_MEDIAMTX_HLS_INTERNAL_URL` / `INFEROPS_MEDIAMTX_WEBRTC_INTERNAL_URL`;
otherwise these derive from the Control API host and ports 8888/8889.
`GET /v1/cameras/{camera_id}/recordings?start=<RFC3339>&end=<RFC3339>&limit=100`
returns timespans and browser-facing MP4 URLs. API/worker responses rebuild public
URLs at the API boundary and omit private upstream addresses and credentials.
Direct mode and local files return 409 for these RTSP-only features.

Live/playback readers use the bundled anonymous-read policy restricted to managed
paths; Control API access uses credentials. Configure MediaMTX authentication
and reverse-proxy access consistently for deployments requiring user login.

## Processing and storage responsibilities

MediaMTX does not resize, infer, annotate or transcode in this integration. Camera
codecs must be browser-compatible. The AI path uses PyAV to decode frames, and
PyAV/FFmpeg/OpenCV provide processing and annotated evidence encoding. Persistent disk
stores verified incident artifacts by default; optional MinIO/S3 uses the same storage port.
PostgreSQL stores lifecycle state; the spool
retains unfinished evidence across crashes. Original recording retention never
deletes the application's incident evidence.

Upstream: [recording](https://mediamtx.org/docs/features/record),
[playback](https://mediamtx.org/docs/features/playback),
[re-encoding](https://mediamtx.org/docs/features/remuxing-reencoding-compression),
and [browser players](https://mediamtx.org/docs/read/web-browsers).

## Verification and capacity

Unit/integration checks cover default configuration, path recovery, recording
settings, PostgreSQL source history, URL boundaries and worker routing. The real
browser scenario requires a dedicated PostgreSQL database:

```bash
.venv/bin/python tests/scenarios/verify_media_router.py --runtime triton \
  --database-url postgresql+asyncpg://postgres:postgres@127.0.0.1:5432/inferops_verification
```

Past reports under `docs/benchmarks` retain their original configuration. New
PostgreSQL/evidence-disk/recording deployments need matching capacity measurements.
