# Source control and measured capacity

Historical measurements: [RTX 4060 data](benchmarks/rtx4060-2026-10-19.json).
Match the recorded workload and software before using these numbers for sizing.

The system does not have a fixed camera count. A stable `camera_id` identifies
an RTSP camera, video or image; each start creates a fresh processing session.
Recent inference results, session history and the latest annotated thumbnail
are retained under that source identity. Incidents and evidence use the existing
pipeline SQL/object storage, connected through session ID and incident
`stream_name` (the camera ID). Removing registration preserves this history.

## Run locally on the RTX 4060

From the repository root, using the existing CUDA/server/video environment:

```bash
.venv/bin/python scripts/serve_local.py --runtime cuda
```

Open [Sources](http://127.0.0.1:8081/ui/cameras). This uses real project models,
PostgreSQL records, persistent-disk evidence and MediaMTX routing. Start dependencies
with `docker compose up -d postgres mediamtx` first. The ordinary server/Compose UI has the same Sources
page at `/ui/cameras` and uses PostgreSQL with persistent-disk evidence by default.
MinIO/S3 remains an optional storage adapter.
For a fresh environment, install `pip install -e '.[server,cuda,video,local]'`.
Run one API process for the source registry. For fixed camera shards with
scoped ownership, use the [static worker deployment](static-workers.md).

1. Choose **Add source** and assign a permanent ID and display name.
2. Select RTSP, video or image, then choose a project pipeline.
3. Enter the RTSP URL, upload a file, or provide a server-local file path.
4. Set inference FPS and a processing resolution limit. Aspect ratio is preserved;
   the system downsizes but does not upscale the source. Set preview FPS/JPEG
   quality independently. Loop applies to video; an image is processed once.
5. Register, then **Start**. Stop before editing. Select **History** to inspect
   successive sessions, recent result JSON and session evidence records.

Each visible tile uses the latest frame actually processed by its pipeline, with
generic detections and the project's existing annotation plugin. Results are not
drawn onto a different, newer video frame. One cached JPEG is shared by viewers.
The current wall refreshes approximately every 1.5 seconds; server preview encoding
is separately capped, default 2 FPS. Hidden/offscreen tiles avoid repeated image
downloads. An inactive tile explicitly shows the age of its last result. These
are annotated analysis previews, not a full-source-FPS video wall.

The registry is operated through the same trusted API boundary as the existing
workspace. Use the existing authenticated gateway/private access arrangement for
remote access. Camera URLs/credentials are omitted from list/edit responses;
leave URI blank when editing to preserve it. Catalog storage contains credentials
and should be included in private backups. Only operator-controlled local paths
should be registered. Uploaded originals are retained until explicitly cleaned up;
source removal does not delete evidence or original files.

Recent frame-result history is written in short batches through a bounded
256-record queue. History reads and graceful shutdown flush pending batches;
an abrupt crash can lose recent uncommitted frame-history entries. Incident and
evidence acceptance uses the existing separate durable repository. Persistent
runtime data should use a native Linux filesystem: this checkout's FUSE/NTFS
mount can make synchronous database/spool writes dominate inference latency.

## Settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `INFEROPS_MAX_ACTIVE_SOURCES` | `8` | Configured admission limit; **not a hardware capacity claim** |
| `INFEROPS_MAX_SESSIONS` | `128` | Existing global session ceiling, including other API clients |
| `INFEROPS_SOURCES_PATH` | `.run/sources` | Catalog, latest preview files and uploads; local launcher uses its data directory |
| `INFEROPS_SOURCE_RESULT_HISTORY` | `1000` | Recent frame-result records retained per camera; older records are evicted |
| `INFEROPS_SOURCE_UPLOAD_MAX_BYTES` | `536870912` | Per-upload size ceiling (512 MiB) |
| `INFEROPS_CAPACITY_REPORTS` | `.run/capacity` | Parent directory of timestamped benchmark reports |
| `INFEROPS_OPENCV_THREADS` | `1` | Native OpenCV threads in the API; limits oversubscription across cameras |

Per-source JSON also supports `rtsp_transport`, `timeout_s`, `preview_width`,
`max_width`, `max_height`, `loop` and `autostart`. The source limit is distinct
from the number registered and from the number of browser viewers. Autostart
opens new sessions after restart; it does not restore tracker/fusion memory.

Compose persists the catalog/uploads/previews in `source-data`, separate from
the evidence spool, and mounts `.run/capacity` read-only for report inspection.
Changing source limits does not automatically retune GPU concurrency or Triton.

## API workflow

| Request | Purpose |
| --- | --- |
| `POST /v1/cameras/media` | Multipart field `file`; validate and retain uploaded image/video |
| `POST /v1/cameras` | Register a `SourceSpec` |
| `GET /v1/cameras` | List registration, current session/state, rates and configured admission |
| `PUT /v1/cameras/{camera_id}` | Edit a stopped source; identity is immutable |
| `POST /v1/cameras/{camera_id}/start` | Open a new session with the camera ID as stream name |
| `POST /v1/cameras/{camera_id}/stop` | Stop capture, close the session and finalize evidence |
| `DELETE /v1/cameras/{camera_id}` | Stop and unregister; retain history |
| `GET /v1/cameras/{camera_id}/preview.jpg` | Latest processed, annotated JPEG |
| `GET /v1/cameras/{camera_id}/live` | MediaMTX native WebRTC/HLS player URLs; original video without annotations |
| `GET /v1/cameras/{camera_id}/runs` | Sessions and terminal status across restarts |
| `GET /v1/cameras/{camera_id}/results` | Bounded recent results, each tagged with camera and session ID |
| `GET /v1/evidence/records?session_id=...` | Authoritative incident evidence for a session |
| `GET /v1/system/hardware` | Actual server CPU/RAM/disk and available NVIDIA telemetry |
| `GET /v1/system/capacity?pipeline=driver_behaviour&width=1920&height=1080&fps=5` | Matching measured profile, confidence and starting recommendation |

Example registration (start is a separate request):

```json
{
  "camera_id": "entrance-01",
  "name": "Main entrance",
  "kind": "rtsp",
  "uri": "rtsp://camera/stream",
  "pipeline": "driver_behaviour",
  "inference_fps": 5,
  "max_width": 1920,
  "max_height": 1080,
  "preview_fps": 2,
  "jpeg_quality": 75,
  "rtsp_transport": "tcp"
}
```

Existing direct image/session, offline upload/job and Python engine interfaces
remain available. Finite files in the registry are paced for live-style viewing;
the ordinary offline job path processes files without real-time pacing. RTSP
readers have dedicated threads, per-context FFmpeg options, a latest-frame slot
and bounded reconnect backoff. Native reads must finish within their configured
I/O timeout before shutdown; a failed shutdown is not reported as success.
The live input is sampled at the configured inference rate before the current
evidence observer; it does not independently capture higher-FPS evidence.

## Capacity: what the calculation means

VRAM is only one constraint. A useful configuration must satisfy all of these:

```text
pipeline demand       = active sources × inference FPS × actual model-call mix
decode demand         = sum(source width × height × source FPS), codec dependent
preview demand        = visible consumers + bounded per-source JPEG encoding
evidence RAM estimate = sources × sample FPS × (pre-window + lag history) × JPEG size
disk growth           = accepted incident rate × evidence bytes + retained uploads/results
```

Decoded frame memory alone is approximately `width × height × 3` bytes per BGR
slot, before decoder buffers, model tensors, JPEG rings and active captures.
Shared model weights are not reloaded once per camera; multiplying model VRAM
by camera count is also an invalid sizing formula. Fixed 576/256 model tensors
do not make source-resolution or CPU face-detection costs disappear.

The calculator uses completed measured profiles matching the CPU/RAM, GPU UUID,
data filesystem, runtime implementation/dependency versions, model checksums,
project pipeline/plugins, resolution limit, inference
FPS, evidence policy and preview-enabled workload. An unavailable GPU probe,
changed model/configuration or untested FPS/resolution returns **Unmeasured**.
It shows the highest tested passing concurrency, the first failing case and a
starting count with 20% count headroom. That headroom is a recommendation, not
a derived guarantee; a single passing camera has no demonstrated count headroom.
It never labels a short local sweep a certified maximum.

Pass for this first local sweep means each camera achieves at least 95% of the
requested FPS and aggregate receive-to-result p95 is at most 1 second, with no
processing errors. Full production acceptance still needs per-camera latency,
long-run stability, real RTSP/network, representative footage, sustained evidence
bursts and the actual deployment services. Spatial resolution and JPEG quality
are configuration controls, not an assessment of detection accuracy.

Reproduce a measured profile:

```bash
.venv/bin/python scripts/benchmark_capacity.py \
  --runtime cuda --video /path/to/representative-1080p15.mp4 \
  --resolutions 1920x1080 --fps 1,5,10 --counts 1,2,3,4 \
  --warmup 5 --seconds 30 --evidence --preview
```

The benchmark uses independent file decoders, isolated sessions, the real
detector/FaceMesh/tracker/fusion/incident pipeline, evidence buffering, bounded
result-history writes and the same annotated preview renderer. CUDA must be
active for every required loaded model. CPU fallback invalidates a CUDA test.
Reports include model hashes, actual providers, per-camera achieved FPS,
latency percentiles, peak server VRAM, RSS, decoded frames, CPU time and model
calls. No live GPU model is guessed from an unavailable `nvidia-smi` command.
The default benchmark repository is PostgreSQL; pass `--database-url` with a
dedicated benchmark database. `--repository memory` is a separate
diagnostic mode. Repository dialect, evidence storage and runtime/preview settings
must match before a report is used by the capacity panel.

The [MediaMTX integration](media-router.md) now supplies shared camera
paths and the upstream browser players, while retaining the current engine and
packaged UI. `preview_enabled: false` disables only annotated preview work.
A raw WebRTC stream does not automatically contain
inference annotations: publishing an annotated stream needs explicit encoding,
or a timestamp-aligned browser overlay. The current JPEG wall already provides
annotations without adding that separate encoder/bandwidth cost.
