# Face Counting project

Counts people by face on each camera and tracks them, so one person is
counted once while they stay in view. It selects models from the shared
[model library](../../models/) and generates one `face_counting` pipeline.

| Counter | Meaning |
| --- | --- |
| `unique_faces` | Distinct people seen in this camera session. One per stay in view. |
| `faces_in_frame` | Counted faces visible right now. |
| `new_faces` | Faces counted for the first time in this frame. |
| `line_in` / `line_out` | Crossings of the camera's counting line (when configured). |
| `zone_entries` / `zone_occupancy` | People who entered the zone / are inside now (when configured). |

The counters appear as live signal chips on the camera preview and in every
pipeline result, and the evidence overlay draws IDs, the line, the zone and a
counter panel, with each newly counted person marked **NEW #n**.

## What counts as one person

- While a person stays in view, short gaps keep the same ID. These include a head
  turn, someone walking in front, or a few missed detections. AuraFace face
  embeddings re-link the face to its lost track (`reid_threshold`), and the
  track is held for `hold_ms`, 5 s by default.
- A face lost **at the frame border** has left the view. Its track closes after
  `exit_ms` (0.5 s), so the person **counts again** when they come back.
- Faces are compared only within one camera session. Nothing is stored. Face
  vectors live in memory inside the tracker and are never published or saved.
- A track is counted only after `min_hits` detections (3) spread over at least
  0.3 s (`min_track_ms`), which filters one-off false detections at any frame
  rate. A second box on a face that is already counted is not counted.
- Small faces (under 40 px, `min_reid_face_px`) are followed by position only,
  because face vectors from such small crops are unreliable. A face that
  reappears within one face-size of where it was lost, under 0.3 s ago, keeps its
  ID even if it no longer overlaps its old box.

Use **`line_in` as the entries count** when people walk past a door or gate. A
person sitting or standing in view for a long time, with their face turned away
longer than `hold_ms`, can be re-counted by `unique_faces`. They cannot re-cross
the line.

Limits of counting by face:

- People who never face the camera are not counted. Aim the camera at the
  direction people walk in.
- Printed faces (posters, screens) are faces too. They never cross a line, so
  `line_in` is unaffected; for `unique_faces`, use a zone that excludes them.
- After a very long absence inside the view (longer than `hold_ms`), the same
  person counts again in `unique_faces`.

## New-person alerts and evidence

Every newly counted person raises one `new_person` alert. Two people counted on
the same frame give two alerts. Each alert becomes an incident with an evidence
clip from before and after the moment. In the clip and on the preview, that
person is boxed in magenta as **NEW #n** for 3 s. `n` is the running number, so
the 12th person on that camera is #12. The alert payload carries `number`,
`track_id`, `box`, `camera` and `unique_faces`.

Set the severity, or turn the alerts off, with the **New person alert** project
setting (`low` by default). The other counters, and occupancy rules you add to
[rules.yaml](rules.yaml), are unaffected.

## Counting line and zone

Every camera's preview shows a counting line, and its crossings are counted as
`line_in` / `line_out`. By default this is a horizontal line through the middle
of the frame, where moving down counts as IN. Change it under **Project settings**:

- **Counting line**: `horizontal`, `vertical` or `off`;
- **Counting line position**: 0–1, the y of a horizontal line or the x of a vertical one;
- **IN direction**: `down`/`up` for a horizontal line, `left`/`right` for a vertical one.

For a different line on one camera, or a slanted line or a zone, set geometry
per camera id (the source's `camera_id`), normalised to 0..1, under the
`face_tracker` stage `counting` parameter. A camera's own entry replaces the
default line. Edit it in **Configuration → Pipelines**, or override the model
in `configs/inferops.yaml`.

```yaml
counting:
  default: {}                     # applies to cameras without their own entry
  cameras:
    door-cam:
      line: {points: [[0.0, 0.55], [1.0, 0.55]], invert: false, margin: 0.01}
      zone: {points: [[0.1, 0.2], [0.9, 0.2], [0.9, 0.9], [0.1, 0.9]]}
```

**IN** means crossing to the right-hand side of the arrow A → B on the image.
For a horizontal line drawn left to right, moving down the image is IN. Set
`invert: true` to flip it; the overlay draws the IN arrow. A track must move
further than `margin` from the line before it changes side. Only crossings
within the segment count, and IN/OUT alternate per track, so jitter on the line
never adds counts.

## Models and settings

| Model | Role | Weights license |
| --- | --- | --- |
| [face_retinaface](../../models/face_retinaface/) | RetinaFace ResNet-34 detector with 5 landmarks (default) | MIT |
| [face_detector](../../models/face_detector/) | YuNet, light detector: `models: [face_detector, face_tracker]` | MIT |
| [face_tracker](../../models/face_tracker/) | AuraFace-v1 embedder + the detect/track/count stage | Apache-2.0 |

The models were chosen to allow commercial use: InsightFace SCRFD/ArcFace and
EdgeFace weights are non-commercial, and YOLO-face is GPL/AGPL. The detectors
were trained on WIDER FACE, a dataset with research-oriented terms; have this
reviewed for your deployment.

Tune these under **Project settings**: `face_detection_confidence` (0.6),
`min_face_px` (20), `min_hits` (3), `reid_threshold` (0.45), `hold_ms` (5000)
and `exit_ms` (500). Calibrate `reid_threshold` on your own cameras:

- lower it if one person gets several IDs;
- raise it if two people share one.

For small, distant faces, raise the RetinaFace input in its `model.yaml` to
`[1, 3, 960, 960]`. Its graph accepts any size, and cost grows with the pixel
count. For more cameras per GPU, switch to the MobileNetV2 RetinaFace (noted in
its `model.yaml`) or to YuNet.

Run it on a GPU (`runtime.mode: cuda` or `triton`). Use
`inference_fps` 5–15 per camera. Lower rates still work, because appearance
re-identification takes over from box overlap, but they need more GPU time per
match.

## Uploaded videos and cameras

Each video run and each camera session starts its own count from zero. Nothing
is compared across videos, cameras or runs: running the same video twice counts
the same people twice.

Counting needs each person's face in several processed frames (`min_hits`, 3).
How many frames that gives depends on how the video is run:

- **Analyze all frames and save evidence** (dashboard video panel, or
  `POST /v1/run/face_counting/upload`) processes every frame of the file,
  dropping none, and saves alerts and evidence. Use it for uploaded clips,
  especially on a CPU-only server.
- **A camera source** (RTSP, or a video registered as a camera) behaves like a
  live camera. It samples `inference_fps` frames per second, and frames that
  wait longer than `max_frame_age_ms` are dropped. People who pass quickly
  need about 10 fps, which requires a GPU. At 1–2 fps, as on a CPU, most
  walkers are seen only once or twice and are not counted.

## Check counts on a video

```bash
python scripts/eval_face_counting.py clip.mp4 --runtime cuda --expected 12 \
  --line "0,0.55,1,0.55" --output annotated.mp4
```

| Folder/file | Edit here to change |
| --- | --- |
| [project.yaml](project.yaml) | Selected models (default or light profile) and the evidence sink |
| [rules.yaml](rules.yaml) | Derived values and optional alerts (for example occupancy) |
| [plugins/frame_annotator.py](plugins/frame_annotator.py) | Evidence overlay: IDs, line, zone, counter panel |
