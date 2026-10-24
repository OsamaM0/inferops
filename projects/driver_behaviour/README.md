# Driver Behaviour project

This block adds driver monitoring to InferOps. Its adapters, model variants,
pipelines, rules, alerts and evidence annotation travel together.

From the repository root:

```bash
cp .env.example .env
docker compose up -d --build
```

Open http://localhost:8080/ui/. CPU/CUDA use trained ONNX models;
Triton use the Triton repository. All modes require real weights. MODE=auto is a CPU alias. Docker prepares the
configured models before inference starts.

| Folder/file | Edit here to change |
| --- | --- |
| [models/rfdetr_driver/model.yaml](models/rfdetr_driver/model.yaml) | RF-DETR tensors, labels, thresholds and artifact source and ONNX/Triton variants |
| [models/rfdetr_driver/adapter.py](models/rfdetr_driver/adapter.py) | RF-DETR output decoding |
| [models/face_mesh/model.yaml](models/face_mesh/model.yaml) | FaceMesh tensors and runtime variants |
| [models/face_mesh/adapter.py](models/face_mesh/adapter.py) | Face crops, landmark decoding and feature extraction |
| configs/pipelines | Stage order, mode choices, tracking and evidence triggers |
| configs/fusion/driver.yaml | Temporal signals and driver safety rules |
| configs/alerts/driver.yaml | Severity, dedupe, cooldown and actions |
| plugins/frame_annotator.py | Project-specific evidence overlay |
| project.yaml | Enabled pipelines, prewarm settings and platform overrides |
| model_repository | Exported weights and Triton tensor configurations |

Use project-relative references such as
`.models.rfdetr_driver.adapter:RFDetrPostprocess` and
`.models.face_mesh.adapter:FaceMeshStage`. They keep working when the folder is
renamed or mounted at /project. Project-level extensions use `.plugins`.

Sample videos and training are kept in
[research/driver_training](../../research/driver_training/).
See the [extension guide](../../docs/project-blocks.md) for adding models and
actions, and the [dashboard guide](../../docs/interfaces.md) for video use.

In **Configuration**, edit settings through forms and follow a pipeline's links
to its models, rules and alert policy. The two models are `rfdetr_driver` and
`face_mesh`; ONNX, TensorRT and Triton are runtime variants of those models.

| UI name | Stable pipeline ID | Purpose |
| --- | --- | --- |
| Full driver monitoring | `driver_behaviour` | Detection, face landmarks, temporal eye/head features, tracking, alerts and evidence |
| Detector-only monitoring | `driver_video` | Detection and phone, seatbelt, smoking and eating rules, alerts and WebSocket output; no FaceMesh, tracking or evidence sink |

Both pipelines accept images, videos and live frames. The older `driver_video`
name describes a reduced workflow, not a restriction to video inputs. IDs stay
unchanged so existing sources and clients keep working.

Hands-off-wheel requires an external `hand_on_wheel` signal producer and is not
a verified detector-only feature. See [pipeline declarations](configs/pipelines/).

## Confirming behavior before alerts and evidence

Both driver profiles require **3 seconds** of continuing positive observations
before a behavior becomes an incident. Each rule exposes `confirm_ms: 3000`,
`min_samples: 3` and `max_gap_ms: 1500` in Configuration. A negative observation,
a missing-observation gap longer than 1.5 seconds, or a media-clock rewind resets
confirmation. Repeated timestamps do not count as additional observations.
Camera sessions have independent confirmation/tracking state. Unsafe state
transitions also require confirmation, so a transient state change cannot bypass
the behavior gate and create evidence. A single still image cannot establish a
three-second behavior. Cooldown/deduplication apply after confirmation.
Incident payloads record the required and observed duration and sample count.
The alert policy's `event_types` list selects behavior incidents and confirmed
critical escalation; normal, recovered and intermediate state notifications
remain events and do not create duplicate evidence clips.

The full driver profile keeps the 80% detector and face confidence thresholds
and disables Haar fallback. Strict mode skips CPU cascade scanning entirely.
FaceMeshV2 uses its designated `Identity_1` face-presence logit, transformed by
sigmoid; an unrelated scalar head cannot promote a rejected crop to a face.
Other model exports can set `face_flag_output` and `face_flag_is_logit` on the
face stage. Model errors remain possible; test representative camera views and
keep the driver in view when validating detections.

## Install as an independent adapter

From the engine root: `pip install -e '.[cpu]'` and
`pip install -e projects/driver_behaviour`. The adapter distribution is named
`inferops-driver-behaviour`; its entry point is `driver_behaviour`. It is excluded
from the InferOps core wheel. Run `inferops adapters init driver_behaviour
./driver_workspace` to create a writable copy before preparing weights. See
[the library guide](../../docs/python-library.md) for runtime selection and
running multiple models and using the driver pipeline for face cropping and fusion.
