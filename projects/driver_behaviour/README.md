# Driver Behaviour project

This block adds driver monitoring to InferOps. It selects models from the shared
[model library](../../models/) and generates one `driver_behaviour` pipeline from
them; its rules, alerts and evidence annotation live here.

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
| [project.yaml](project.yaml) | Selected models, tracking, evidence sink and confirmation settings |
| [rules.yaml](rules.yaml) | Derived values, safety rules, driver states and alert actions |
| [plugins/frame_annotator.py](plugins/frame_annotator.py) | Project-specific evidence overlay |
| [../../models/rfdetr_driver/](../../models/rfdetr_driver/) | RF-DETR tensors, labels, decoding and its phone/seatbelt/smoking/eating signals |
| [../../models/face_detector/](../../models/face_detector/) | YuNet face detector, used by face_mesh to find the face |
| [../../models/face_mesh/](../../models/face_mesh/) | Face stage: tracking, head pose, eyes, mouth activity and its signals |
| [../../models/face_blendshapes/](../../models/face_blendshapes/) | MediaPipe blendshapes, used by face_mesh for eye state |
| [../../models/model_repository/](../../models/model_repository/) | Exported weights and Triton tensor configurations |

References such as `.models.face_mesh.adapter:FaceMeshStage` resolve into the
library, mounted as the project's `models` package; project extensions use
`.plugins`. Both keep working when the folder is renamed or mounted at /project.

Sample videos and training are kept in
[research/driver_training](../../research/driver_training/).
See the [extension guide](../../docs/project-blocks.md) for adding models and
actions, and the [dashboard guide](../../docs/interfaces.md) for video use.

Pick the models under **Configuration → Project settings → Models**, or set
`projects.driver_behaviour.models` in `configs/inferops.yaml`. ONNX, TensorRT
and Triton are runtime variants of each model.

| Models | Rules that run |
| --- | --- |
| all four (default) | everything: objects, head direction, eyes, drowsiness, distraction |
| `[rfdetr_driver]` | phone, seatbelt, smoking, eating; states and evidence still apply |
| `[face_detector, face_mesh, face_blendshapes]` | head direction, eyes closed, drowsiness, distraction |

With both RF-DETR and FaceMesh, smoking and eating are confirmed by overlap with
the mouth. Hands-off-wheel requires an external `hand_on_wheel` signal producer.

## Confirming behavior before alerts and evidence

Both driver profiles require **3 seconds** of continuing positive observations
before a behavior becomes an incident. The `confirm_ms` (3000), `min_samples` (3)
and `max_gap_ms` (1500) project settings apply to every rule and unsafe state
transition; change them under **Project settings**. A negative observation,
a missing-observation gap longer than 1.5 seconds, or a media-clock rewind resets
confirmation. Repeated timestamps do not count as additional observations.
Camera sessions have independent confirmation/tracking state. Unsafe state
transitions also require confirmation, so a transient state change cannot bypass
the behavior gate and create evidence. A single still image cannot establish a
three-second behavior. Cooldown/deduplication apply after confirmation.
Incident payloads record the required and observed duration and sample count.
The alert policy (`rated_only`) turns rule events and the confirmed critical
escalation into incidents; normal, recovered and intermediate state
notifications remain events and do not create duplicate evidence clips.

## Face direction and eye state

"Front" (looking at the road) is the only safe head state. Sustained looking
**left, right, up or down** raises `looking_left`, `looking_right`, `looking_up`
or `looking_down`; closed eyes raise `eyes_closed`; a driver face that turns out
of view raises `distraction`; long closures and high PERCLOS raise `drowsiness`.

Per frame the face stage:

1. **Tracks** the face from the previous frame's landmarks. The YuNet
   `face_detector` only runs to acquire it and every `redetect_interval_ms`;
   after a loss it re-acquires near the driver's last position, never a
   passenger's face.
2. Runs **FaceMeshV2** on a square, roll-aligned crop and keeps all 478
   landmarks, including the iris. FaceMeshV2 uses its designated `Identity_1`
   face-presence logit; an unrelated scalar head cannot promote a rejected crop.
3. Fits the landmarks to MediaPipe's canonical face (weighted Procrustes) for
   **yaw / pitch / roll**, smoothed with a One-Euro filter.
4. Compares the pose with the stream's **neutral pose**. Cab cameras are rarely
   on the driver's axis, so `neutral: {mode: auto}` learns it from attentive
   frames (eyes open, not looking down; raw pose, so brief road glances count).
   It never falls back to the camera axis and cannot be dragged by long glances.
   Set `mode: fixed` with `yaw`/`pitch` for a known installation. Until it is
   learned, `head_direction` is `calibrating` and only eye-based "down" counts.
5. Classifies the offset with hysteresis (`thresholds`, in degrees).
6. Judges the eyes from `face_blendshapes` (`eyeBlink*`) and a per-driver
   eye-aspect ratio. Lowered lids with eyes looking down (`eyeLookDown*`) are a
   phone/dash gaze (`down`), not closed eyes. Eyes narrower than `min_eye_px`
   (10 px) are `unknown` and never raise closed-eye alerts.

Direction rules fire on 3 s of continuous looking, or when 70% of the last
5 s was spent looking that way, which catches phone use made of short
glances. A face the detector still sees but landmarks cannot read is
`unknown` (no alarm); an established face that disappears, or a detection in
profile, is `away`. Dropouts shorter than `away_after_ms` hold the last state.

Every model call goes through the model registry, so ONNX Runtime CPU/CUDA,
TensorRT and Triton runs make the same decisions. Evaluate on videos with
`scripts/eval_face_direction.py` (`--mode cpu|cuda|triton`, or `--api URL` for a
running server; `--compare RUN_A RUN_B` checks parity between runs).

The full driver profile keeps the 80% detector and face confidence thresholds. Other
FaceMesh exports can set `face_flag_output` and `face_flag_is_logit` on the face
stage. `allow_haar_fallback` is accepted for old configurations but ignored:
YuNet replaced the Haar cascades. Model errors remain possible; test
representative camera views and keep the driver in view when validating.

## Install as an independent adapter

From the engine root: `pip install -e '.[cpu]'` and
`pip install -e projects/driver_behaviour`. The adapter distribution is named
`inferops-driver-behaviour`; its entry point is `driver_behaviour`. It is excluded
from the InferOps core wheel. Run `inferops adapters init driver_behaviour
./driver_workspace` to create a writable copy before preparing weights. See
[the library guide](../../docs/python-library.md) for runtime selection and
running multiple models and using the driver pipeline for face cropping and fusion.
