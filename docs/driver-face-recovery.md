# Driver-face recovery and activation

The second iteration improves how the detector searches the camera image and
validates borderline observations. It is active on the existing `driver_face`
camera pipeline. The saved camera address and driver region were retained.

## Detection changes

- The camera's `driver_roi` is cropped **before** resizing to the detector's
  input. This gives the driver more input pixels. Context searches still reject
  face centers outside the region, preventing a visible passenger from being
  selected instead.
- Recently measured face boxes guide a local search. Full-region searches
  periodically reacquire the driver. A large box change, region change, frame
  size change, clock rewind or long gap invalidates tracking state.
- The detector proposes candidates at 0.5. Candidates below 0.8 require at least
  0.8 FaceMesh presence plus nondegenerate landmark geometry. Detector scores
  remain their actual values; verification does not inflate them. Strong
  detections retain independent head-pose analysis when eyelid landmarks fail.
- Overlapping crops and a contrast-enhanced view can recover difficult faces.
  Recovery advances across misses, with at most three search views per attempt
  and a 350 ms cache for negative results. This bounds the cost when the face
  stays obscured. New sightings can be delayed by up to this negative cache
  interval; an old positive is never substituted for a negative observation.
- Verified landmark tensors are consumed by the downstream mesh stage, avoiding
  a second landmark inference. The payload sent to clients excludes these
  internal tensors.

`driver_face` runs face detection, learned head pose, eyelid features and the
existing confirmed behavior rules without RF-DETR object detection. The full
`driver_behaviour` profile retains RF-DETR. Both profiles share the recovery
settings. `SourceSpec.driver_roi` is validated, stored with the camera and
carried into each processed frame.

The learned pose's published up-positive pitch convention is translated to the
application's down-positive convention. Direction defaults are 25° yaw and 18°
pitch, exposed as `direction_yaw_threshold` and `direction_pitch_threshold` on
the mesh stage. Neutral camera offsets remain configurable. These defaults
need camera-specific validation and do not estimate eye gaze.

## Evaluation

A larger SCRFD-34G ONNX model was compared with the Buffalo SCRFD-10G model on
the same 20 frames. At 0.8, larger-model full-frame face availability changed
only from 12 to 13 frames. At a proposal threshold of 0.5, it changed from 17 to
18. This small sample did not justify replacing the deployed model solely on
size. Upstream's [SCRFD project](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)
documents the model variants and benchmark conditions; the local comparison is
not a measured detection-accuracy result.

The complete new cascade was replayed on CPU and on the existing Triton GPU
server. With explicitly selected driver regions, it produced verified face and
pose observations in 13 of the 20 mixed sample frames, including views that
previously selected a passenger. The region settings and acceptance process
differ from the first iteration, so the aggregate counts must not be treated as
an accuracy improvement percentage.

An evenly spaced replay of the active phone-use video produced face and pose
observations in 44/61 samples; the control video produced them in 63/63. Many
missing phone-video intervals show the driver's hands covering her face. These
unlabeled availability counts do not measure precision, recall or eye-state
accuracy. Covered or unusable eyes remain `unknown`.

Checks passed 56 driver-adapter tests (including HTTP and the real 187-frame
video) and 82 affected core tests, plus Ruff/whitespace checks. Regression
coverage includes cropping and inverse box mapping, candidate verification,
private tensor reuse, region/clock cache invalidation, loss of face, rotating
recovery views, negative-cache backoff, persistent ROI propagation, eye-state
hysteresis and temporal incident confirmation.

After activating the bounded recovery schedule, the running camera was healthy
on Triton at approximately 24.5 FPS against its 25 FPS source cadence. This is a
live observation for this source/server, not a capacity guarantee. A montage of
the 17 unavailable phone-video samples is in
`.run/face-recovery-v2/missed-phone-frames.jpg`: these show hands covering the
face or the driver turned away, rather than visible frontal eyes being labeled
closed or open without evidence.

Review artifacts are in `.run/face-recovery-v2/` and
`.run/face-recovery-v2-triton/`. The native application was restored through
`runpod-start.sh`; the final recovery scheduling change was activated with
`POST /v1/configs/reload?apply_to_running=true`, which reconnected the camera
without restarting its service dependencies. Use this reload endpoint for
pipeline-only changes. Stopping the supervisor-owned API also stops its local
dependencies, so model/runtime changes need the complete native launcher.

The model comparison, licensing and remaining eye-classifier options are in
[the face-model guide](driver-face-models.md). No trained-from-scratch model or
ground-truth accuracy claim is implied by these changes.
