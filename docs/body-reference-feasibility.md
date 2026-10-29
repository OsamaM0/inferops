# Driver torso reference: BUS-101-CAM1 feasibility test

The desired `forward` direction means the vehicle's road-forward direction, not
the camera optical axis. A head-pose network alone does not establish that
reference. Lowering the camera-relative display thresholds to 10 degrees yaw /
8 degrees pitch caused false right labels reported by the user. Defaults are
restored to 25 / 18 while the reference problem is investigated. This rollback
does not establish road-relative accuracy.

## Experiment, 2026-10-29

Run on the RunPod server, separately from the serving virtual environment and
without adding body inference to either live pipeline. The active source named
BUS-101-CAM1 uses `test.mp4`, duration 14.97 seconds. Twenty-four evenly spaced
frames were tested with the official MediaPipe Pose Landmarker Full and Heavy
float16 model bundles, CPU delegate, IMAGE mode, maximum input height 720 pixels.
The live SCRFD and 6DRepNet360 models supplied face ownership and head rotation.

| Body model | Pose detected | Torso quality gate passed | Body CPU median | Body CPU p95 |
| --- | ---: | ---: | ---: | ---: |
| Full | 24/24 | 6/24 | 55.3 ms | 67.4 ms |
| Heavy | 24/24 | 14/24 | 99.8 ms | 104.8 ms |

These timings exclude the first inference and measure only `detect()`, with the
live camera also running. They are not whole-pipeline FPS or accuracy scores.
The quality gate requires both shoulders and hips: visibility >= 0.65, presence
>= 0.70, in-frame positions, shoulder span >= 30 original-image pixels,
nondegenerate 3D trunk, and a pose nose associated with the detected face.

The shoulders/hips form an experimental 3D torso basis. Tests compared head
orientation against the instantaneous torso and an averaged, frozen torso
rotation. This coordinate conversion has not been independently validated with
labelled road directions. Even among frames passing the visibility gate, torso
pitch estimates ranged from 6.4 to 43.2 degrees for Full and 0.2 to 26.8 for Heavy.
Median torso yaw was -12.1 degrees for Full and -21.2 for Heavy. Visibility alone
therefore does not make this a sufficiently stable angular reference for small
head turns. Treating a reclining torso's forward axis as neutral head direction
also produces a substantial pitch offset.

## Deployment decision

Do not deploy the tested torso subtraction as a road-direction classifier.
Body landmarks can help associate a face with the driver's seat and reject
ambiguous calibration samples, but this test does not validate them as the sole
road reference. A fixed per-camera vehicle reference, established from a known
normal driving pose or an independently validated calibration method, is still
needed. Do not continuously redefine forward when the driver turns their torso.
BlinkCNN eye classification is unchanged.

Full reports, per-frame landmark visibility, model SHA256 values and contact
sheets are retained on the server under
`/workspace/inferops-data/experiments/body-reference/`. No body model or its
dependencies are installed in the serving environment.

Model and API documentation:
[MediaPipe Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker),
[Python guide](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python).
