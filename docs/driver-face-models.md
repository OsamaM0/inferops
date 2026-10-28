# Driver face, eye-state and head-pose upgrade

Research and implementation: 28 October 2026. Target runtimes: CPU, CUDA and
Triton, plus the existing TensorRT library variant.

The [recovery and activation report](driver-face-recovery.md) describes the
subsequent driver-region search, candidate verification and live activation.

The recommended stack for this project is **SCRFD-10G face detection →
6DRepNet head pose + FaceMesh eyelid landmarks → temporal confirmation**.
RF-DETR continues to detect phones, seatbelts, food and cigarettes. These are
separate tasks: identity recognition accuracy does not establish eye-closure
or head-direction accuracy.

## Models considered

| Model/project | Relevant output | Recommendation for this project |
| --- | --- | --- |
| [InsightFace Buffalo](https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md) | `buffalo_l` bundles SCRFD-10GF, identity embeddings, 106-point 2D/68-point 3D alignment and attributes; `buffalo_sc` omits dense alignment. | Use the face detector alone. Recognition and age/gender are unnecessary for the requested monitoring. |
| [SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd) | Face boxes and optionally five keypoints; several compute sizes. | The 10G KPS model is integrated. Consider 2.5G for constrained CPUs, after qualifying that export's tensor contract and footage. Five keypoints alone cannot measure eyelid closure. |
| [RetinaFace](https://arxiv.org/abs/1905.00641) | Face detection and five facial keypoints. | A useful alternative benchmark; it still needs a separate eye/pose stage. Do not replace the current stack solely on a generic face benchmark. |
| [OpenCV YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | Lightweight face detector with facial keypoints. | CPU-oriented alternative, with MIT-licensed files in the model directory. Compare face recall on the actual camera, especially profiles and small faces. |
| [MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker) | 478 landmarks, optional 52 blendshape coefficients and facial transformation matrices. | Retain the existing FaceMeshV2 ONNX for eyelid geometry. The standalone ONNX in this repository does **not** produce the full bundle's blink blendshapes or transformation matrices. |
| [6DRepNet](https://github.com/thohemp/6DRepNet) / [6DRepNet360](https://github.com/thohemp/6DRepNet360) | Learned head orientation; the 360 project addresses full-range rotation. | 6DRepNet is integrated using [Qualcomm's ONNX export](https://huggingface.co/qualcomm/SixDRepNet). Its RGB input is in [0,1], with ImageNet normalization embedded in the graph. 360 is a further candidate for severe turns; it has not been integrated here. |
| [3DDFA_V2](https://github.com/cleardusk/3DDFA_V2) | Dense 3D alignment and pose, with ONNX support. | Useful alternative for large-pose alignment. Its author notes tracking can fail above 90° or under fast motion. It is not a dedicated eye-closure classifier. |
| [OpenVINO open-closed-eye-0001](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/open-closed-eye-0001/README.md) | Two-class open/closed result per 32×32 BGR eye crop. | Next candidate if calibrated EAR remains insufficient. The published 95.84% is on its MRL Eye Dataset split, not this driver's camera. Export/runtime qualification and evaluation are still required. |

These recommendations are engineering judgments based on the task fit and
project replay, not a claim that one model is universally the most accurate.

InsightFace's code is MIT, while its published pretrained models are restricted
to non-commercial research. Commercial deployment needs appropriate model
rights or replacement weights. See the [upstream license policy](https://github.com/deepinsight/insightface#license).

## Implemented changes

The full `driver_behaviour` pipeline now runs an independent SCRFD face detector
before the pose and mesh stages. Its result is authoritative: a rejected face
does not silently become a face through the RF-DETR behavior box. The configured
strong-face and mesh confidence thresholds are 0.8; borderline 0.5 detector
proposals need independent mesh verification. Haar remains disabled.

6DRepNet estimates head orientation independently of mesh success. Missing eye
landmarks can therefore coexist with a valid head-direction estimate. Rotation
matrices are checked for finite values, orthogonality and proper determinant.
The artifact installer downloads Qualcomm release v0.65.0 and embeds its external
ONNX tensors into a single model file for the existing serving workflow.

Face crops preserve square geometry, including padding at frame edges. The mesh
parser retains all 478 points when available. The legacy landmark-only path uses
an OpenCV 3D pose fit instead of distance-ratio pseudo-angles; see
[OpenCV's PnP model](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html).

The old EAR score crossed the fusion threshold only below 0.105 despite a 0.21
setting. The old pitch score was capped below the alert threshold. Scores now
cross the 0.5 decision boundary at the configured physical thresholds.

Eye states use separate close/reopen EAR thresholds (0.21/0.24 by default),
bilateral closure, and per-stream duration. A wink does not establish bilateral
closure. Tiny, off-screen or extreme-pose eyes produce `unknown`. Missing
observations, long gaps, rewinds and large face-box changes reset eye history.
The existing three-second continuous-positive confirmation still gates incidents;
an instantaneous `closed` eye state is not itself a drowsiness incident.

The dashboard receives `head_direction` (`forward`, `up`, `down`, `left`, `right`,
`unknown`), yaw/pitch/roll, `head_pose_source`, `eyes_state`, anatomical
left/right eye states, EAR values and `eyes_closed_ms`. Left/right assume an
upright, unmirrored camera. Head orientation is distinct from eye gaze.

## Camera configuration and runtime preparation

Set `driver_region` on the `detect_face` stage when other people are visible:
normalized `[x1, y1, x2, y2]` in the upright input frame. Without it, the largest
detected face is selected; this can be a passenger. The region filters detected
face centers, crops the selected area before model resizing, and restricts
context recovery results to that area.

Set `neutral_yaw_deg` and `neutral_pitch_deg` on the mesh stage from a driver
facing the road in that camera mounting. Defaults are camera-relative zero.
Calibrate the EAR thresholds against visible open/closed eyes for that driver;
they are initial settings, not universal physiological constants. Eye visibility
checks use the physical camera angle before subtracting the neutral offset.

The logical models are `rfdetr_driver`, `buffalo_detector`, `head_pose` and
`face_mesh`. Each has ONNX, TensorRT and Triton variants. Existing model
preparation discovers all four through the pipeline's `model` parameters.
From the repository root, with the runtime and artifact dependencies installed:

```bash
INFEROPS_EXECUTION_MODE=cpu inferops project prepare --domain-root projects/driver_behaviour
INFEROPS_EXECUTION_MODE=cuda inferops project prepare --domain-root projects/driver_behaviour
INFEROPS_EXECUTION_MODE=triton inferops project prepare --domain-root projects/driver_behaviour
```

Use the line for the intended runtime. Triton serves batch-one detector and pose
ONNX models using the supplied `config.pbtxt` files. Prepare and reload the model
server with the application when deploying the new pipeline. Model files remain
downloaded artifacts rather than Git content. Fresh preparation needs outbound
HTTPS. The recovery iteration has been activated on the running camera; see
the [activation report](driver-face-recovery.md).

## Validation and remaining limits

On 20 existing sample frames, at the configured 0.8 thresholds, the RF-DETR face
class was available in 0 frames and SCRFD in 12. The new mesh and learned pose
were available in 12 frames; an eye state was available in 7 (including one
mixed-eye result). These clips include split screens, tiny faces,
blur, profile views and passengers. These are **availability counts, not driver
detection accuracy**; no annotated face boxes, head angles or eyelid labels were
provided. The contact sheet must be reviewed with a driver region selected
before evaluating driver-specific behavior.

Real weights were exercised on CPU and subsequently on the live Triton GPU
server. Runtime resolution was checked for CPU/CUDA/TensorRT/Triton. A dedicated
pose model adds compute and memory. Current
100 ms sampling/caching does not guarantee any live FPS.

Regression tests cover detector decoding/NMS and padding, stream/clock cache
isolation, driver-region validation, synthetic 3D rotations, invalid learned
rotations, independent pose when mesh fails, export normalization, eye
hysteresis/winks and blink-versus-sustained-closure confirmation. Real video/API
checks exercise the complete pipeline. For deployment, collect labeled camera
clips covering all requested directions, closed/open eyes, glasses, night
lighting, occlusion and face loss; measure missed closures and false incidents.
PERCLOS and a learned eye-state classifier remain future enhancements.

The final checks passed 48 driver-adapter tests and 81 affected core tests
(129 total), plus Ruff checks on the changed Python files. Local review artifacts
are in `.run/face-upgrade-2026-10-28/` (`results.json` and `contact-sheet.jpg`).
The local detector was verified against the Buffalo archive, and the pose
installer reproduced the exact embedded model used for the CPU replay.
