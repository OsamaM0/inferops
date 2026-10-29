# Driver face, head direction and eye state

The active workflow selects the driver's face with **SCRFD**, estimates head
orientation with **6DRepNet360 Full Rotation**, locates the eyes with **InsightFace
2D106**, and scores each eye with **BlinkCNN-ONNX**. The full `driver_behaviour`
pipeline also runs RF-DETR for behavior objects. `driver_face` contains only
face/head/eye monitoring. No identity-recognition network is used.

## Runtime and preparation

All four logical face models expose ONNX, TensorRT and Triton ports. Runtime
selection is shared with RF-DETR. In native Triton deployment, five ONNX graphs
are served: RF-DETR, SCRFD, 6DRepNet360, 2D106 and BlinkCNN. The MediaPipe stage
is no longer part of either active face pipeline; its model declaration is kept
as `models/face_landmarker/model.yaml.example` for reference.

Install the `driver_face` extra for Pillow, e.g. `pip install -e '.[driver_face]'`.
Model preparation downloads the pinned 2D106 and BlinkCNN graphs and checks SHA-256.
The full-rotation head graph is prepared once from the official checkpoint using
`models/head_pose_360/prepare.py`. Export needs `torch==2.8.0`,
`torchvision==0.23.0` and `onnx`; serving the prepared graph uses ONNX only.
The optional `head_pose_export` extra declares export dependencies. A separate CPU
export environment can use the PyTorch CPU wheel index and then install ONNX.

The prepared artifacts live in `model_repository/{scrfd,head_pose_360,landmarks_106,blinkcnn}_onnx/1/model.onnx`.
Pinned hashes and download revisions are in each `models/*/model.yaml`.

## Head direction

6DRepNet360 receives the square-padded face crop (25% padding), RGB pixels,
Resize 256 / CenterCrop 224 and ImageNet normalization. Its predicted rotation
matrix is validated before use. The forward axis gives camera-relative yaw and
pitch without folding yaw at 90 degrees. Positive yaw means the driver's left;
positive pitch means down in an upright, unmirrored image.

`neutral_yaw_deg` and `neutral_pitch_deg` calibrate the installed camera. The
pipeline classifies yaw first at 25 degrees, then pitch at 18 degrees. The labels
are `forward`, `left`, `right`, `up`, `down`, or `unknown` when pose is unavailable.
They describe head orientation, not eye gaze. A visible face retains its box even
if its head or eyes cannot be resolved.

## Eyes

2D106 receives a loose 192x192 face crop using the upstream affine transform.
Its normalization is inside the pinned graph. Landmarks are mapped back to the
original frame before extracting two rotation-aligned 64x64 eye crops.
BlinkCNN receives RGB in `[0,1]`, shaped `[2,1,3,64,64]`: two eyes and one frame.
ImageNet normalization is inside this graph too and must not be applied twice.

The sigmoid outputs are per-eye closure scores, not calibrated probabilities for
this camera. A score at least **0.53** establishes closure; at most **0.1325**
establishes opening; the intermediate band preserves a valid previous state.
Without valid history the intermediate band is `unknown`. These defaults use
the author's published high/low operating point; the pipeline exposes both
thresholds for later camera-specific validation.

Both eyes must be closed for `drowsy_ear=1`. A wink is `mixed` and does not imply
bilateral closure. EAR remains diagnostic metadata and does not override BlinkCNN.
The existing three-second incident confirmation remains unchanged. Missing faces,
invalid head rotations, eyes smaller than 6 pixels, off-frame crops, absolute yaw
above 40 degrees or pitch above 35 degrees produce `unknown`, not `closed`.
Large face changes, rewinds and gaps over one second reset closure history.

## Driver selection and outputs

A source's `driver_roi` overrides the detector's `driver_region`. Without a
region, selection uses the largest face and short-term spatial continuity; set
the driver's region when passengers are visible. SCRFD candidates below 0.8 are
rejected when no independent presence verifier is configured. A landmark regressor
alone is not treated as a face-presence classifier.

The API preserves `face_box`, `face_detected`, SCRFD confidence and search age,
`head_direction`, `head_yaw_deg`, `head_pitch_deg`, `head_roll_deg`, `eyes_state`,
per-eye states and `eyes_closed_ms`. Source metadata identifies `6drepnet360` and
`blinkcnn`. Per-stage ages show cached measurements. Food/cigarette associations
still use the selected lower face box. Raw 106-point coordinates use the separate
`face_landmarks_106` field, not the legacy 478-point mesh field.

## References and validation limits

- [6DRepNet360 official model and paper](https://github.com/thohemp/6DRepNet360)
- [InsightFace 106-point model and markup](https://github.com/deepinsight/insightface/tree/master/alignment/coordinate_reg)
- [BlinkCNN weights and normalization](https://huggingface.co/fodorad/blink_detection)

Unit checks cover all five directions, full yaw range, malformed rotations,
original-frame eye geometry, normalization, bilateral closure, hysteresis, stream
isolation, time gaps, missing/hidden eyes and runtime selection. Real-video and
live-API verification establish that the workflow runs; the supplied videos do
not contain comprehensive labeled head angles or eye states, so availability
counts are not an accuracy estimate.
