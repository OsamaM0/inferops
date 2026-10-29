# Driver Behaviour

RF-DETR monitors behavior objects. The face workflow uses SCRFD for face selection,
6DRepNet360 Full Rotation for head direction, InsightFace 2D106 for eye locations,
and BlinkCNN for per-eye open/closed state. The face-only pipeline is `driver_face`;
`driver_behaviour` also includes behavior objects and tracking, while `driver_video`
retains detector-only monitoring.

Both active face pipelines publish head direction (forward/left/right/up/down),
head angles, bilateral/per-eye state and closure duration through the existing
signals and overlays. The driver's camera region can be configured as `driver_roi`.
Hidden/tiny eyes return unknown; a wink is mixed. EAR is diagnostic, and BlinkCNN
is authoritative for eye state. Existing sustained-incident confirmation remains.

The logical ports support ONNX, CUDA, TensorRT and Triton. Install the engine's
`driver_face` extra for image preprocessing. All serving graphs are ONNX; head
export alone needs a separate PyTorch environment. See
[model preparation, thresholds and head calibration](../../docs/driver-face-models.md).

## Install as an independent adapter

From the engine root: `pip install -e '.[cpu]'` and
`pip install -e projects/driver_behaviour`. The adapter distribution is named
`inferops-driver-behaviour`; its entry point is `driver_behaviour`. It is excluded
from the InferOps core wheel. Run `inferops adapters init driver_behaviour
./driver_workspace` to create a writable copy before preparing weights. See
[the library guide](../../docs/python-library.md) for runtime selection and
running multiple models and using the driver pipeline for face cropping and fusion.
