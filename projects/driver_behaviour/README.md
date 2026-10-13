# Driver Behaviour project

This block adds driver monitoring to InferOps. Its adapters, model variants,
pipelines, rules, alerts and evidence annotation travel together.

From the repository root:

```bash
cp .env.example .env
docker compose up -d --build
```

Open http://localhost:8080/ui/. CPU/CUDA use trained ONNX models;
Triton/DeepStream use the Triton repository. All modes require real weights. MODE=auto is a CPU alias. Docker prepares the
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

## Install as an independent adapter

From the engine root: `pip install -e '.[cpu]'` and
`pip install -e projects/driver_behaviour`. The adapter distribution is named
`inferops-driver-behaviour`; its entry point is `driver_behaviour`. It is excluded
from the InferOps core wheel. Run `inferops adapters init driver_behaviour
./driver_workspace` to create a writable copy before preparing weights. See
[the library guide](../../docs/python-library.md) for runtime selection and
running multiple models and using the driver pipeline for face cropping and fusion.
