# Checkpoint conversion

The dashboard's **Pipeline & models** view now accepts a trained `.pth`/`.pt` upload or an absolute checkpoint path inside the active project. The initial adapter supports RF-DETR detection models. Exporting arbitrary model families needs their architecture/export adapter; weights alone cannot guarantee a valid conversion.

## Optional prerequisite

Keep PyTorch and CPU ONNX Runtime in a separate environment from the API's CUDA ONNX Runtime:

```bash
uv venv --python .venv/bin/python .run/export-env
uv pip install --python .run/export-env/bin/python --torch-backend cpu \
  -r requirements/export-rfdetr.txt -e '.[cpu,artifacts]'
export INFEROPS_EXPORT_PYTHON="$PWD/.run/export-env/bin/python"
python scripts/serve_local.py --runtime cuda
```

The dashboard does not install dependencies. Its conversion controls show the missing prerequisite when the worker is not configured. In Docker, this environment must be installed in or mounted into the API container at a usable path; the default small API image does not include it. CPU export does not require a GPU.

Standalone conversion:

```bash
.run/export-env/bin/python -m inferops.adapters.outbound.inference.checkpoint_export \
  --checkpoint projects/driver_behaviour/model_repository/rfdetr_driver_behaviour_pth/checkpoint_best_regular_driver_behivour_85.pth \
  --output .run/checkpoint-driver --variant auto --resolution 576 --max-batch 4
```

## What validation means

The worker uses PyTorch's restricted `weights_only=True` loader, with an explicit `argparse.Namespace` allowance for legacy training metadata. It reconstructs a known RF-DETR variant and strictly matches all learned weights. The supplied legacy Medium checkpoint needs only the new SDK's empty, nonlearned `_kp_active_mask` buffer initialized; this is recorded in the report.

The SDK performs a genuine dynamic-batch export; the converter does not merely rename an ONNX dimension. It checks model I/O, finite outputs, and strict ONNX batch 1/2/4 separation against independent single-image executions. PyTorch foreground comparisons use this project's RF-DETR contract: sigmoid, highest-scoring class, foreground threshold 0.35, and exclusion of the extra background class. Foreground count, classes, boxes and scores must agree; reordered proposals are matched one-to-one. Raw PyTorch tensor parity is reported separately because low-information inputs can produce different background proposals. The synthetic export test produced no foreground detections, so its foreground comparison alone is not a representative validation or an accuracy result.

Before serving, qualify with representative video, and reject a foreground comparison with no detections:

```bash
# Put the exported detector at SOURCE/rfdetr_driver_behaviour_onnx/1/model.onnx
# and the original FaceMesh at SOURCE/face_mesh_onnx/1/model.onnx.
python scripts/prepare_triton.py --source SOURCE --output .run/triton/checkpoint-models \
  --device cuda --max-batch-size 4 --rfdetr-model rfdetr_driver_behaviour_onnx \
  --sample-video research/driver_training/test.mp4
python scripts/verify_triton.py --reference-repository .run/triton/checkpoint-models
```

The RF-DETR comparison is explicitly selected; other models retain strict raw-tensor qualification. Check `qualification.json` for the actual comparison, detections and enabled batch size. The threshold is currently 0.35; changing postprocessor thresholds or preprocessing requires requalification. A supplied checkpoint's class order must match the project, even when its display names differ. Task accuracy still requires labeled representative data.

Jobs are persisted under the selected source data directory's `models/` folder, have a bounded queue of two and one active CPU worker, and time out after 15 minutes. Restart/shutdown marks unfinished jobs interrupted and stops their child process. Failed exports retain a log. Successful jobs expose their ONNX, report, artifact path and content hashes. Upload limit defaults to 512 MiB.

## Activation

Exports are staged separately. To use one locally, change the selected model's `artifact.path` and `artifact.sha256` in Configuration, remove an old download source pointing at different weights, and reload the configuration. Existing sessions retain their original revision. To use it in Triton, prepare a separate repository, select `INFEROPS_TRITON_REPOSITORY`, restart/load the server explicitly, then verify the live model and counters. The generated config includes the serving ONNX hash so capacity profiles distinguish different remote weights.

Do not confuse conversion readiness with measured camera capacity. Run the evidence-enabled pipeline benchmark after activation.
