"""Optional RF-DETR checkpoint adapter. Run in the isolated CPU export environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np


def infer_variant(state):
    """Infer only known detection architectures; loading remains strictly checked."""
    patch = state["backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight"].shape[-1]
    positions = state["backbone.0.encoder.encoder.embeddings.position_embeddings"].shape[1] - 1
    side = math.isqrt(positions)
    layers = len(set(re.findall(r"transformer\.decoder\.layers\.(\d+)\.", " ".join(state))))
    if side * side != positions:
        raise ValueError("Unrecognized position embedding shape; supply a supported architecture")
    variant = {(16, 2): "nano", (16, 3): "small", (14, 3): "base"}.get((patch, layers))
    if patch == 16 and layers == 4:
        variant = {36: "medium", 44: "large"}.get(side)
    if variant is None:
        raise ValueError("Cannot infer RF-DETR variant; select architecture explicitly")
    return variant


def compare_detections(actual, expected, *, classes, threshold=0.35):
    """Compare the RF-DETR foreground contract, allowing proposal permutation.

    Background proposals can change when top-k rounds near tied logits. They
    never reach this project's detector postprocessor. Foreground count, class,
    confidence and box must still agree; threshold crossings fail validation.
    """
    def foreground(values):
        boxes, logits = (x[0] for x in values)
        scores = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
        labels = scores.argmax(-1)
        confidence = scores[np.arange(len(scores)), labels]
        keep = (labels < classes) & (confidence >= threshold)
        return labels[keep], confidence[keep], boxes[keep]

    al, ac, ab = foreground(actual)
    el, ec, eb = foreground(expected)
    if len(al) != len(el):
        raise ValueError("PyTorch/ONNX foreground detection counts differ")
    compatible = (al[:, None] == el[None]) & np.isclose(ac[:, None], ec[None], rtol=1e-3, atol=1e-4)
    compatible &= np.isclose(ab[:, None], eb[None], rtol=1e-3, atol=1e-4).all(-1)
    matches = {}

    def assign(left, visited):
        for right in np.flatnonzero(compatible[left]):
            if right in visited:
                continue
            visited.add(right)
            if right not in matches or assign(matches[right], visited):
                matches[right] = left
                return True
        return False

    if not all(assign(left, set()) for left in range(len(al))):
        raise ValueError("PyTorch/ONNX foreground classes, boxes or confidence differ")
    return len(al)


def export_checkpoint(checkpoint, output, *, variant="auto", resolution=576, max_batch=4):
    import importlib.metadata

    import onnx
    import onnxruntime as ort
    import rfdetr
    import torch

    if max_batch not in {2, 4, 8} or resolution < 32 or resolution > 2048:
        raise ValueError("Unsupported batch/resolution")
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # No fallback to executable pickle objects, including for legacy checkpoints.
    with torch.serialization.safe_globals([argparse.Namespace]):
        saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    state = saved.get("model")
    if not isinstance(state, dict) or "class_embed.weight" not in state:
        raise ValueError("Expected an RF-DETR detection checkpoint containing model weights")
    architecture = infer_variant(state) if variant == "auto" else variant
    classes = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium",
               "large": "RFDETRLarge", "base": "RFDETRBase"}
    if architecture not in classes:
        raise ValueError("Unsupported architecture")
    torch.set_num_threads(4)
    torch.manual_seed(20261019)
    model = getattr(rfdetr, classes[architecture])(
        pretrain_weights=None, device="cpu",
        num_classes=state["class_embed.weight"].shape[0] - 1, resolution=resolution,
    )
    # SDK 1.11 adds a nonlearned keypoint mask, unused by detection checkpoints.
    # Supply only this known compatibility buffer, then require every weight.
    compatibility_buffers = []
    if "_kp_active_mask" not in state and not model.model_config.use_grouppose_keypoints:
        state["_kp_active_mask"] = model.model.model.state_dict()["_kp_active_mask"]
        compatibility_buffers.append("_kp_active_mask")
    model.model.model.load_state_dict(state, strict=True)
    model.model.model.eval()
    del state
    args = saved.get("args", {})
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    names = args.get("class_names", [])
    del saved
    exported = model.export(output_dir=str(output / "export"), shape=(resolution, resolution),
                            dynamic_batch=True, opset_version=18, verbose=False, fp16=False)
    graph = onnx.load(exported, load_external_data=True)
    onnx.save_model(graph, output / "model.onnx", save_as_external_data=False)
    onnx.checker.check_model(graph)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(output / "model.onnx"), sess_options=opts,
                                   providers=["CPUExecutionProvider"])
    if [x.name for x in session.get_outputs()] != ["dets", "labels"]:
        raise ValueError("Unsupported detector output contract")
    rng = np.random.default_rng(20261019)
    checks = []
    for batch in sorted({1, 2, max_batch}):
        images = rng.uniform(-1, 1, (batch, 3, resolution, resolution)).astype(np.float32)
        actual = session.run(None, {"input": images})
        raw_parity = True
        detections = []
        for slot in range(batch):
            with torch.inference_mode():
                reference = model.model.model(torch.from_numpy(images[slot:slot + 1]))
            expected = [reference["pred_boxes"].numpy(), reference["pred_logits"].numpy()]
            singles = session.run(None, {"input": images[slot:slot + 1]})
            for value, baseline, pytorch in zip(actual, singles, expected, strict=True):
                if value.shape != (batch, *pytorch.shape[1:]) or not np.isfinite(value).all():
                    raise ValueError("Invalid output shape or nonfinite prediction")
                raw_parity &= bool(np.allclose(value[slot:slot + 1], pytorch, rtol=1e-3, atol=1e-4))
                np.testing.assert_allclose(value[slot:slot + 1], baseline, rtol=1e-3, atol=1e-4)
            detections.append(compare_detections([x[slot:slot + 1] for x in actual], expected,
                                                 classes=model.model_config.num_classes))
        checks.append({"batch": batch, "foreground_pytorch_parity": True,
                       "raw_pytorch_parity": raw_parity, "foreground_counts": detections,
                       "independent_samples_match": True})
    report = {
        "adapter": "rfdetr", "architecture": architecture, "strict_weights": True,
        "initialized_nonlearned_buffers": compatibility_buffers,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256((output / "model.onnx").read_bytes()).hexdigest(),
        "class_names": names, "classes": model.model_config.num_classes,
        "inputs": [{"name": x.name, "shape": x.shape, "dtype": x.type} for x in session.get_inputs()],
        "outputs": [{"name": x.name, "shape": x.shape, "dtype": x.type} for x in session.get_outputs()],
        "max_batch_size": max_batch, "checks": checks, "rtol": 1e-3, "atol": 1e-4,
        "pytorch_reference": "Independent ordinary single-image forwards, without export transformations",
        "foreground_threshold": 0.35,
        "validation_scope": "Synthetic inputs: strict ONNX batch separation; PyTorch foreground detection parity. Raw proposal parity is reported separately. Representative footage and task accuracy require separate validation.",
        "versions": {key: importlib.metadata.version(key) for key in ("torch", "rfdetr", "onnx", "onnxruntime")},
        "status": "validated",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", default="auto", choices=["auto", "nano", "small", "medium", "large", "base"])
    parser.add_argument("--resolution", type=int, default=576)
    parser.add_argument("--max-batch", type=int, default=4)
    options = parser.parse_args()
    print(json.dumps(export_checkpoint(**vars(options)), indent=2))
