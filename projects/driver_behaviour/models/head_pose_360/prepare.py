"""Export the official full-rotation checkpoint; no PyTorch is needed to serve it."""

from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import urllib.request
from pathlib import Path

CHECKPOINT_URL = (
    "https://cloud.ovgu.de/s/TewGC9TDLGgKkmS/download/6DRepNet360_Full-Rotation_300W_LP%2BPanoptic.pth"
)
CHECKPOINT_SHA256 = "3ee08f1e04b8d452a6c4a40926a6f38051894ae6d0aaa6d191fe6d8bc6e4f9c6"


def install(target: Path) -> None:
    try:
        import onnx
        import torch
        from torchvision.models.resnet import Bottleneck
    except ImportError as exc:
        raise RuntimeError(
            "Exporting head_pose_360 needs torch==2.8.0, torchvision==0.23.0 and onnx. "
            "Prepare the ONNX artifact once in an export environment; serving uses ONNX only."
        ) from exc
    reference_spec = importlib.util.spec_from_file_location(
        "inferops_6drepnet360_reference", Path(__file__).with_name("reference.py")
    )
    reference = importlib.util.module_from_spec(reference_spec)
    reference_spec.loader.exec_module(reference)
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="6drepnet360-") as directory:
        checkpoint = Path(directory) / "full-rotation.pth"
        with urllib.request.urlopen(CHECKPOINT_URL, timeout=120) as source, checkpoint.open("wb") as handle:
            while chunk := source.read(1024 * 1024):
                handle.write(chunk)
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
            raise RuntimeError("Unexpected 6DRepNet360 checkpoint checksum")
        model = reference.SixDRepNet360(Bottleneck, [3, 4, 6, 3], 6)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("model_state_dict", state), strict=True)
        model.eval()
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        with torch.inference_mode():
            torch.onnx.export(
                model,
                torch.randn(1, 3, 224, 224),
                str(target),
                input_names=["image"],
                output_names=["rotation"],
                opset_version=17,
                dynamo=False,
            )
        onnx.checker.check_model(str(target))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    install(parser.parse_args().output)
