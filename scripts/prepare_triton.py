"""Prepare isolated, parity-qualified Triton models without changing originals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inferops.adapters.outbound.inference.triton_repository import prepare_repository

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="projects/driver_behaviour/model_repository")
    parser.add_argument("--output", default=".run/triton/models")
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--queue-delay-us", type=int, default=2000)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--rfdetr-model", action="append", default=[], help="Explicit RF-DETR foreground contract (threshold 0.35); requires representative video")
    parser.add_argument("--sample-video")
    args = parser.parse_args()
    providers = ["CPUExecutionProvider"]
    if args.device == "cuda":
        import onnxruntime as ort

        ort.preload_dlls()
        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "arena_extend_strategy": "kSameAsRequested",
                    "use_tf32": 0,
                    "cudnn_conv_use_max_workspace": 0,
                },
            ),
            "CPUExecutionProvider",
        ]
    print(
        json.dumps(
            prepare_repository(
                args.source,
                args.output,
                max_batch_size=args.max_batch_size,
                providers=providers,
                queue_delay_us=args.queue_delay_us,
                detector_models=args.rfdetr_model,
                sample_video=args.sample_video,
            ),
            indent=2,
        )
    )
