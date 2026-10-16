"""Create editable model ports without changing the InferOps engine."""
from __future__ import annotations

from pathlib import Path

import yaml

from inferops.adapters.outbound.projects.models import model_config_files


def scaffold_model_port(
    root: Path, name: str, labels: list[str], *, format: str = "yolo",
    variant_names: dict[str, str] | None = None,
) -> list[Path]:
    """Generate editable ONNX and Triton variants for a real detector."""
    if not name.isidentifier():
        raise ValueError("model name must be a Python identifier")
    if not labels:
        raise ValueError("at least one label is required")
    if format not in {"yolo", "detr", "rfdetr"}:
        raise ValueError("format must be yolo, detr, or rfdetr")
    legacy = root / "configs/models"
    base = legacy if not (root / "models").exists() and model_config_files(legacy) else root / "models"
    folder = base / name
    if folder.exists():
        raise FileExistsError(f"{folder} already exists")
    prefix = ".configs.models" if base == legacy else ".models"
    adapter = f"{prefix}.{name}.adapter"
    def tensor(name: str, shape: list[int]) -> dict:
        return {"name": name, "shape": shape, "dtype": "float32", "layout": "ND", "color": "NONE"}
    detr_outputs = [tensor("logits", [1, 300, len(labels)]), tensor("boxes", [1, 300, 4])]
    names = variant_names or {}
    repository_name = names.get("onnx", name)
    variants = {
        "onnx": {"backend": "onnxruntime", "artifact": {
            "path": f"model_repository/{repository_name}/1/model.onnx"}, "warmup": {"iterations": 1}},
        "triton": {"backend": "triton", "artifact": {"triton_model_name": repository_name}},
    }
    for variant, runtime_name in names.items():
        variants[variant]["name"] = runtime_name
    spec = {
        "name": name,
        "defaults": {
            "version": 1, "task": "detection", "device": {"preferred": "gpu", "gpu_id": 0},
            "inputs": [{"name": "images", "shape": [1, 3, 640, 640], "dtype": "float32",
                        "layout": "NCHW", "color": "RGB"}],
            "outputs": detr_outputs if format in {"detr", "rfdetr"} else [tensor("output0", [1, 4 + len(labels), 8400])],
            "labels": {"inline": labels},
            "preprocess": {"plugin": f"{adapter}:Preprocess", "params": {}},
            "postprocess": {"plugin": f"{adapter}:Postprocess", "params": {"conf_threshold": 0.25}},
            "batching": {"enabled": False, "max_batch_size": 1},
        },
        "variants": variants,
    }
    if format == "detr":
        spec["defaults"]["postprocess"]["params"]["background_index"] = None
    pre = ("resize_normalize", "ResizeNormalize") if format in {"detr", "rfdetr"} else ("image_letterbox", "Letterbox")
    post = ("detr_decode", "DetrDecode") if format in {"detr", "rfdetr"} else ("yolo_decode", "YoloDecode")
    if format == "rfdetr":
        post = ("rfdetr_decode", "RFDetrDecode")
        spec["defaults"]["postprocess"]["params"].update(boxes_out="boxes", logits_out="logits")
        spec["defaults"]["preprocess"]["params"] = {"size": 640, "normalize": True}
        pre = ("resize_normalize", "ResizeNormalize")
    content = {
        "__init__.py": "",
        "model.yaml": "# Edit the tensor names and shapes to match your exported graph.\n"
                      "# Variant dictionaries override defaults; lists replace the whole list.\n"
                      + yaml.safe_dump(spec, sort_keys=False),
        "adapter.py": f'''"""Model-specific preprocessing and output decoding. Customize these classes."""
from inferops.adapters.outbound.inference.plugins.{pre[0]} import {pre[1]}
from inferops.adapters.outbound.inference.plugins.{post[0]} import {post[1]}


class Preprocess({pre[1]}):
    """Prepare frames for this model's input contract."""


class Postprocess({post[1]}):
    """Map this model's outputs to InferOps detections."""
''',
        "README.md": f'''# {name}

1. Put your exported weights at `model_repository/{repository_name}/1/model.onnx`
   relative to the project folder, or set `artifact.path` to a file here.
2. Edit `model.yaml` to match the graph's tensor names, shapes and labels.
3. Customize `adapter.py` if the built-in {format.upper()} decoding needs changes.
4. Add an `infer` stage to `configs/pipelines/<pipeline>.yaml`:

```yaml
- name: {name}
  stage: infer
  params:
    model: {name}
```

Declare `defaults.artifact.source` with kind `url`, `google_drive`, or `huggingface`,
or supply the local weights yourself. No synthetic weights are generated.
For custom Triton tuning, add `model_repository/{repository_name}/config.pbtxt` with the
matching tensor contract; otherwise preparation generates a basic config.
Enable the pipeline in `project.yaml`, then run `inferops project prepare`
and `make preflight` with your project selected.
''',
    }
    folder.mkdir(parents=True)
    (base / "__init__.py").touch(exist_ok=True)
    created = [base / "__init__.py"]
    for relative, text in content.items():
        path = folder / relative
        path.write_text(text, encoding="utf-8")
        created.append(path)
    return created
