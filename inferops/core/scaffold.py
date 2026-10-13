"""Create an empty project without choosing or inventing model weights."""
from __future__ import annotations

from pathlib import Path

import yaml


def scaffold_domain(root: Path, name: str, labels: list[str], *, label: str | None = None) -> list[Path]:
    if root.exists():
        raise FileExistsError(f"{root} already exists")
    if not name.isidentifier():
        raise ValueError("project name must be a Python identifier")
    manifest = {"name": name, "label": label or name.replace("_", " ").title(),
                "version": 1, "pipelines": [], "prewarm_pipelines": []}
    files = {"project.yaml": yaml.safe_dump(manifest, sort_keys=False),
             "__init__.py": "", "models/__init__.py": "", "plugins/__init__.py": "",
             "README.md": f"""# {manifest['label']}

This project opens an empty dashboard until you configure a real model.

1. Add a model: `make new-model PROJECT={name} NAME=detector`.
2. Edit its `model.yaml` tensors, adapter, artifact path and download source.
3. Create a pipeline in `configs/pipelines/` using `model: detector`.
4. Add that pipeline name to `project.yaml` under `pipelines`.
5. Run `inferops project prepare --domain-root {root}` before starting locally,
   or mount this project through `.env` and run `docker compose up -d --build`.

Add fusion rules, alert actions and evidence stages as needed. No synthetic
model or automatic model conversion is supplied.
"""}
    root.mkdir(parents=True)
    for directory in ("configs/pipelines", "configs/fusion", "configs/alerts", "model_repository", "tests"):
        (root / directory).mkdir(parents=True)
    created = []
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        created.append(path)
    return created
