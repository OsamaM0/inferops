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
                "version": 1, "prewarm_pipelines": []}
    files = {"project.yaml": yaml.safe_dump(manifest, sort_keys=False),
             "__init__.py": "", "models/__init__.py": "", "plugins/__init__.py": "",
             "README.md": f"""# {manifest['label']}

This project opens an empty dashboard until you configure a real model.
Open Configuration to add and edit models, pipelines, rules and alert actions
through forms. These forms save the same YAML files used by the CLI and library.
Pipelines are discovered automatically; add a `pipelines` list to project.yaml
only when you want to restrict which pipelines are enabled.

1. Add a model: `make new-model PROJECT={name} NAME=detector`.
2. Edit its `model.yaml` tensors, adapter, artifact path and download source.
3. Add a pipeline in Configuration using `model: detector`, or create it in
   `configs/pipelines/`. Use a display label and description to explain its purpose.
4. Run `inferops project prepare --domain-root {root}` before starting locally,
   or mount this project through `.env` and run `docker compose up -d --build`.

Model runtime variants share one contract. Use the logical model name in
pipelines; you do not need a separate pipeline for each runtime.
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
