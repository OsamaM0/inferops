"""Domain packs: the unit that turns InferOps into a concrete CV product.

A domain pack is a directory with this layout::

    <root>/
      domain.yaml            # manifest (this module)
      __init__.py            # makes the pack importable as a package
      configs/
        models/*.yaml
        pipelines/*.yaml
        fusion/*.yaml
        alerts/*.yaml
        platform.yaml        # optional overrides merged over core platform.yaml
      plugins/               # stages, pre/post-processors, backends, annotators
      model_repository/      # optional; replaces the core repository when present
      scripts/               # optional asset installers referenced by the manifest
      tests/                 # optional; run with INFEROPS_ACTIVE_DOMAIN=<name>

Nothing in the core package knows a domain's models, classes, or signals: the
manifest declares what the tooling (``make``, ``setup``, ``serve``, the e2e
scenario) needs, and YAML references plugins through ``<package>.plugins...``
or the pack-relative form ``.plugins.<module>:<Class>``.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

MANIFEST_FILENAME = "domain.yaml"
_PACKAGE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Package name of the active domain pack, set by activate_domain_imports().
# Used to resolve pack-relative plugin references (".plugins.x:Y").
_active_package: str | None = None


class DomainAsset(BaseModel):
    """A model artifact the domain needs on disk before it can serve."""

    path: str
    installer: str | None = None
    description: str = ""
    required: bool = False


class DomainScenario(BaseModel):
    """What the browser end-to-end scenario should drive for this domain."""

    pipeline: str | None = None
    sample_video: str | None = None
    expect_alert: bool = False


class DomainManifest(BaseModel):
    name: str
    label: str | None = None
    description: str = ""
    version: int = 1
    python_extras: list[str] = Field(default_factory=list)
    triton: bool = False
    prewarm_pipelines: list[str] = Field(default_factory=list)
    prewarm_models: list[str] = Field(default_factory=list)
    assets: list[DomainAsset] = Field(default_factory=list)
    scenario: DomainScenario = Field(default_factory=DomainScenario)
    tags: dict[str, str] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not _PACKAGE_RE.fullmatch(value):
            raise ValueError("domain name must be a valid Python identifier (letters, digits, underscore)")
        return value

    @property
    def display_label(self) -> str:
        return self.label or self.name.replace("_", " ").title()


def manifest_path(domain_root: str | Path) -> Path:
    return Path(domain_root) / MANIFEST_FILENAME


def load_domain_manifest(domain_root: str | Path | None) -> DomainManifest | None:
    """Return the manifest for ``domain_root``.

    A pack without ``domain.yaml`` still works: it gets a manifest with only
    its directory name, so pre-manifest packs keep loading.
    """
    if domain_root is None:
        return None
    root = Path(domain_root)
    if not root.is_dir():
        return None
    path = manifest_path(root)
    if not path.exists():
        return DomainManifest(name=root.name)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: manifest must be a YAML mapping")
    data.setdefault("name", root.name)
    return DomainManifest(**data)


def discover_domains(configs_dir: str | Path = "configs") -> list[tuple[str, Path]]:
    """List ``(name, root)`` for every pack under ``examples/`` and ``domains/``."""
    repo_root = Path(configs_dir).resolve().parent
    found: dict[str, Path] = {}
    for parent in ("examples", "domains"):
        base = repo_root / parent
        if not base.is_dir():
            continue
        for child in sorted(base.iterdir()):
            if child.is_dir() and (child / "configs").is_dir() and child.name not in found:
                found[child.name] = child
    return list(found.items())


def domain_package_name(domain_root: str | Path) -> str:
    name = Path(domain_root).resolve().name
    if not _PACKAGE_RE.fullmatch(name):
        raise ValueError(
            f"domain directory {name!r} is not importable as a Python package; "
            "use letters, digits and underscores"
        )
    return name


def activate_domain_imports(domain_root: str | Path | None) -> str | None:
    """Make ``<pack>.plugins...`` importable and remember the pack for ``.plugins`` refs.

    The pack's *parent* directory goes on ``sys.path`` so a domain can live
    anywhere — ``examples/``, ``domains/``, or a path given through
    ``INFEROPS_DOMAIN_ROOT`` outside the repository.
    """
    global _active_package
    if domain_root is None:
        return None
    root = Path(domain_root).resolve()
    if not root.is_dir():
        return None
    package = domain_package_name(root)
    parent = str(root.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    _active_package = package
    return package


def active_package() -> str | None:
    return _active_package


def resolve_plugin_ref(ref: str) -> str:
    """Expand a pack-relative ``.plugins.mod:Cls`` reference to an absolute one."""
    if not ref.startswith("."):
        return ref
    package = _active_package or os.environ.get("INFEROPS_DOMAIN_PACKAGE")
    if not package:
        raise ValueError(
            f"plugin reference {ref!r} is relative to a domain pack but no domain is active"
        )
    return f"{package}{ref}"


def scaffold_domain(root: Path, name: str, labels: list[str], *, label: str | None = None) -> list[Path]:
    """Write a runnable detection domain pack under ``root``.

    The pack runs immediately on the synthetic detector (no weights needed) and
    documents where to drop a real ONNX/Triton model; every file is meant to be
    edited, not generated again.
    """
    if root.exists():
        raise FileExistsError(f"{root} already exists")
    if not _PACKAGE_RE.fullmatch(name):
        raise ValueError("domain name must be a valid Python identifier")
    if not labels:
        raise ValueError("at least one label is required")
    display = label or name.replace("_", " ").title()
    first = labels[0]
    label_list = ", ".join(labels)

    files: dict[str, str] = {
        "domain.yaml": f"""\
# Domain pack manifest — see inferops/core/domain.py for every field.
name: {name}
label: {display}
description: {display} detection pipeline scaffolded by `inferops domain init`.
version: 1

python_extras: []          # e.g. [triton] — extras from inferops/pyproject.toml
triton: false              # true: `make serve DOMAIN={name}` starts Triton

prewarm_pipelines: [{name}]
prewarm_models: [{name}_detector]

# Drop your exported model at the path below (or declare an installer script).
assets:
  - path: model_repository/{name}_detector/1/model.onnx
    description: {display} detector (ONNX)
    required: false

scenario:
  pipeline: {name}
  expect_alert: true
""",
        "__init__.py": "",
        "plugins/__init__.py": "",
        "README.md": f"""\
# {display} domain pack

Runs out of the box on the synthetic detector. To use a real model:

1. Export your detector to ONNX and copy it to
   `model_repository/{name}_detector/1/model.onnx`
   (or set `assets[].installer` in `domain.yaml` to a download script).
2. Edit `configs/models/{name}_detector.yaml`: input name/shape, labels, and the
   postprocess plugin (`yolo_decode` for YOLO exports, `detr_decode` for DETR).
3. Tune `configs/fusion/{name}.yaml` (signals, windows, rules) and
   `configs/alerts/{name}.yaml` (severity, dedupe, cooldown).
4. `make validate DOMAIN={name}` → `make serve-mock DOMAIN={name}` → open the
   dashboard at http://localhost:8090/dashboard.html.

Custom stages, postprocessors, annotators or backends go in `plugins/` and are
referenced from YAML as `.plugins.<module>:<Class>`.
""",
        f"configs/models/{name}_detector.yaml": f"""\
# Real detector served in-process by ONNX Runtime. Falls back to
# {name}_synth (see the pipeline) while model.onnx is absent.
name: {name}_detector
version: 1
task: detection

backend: onnxruntime
artifact:
  path: model_repository/{name}_detector/1/model.onnx

device:
  preferred: gpu       # silently falls back to CPU; check /v1/system/status
  gpu_id: 0

inputs:
  - name: images       # must match the exported graph's input name
    shape: [1, 3, 640, 640]
    dtype: float32
    layout: NCHW
    color: RGB

outputs:
  - name: output0
    shape: [1, 84, 8400]
    dtype: float32
    layout: ND
    color: NONE

labels:
  inline: [{label_list}]

preprocess:
  plugin: inferops.plugins.builtin.image_letterbox:Letterbox
  params: {{}}

postprocess:
  plugin: inferops.plugins.builtin.yolo_decode:YoloDecode
  params:
    conf_threshold: 0.25
    nms_iou: 0.45

batching:
  enabled: false
  max_batch_size: 1

warmup:
  iterations: 1
""",
        f"configs/models/{name}_synth.yaml": f"""\
# Deterministic stand-in so the pipeline runs with no weights, no GPU.
name: {name}_synth
version: 1
task: detection

backend: synthetic
device:
  preferred: cpu

inputs:
  - name: images
    shape: [1, 3, 640, 640]
    dtype: float32
    layout: NCHW
    color: RGB

outputs:
  - name: logits
    shape: [1, 20, {len(labels)}]
    dtype: float32
    layout: ND
    color: NONE
  - name: boxes
    shape: [1, 20, 4]
    dtype: float32
    layout: ND
    color: NONE

labels:
  inline: [{label_list}]

preprocess:
  plugin: inferops.plugins.builtin.resize_normalize:ResizeNormalize
  params:
    normalize: false

postprocess:
  plugin: inferops.plugins.builtin.detr_decode:DetrDecode
  params:
    conf_threshold: 0.5
    background_index: null

# label=every_n_frames[@cx,cy,w,h]; ';'-separated.
tags:
  queries: "20"
  emit: "{first}=2@0.5,0.5,0.3,0.4"
  seed: "0"
""",
        f"configs/pipelines/{name}.yaml": f"""\
name: {name}
version: 1
tags:
  label: {display}

source:
  name: src
  stage: source_rest

stages:
  - name: decode
    stage: decode_any
    params: {{ every_n: 1 }}

  - name: detect
    stage: infer
    params: {{ model: {name}_detector, fallback_model: {name}_synth }}

  - name: fuse
    stage: fuse
    params: {{ fusion: {name}_fusion }}

  - name: alert
    stage: alert
    params: {{ alerts: {name}_alerts }}

sinks:
  - name: evidence
    stage: sink_s3
    params:
      evidence_mode: both
      trigger_secs: 2.0
      pre_frames: 50
      post_frames: 50
      # Keep the evidence timer running while the condition holds on raw
      # detections (not throttled by alert cooldowns).
      trigger_rules:
        - {{ flag: {first}_present, class: {first}, min_score: 0.5 }}

limits: {{ max_concurrency: 4, queue_size: 32 }}

tracking:
  enabled: true
  algorithm: iou_kalman
  by: stream_id
  max_age: 30
  min_hits: 3
  iou_threshold: 0.3
  per_class: true
""",
        f"configs/fusion/{name}.yaml": f"""\
name: {name}_fusion
entity_key: stream_id
signals: [{first}_score]
extract:
  # Highest-confidence "{first}" detection on this frame (0 when absent).
  {first}_score: "max((d.score for r in results if r.detections for d in r.detections if d.label == '{first}'), default=0.0)"
derived:
  {first}_present: "{first}_score.fraction_above(0.5)"
windows:
  {first}_score: {{ kind: rolling_time, duration_ms: 3000, reducers: [latest, max, fraction_above] }}
rules:
  - name: {first}_present
    when: "{first}_present >= 0.5"
    severity: medium
    cooldown_ms: 15000
    emit: {{ type: {first}_present }}
outputs:
  events_topic: inferops.events.{name}
""",
        f"configs/alerts/{name}.yaml": f"""\
name: {name}_alerts
severity:
  levels:
    - {{ name: low,      rank: 1, color: "#9cf" }}
    - {{ name: medium,   rank: 2, color: "#fc6" }}
    - {{ name: high,     rank: 3, color: "#f63" }}
    - {{ name: critical, rank: 4, color: "#f00" }}
  default: low
  mapping:
    {first}_present: medium
dedupe:
  strategy: by_key_template
  key_template: "{{type}}:{{entity_id}}"
  window_ms: 30000
cooldown:
  per_dedupe_key_ms: 15000
sinks:
  - {{ name: ws,     kind: ws,       params: {{ channel: "alerts.{{entity_id}}" }} }}
  - {{ name: pg,     kind: postgres, params: {{}} }}
  - {{ name: stdout, kind: stdout,   params: {{}} }}
persistence_enabled: true
""",
        "model_repository/.gitkeep": "",
        "tests/conftest.py": f"""\
from inferops.testing import configure_test_env

configure_test_env(domain="{name}")

from inferops.testing.fixtures import *  # noqa: E402,F403
""",
        f"tests/test_{name}_smoke.py": f"""\
import io

import cv2
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from inferops.api.main import create_app


@pytest.mark.asyncio
async def test_pipeline_runs_through_the_api():
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac, \\
            app.router.lifespan_context(app):
        r = await ac.get("/v1/pipelines")
        assert any(p["name"] == "{name}" for p in r.json())
        img = (np.random.rand(64, 64, 3) * 255).astype("uint8")
        ok, buf = cv2.imencode(".jpg", img)
        assert ok
        for _ in range(3):
            r = await ac.post("/v1/infer/{name}",
                              files={{"image": ("f.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")}},
                              data={{"stream_id": "t"}})
            assert r.status_code == 200, r.text
        assert any(res["kind"] == "detections" for res in r.json()[0]["results"])
""",
    }
    created: list[Path] = []
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        created.append(path)
    return created


def manifest_env(manifest: DomainManifest, domain_root: Path, model_repository: Path) -> dict[str, Any]:
    """Shell-friendly view of the manifest consumed by the Makefile."""
    return {
        "INFEROPS_DOMAIN_NAME": manifest.name,
        "INFEROPS_DOMAIN_ROOT": str(domain_root.resolve()),
        "INFEROPS_DOMAIN_PACKAGE": domain_package_name(domain_root),
        "INFEROPS_DOMAIN_TRITON": "1" if manifest.triton else "0",
        "INFEROPS_DOMAIN_MODEL_REPOSITORY": str(model_repository.resolve()),
        "INFEROPS_PREWARM_PIPELINES": ",".join(manifest.prewarm_pipelines),
        "INFEROPS_PREWARM_MODELS": ",".join(manifest.prewarm_models),
        "INFEROPS_DOMAIN_EXTRAS": ",".join(manifest.python_extras),
        "INFEROPS_DOMAIN_HAS_TESTS": "1" if (domain_root / "tests").is_dir() else "0",
    }
