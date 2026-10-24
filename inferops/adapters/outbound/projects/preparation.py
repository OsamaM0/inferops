"""Prepare selected real model artifacts before opening runtime resources."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from html.parser import HTMLParser
from http.cookiejar import CookieJar
from pathlib import Path

from filelock import FileLock

from inferops.adapters.outbound.projects.artifacts import artifact_candidates
from inferops.adapters.outbound.projects.models import (
    expand_model_config,
    model_config_files,
)
from inferops.adapters.outbound.projects.resolver import ModelResolver, canonical_mode
from inferops.adapters.outbound.projects.settings import ConfigStack, settings
from inferops.adapters.outbound.projects.validation import validate_config_stack
from inferops.adapters.outbound.projects.yaml import load_yaml
from inferops.domain.errors import ConfigError
from inferops.domain.models import ArtifactSource, ModelSpec


def selected_surface(stack: ConfigStack, mode: str):
    """Return resolved enabled pipelines and selected model specs with owners."""
    surface = validate_config_stack(stack)
    overrides = stack.configuration_overrides()
    resolver = ModelResolver(stack.spec_dir("models").resolve(), overrides=overrides)
    pipelines = [spec.for_mode(mode, resolver) for spec in surface["pipelines"].values()
                 if stack.enabled_pipeline(spec.name)]
    enabled = {pipeline.name for pipeline in pipelines}
    for name in os.environ.get("INFEROPS_PREWARM_PIPELINES", "").split(","):
        if name.strip() and name.strip() not in enabled:
            raise ConfigError(f"prewarm pipeline {name.strip()!r} is not enabled")
    names = {stage.params["model"] for spec in pipelines
             for stage in ([spec.source] if spec.source else []) + spec.stages + spec.sinks
             if stage.params.get("model")}
    manifest = stack.manifest()
    explicit = (os.environ["INFEROPS_PREWARM_MODELS"].split(",") if "INFEROPS_PREWARM_MODELS" in os.environ
                else manifest.prewarm_models if manifest else [])
    names.update(resolver.resolve(name.strip(), mode) for name in explicit if name.strip())
    directory = stack.spec_dir("models").resolve()
    paths = set(model_config_files(directory)) | {path for path in overrides if path.parent == directory
            or (path.name == "model.yaml" and path.parent.parent == directory)}
    sources = {spec.name: (path, surface["models"][spec.name]) for path in paths
               for spec in expand_model_config(overrides[path] if path in overrides else load_yaml(path))}
    selected = [sources[name] for name in sorted(names)]
    for _path, spec in selected:
        if spec.backend == "synthetic" or "Synthetic" in spec.backend or "synthetic" in spec.backend:
            raise ConfigError(f"model {spec.name!r}: synthetic models are not supported in runnable projects")
        if not spec.artifact.path and not spec.artifact.triton_model_name:
            raise ConfigError(f"model {spec.name!r}: real weights are required; configure artifact.path")
    return pipelines, selected


def artifact_target(spec: ModelSpec, owner: Path, stack: ConfigStack) -> Path:
    repository = stack.model_repository(settings.model_repository)
    if spec.artifact.path:
        candidates = artifact_candidates(spec.artifact.path, base_dir=owner.parent,
                                         model_repository=repository, domain_root=stack.domain_root,
                                         configs_dir=stack.configs_dir)
        for path in candidates:
            if path.is_file():
                return path.resolve()
        path = Path(spec.artifact.path)
        if path.is_absolute():
            return path
        if path.parts[0] == repository.name:
            return (repository.parent / path).resolve()
        return (owner.parent / path).resolve()
    if spec.artifact.triton_model_name:
        return (repository / spec.artifact.triton_model_name / str(spec.version) / "model.onnx").resolve()
    raise ConfigError(f"model {spec.name!r}: no artifact path configured")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_onnx(path: Path, spec: ModelSpec) -> None:
    """Validate the graph and its configured names, dtypes and dimensions."""
    import onnx

    if not path.is_file() or not path.stat().st_size:
        raise ConfigError(f"model {spec.name!r}: missing or empty weights at {path}")
    if spec.artifact.sha256 and sha256(path) != spec.artifact.sha256.lower():
        raise ConfigError(f"model {spec.name!r}: SHA-256 mismatch at {path}")
    try:
        onnx.checker.check_model(str(path))
        graph = onnx.load(str(path), load_external_data=False).graph
        initializers = {item.name for item in graph.initializer}
        for kind, declared, actual in (("input", spec.inputs, [item for item in graph.input if item.name not in initializers]),
                                       ("output", spec.outputs, list(graph.output))):
            tensors = {item.name: item.type.tensor_type for item in actual}
            if declared and {item.name for item in declared} != set(tensors):
                raise ValueError(f"{kind} names: configured {[item.name for item in declared]}, graph {list(tensors)}")
            for item in declared:
                tensor = tensors[item.name]
                dtype = str(onnx.helper.tensor_dtype_to_np_dtype(tensor.elem_type))
                if item.dtype != dtype:
                    raise ValueError(f"{kind} {item.name}: dtype {dtype}, configured {item.dtype}")
                dims = [dim.dim_value if dim.HasField("dim_value") else None for dim in tensor.shape.dim]
                if item.shape and (len(item.shape) != len(dims) or any(
                        expected > 0 and actual is not None and expected != actual
                        for expected, actual in zip(item.shape, dims, strict=False))):
                    raise ValueError(f"{kind} {item.name}: shape {dims}, configured {item.shape}")
    except Exception as exc:
        raise ConfigError(f"model {spec.name!r}: invalid ONNX artifact {path}: {exc}") from exc


class _DriveForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action: str | None = None
        self.params: dict[str, str] = {}
        self.inside = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form" and attrs.get("id") == "download-form":
            self.action = attrs.get("action")
            self.inside = True
        if self.inside and tag == "input" and attrs.get("name"):
            self.params[attrs["name"]] = attrs.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self.inside = False


def download_url(url: str, destination: Path, *, drive: bool = False) -> None:
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
    for _attempt in range(4):
        request = urllib.request.Request(url, headers={"User-Agent": "InferOps/1.0"})
        with opener.open(request, timeout=60) as response:
            if "text/html" in response.headers.get("Content-Type", "").lower():
                if not drive:
                    raise ConfigError("download returned HTML instead of model weights")
                form = _DriveForm()
                form.feed(response.read(1024 * 1024).decode("utf-8", errors="replace"))
                parsed = urllib.parse.urlparse(form.action or "")
                if parsed.scheme != "https" or parsed.hostname not in {"drive.google.com", "drive.usercontent.google.com"}:
                    raise ConfigError("Google Drive returned no public download confirmation; check file access")
                url = f"{form.action}?{urllib.parse.urlencode(form.params)}"
                continue
            with destination.open("wb") as handle:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    handle.write(chunk)
            return
    raise ConfigError("Google Drive download exceeded confirmation limit")


def run_installer(installer: str, target: Path, root: Path) -> None:
    if ":" in installer and not installer.endswith(".py"):
        from inferops.adapters.outbound.projects.plugins import load_object

        load_object(installer)(target)
    else:
        subprocess.run([sys.executable, str(root / installer), "--output", str(target)], cwd=root, check=True)


def fetch_source(source: ArtifactSource, target: Path, root: Path) -> None:
    downloaded = target.with_suffix(".download") if source.archive_member else target
    if source.kind == "url":
        download_url(source.url, downloaded)
    elif source.kind == "google_drive":
        download_url("https://drive.google.com/uc?export=download&id=" + urllib.parse.quote(source.file_id),
                     downloaded, drive=True)
    elif source.kind == "huggingface":
        import shutil

        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(repo_id=source.repo_id, filename=source.filename,
                                 revision=source.revision, token=os.environ.get("HF_TOKEN") or None)
        shutil.copyfile(cached, downloaded)
    else:
        run_installer(source.installer, downloaded, root)
    if source.archive_member:
        import shutil

        # Extract only the requested regular member, never archive paths.
        if zipfile.is_zipfile(downloaded):
            with zipfile.ZipFile(downloaded) as archive, archive.open(source.archive_member) as member, target.open("wb") as handle:
                shutil.copyfileobj(member, handle)
        else:
            with tarfile.open(downloaded, "r:*") as archive:
                info = archive.getmember(source.archive_member)
                if not info.isfile():
                    raise ConfigError("archive member must be a regular model file")
                with archive.extractfile(info) as member, target.open("wb") as handle:
                    shutil.copyfileobj(member, handle)


def prepare_artifact(spec: ModelSpec, target: Path, root: Path, *, force: bool = False) -> None:
    source = spec.artifact.source
    if source is None and spec.artifact.url:
        source = ArtifactSource(kind="url", url=spec.artifact.url)
    fingerprint = hashlib.sha256(json.dumps(source.model_dump() if source else None, sort_keys=True).encode()).hexdigest()
    metadata = target.with_name(target.name + ".source.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(target) + ".lock", timeout=600):
        try:
            previous = json.loads(metadata.read_text()) if metadata.is_file() else {}
        except (ValueError, OSError):
            previous = {}
        if not isinstance(previous, dict):
            previous = {}
        def record_source():
            with tempfile.TemporaryDirectory(prefix=".prepare-", dir=target.parent) as directory:
                staged_metadata = Path(directory) / "source.json"
                staged_metadata.write_text(json.dumps({"source": fingerprint}) + "\n")
                staged_metadata.replace(metadata)
        changed = bool(previous) and previous.get("source") != fingerprint
        if target.is_file() and not force and not changed:
            validate_onnx(target, spec)
            record_source()
            print(f"Validated existing weights: {spec.name}", flush=True)
            return
        if source is None:
            if target.is_file():
                validate_onnx(target, spec)
                return
            raise ConfigError(f"model {spec.name!r}: weights missing at {target}; configure artifact.source or supply the file")
        print(f"Downloading {spec.name} from {source.kind}", flush=True)
        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=target.parent) as directory:
            staged = Path(directory) / "model.onnx"
            fetch_source(source, staged, root)
            validate_onnx(staged, spec)
            staged.replace(target)
        record_source()


def generate_triton_config(spec: ModelSpec, repository: Path) -> None:
    name = spec.artifact.triton_model_name
    if not name or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ConfigError(f"invalid Triton model name {name!r}")
    path = repository / name / "config.pbtxt"
    if path.exists():
        return
    dtype = {"float32": "FP32", "float16": "FP16", "float64": "FP64", "int64": "INT64",
             "int32": "INT32", "uint8": "UINT8", "bool": "BOOL"}
    lines = [f'name: "{name}"', 'platform: "onnxruntime_onnx"', "max_batch_size: 0"]
    for kind, tensors in (("input", spec.inputs), ("output", spec.outputs)):
        lines.append(f"{kind} [")
        for tensor in tensors:
            if tensor.dtype not in dtype:
                raise ConfigError(f"unsupported Triton dtype {tensor.dtype!r}")
            lines.append(f'  {{ name: "{tensor.name}" data_type: TYPE_{dtype[tensor.dtype]} dims: {tensor.shape} }}')
        lines.append("]")
    lines.append(f"instance_group [ {{ kind: KIND_GPU count: 1 gpus: [{spec.device.gpu_id}] }} ]")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Avoid replacing user tuning or another preparation process's config.
    try:
        with path.open("x") as handle:
            handle.write("\n".join(lines) + "\n")
    except FileExistsError:
        pass


def prepare_project(stack: ConfigStack, mode: str, *, force: bool = False) -> int:
    _pipelines, selected = selected_surface(stack, mode)
    root = stack.domain_root or stack.configs_dir.parent
    manifest = stack.manifest()
    targets = {artifact_target(spec, owner, stack) for owner, spec in selected
               if spec.backend in {"onnxruntime", "triton"} or spec.artifact.path or spec.artifact.source}
    # Legacy assets retain their original script interface.
    for asset in manifest.assets if manifest else []:
        target = (root / asset.path).resolve()
        if target not in targets and not asset.required:
            continue
        if asset.installer and (force or not target.is_file()):
            if ":" in asset.installer and not asset.installer.endswith(".py"):
                from inferops.adapters.outbound.projects.plugins import load_object
                load_object(asset.installer)(target)
            else:
                subprocess.run([sys.executable, str(root / asset.installer)], cwd=root, check=True)
        if asset.required and not target.is_file():
            raise ConfigError(f"required legacy asset missing: {target}")
    prepared = set()
    for owner, spec in selected:
        if spec.backend not in {"onnxruntime", "triton"} and not spec.artifact.path and not spec.artifact.source:
            continue
        target = artifact_target(spec, owner, stack)
        prepare_artifact(spec, target, root, force=force and target not in prepared)
        prepared.add(target)
        if canonical_mode(mode) in {"triton"} and spec.backend == "triton":
            repository = stack.model_repository(settings.model_repository).resolve()
            expected = repository / spec.artifact.triton_model_name / str(spec.version) / "model.onnx"
            if expected.resolve() != target:
                raise ConfigError(f"Triton artifact must be stored at {expected}; configured artifact resolves to {target}")
            generate_triton_config(spec, repository)
    return len(prepared)
