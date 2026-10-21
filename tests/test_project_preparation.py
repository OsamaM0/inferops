"""Artifact preparation, logical runtime selection and empty project contracts."""
from __future__ import annotations

import io
import tarfile
import zipfile

import pytest

from inferops.adapters.outbound.projects.manifest import load_domain_manifest, scaffold_domain
from inferops.adapters.outbound.projects.models import load_model_file
from inferops.adapters.outbound.projects.preparation import (
    download_url,
    fetch_source,
    generate_triton_config,
    prepare_artifact,
    prepare_project,
    run_installer,
    selected_surface,
    sha256,
    validate_onnx,
)
from inferops.adapters.outbound.projects.resolver import ModelResolver
from inferops.adapters.outbound.projects.settings import ConfigStack
from inferops.domain.errors import ConfigError
from inferops.domain.models import ArtifactSource
from inferops.domain.pipeline import PipelineSpec
from inferops.testing import PROJECT_ROOT, TINY_MODEL_PATH


@pytest.fixture
def weights():
    return TINY_MODEL_PATH


@pytest.fixture
def spec(weights):
    model = load_model_file(PROJECT_ROOT / "inferops/testing/configs/models/tiny_classifier.yaml")
    model.artifact.path = str(weights)
    model.artifact.source = ArtifactSource(kind="url", url="https://example.test/model.onnx")
    return model


def test_manifest_prefers_project_and_merges_platform(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "domain.yaml").write_text("name: legacy\n")
    (root / "project.yaml").write_text("name: current\npipelines: []\nplatform: {obs: {log_level: DEBUG}}\n")
    assert load_domain_manifest(root).name == "current"
    assert ConfigStack(domain_root=root).platform()["obs"]["log_level"] == "DEBUG"


@pytest.mark.parametrize("mode,expected", [("auto", "onnx"), ("cpu", "onnx"), ("cuda", "onnx"),
                                         ("triton", "triton")])
def test_logical_models_and_tracking_share_selection(mode, expected):
    stack = ConfigStack(PROJECT_ROOT / "configs", active_domain="driver_behaviour")
    pipelines, models = selected_surface(stack, mode)
    primary = next(item for item in pipelines if item.name == "driver_behaviour")
    assert primary.stages[1].params["model"] == f"rfdetr_driver_{expected}"
    assert primary.stages[2].params["model"] == f"face_mesh_{expected}"
    assert primary.tracking.model_filter == [f"rfdetr_driver_{expected}"]
    assert {model.name for _, model in models} == {f"rfdetr_driver_{expected}", f"face_mesh_{expected}"}
    resolver = ModelResolver(stack.spec_dir("models"))
    assert resolver.resolve("rfdetr_driver_triton", "cpu") == "rfdetr_driver_triton"


def test_empty_scaffold_prepares_no_models(tmp_path):
    root = tmp_path / "empty"
    scaffold_domain(root, "empty", ["person"])
    # New projects discover pipelines as they are added; an empty directory still
    # prepares nothing, without a second manifest list to maintain.
    assert load_domain_manifest(root).pipelines is None
    assert prepare_project(ConfigStack(domain_root=root), "cpu") == 0
    assert not list((root / "models").glob("*/model.yaml"))


def test_legacy_auto_override_is_a_cpu_choice():
    pipeline = PipelineSpec(name="legacy", stages=[{"name": "detect", "stage": "infer", "params": {
        "model": "default", "model_modes": {"auto": "chosen"},
    }}])
    assert pipeline.for_mode("auto").stages[0].params["model"] == "chosen"
    assert pipeline.for_mode("cpu").stages[0].params["model"] == "chosen"


def test_unused_model_requires_no_download(tmp_path):
    root = tmp_path / "empty"
    scaffold_domain(root, "empty", [])
    directory = root / "models/unselected"
    directory.mkdir()
    (directory / "model.yaml").write_text("name: unused\ninputs: []\nartifact: {path: missing.onnx}\n")
    assert prepare_project(ConfigStack(domain_root=root), "cpu") == 0


def test_selected_custom_backend_requires_weight_configuration(tmp_path):
    root = tmp_path / "custom_project"
    scaffold_domain(root, "custom_project", [])
    (root / "project.yaml").write_text("name: custom_project\npipelines: [custom]\n")
    (root / "models/custom.yaml").write_text("name: custom\nbackend: custom:Backend\ninputs: []\n")
    (root / "configs/pipelines/custom.yaml").write_text(
        "name: custom\nstages: [{name: detect, stage: infer, params: {model: custom}}]\n")
    with pytest.raises(ConfigError, match="real weights are required"):
        prepare_project(ConfigStack(domain_root=root), "cpu")


def test_cache_reuses_offline_and_refreshes_changed_source(tmp_path, weights, spec, monkeypatch):
    target = tmp_path / "model.onnx"
    calls = []
    def fetch(source, staged, root):
        calls.append(source.url)
        staged.write_bytes(weights.read_bytes())
    monkeypatch.setattr("inferops.adapters.outbound.projects.preparation.fetch_source", fetch)
    prepare_artifact(spec, target, tmp_path)
    assert len(calls) == 1
    prepare_artifact(spec, target, tmp_path)
    assert len(calls) == 1
    spec.artifact.source.url = "https://example.test/new.onnx"
    prepare_artifact(spec, target, tmp_path)
    assert len(calls) == 2
    prepare_artifact(spec, target, tmp_path, force=True)
    assert len(calls) == 3


def test_interrupted_download_preserves_previous_weights(tmp_path, weights, spec, monkeypatch):
    target = tmp_path / "model.onnx"
    target.write_bytes(weights.read_bytes())
    original = sha256(target)
    def interrupted(source, staged, root):
        staged.write_bytes(b"partial download")
        raise OSError("connection interrupted")
    monkeypatch.setattr("inferops.adapters.outbound.projects.preparation.fetch_source", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        prepare_artifact(spec, target, tmp_path, force=True)
    assert sha256(target) == original
    assert not list(tmp_path.glob(".prepare-*"))


def test_interrupted_source_record_recovers_offline(tmp_path, weights, spec, monkeypatch):
    target = tmp_path / "model.onnx"
    target.write_bytes(weights.read_bytes())
    metadata = target.with_name(target.name + ".source.json")
    metadata.write_text('{"source":')
    def offline(*args):
        raise OSError("offline")
    monkeypatch.setattr("inferops.adapters.outbound.projects.preparation.fetch_source", offline)
    prepare_artifact(spec, target, tmp_path)
    import json

    assert json.loads(metadata.read_text())["source"]
    assert not list(tmp_path.glob(".prepare-*"))


@pytest.mark.parametrize("problem", ["shape", "dtype", "name", "checksum", "html"])
def test_invalid_artifacts_fail_without_replacing_weights(tmp_path, weights, spec, problem):
    target = tmp_path / "model.onnx"
    target.write_bytes(weights.read_bytes())
    if problem == "shape":
        spec.inputs[0].shape[-1] = 999
    elif problem == "dtype":
        spec.inputs[0].dtype = "int64"
    elif problem == "name":
        spec.inputs[0].name = "missing"
    elif problem == "checksum":
        spec.artifact.sha256 = "0" * 64
    else:
        target.write_text("<!DOCTYPE html><html>Access denied</html>")
    with pytest.raises(ConfigError, match=r"mismatch|invalid ONNX"):
        validate_onnx(target, spec)


def test_missing_local_model_is_an_error(tmp_path, spec):
    spec.artifact.source = None
    with pytest.raises(ConfigError, match="weights missing"):
        prepare_artifact(spec, tmp_path / "missing.onnx", tmp_path)
    assert not (tmp_path / "missing.onnx").exists()


def test_invalid_fresh_download_keeps_valid_weights(tmp_path, weights, spec, monkeypatch):
    target = tmp_path / "model.onnx"
    target.write_bytes(weights.read_bytes())
    def invalid(source, staged, root):
        staged.write_bytes(b"invalid downloaded graph")
    monkeypatch.setattr("inferops.adapters.outbound.projects.preparation.fetch_source", invalid)
    with pytest.raises(ConfigError, match="invalid ONNX"):
        prepare_artifact(spec, target, tmp_path, force=True)
    assert target.read_bytes() == weights.read_bytes()
    assert not list(tmp_path.glob(".prepare-*"))


def test_huggingface_repository_revision_and_token(tmp_path, weights, monkeypatch):
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return weights
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    monkeypatch.setenv("HF_TOKEN", "test-only-token")
    target = tmp_path / "model.onnx"
    fetch_source(ArtifactSource(kind="huggingface", repo_id="owner/model", filename="onnx/model.onnx",
                                revision="pinned-commit"), target, tmp_path)
    assert calls == [{"repo_id": "owner/model", "filename": "onnx/model.onnx",
                      "revision": "pinned-commit", "token": "test-only-token"}]
    assert target.read_bytes() == weights.read_bytes()


def test_url_archive_extracts_only_selected_regular_member(tmp_path, weights, monkeypatch):
    archive_path = tmp_path / "source.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        payload = weights.read_bytes()
        info = tarfile.TarInfo("weights/model.onnx")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
        dangerous = tarfile.TarInfo("../../escaped")
        dangerous.size = 1
        archive.addfile(dangerous, io.BytesIO(b"x"))
    def download(url, target):
        target.write_bytes(archive_path.read_bytes())
    monkeypatch.setattr("inferops.adapters.outbound.projects.preparation.download_url", download)
    target = tmp_path / "model.onnx"
    fetch_source(ArtifactSource(kind="url", url="https://example.test/weights.tar.gz",
                                archive_member="weights/model.onnx"), target, tmp_path)
    assert target.read_bytes() == weights.read_bytes()
    assert not (tmp_path.parent.parent / "escaped").exists()


def test_zip_archive_extracts_selected_member(tmp_path, weights, monkeypatch):
    archive_path = tmp_path / "source.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("weights/model.onnx", weights.read_bytes())
        archive.writestr("../../escaped", b"x")
    monkeypatch.setattr("inferops.adapters.outbound.projects.preparation.download_url",
                        lambda url, target: target.write_bytes(archive_path.read_bytes()))
    target = tmp_path / "model.onnx"
    fetch_source(ArtifactSource(kind="url", url="https://example.test/weights.zip",
                                archive_member="weights/model.onnx"), target, tmp_path)
    assert target.read_bytes() == weights.read_bytes()


def test_archive_link_is_rejected(tmp_path, monkeypatch):
    archive_path = tmp_path / "source.tar"
    with tarfile.open(archive_path, "w") as archive:
        info = tarfile.TarInfo("model.onnx")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        archive.addfile(info)
    monkeypatch.setattr("inferops.adapters.outbound.projects.preparation.download_url",
                        lambda url, target: target.write_bytes(archive_path.read_bytes()))
    with pytest.raises(ConfigError, match="regular model file"):
        fetch_source(ArtifactSource(kind="url", url="https://example.test/weights.tar",
                                    archive_member="model.onnx"), tmp_path / "model.onnx", tmp_path)


def test_custom_installer_receives_temporary_output(tmp_path, weights, spec):
    installer = tmp_path / "install.py"
    installer.write_text("import argparse,shutil\n"
                         "parser=argparse.ArgumentParser()\nparser.add_argument('--output')\n"
                         f"shutil.copyfile({str(weights)!r},parser.parse_args().output)\n")
    spec.artifact.source = ArtifactSource(kind="installer", installer="install.py")
    target = tmp_path / "model.onnx"
    prepare_artifact(spec, target, tmp_path)
    assert target.read_bytes() == weights.read_bytes()


def test_custom_module_installer_receives_target(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr("inferops.adapters.outbound.projects.plugins.load_object", lambda ref: lambda target: called.append((ref, target)))
    target = tmp_path / "model.onnx"
    run_installer("project.install:download", target, tmp_path)
    assert called == [("project.install:download", target)]


def test_google_drive_confirmation_and_streaming(tmp_path, monkeypatch):
    class Response(io.BytesIO):
        def __init__(self, content, content_type):
            super().__init__(content)
            self.headers = {"Content-Type": content_type}
    responses = iter([
        Response(b'<form action="https://drive.usercontent.google.com/download" id="download-form">'
                 b'<input value="abc" name="id"><input name="confirm" value="yes"></form>', "text/html"),
        Response(b"model bytes", "application/octet-stream"),
    ])
    urls = []
    class Opener:
        def open(self, request, timeout):
            urls.append(request.full_url)
            return next(responses)
    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
    target = tmp_path / "model.onnx"
    fetch_source(ArtifactSource(kind="google_drive", file_id="abc"), target, tmp_path)
    assert "id=abc" in urls[0]
    assert "confirm=yes" in urls[1]
    assert target.read_bytes() == b"model bytes"


def test_html_download_is_rejected(tmp_path, monkeypatch):
    class Response(io.BytesIO):
        @property
        def headers(self):
            return {"Content-Type": "text/html"}
    class Opener:
        def open(self, *args, **kwargs):
            return Response(b"<html>Login</html>")
    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
    with pytest.raises(ConfigError, match="HTML"):
        download_url("https://example.test/model", tmp_path / "model.onnx")


def test_triton_generation_preserves_custom_tuning(tmp_path, spec):
    spec.artifact.triton_model_name = "classifier"
    generate_triton_config(spec, tmp_path)
    path = tmp_path / "classifier/config.pbtxt"
    assert 'name: "input"' in path.read_text()
    path.write_text("custom tuning\n")
    generate_triton_config(spec, tmp_path)
    assert path.read_text() == "custom tuning\n"


@pytest.mark.parametrize("source", [{"kind": "url"}, {"kind": "google_drive"},
                                    {"kind": "huggingface", "repo_id": "owner/model"}])
def test_source_requires_provider_identifiers(source):
    with pytest.raises(ValueError, match="requires"):
        ArtifactSource(**source)
