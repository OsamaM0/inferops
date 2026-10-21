"""The generic settings UI and YAML API share one project configuration."""
from __future__ import annotations

import shutil

import pytest
import yaml
from httpx import ASGITransport, AsyncClient

from inferops.adapters.inbound.http.main import create_app
from inferops.adapters.outbound.projects.loader import ProjectLoader
from inferops.adapters.outbound.projects.manifest import scaffold_domain
from inferops.adapters.outbound.projects.settings import ConfigStack
from inferops.bootstrap import build_engine
from inferops.testing import PROJECT_ROOT


@pytest.fixture
async def workspace(tmp_path):
    root = tmp_path / "inspection"
    scaffold_domain(root, "inspection", [])
    core = tmp_path / "configs"
    core.mkdir()
    (core / "platform.yaml").write_text("api:\n  port: 8080\nobs:\n  log_level: INFO\n")
    (root / "models/detector").mkdir()
    (root / "models/detector/model.yaml").write_text(yaml.safe_dump({
        "name": "objects", "defaults": {"inputs": [{"name": "image"}]},
        "variants": {"onnx": {"backend": "onnxruntime"}, "triton": {"backend": "triton"}},
    }))
    (root / "configs/fusion/policy.yaml").write_text("name: inspection_rules\nsignals: []\n")
    (root / "configs/pipelines/process.yaml").write_text(yaml.safe_dump({
        "name": "inspection", "tags": {"label": "Inspect objects"},
        "stages": [{"name": "detect", "stage": "infer", "params": {"model": "objects"}},
                   {"name": "decide", "stage": "fuse", "params": {"fusion": "inspection_rules"}}],
    }))
    engine = build_engine(root, configs_dir=core)
    app = create_app(engine=engine)
    app.state.engine = engine
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield root, core, engine, client
    await engine.close()


async def test_catalog_links_logical_names_to_editor_ids_and_groups_variants(workspace):
    _, _, _, client = workspace
    response = await client.get("/v1/configs")
    assert response.status_code == 200, response.text
    catalog = response.json()
    assert catalog["domains"][0]["config_links"] == [
        {"section": "pipelines", "name": "process", "title": "Inspect objects"},
        {"section": "models", "name": "detector", "title": "objects"},
        {"section": "fusion", "name": "policy", "title": "inspection_rules"},
    ]
    models = [entry for entry in catalog["files"] if entry["section"] == "models"]
    assert len(models) == 1 and models[0]["variants"] == ["onnx", "triton"]
    for link in catalog["domains"][0]["config_links"]:
        entry = (await client.get(f"/v1/configs/{link['section']}/{link['name']}")).json()
        assert entry["schema"]["properties"] and entry["data"]["name"]
    system = (await client.get("/v1/system/workspace")).json()
    assert {model["logical_name"] for model in system["models"]} == {"objects"}
    assert [model["name"] for model in system["models"] if model["selected"]] == ["objects_onnx"]


async def test_pipeline_form_uses_registered_steps_and_shared_required_parameters(workspace):
    from inferops.adapters.outbound.stages.registry import APPLICATION_STAGES, BUILTINS

    _, _, _, client = workspace
    schema = (await client.get("/v1/configs/schema/pipelines")).json()
    steps = schema["x-stages"]
    assert set(steps) == set(BUILTINS) | APPLICATION_STAGES
    assert steps["infer"]["category"] == "processor"
    assert steps["infer"]["params"]["properties"]["model"]["x-reference"] == "models"
    assert steps["source_folder"]["category"] == "source"
    assert steps["sink_file"]["category"] == "sink"
    assert schema["$defs"]["StageDecl"]["x-stage-declaration"]
    for alias, parameter in (("infer", "model"), ("fuse", "fusion"), ("alert", "alerts")):
        response = await client.post("/v1/configs/pipelines/invalid_step", json={"data": {
            "name": "invalid_step", "stages": [{"name": "run", "stage": alias}],
        }, "validate_only": True})
        assert response.status_code == 400 and f"{parameter} is required" in response.text


async def test_stage_schema_discovery_never_imports_project_plugins(workspace):
    from inferops.adapters.outbound.stages.registry import stage_configuration

    root, _, _, client = workspace
    (root / "plugins/custom.py").write_text("raise RuntimeError('Plugin must not execute during schema discovery')\n")
    response = await client.get("/v1/configs/schema/pipelines")
    assert response.status_code == 200
    candidate = await client.post("/v1/configs/pipelines/custom", json={"data": {
        "name": "custom", "stages": [{"name": "custom", "stage": ".plugins.custom:Stage", "params": {"threshold": 0.4}}],
    }, "validate_only": True})
    assert candidate.status_code == 201 and candidate.json()["valid"]
    # A caller can customize its copy without changing the shared contract.
    stage_configuration()["infer"]["params"]["required"].clear()
    schema = (await client.get("/v1/configs/schema/pipelines")).json()
    assert schema["x-stages"]["infer"]["params"]["required"] == ["model"]


async def test_form_edits_validate_without_saving_reject_stale_and_preserve_sessions(workspace):
    root, _, engine, client = workspace
    url = "/v1/configs/pipelines/process"
    entry = (await client.get(url)).json()
    data = entry["data"]
    data["limits"] = {"timeout_ms": 1234}
    original = (root / "configs/pipelines/process.yaml").read_text()
    response = await client.put(url, json={"data": data, "version": entry["version"], "validate_only": True})
    assert response.status_code == 200 and response.json()["saved"] is False
    assert (root / "configs/pipelines/process.yaml").read_text() == original
    old_session = await engine.open_session("inspection")
    saved = await client.put(url, json={"data": data, "version": entry["version"]})
    assert saved.status_code == 200, saved.text
    assert saved.json()["reload"]["reloaded"] == 1
    new_session = await engine.open_session("inspection")
    assert old_session.models is not new_session.models
    assert old_session.spec.limits.timeout_ms is None
    assert new_session.spec.limits.timeout_ms == 1234
    conflict = await client.put(url, json={"data": entry["data"], "version": entry["version"]})
    assert conflict.status_code == 409
    data["stages"][0]["params"]["model"] = "does_not_exist"
    rejected = await client.put(url, json={"data": data})
    assert rejected.status_code == 400
    assert (await client.get(url)).json()["data"]["stages"][0]["params"]["model"] == "objects"
    await old_session.close()
    await new_session.close()


async def test_interpolation_remains_literal_and_invalid_platform_is_not_saved(workspace, monkeypatch):
    root, core, _, client = workspace
    path = core / "platform.yaml"
    path.write_text("api:\n  port: ${CONFIG_EDITOR_PORT:-8091}\n")
    monkeypatch.setenv("CONFIG_EDITOR_PORT", "8092")
    entry = (await client.get("/v1/configs/platform/platform")).json()
    assert entry["data"]["api"]["port"] == "${CONFIG_EDITOR_PORT:-8091}"
    response = await client.put("/v1/configs/platform/platform", json={"data": entry["data"]})
    assert response.status_code == 200, response.text
    assert "${CONFIG_EDITOR_PORT:-8091}" in path.read_text()
    saved = path.read_text()
    rejected = await client.put("/v1/configs/platform/platform", json={
        "data": {"runtime": {"admission": {"max_pending": 0}}},
    })
    assert rejected.status_code == 400
    assert path.read_text() == saved
    (root / "configs/platform.yaml").write_text("obs:\n  log_level: DEBUG\n")
    manifest = yaml.safe_load((root / "project.yaml").read_text())
    manifest["platform"] = {"api": {"port": 8093}}
    (root / "project.yaml").write_text(yaml.safe_dump(manifest))
    stack = ConfigStack(core, domain_root=root)
    definition = ProjectLoader(root, configs_dir=core).load()
    assert dict(definition.platform) == stack.platform() == {"api": {"port": 8093}, "obs": {"log_level": "DEBUG"}}
    revision = definition.revision
    path.write_text(saved + "\ntransport:\n  max_value_bytes: 1200\n")
    assert ProjectLoader(root, configs_dir=core).load().revision != revision


async def test_create_generic_pipeline_and_prevent_overwrite_and_path_escape(workspace):
    root, _, engine, client = workspace
    response = await client.post("/v1/configs/pipelines/count", json={"data": {
        "name": "count", "stages": [{"name": "detect", "stage": "infer", "params": {"model": "objects"}}],
    }})
    assert response.status_code == 201, response.text
    assert "count" in engine.application.definition.pipelines
    assert (root / "configs/pipelines/count.yaml").is_file()
    assert (await client.post("/v1/configs/pipelines/count", json={"data": {"name": "count"}})).status_code == 409
    assert (await client.post("/v1/configs/pipelines/bad%20name", json={"data": {"name": "bad"}})).status_code == 400
    outside = root.parent / "outside.yaml"
    outside.write_text("name: outside\n")
    (root / "configs/pipelines/link.yaml").symlink_to(outside)
    assert (await client.put("/v1/configs/pipelines/link", json={"data": {"name": "changed"}})).status_code == 400
    assert outside.read_text() == "name: outside\n"


async def test_pending_artifact_change_cannot_be_applied_by_reload_or_another_save(workspace):
    _, _, engine, client = workspace
    original_revision = engine.application.definition.revision
    entry = (await client.get("/v1/configs/models/detector")).json()
    entry["data"]["defaults"]["artifact"] = {"path": "model_repository/replacement.onnx"}
    response = await client.put("/v1/configs/models/detector", json={"data": entry["data"]})
    assert response.json()["preparation_required"]
    reload = await client.post("/v1/configs/reload")
    assert reload.json()["restart_required"] and reload.json()["reloaded"] == 0
    pipeline = (await client.get("/v1/configs/pipelines/process")).json()
    saved = await client.put("/v1/configs/pipelines/process", json={"data": pipeline["data"]})
    assert saved.json()["restart_required"] and saved.json()["reload"]["reloaded"] == 0
    assert engine.application.definition.revision == original_revision


async def test_legacy_yaml_contract_and_invalid_documents(workspace):
    _, _, _, client = workspace
    url = "/v1/configs/fusion/policy"
    original = (await client.get(url)).json()["content"]
    updated = original + "\n# Keep advanced YAML comments\n"
    response = await client.put(url, json={"content": updated, "reload": False})
    assert response.status_code == 200 and response.json()["content"] == updated
    for content in ("name: [", "[]", "false"):
        assert (await client.put(url, json={"content": content})).status_code == 400
    assert (await client.put(url, json={"data": {}, "content": updated})).status_code == 422
    assert (await client.get(url)).json()["content"] == updated


def test_driver_profiles_keep_distinct_behaviors_when_relocated(tmp_path):
    root = tmp_path / "monitoring"
    shutil.copytree(PROJECT_ROOT / "projects/driver_behaviour", root,
                    ignore=shutil.ignore_patterns("model_repository", "__pycache__", "tests"))
    definition = ProjectLoader(root, configs_dir=tmp_path / "configs").load()
    assert set(definition.resolver.ports) == {"rfdetr_driver", "face_mesh"}
    full, reduced = [definition.pipelines[name] for name in ("driver_behaviour", "driver_video")]
    assert full.tags["label"] == "Full driver monitoring"
    assert reduced.tags["label"] == "Detector-only monitoring"
    assert any(stage.stage == "track" for stage in full.stages)
    assert all(stage.stage != "track" for stage in reduced.stages)
    assert all("face_mesh" not in stage.stage for stage in reduced.stages)
