"""Projects remain runnable when copied, renamed, mounted, or replaced."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest
import yaml

from inferops.adapters.outbound.projects.manifest import resolve_plugin_ref, scaffold_domain
from inferops.adapters.outbound.projects.settings import ConfigStack
from inferops.adapters.outbound.projects.validation import validate_config_stack
from inferops.domain.errors import ConfigError
from inferops.domain.pipeline import PipelineSpec
from inferops.testing import PROJECT_ROOT
from inferops.testing.postgres import test_database_url


def test_selected_project_never_inherits_core_sections(tmp_path):
    core = tmp_path / "configs"
    (core / "alerts").mkdir(parents=True)
    (core / "alerts/unrelated.yaml").write_text("name: unrelated\n")
    pack = tmp_path / "empty_project"
    pack.mkdir()
    stack = ConfigStack(core, domain_root=pack)
    assert stack.spec_dir("alerts") == pack / "configs/alerts"
    assert not stack.spec_dir("alerts").exists()
    assert stack.model_repository() == pack / "model_repository"
    assert validate_config_stack(stack)["alerts"] == {}
    ConfigStack(core)
    with pytest.raises(ValueError, match="no domain is active"):
        resolve_plugin_ref(".plugins.missing:Missing")


def test_missing_project_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="project block does not exist"):
        ConfigStack(tmp_path / "configs", active_domain="typo")


def test_invalid_yaml_is_reported_and_can_be_repaired_with_a_prospective_edit(tmp_path):
    root = tmp_path / "repairable"
    _configured_project(root, "repairable", ["person"])
    stack = ConfigStack(tmp_path / "configs", domain_root=root)
    path = root / "configs/pipelines/repairable.yaml"
    original = yaml.safe_load(path.read_text())
    path.write_text("name: [\n")
    with pytest.raises(ConfigError, match="Invalid project configuration"):
        validate_config_stack(stack)
    assert "repairable" in validate_config_stack(stack, overrides={path: original})["pipelines"]


@pytest.mark.parametrize("reference", ["model", "fusion", "alerts", "mode", "duplicate"])
def test_invalid_project_graph_fails_before_startup(tmp_path, reference):
    root = tmp_path / "new_project"
    _configured_project(root, "new_project", ["person"])
    path = root / "configs/pipelines/new_project.yaml"
    data = yaml.safe_load(path.read_text())
    if reference == "duplicate":
        shutil.copytree(root / "models/new_project_detector", root / "models/duplicate")
    elif reference == "mode":
        data["stages"][1]["params"]["model_modes"] = {"cpu": "missing_model"}
    else:
        index = {"model": 1, "fusion": 2, "alerts": 3}[reference]
        data["stages"][index]["params"][reference] = "missing_config"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match=r"duplicate|unknown"):
        validate_config_stack(ConfigStack(tmp_path / "configs", domain_root=root))


def test_driver_plugins_survive_a_relocated_and_renamed_pack(tmp_path):
    source = PROJECT_ROOT / "projects/driver_behaviour"
    relocated = tmp_path / "mounted_driver"
    shutil.copytree(source, relocated, ignore=shutil.ignore_patterns("model_repository", "__pycache__", "tests"))
    stack = ConfigStack(tmp_path / "configs", domain_root=relocated)
    specs = validate_config_stack(stack)
    plugin = specs["models"]["rfdetr_driver_onnx"].postprocess.plugin
    assert plugin.startswith(".models.")
    from inferops.adapters.outbound.projects.loader import ProjectPluginResolver
    resolver = ProjectPluginResolver(relocated, "test")
    assert resolver.load_object(plugin).__name__ == "RFDetrPostprocess"
    stage = specs["pipelines"]["driver_behaviour"].stages[2].stage
    assert resolver.load_object(stage).__name__ == "FaceMeshStage"
    assert stack.active_domain == "driver_behaviour"
    ConfigStack(PROJECT_ROOT / "configs")


def test_external_project_runs_inference_rules_alerts_and_custom_action(tmp_path):
    """Exercise the full API contract in a fresh process with no core edits."""
    original = tmp_path / "inspection"
    _configured_project(original, "inspection", ["person", "helmet"])
    mounted = tmp_path / "mounted_project"
    original.rename(mounted)
    (mounted / "plugins/record_alert.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        "from inferops.ports.extensions import AlertSink\n"
        "class RecordAlert(AlertSink):\n"
        "    async def emit(self, alert):\n"
        "        with Path(self.spec.params['path']).open('a') as output:\n"
        "            output.write(json.dumps(alert.model_dump(mode='json')) + '\\n')\n"
    )
    path = mounted / "configs/alerts/inspection.yaml"
    alerts = yaml.safe_load(path.read_text())
    action_output = tmp_path / "actions.jsonl"
    alerts["persistence_enabled"] = True  # Durable HTTP server configuration.
    alerts["sinks"] = [{"name": "record", "kind": "plugin", "params": {
        "target": ".plugins.record_alert:RecordAlert", "path": str(action_output),
    }}]
    path.write_text(yaml.safe_dump(alerts))
    path = mounted / "configs/pipelines/inspection.yaml"
    pipeline = yaml.safe_load(path.read_text())
    pipeline["sinks"] = []
    path.write_text(yaml.safe_dump(pipeline))
    env = {key: value for key, value in os.environ.items() if not key.startswith("INFEROPS_")}
    env.update({
        "INFEROPS_CONFIGS_DIR": str(PROJECT_ROOT / "configs"),
        "INFEROPS_DOMAIN_ROOT": str(mounted),
        "INFEROPS_ACTIVE_DOMAIN": "inspection",
        "INFEROPS_EXECUTION_MODE": "auto",
        "INFEROPS_TRANSPORT_KIND": "inproc",
        "INFEROPS_DATABASE_URL": test_database_url(tmp_path),
        "INFEROPS_STORAGE_KIND": "filesystem",
        "INFEROPS_DATA_PATH": str(tmp_path / "data"),
        "INFEROPS_SPOOL_PATH": str(tmp_path / "spool"),
        "INFEROPS_SOURCES_PATH": str(tmp_path / "sources"),
        "INFEROPS_MEDIA_ROUTER": "direct",
        "INFEROPS_PREWARM_MODELS": "",
        "INFEROPS_PREWARM_PIPELINES": "",
        "PYTHONPATH": str(PROJECT_ROOT),
    })
    script = """
import asyncio, json
import cv2
import numpy as np
import yaml
from httpx import ASGITransport, AsyncClient
from inferops.adapters.inbound.http.main import create_app
async def run():
    app = create_app()
    async with app.router.lifespan_context(app), AsyncClient(
        transport=ASGITransport(app=app), base_url='http://test'
    ) as client:
        config = (await client.get('/v1/configs')).json()
        assert config['active_domain'] == 'inspection'
        names = {model['name'] for model in (await client.get('/v1/models')).json()}
        assert names == {'inspection_detector_onnx', 'inspection_detector_triton'}
        port = 'inspection_detector'
        assert any(f['section'] == 'models' and f['name'] == port for f in config['files'])
        assert config['domains'][0]['config_names']['model'] == port
        url = f'/v1/configs/models/{port}'
        original = (await client.get(url)).json()['content']
        updated = original + '\\n# model-port-editor-check\\n'
        saved = await client.put(url, json={'content': updated})
        assert saved.status_code == 200, saved.text
        assert saved.json()['reload']['reloaded'] == 2
        invalid = yaml.safe_load(original)
        invalid['name'] = 'broken_reference'
        rejected = await client.put(url, json={'content': yaml.safe_dump(invalid)})
        assert rejected.status_code == 400, rejected.text
        assert (await client.get(url)).json()['content'] == updated
        unsupported = yaml.safe_load(original)
        unsupported['variants'].pop('onnx')
        assert (await client.put(url, json={'content': yaml.safe_dump(unsupported)})).status_code == 400
        assert (await client.get(url)).json()['content'] == updated
        restored = await client.put(url, json={'content': original})
        assert restored.status_code == 200, restored.text
        project_url = '/v1/configs/project/project'
        project_content = (await client.get(project_url)).json()['content']
        invalid_project = yaml.safe_load(project_content)
        invalid_project['pipelines'] = ['missing']
        assert (await client.put(project_url, json={'content': yaml.safe_dump(invalid_project)})).status_code == 400
        project_saved = await client.put(project_url, json={'content': project_content + '\\n# project-editor-check\\n'})
        assert project_saved.json()['reload']['restart_required'] is True
        assert project_saved.json()['preparation_required'] is True
        new_source = yaml.safe_load(original)
        new_source['defaults']['artifact'] = {'source': {'kind': 'url', 'url': 'https://example.test/model.onnx'}}
        source_saved = await client.put(url, json={'content': yaml.safe_dump(new_source)})
        assert source_saved.json()['preparation_required'] is True
        assert source_saved.json()['reload']['reloaded'] == 0
        assert (await client.put(url, json={'content': original})).status_code == 200
        ok, jpeg = cv2.imencode('.jpg', np.zeros((64, 64, 3), dtype=np.uint8))
        assert ok
        for _ in range(4):
            response = await client.post('/v1/infer/inspection',
                files={'image': ('f.jpg', jpeg.tobytes(), 'image/jpeg')},
                data={'stream_id': 'inspection-test'})
            assert response.status_code == 200, response.text
        assert any(result['kind'] == 'detections' for result in response.json()[0]['results'])
asyncio.run(run())
"""
    proc = subprocess.run([sys.executable, "-c", script], cwd=PROJECT_ROOT, env=env,
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    import json

    recorded = [json.loads(line) for line in action_output.read_text().splitlines()]
    assert any(alert["type"] == "person_present" and alert["severity"] == "medium" for alert in recorded)


def test_launcher_resolves_an_external_project_without_driver_paths(tmp_path, monkeypatch):
    from scripts.run_mode import runtime_environment

    root = tmp_path / "warehouse"
    scaffold_domain(root, "warehouse", ["forklift"])
    monkeypatch.setenv("INFEROPS_ACTIVE_DOMAIN", "driver_behaviour")
    env = runtime_environment("cpu", 18080, domain_root=str(root))
    assert env["INFEROPS_ACTIVE_DOMAIN"] == "warehouse"
    assert env["INFEROPS_DOMAIN_ROOT"] == str(root)
    assert env["INFEROPS_PROJECT_ROOT"] == str(root)
    assert env["INFEROPS_MODEL_REPOSITORY"] == str(root / "model_repository")
    assert env["INFEROPS_TRITON_REPOSITORY"] == str(root / "model_repository")


def test_production_mode_rejects_unsupported_choices_and_disables_fallback():
    spec = PipelineSpec(name="portable", stages=[{"name": "detect", "stage": "infer", "params": {
        "model": "real_model", "fallback_model": "demo_model", "model_modes": {"cpu": "real_model"},
    }}])
    with pytest.raises(ConfigError, match="no model configured for mode 'triton'"):
        spec.for_mode("triton")
    spec.stages[0].params.pop("model_modes")
    assert spec.for_mode("cpu").stages[0].params["fallback_model"] is None
    assert spec.for_mode("auto").stages[0].params["fallback_model"] is None


def test_empty_project_opens_dashboard_without_inference(tmp_path):
    root = tmp_path / "empty"
    scaffold_domain(root, "empty", [])
    env = {key: value for key, value in os.environ.items() if not key.startswith("INFEROPS_")}
    env.update({"INFEROPS_DOMAIN_ROOT": str(root), "INFEROPS_CONFIGS_DIR": str(PROJECT_ROOT / "configs"),
                "INFEROPS_DATABASE_URL": test_database_url(tmp_path), "INFEROPS_TRANSPORT_KIND": "inproc",
                "INFEROPS_STORAGE_KIND": "filesystem", "INFEROPS_DATA_PATH": str(tmp_path / "data"),
                "INFEROPS_SPOOL_PATH": str(tmp_path / "spool"), "INFEROPS_SOURCES_PATH": str(tmp_path / "sources"),
                "INFEROPS_MEDIA_ROUTER": "direct",
                "PYTHONPATH": str(PROJECT_ROOT)})
    script = """
import asyncio
from httpx import ASGITransport, AsyncClient
from inferops.adapters.inbound.http.main import create_app
async def run():
    app = create_app()
    async with app.router.lifespan_context(app), AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
        assert (await client.get('/frontend/')).status_code == 200
        assert (await client.get('/v1/models')).json() == []
        assert (await client.get('/v1/pipelines')).json() == []
asyncio.run(run())
"""
    proc = subprocess.run([sys.executable, "-c", script], cwd=PROJECT_ROOT, env=env,
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stdout + proc.stderr

def _configured_project(root, name, labels):
    """Test-only constant ONNX detector; production scaffolds remain empty."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from inferops.adapters.outbound.projects.model_scaffold import scaffold_model_port

    scaffold_domain(root, name, labels)
    port = f"{name}_detector"
    scaffold_model_port(root, port, labels, format="detr")
    path = root / "models" / port / "model.yaml"
    data = yaml.safe_load(path.read_text())
    data["defaults"]["outputs"] = [
        {"name": "logits", "shape": [1, 1, len(labels)], "layout": "ND", "color": "NONE"},
        {"name": "boxes", "shape": [1, 1, 4], "layout": "ND", "color": "NONE"},
    ]
    path.write_text(yaml.safe_dump(data))
    logits = np.full((1, 1, len(labels)), -6.0, dtype=np.float32)
    logits[0, 0, 0] = 6.0
    boxes = np.array([[[0.5, 0.5, 0.3, 0.4]]], dtype=np.float32)
    graph = helper.make_graph([
        helper.make_node("Constant", [], ["logits"], value=numpy_helper.from_array(logits)),
        helper.make_node("Constant", [], ["boxes"], value=numpy_helper.from_array(boxes)),
    ], "test_detector", [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
       [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 1, len(labels)]),
        helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 1, 4])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    artifact = root / data["variants"]["onnx"]["artifact"]["path"]
    artifact.parent.mkdir(parents=True)
    onnx.save(model, artifact)
    pipeline = {"name": name, "source": {"name": "src", "stage": "source_rest"}, "stages": [
        {"name": "decode", "stage": "decode_any"},
        {"name": "detect", "stage": "infer", "params": {"model": port}},
        {"name": "fuse", "stage": "fuse", "params": {"fusion": f"{name}_fusion"}},
        {"name": "alert", "stage": "alert", "params": {"alerts": f"{name}_alerts"}},
    ], "sinks": []}
    fusion = {"name": f"{name}_fusion", "entity_key": "stream_id", "signals": ["score"],
              "extract": {"score": "max((d.score for r in results if r.detections for d in r.detections), default=0.0)"},
              "windows": {"score": {"kind": "rolling_time", "duration_ms": 3000, "reducers": ["latest"]}},
              "rules": [{"name": "present", "when": "score.latest() > 0.5", "severity": "medium",
                         "emit": {"type": f"{labels[0]}_present"}}]}
    alerts = {"name": f"{name}_alerts", "sinks": [{"name": "stdout", "kind": "stdout"}],
              "severity": {"levels": [{"name": "low", "rank": 1}, {"name": "medium", "rank": 2}], "mapping": {f"{labels[0]}_present": "medium"}}, "persistence_enabled": False}
    for section, content in [("pipelines", pipeline), ("fusion", fusion), ("alerts", alerts)]:
        (root / "configs" / section / f"{name}.yaml").write_text(yaml.safe_dump(content))
    manifest = yaml.safe_load((root / "project.yaml").read_text())
    manifest["pipelines"] = [name]
    (root / "project.yaml").write_text(yaml.safe_dump(manifest))
