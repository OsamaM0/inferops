"""Project/model isolation, clean imports, bounded runtime ownership, pure rules."""
from __future__ import annotations

import asyncio
import subprocess
import sys

import numpy as np
import pytest
import yaml

from inferops.adapters.outbound.projects import ProjectLoader
from inferops.adapters.outbound.projects import manifest as domain
from inferops.application.models import ModelPool
from inferops.domain.fusion import FusionEngine
from inferops.domain.fusion.spec import FusionSpec
from inferops.domain.incident_spec import AlertCandidate, AlertSpec
from inferops.domain.incidents import IncidentPolicy
from inferops.domain.models import ModelSpec
from inferops.domain.types import StageItem


def project(root, *, number=1):
    root.mkdir(parents=True)
    (root / "models").mkdir()
    (root / "__init__.py").write_text("")
    (root / "project.yaml").write_text("name: sample\n")
    (root / "backend.py").write_text(
        "import asyncio\n"
        "class Backend:\n"
        "    loads = closes = active = peak = 0\n"
        "    spec = None\n"
        "    device = 'cpu'\n"
        "    device_info = {}\n"
        "    async def load(self, spec):\n"
        "        self.spec = spec\n"
        "        type(self).loads += 1\n"
        "    async def infer(self, inputs):\n"
        "        type(self).active += 1\n"
        "        type(self).peak = max(type(self).peak, type(self).active)\n"
        "        try:\n"
        "            await asyncio.sleep(0.01)\n"
        f"            return {{key: value * {number} for key, value in inputs.items()}}\n"
        "        finally: type(self).active -= 1\n"
        "    async def close(self): type(self).closes += 1\n")
    (root / "models" / "identity.yaml").write_text(yaml.safe_dump({
        "name": "identity", "backend": ".backend:Backend", "inputs": [{"name": "images"}],
        "preprocess": {"plugin": "inferops.adapters.outbound.inference.plugins.resize_normalize:ResizeNormalize",
                       "params": {"size": 2}},
    }))
    return root


def test_new_domain_application_imports_have_no_infrastructure_side_effects():
    source = """
import sys
from inferops.domain.fusion import FusionEngine
from inferops.domain.incidents import IncidentPolicy
from inferops.application.models import ModelPool
from inferops.adapters.outbound.projects import ProjectLoader
for name in ('inferops.adapters.outbound.projects.settings', 'sqlalchemy', 'asyncpg', 'boto3', 'redis', 'aiokafka'):
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-c", source], check=True, capture_output=True, text=True)


def test_public_library_project_loading_has_no_settings_or_active_project(tmp_path):
    root = project(tmp_path / "sample")
    source = """
import sys
from inferops import Engine, Model, InferenceSession, ModelSpec
from inferops.adapters.outbound.projects import manifest as domain
from inferops.adapters.outbound.projects import ProjectLoader
before = list(sys.path)
definition = ProjectLoader(sys.argv[1]).load()
session = InferenceSession.from_project(sys.argv[1])
model = Model(ModelSpec(name='standalone', inputs=[{'name': 'x'}]))
assert not hasattr(domain, "_active_package")
assert sys.path == before
assert session.models['identity'].spec.backend.startswith('_inferops_project_')
for name in ('inferops.adapters.outbound.projects.settings', 'sqlalchemy', 'asyncpg', 'boto3', 'redis', 'aiokafka'):
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-c", source, str(root)], check=True, capture_output=True, text=True)


async def test_same_external_project_name_has_independent_plugins_and_models(tmp_path):
    first = project(tmp_path / "a" / "sample", number=2)
    second = project(tmp_path / "b" / "sample", number=3)
    paths = list(sys.path)
    definitions = [ProjectLoader(root).load() for root in (first, second)]
    assert not hasattr(domain, "_active_package")
    assert sys.path == paths
    assert definitions[0].models["identity"].backend != definitions[1].models["identity"].backend
    pools = [ModelPool(definition) for definition in definitions]
    try:
        image = np.full((4, 4, 3), 255, dtype=np.uint8)
        first_result, second_result = await asyncio.gather(*(pool.predict("identity", image) for pool in pools))
        assert np.asarray(first_result.raw["images"]).mean() == 2
        assert np.asarray(second_result.raw["images"]).mean() == 3
    finally:
        await asyncio.gather(*(pool.close() for pool in pools))


def test_reloading_plugin_revision_preserves_old_project_definition(tmp_path):
    root = project(tmp_path / "sample", number=2)
    old = ProjectLoader(root).load()
    old_class = old.object_loader(old.models["identity"].backend)
    path = root / "backend.py"
    path.write_text(path.read_text().replace("value * 2", "value * 999"))
    new = ProjectLoader(root).load()
    new_class = new.object_loader(new.models["identity"].backend)
    assert old.revision != new.revision
    assert old_class is not new_class
    assert old.object_loader(old.models["identity"].backend) is old_class


async def test_late_plugin_import_uses_original_source_without_stale_bytecode(tmp_path):
    root = project(tmp_path / "sample", number=2)
    old = ProjectLoader(root).load()
    path = root / "backend.py"
    stat = path.stat()
    path.write_text(path.read_text().replace("value * 2", "value * 3"))
    import os
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    new = ProjectLoader(root).load()
    old_backend = old.backend_factory(old.models["identity"], None)
    new_backend = new.backend_factory(new.models["identity"], None)
    result_old, result_new = await asyncio.gather(
        old_backend.infer({"x": np.ones(1)}), new_backend.infer({"x": np.ones(1)}))
    assert result_old["x"].item() == 2
    assert result_new["x"].item() == 3


async def test_legacy_sessions_use_independent_project_contexts(tmp_path):
    from inferops import InferenceSession
    first = project(tmp_path / "one" / "sample", number=2)
    second = project(tmp_path / "two" / "sample", number=3)
    async with InferenceSession.from_project(first) as left, InferenceSession.from_project(second) as right:
        image = np.full((4, 4, 3), 255, dtype=np.uint8)
        a, b = await asyncio.gather(left.predict(image), right.predict(image))
        assert np.asarray(a["identity"].raw["images"]).mean() == 2
        assert np.asarray(b["identity"].raw["images"]).mean() == 3


def test_allowlist_tracking_and_explicit_model_groups(tmp_path):
    root = project(tmp_path / "sample")
    folder = root / "configs" / "pipelines"
    folder.mkdir(parents=True)
    (root / "project.yaml").write_text("name: sample\npipelines: [enabled]\n")
    for name in ["enabled", "disabled"]:
        (folder / f"{name}.yaml").write_text(yaml.safe_dump({
            "name": name,
            "stages": [{"name": "run", "stage": "infer", "params": {"model": "identity"}},
                       {"name": "group", "stage": "model_group", "params": {"models": ["identity"]}}],
            "tracking": {"enabled": True, "model_filter": ["identity"]},
        }))
    definition = ProjectLoader(root).load()
    assert list(definition.pipelines) == ["enabled"]
    pipeline = definition.pipelines["enabled"]
    assert [stage.stage for stage in pipeline.stages] == ["infer", "track", "model_group"]
    assert pipeline.stages[2].params["models"] == ["identity"]


def test_project_mode_and_artifact_path_ignore_process_settings(tmp_path, monkeypatch):
    root = project(tmp_path / "sample")
    path = root / "models" / "identity.yaml"
    data = yaml.safe_load(path.read_text())
    data["backend"] = "onnxruntime"
    data["artifact"] = {"path": "model_repository/identity/1/model.onnx"}
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setenv("INFEROPS_EXECUTION_MODE", "cuda")
    cpu = ProjectLoader(root, mode="cpu").load()
    gpu = ProjectLoader(root, mode="cuda").load()
    assert cpu.models["identity"].device.preferred == "cpu"
    assert gpu.models["identity"].device.preferred == "gpu"
    assert cpu.models["identity"].artifact.path == str(root / "model_repository" / "identity" / "1" / "model.onnx")
    backend = cpu.backend_factory(cpu.models["identity"], None)
    assert backend.execution_mode == "cpu"
    assert backend.respect_env is False


async def test_pool_shares_lazy_load_bounds_concurrency_and_closes_once(tmp_path):
    definition = ProjectLoader(project(tmp_path / "sample")).load()
    backend_class = definition.object_loader(definition.models["identity"].backend)
    pool = ModelPool(definition, max_concurrency=2)
    backends = await asyncio.gather(*(pool.get_backend("identity") for _ in range(8)))
    assert all(backend is backends[0] for backend in backends)
    assert backend_class.loads == 1
    results = await asyncio.gather(*(backends[0].infer({"x": np.ones(2)}) for _ in range(8)))
    assert len(results) == 8
    assert backend_class.peak == 2
    await pool.close()
    await pool.close()
    assert backend_class.closes == 1
    with pytest.raises(RuntimeError, match="closed"):
        await pool.get_backend("identity")


async def test_cancelled_backend_call_finishes_before_teardown(tmp_path):
    definition = ProjectLoader(project(tmp_path / "sample")).load()
    backend_class = definition.object_loader(definition.models["identity"].backend)
    pool = ModelPool(definition)
    backend = await pool.get_backend("identity")
    entered, release = asyncio.Event(), asyncio.Event()

    async def delayed_infer(inputs):
        entered.set()
        await release.wait()
        return inputs

    backend._backend.infer = delayed_infer
    task = asyncio.create_task(backend.infer({"x": np.ones(2)}))
    await entered.wait()
    task.cancel()
    close = asyncio.create_task(pool.close())
    await asyncio.sleep(0)
    assert not task.done()
    assert backend_class.closes == 0
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await close
    assert backend_class.closes == 1


async def test_failed_load_releases_partial_backend_and_can_retry(tmp_path):
    definition = ProjectLoader(project(tmp_path / "sample")).load()
    backend_class = definition.object_loader(definition.models["identity"].backend)
    load = backend_class.load

    async def fail(self, spec):
        raise RuntimeError("load failed")

    backend_class.load = fail
    pool = ModelPool(definition)
    with pytest.raises(RuntimeError, match="load failed"):
        await pool.get_backend("identity")
    assert backend_class.closes == 1
    backend_class.load = load
    await pool.get_backend("identity")
    await pool.close()
    assert backend_class.closes == 2


async def test_pure_fusion_uses_zero_media_timestamp_and_session_reset():
    spec = FusionSpec(name="motion", signals=["value"],
                      windows={"value": {"duration_ms": 100}},
                      extract={"value": "payload['value']"},
                      rules=[{"name": "high", "when": "value.latest() > 0.5", "cooldown_ms": 100}],
                      outputs={"events_topic": "unused"}, persistence_enabled=True)
    first, second = FusionEngine(spec), FusionEngine(spec)
    item = StageItem(stream_id="same", ts_ms=0, payload={"value": 1.0})
    assert (await first.ingest_item(item))[0].ts_ms == 0
    assert await first.ingest_item(item.model_copy(update={"ts_ms": 50})) == []
    assert (await second.ingest_item(item))[0].ts_ms == 0
    first.reset()
    assert (await first.ingest_item(item))[0].ts_ms == 0


async def test_fusion_checkpoint_restores_windows_cooldowns_and_state_machine():
    spec = FusionSpec(
        name="motion", signals=["value"], windows={"value": {"duration_ms": 100}},
        extract={"value": "payload['value']"},
        rules=[{"name": "high", "when": "value.latest() > 0.5", "cooldown_ms": 100}],
        state_machine={"initial": "idle", "states": ["idle", "active"],
                       "transitions": [{"from_state": "idle", "to_state": "active",
                                        "when": "value.latest() > 0.5"}]})
    engine = FusionEngine(spec)
    item = StageItem(stream_id="same", ts_ms=0, payload={"value": 1.0})
    checkpoint = engine.checkpoint()
    assert [event.type for event in await engine.ingest_item(item)] == ["high", "STATE_active"]
    assert await engine.ingest_item(item.model_copy(update={"ts_ms": 50})) == []
    engine.restore(checkpoint)
    assert [event.type for event in await engine.ingest_item(item)] == ["high", "STATE_active"]
    engine.restore(checkpoint)
    assert not engine._windows
    assert not engine._cooldowns
    assert engine._fsm.state("same") == "idle"


async def test_incident_policy_rollback_and_independent_media_cooldowns():
    spec = AlertSpec(name="alerts", severity={"levels": [{"name": "high", "rank": 1}],
                                             "default": "high"},
                     dedupe={"window_ms": 100}, cooldown={"per_dedupe_key_ms": 0},
                     sinks=[{"name": "unused", "kind": "postgres"}], persistence_enabled=True)
    first, second = IncidentPolicy(spec), IncidentPolicy(spec)
    candidate = AlertCandidate(type="high", entity_id="same", ts_ms=0)
    checkpoint = first.checkpoint()
    assert (await first.submit(candidate)).ts_ms == 0
    assert await first.submit(candidate.model_copy(update={"ts_ms": 50})) is None
    assert await second.submit(candidate) is not None
    first.restore(checkpoint)
    assert await first.submit(candidate) is not None
    assert await first.submit(candidate.model_copy(update={"ts_ms": 100})) is not None


def test_specs_can_be_copied_without_mutating_compiled_model_definition(tmp_path):
    definition = ProjectLoader(project(tmp_path / "sample")).load()
    pool = ModelPool(definition)
    copy = pool.get_spec("identity")
    copy.inputs[0].name = "changed"
    assert definition.models["identity"].inputs[0].name == "images"
    assert isinstance(copy, ModelSpec)


def test_candidate_overrides_validate_references_without_writing_or_mutating(tmp_path):
    from inferops.domain.errors import ConfigError
    root = project(tmp_path / "sample")
    folder = root / "configs" / "pipelines"
    folder.mkdir(parents=True)
    path = folder / "driver.yaml"
    original = {"name": "driver", "stages": [{"name": "infer", "stage": "infer", "params": {"model": "identity"}}]}
    path.write_text(yaml.safe_dump(original))
    loader = ProjectLoader(root)
    baseline = loader.load()
    bad = {"name": "driver", "stages": [{"name": "fuse", "stage": "fuse", "params": {"fusion": "missing"}}]}
    with pytest.raises(ConfigError, match="unknown fusion"):
        loader.load({path: bad})
    assert yaml.safe_load(path.read_text()) == original
    changed = {"name": "driver", "stages": [{"name": "group", "stage": "model_group",
                                               "params": {"models": ["identity"]}}]}
    candidate = loader.load({path: changed})
    assert candidate.revision != baseline.revision
    assert candidate.pipelines["driver"].stages[0].stage == "model_group"
    assert baseline.pipelines["driver"].stages[0].stage == "infer"
    assert yaml.safe_load(path.read_text()) == original
    assert changed["stages"][0]["params"]["models"] == ["identity"]
    path.write_text("# candidate saved with different formatting\n" + yaml.safe_dump(changed, sort_keys=False))
    assert loader.load().revision == candidate.revision


def test_candidate_new_model_port_and_manifest_use_same_resolver(tmp_path):
    root = project(tmp_path / "sample")
    path = root / "models" / "new" / "model.yaml"
    config = {"name": "new", "defaults": {"inputs": [{"name": "x"}]}, "variants": {"onnx": {}}}
    definition = ProjectLoader(root).load({path: config,
                                          root / "project.yaml": {"name": "sample", "prewarm_models": ["new"]}})
    assert definition.resolve_model("new") == "new_onnx"
    assert "new_onnx" in definition.models
    assert not path.exists()


@pytest.mark.parametrize("declaration, error", [
    ({"name": "group", "stage": "model_group", "params": {"models": []}}, "nonempty models"),
    ({"name": "group", "stage": "model_group", "params": {"models": ["missing"]}}, "unknown models"),
    ({"name": "alert", "stage": "alert", "params": {"alerts": "missing"}}, "unknown alerts"),
    ({"name": "decode", "stage": "decode_any", "params": {"every_n": 0}}, "positive integer"),
    ({"name": "typo", "stage": "typo"}, "unknown builtin"),
])
def test_compile_rejects_broken_builtin_contracts(tmp_path, declaration, error):
    from inferops.domain.errors import ConfigError
    root = project(tmp_path / "sample")
    path = root / "configs" / "pipelines" / "driver.yaml"
    with pytest.raises(ConfigError, match=error):
        ProjectLoader(root).load({path: {"name": "driver", "stages": [declaration]}})


def test_relative_alert_targets_and_capture_hooks_are_instance_scoped(tmp_path):
    root = project(tmp_path / "sample")
    definition = ProjectLoader(root).load({
        root / "configs" / "pipelines" / "driver.yaml": {
            "name": "driver", "sinks": [{"name": "capture", "stage": "sink_s3",
                                           "params": {"trigger_flags": ".hooks:flags"}}]},
        root / "configs" / "alerts" / "driver.yaml": {
            "name": "driver", "severity": {"levels": [{"name": "high", "rank": 1}], "default": "high"},
            "sinks": [{"name": "custom", "kind": "plugin", "params": {"target": ".hooks:Action"}}]},
    })
    target = definition.alerts["driver"].sinks[0].params["target"]
    hook = definition.pipelines["driver"].sinks[0].params["trigger_flags"]
    assert target.startswith(definition.plugin_resolver.package)
    assert hook.startswith(definition.plugin_resolver.package)
