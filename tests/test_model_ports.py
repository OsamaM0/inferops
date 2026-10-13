"""Model ports work with local assets, reload failures and legacy projects."""
from __future__ import annotations

import shutil

import numpy as np
import pytest
import yaml

from inferops.core.config import ConfigStack, settings
from inferops.core.errors import ConfigError
from inferops.core.plugins import load_object
from inferops.registry.loader import load_model_variants, load_models_dir
from inferops.registry.scaffold import scaffold_model_port
from inferops.registry.store import ModelRegistry
from inferops.testing import PROJECT_ROOT


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["cpu", "auto", "cuda", "tensorrt", "triton", "deepstream"])
async def test_logical_models_group_variants_and_select_runtime(mode, monkeypatch):
    monkeypatch.setattr(settings, "execution_mode", mode)
    registry = ModelRegistry()
    try:
        assert await registry.reload_from(PROJECT_ROOT / "projects/driver_behaviour/models") == 6
        models = registry.logical_models()
        assert [model["name"] for model in models] == ["face_mesh", "rfdetr_driver"]
        suffix = "triton" if mode in {"triton", "deepstream"} else "tensorrt" if mode == "tensorrt" else "onnx"
        assert all(model["selected_variant"] == f"{model['name']}_{suffix}" for model in models)
        assert all(len(model["variants"]) == 3 and not model["loaded_variants"] for model in models)
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_port_local_assets_run_and_invalid_reload_preserves_backend(tmp_path):
    folder = tmp_path / "models/classifier"
    folder.mkdir(parents=True)
    shutil.copy(PROJECT_ROOT / "model_repository/tiny_classifier/1/model.onnx", folder / "model.onnx")
    (folder / "labels.txt").write_text("\n".join(f"local{i}" for i in range(10)))
    data = yaml.safe_load((PROJECT_ROOT / "inferops/testing/configs/models/tiny_classifier.yaml").read_text())
    data.pop("name")
    data["labels"] = {"from_file": "labels.txt"}
    data["artifact"] = {"path": "model.onnx"}
    path = folder / "model.yaml"
    path.write_text(yaml.safe_dump({"name": "local", "defaults": data, "variants": {"cpu": {}}}))
    registry = ModelRegistry()
    try:
        assert await registry.reload_from(folder.parent) == 1
        assert registry.logical_models() == [{"name": "local", "variants": ["local_cpu"],
                                              "selected_variant": "local_cpu", "loaded_variants": []}]
        assert registry.get_spec("local_cpu").label_list()[0] == "local0"
        backend = await registry.get_backend("local_cpu")
        assert registry.logical_models()[0]["loaded_variants"] == ["local_cpu"]
        outputs = await backend.infer({"input": np.zeros((1, 3, 32, 32), dtype=np.float32)})
        assert outputs["logits"].shape == (1, 10)
        path.write_text("name: local\nvariants: {}\n")
        with pytest.raises(ConfigError):
            await registry.reload_from(folder.parent)
        assert await registry.get_backend("local_cpu") is backend
        assert registry.get_spec("local_cpu").label_list()[0] == "local0"
    finally:
        await registry.close()


def test_add_port_to_legacy_project_keeps_existing_models(tmp_path):
    root = tmp_path / "legacy_project"
    directory = root / "configs/models"
    directory.mkdir(parents=True)
    (root / "__init__.py").touch()
    (directory / "old.yaml").write_text("name: existing\nbackend: synthetic\ninputs: []\n")
    scaffold_model_port(root, "helmets", ["person", "helmet"])
    stack = ConfigStack(tmp_path / "configs", domain_root=root)
    try:
        specs = load_models_dir(stack.spec_dir("models"))
        assert {spec.name for spec in specs} == {"existing", "helmets_onnx", "helmets_triton"}
        adapter = next(spec for spec in specs if spec.name == "helmets_onnx").postprocess.plugin
        assert load_object(adapter).__name__ == "Postprocess"
        with pytest.raises(FileExistsError):
            scaffold_model_port(root, "helmets", ["person"])
    finally:
        ConfigStack(PROJECT_ROOT / "configs")


def test_variant_error_identifies_the_model_and_runtime(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text("name: detector\ndefaults: {inputs: []}\nvariants: {cuda: {device: {preferred: wrong}}}\n")
    with pytest.raises(ConfigError, match=r"detector.*cuda"):
        load_model_variants(path)
