"""First boot must export head pose without adding Torch to the API environment."""

import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

path = Path(__file__).resolve().parents[1] / "models/head_pose_360/prepare.py"
spec = importlib.util.spec_from_file_location("head_pose_preparation_tests", path)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


@pytest.fixture
def exporter(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "__file__", str(tmp_path / "project/models/head_pose_360/prepare.py"))
    state = SimpleNamespace(ready=False, fail_install=False, reject=False, commands=[])

    def run(command, **kwargs):
        state.commands.append(command)
        if command[1:3] == ["-m", "venv"]:
            python = Path(command[-1]) / "bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
        elif command[1:3] == ["-m", "pip"]:
            if state.fail_install:
                raise subprocess.CalledProcessError(1, command)
            if "onnx==1.23.2" in command:
                state.ready = True
        elif command[1] == "-c":
            return SimpleNamespace(returncode=int(not state.ready or state.reject))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(prepare.subprocess, "run", run)
    return state


def test_fresh_boot_installs_cpu_exporter_separately_and_reuses_it_offline(exporter):
    python = prepare.export_python()
    installers = [c for c in exporter.commands if c[1:3] == ["-m", "pip"]]
    assert installers and all(c[0] == str(python) for c in installers)
    assert all(c[0] != prepare.sys.executable for c in installers)
    torch = next(c for c in installers if "torch==2.8.0" in c)
    assert torch[-2:] == ["--index-url", "https://download.pytorch.org/whl/cpu"]
    assert "torchvision==0.23.0" in torch
    exporter.commands.clear()
    assert prepare.export_python() == python
    assert not any(c[1:3] == ["-m", "pip"] for c in exporter.commands)


def test_interrupted_dependency_install_is_retried_on_next_boot(exporter):
    exporter.fail_install = True
    with pytest.raises(subprocess.CalledProcessError):
        prepare.export_python()
    exporter.fail_install = False
    assert prepare.export_python().is_file() and exporter.ready


def test_invalid_export_environment_does_not_report_success(exporter):
    exporter.reject = True
    with pytest.raises(RuntimeError, match="version/CPU check"):
        prepare.export_python()


def test_installer_delegates_export_and_propagates_failure(tmp_path, monkeypatch):
    python = tmp_path / "isolated/bin/python"
    monkeypatch.setattr(prepare, "export_python", lambda: python)
    output = tmp_path / "model.onnx"

    def fail(command, **kwargs):
        assert command[0] == str(python)
        assert command[-3:] == ["--export-only", "--output", str(output)]
        assert kwargs["check"] is True
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(prepare.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        prepare.install(output)
    assert not output.exists()
