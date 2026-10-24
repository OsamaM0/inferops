"""The documented .env selects the complete stack without manual -f flags."""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess

import pytest

from inferops.testing import PROJECT_ROOT


@pytest.mark.parametrize(("profiles", "configured", "expected"), [
    ("", None, "filesystem"),
    ("storage", "filesystem", "s3"),
    ("", "s3", "s3"),
])
def test_launcher_selects_disk_by_default_and_s3_when_requested(monkeypatch, profiles, configured, expected):
    from scripts import run_mode

    environment = {} if configured is None else {"INFEROPS_STORAGE_KIND": configured}
    monkeypatch.setattr(run_mode, "runtime_environment", lambda *args, **kwargs: environment.copy())
    monkeypatch.setattr(run_mode.sys, "argv", ["run_mode.py", "--compose-check", "--profiles", profiles])
    calls = []
    monkeypatch.setattr(run_mode.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs["env"])))
    run_mode.main()
    command, selected = calls.pop()
    assert selected["INFEROPS_STORAGE_KIND"] == expected
    assert ("storage" in command) == (profiles == "storage")


@pytest.mark.parametrize("owned", [False, True])
def test_container_start_diagnoses_an_existing_port_without_killing_other_processes(monkeypatch, owned):
    from scripts import run_mode

    monkeypatch.setattr(run_mode, "stop_native_api", lambda **kwargs: False)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        payload = [{"State": "running", "Publishers": [{"PublishedPort": port}]}] if owned else []
        monkeypatch.setattr(run_mode.subprocess, "check_output", lambda *args, **kwargs: json.dumps(payload))
        if owned:
            run_mode.ensure_container_port(port)
        else:
            with pytest.raises(SystemExit, match=f"Port {port} is already in use"):
                run_mode.ensure_container_port(port)


@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker Compose is unavailable")
@pytest.mark.parametrize("mode", ["cpu", "auto", "cuda", "triton"])
@pytest.mark.parametrize("image", [None, "osamam0/inferops-driverbeh:1.2.3-cpu"])
def test_central_yaml_selects_runtime_and_orders_preparation(tmp_path, mode, image):
    import yaml

    from scripts.configure import render

    config = tmp_path / "configs/inferops.yaml"
    config.parent.mkdir()
    data = yaml.safe_load((PROJECT_ROOT / "configs/inferops.yaml").read_text())
    data["runtime"].update(mode=mode, api_port=8099, services=["storage", "broker", "monitoring"])
    data["storage"]["kind"] = "s3"
    data["projects"]["driver_behaviour"]["root"] = str(PROJECT_ROOT / "projects/driver_behaviour")
    if image:
        data["docker"]["image"] = image
    config.write_text(yaml.safe_dump(data))
    (tmp_path / ".env").write_text("INFEROPS_S3_ACCESS_KEY=verification-key\nINFEROPS_S3_SECRET_KEY=verification-secret\n")
    output = tmp_path / "generated"
    render(config_file=config, output=output, environ={})
    environment = {key: value for key, value in os.environ.items() if not key.startswith(("COMPOSE_", "INFEROPS_"))}
    root = tmp_path / "compose.yaml"
    root.write_text(yaml.safe_dump({"name": "inferops", "include": [str(output / "compose.yaml")]}))
    command = ["docker", "compose", "--env-file", str(output / "compose.env"), "-f", str(root)]
    stack = json.loads(subprocess.check_output([*command, "config", "--format", "json"], env=environment, text=True))
    services = stack["services"]
    assert {"api", "prepare", "postgres", "mediamtx", "minio", "redpanda", "grafana", "prometheus"} <= services.keys()
    assert services["api"]["image"] == services["prepare"]["image"] == (image or data["docker"]["image"])
    assert services["api"]["environment"]["INFEROPS_S3_SECRET_KEY"] == "verification-secret"
    assert services["minio"]["environment"]["MINIO_ROOT_PASSWORD"] == "verification-secret"
    assert services["api"]["ports"][0]["published"] == "8099"
    assert services["api"]["environment"]["INFEROPS_API_PORT"] == "8080"
    assert services["api"]["environment"]["INFEROPS_STORAGE_KIND"] == "s3"
    assert services["api"]["depends_on"]["prepare"]["condition"] == "service_completed_successfully"
    assert services["api"]["depends_on"]["minio-init"]["required"] is False
    config_mount = next(mount for mount in services["api"]["volumes"] if mount["target"] == "/app/configs")
    assert config_mount["source"] == str(config.parent) and not config_mount.get("read_only", False)
    project_mount = next(mount for mount in services["api"]["volumes"] if mount["target"] == "/project")
    assert project_mount["source"] == str(PROJECT_ROOT / "projects/driver_behaviour")
    if mode == "triton":
        assert services["triton"]["depends_on"]["prepare"]["condition"] == "service_completed_successfully"
        assert services["api"]["depends_on"]["triton"]["condition"] == "service_healthy"
    else:
        assert "triton" not in services
    if mode == "cuda":
        assert services["api"]["deploy"]["resources"]["reservations"]["devices"][0]["capabilities"] == ["gpu"]
        assert services["prepare"]["build"]["args"]["INFERENCE_EXTRAS"] == "cuda"
    # Root include and direct base+overlay produce identical effective services.
    explicit = ["docker", "compose", "--env-file", str(output / "compose.env"), "--project-directory", str(PROJECT_ROOT),
                "-f", str(PROJECT_ROOT / "deploy/compose.yaml"), "-f", str(PROJECT_ROOT / f"deploy/compose/{mode}.yaml")]
    direct = json.loads(subprocess.check_output([*explicit, "config", "--format", "json"], env=environment, text=True))
    assert direct["services"] == services
