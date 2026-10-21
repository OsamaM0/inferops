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
def test_env_selects_runtime_and_orders_preparation(tmp_path, mode):
    root = PROJECT_ROOT
    environment = {key: value for key, value in os.environ.items() if not key.startswith(("COMPOSE_", "INFEROPS_"))}
    environment["INFEROPS_EXECUTION_MODE"] = mode
    environment["INFEROPS_S3_ACCESS_KEY"] = "verification-key"
    environment["INFEROPS_S3_SECRET_KEY"] = "verification-secret"
    env_file = tmp_path / ".env"
    env_file.write_text((root / ".env.example").read_text())
    result = subprocess.run(["docker", "compose", "--env-file", str(env_file), "config", "--format", "json"],
                            cwd=root, env=environment, capture_output=True, text=True, check=True)
    services = json.loads(result.stdout)["services"]
    assert {"prepare", "api"} <= services.keys()
    assert {"postgres", "mediamtx"} <= services.keys()
    assert not {"minio", "minio-init", "redis", "redpanda", "grafana", "prometheus"} & services.keys()
    assert services["api"]["environment"]["INFEROPS_TRANSPORT_KIND"] == "inproc"
    assert services["api"]["environment"]["INFEROPS_STORAGE_KIND"] == "filesystem"
    evidence_mount = next(mount for mount in services["api"]["volumes"]
                          if mount["target"] == "/var/lib/inferops/data")
    assert evidence_mount["type"] == "volume" and evidence_mount["source"] == "engine-data"
    environment["COMPOSE_PROFILES"] = f"{mode},storage,broker,monitoring"
    environment["INFEROPS_STORAGE_KIND"] = "s3"
    result = subprocess.run(["docker", "compose", "--env-file", str(env_file), "config", "--format", "json"],
                            cwd=root, env=environment, capture_output=True, text=True, check=True)
    services = json.loads(result.stdout)["services"]
    assert services["minio"]["environment"]["MINIO_ROOT_USER"] == services["api"]["environment"]["INFEROPS_S3_ACCESS_KEY"]
    assert services["minio"]["environment"]["MINIO_ROOT_PASSWORD"] == services["api"]["environment"]["INFEROPS_S3_SECRET_KEY"]
    assert services["prepare"]["command"] == ["inferops", "project", "prepare"]
    assert services["api"]["depends_on"]["prepare"]["condition"] == "service_completed_successfully"
    assert services["api"]["depends_on"]["minio-init"]["condition"] == "service_completed_successfully"
    assert services["api"]["depends_on"]["minio-init"]["required"] is False
    assert services["api"]["environment"]["INFEROPS_STORAGE_KIND"] == "s3"
    assert "redis" not in services
    assert services["postgres"]["volumes"][0]["type"] == "volume"
    if mode in {"triton"}:
        assert services["triton"]["depends_on"]["prepare"]["condition"] == "service_completed_successfully"
        assert services["api"]["depends_on"]["triton"]["condition"] == "service_healthy"
    else:
        assert "triton" not in services
    if mode in {"cuda"}:
        assert services["api"]["deploy"]["resources"]["reservations"]["devices"][0]["capabilities"] == ["gpu"]
    else:
        assert "deploy" not in services["api"]

    # The thin root entry point selects the same definitions as an explicit
    # base/overlay merge, with all paths resolved from the project root.
    root_args = ["docker", "compose", "--env-file", str(env_file), "-f", "compose.yaml",
                 "--profile", mode, "--profile", "observability", "config", "--format", "json"]
    explicit_args = ["docker", "compose", "--env-file", str(env_file), "--project-directory", str(root),
                     "-f", "deploy/compose.yaml", "-f", f"deploy/compose/{mode}.yaml",
                     "--profile", mode, "--profile", "observability", "config", "--format", "json"]
    root_stack = json.loads(subprocess.check_output(root_args, cwd=root, env=environment, text=True))
    explicit_stack = json.loads(subprocess.check_output(explicit_args, cwd=root, env=environment, text=True))
    assert explicit_stack["services"] == root_stack["services"]
    assert explicit_stack["volumes"] == root_stack["volumes"]
    project_mount = next(mount for mount in services["api"]["volumes"] if mount["target"] == "/project")
    assert project_mount["source"] == str(PROJECT_ROOT / "projects" / "driver_behaviour")
    assert {"grafana", "prometheus"} <= services.keys()
    assert services["minio"]["image"] == "inferops-minio:console"
    assert services["minio"]["build"]["dockerfile"] == "deploy/docker/minio.Dockerfile"
    assert services["api"]["build"]["context"] == str(PROJECT_ROOT)
