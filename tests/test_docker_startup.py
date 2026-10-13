"""The documented .env selects the complete stack without manual -f flags."""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess

import pytest

from inferops.testing import PROJECT_ROOT


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
@pytest.mark.parametrize("mode", ["cpu", "auto", "cuda", "triton", "deepstream"])
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
    assert services["minio"]["environment"]["MINIO_ROOT_USER"] == services["api"]["environment"]["INFEROPS_S3_ACCESS_KEY"]
    assert services["minio"]["environment"]["MINIO_ROOT_PASSWORD"] == services["api"]["environment"]["INFEROPS_S3_SECRET_KEY"]
    assert "deepstream" not in services  # All modes retain one API service.
    assert services["prepare"]["command"] == ["inferops", "project", "prepare"]
    assert services["api"]["depends_on"]["prepare"]["condition"] == "service_completed_successfully"
    assert services["api"]["depends_on"]["minio-init"]["condition"] == "service_completed_successfully"
    assert services["postgres"]["volumes"][0]["type"] == "volume"
    if mode in {"triton", "deepstream"}:
        assert services["triton"]["depends_on"]["prepare"]["condition"] == "service_completed_successfully"
        assert services["api"]["depends_on"]["triton"]["condition"] == "service_healthy"
    else:
        assert "triton" not in services
    if mode in {"cuda", "deepstream"}:
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
