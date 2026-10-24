"""Exercise the Docker-only launcher without touching real services or user settings."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launcher(tmp_path):
    workspace = tmp_path / "workspace with spaces"
    workspace.mkdir()
    shutil.copy2(ROOT / "run.sh", workspace / "run.sh")
    (workspace / ".env.example").write_text("# Credentials only\n")
    (workspace / "configs").mkdir()
    (workspace / "configs/inferops.yaml").write_text("runtime:\n  mode: cpu\n  api_port: 55981\n")
    binary = tmp_path / "bin"
    binary.mkdir()
    docker = binary / "docker"
    docker.write_text(f"#!{sys.executable}\n" + '''
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
with Path(os.environ["DOCKER_CALLS"]).open("a") as log:
    log.write(json.dumps(args) + "\\n")
if args[0] == "info" and os.getenv("DOCKER_UNAVAILABLE"):
    sys.exit(1)
elif args[0] == "inspect":
    print(os.getenv("API_NETWORKS", "1"))
elif args[0] == "ps":
    print(os.getenv("PORT_OWNER", ""))
elif args[0] == "run":
    # Simulate only the configuration helper, without an image or daemon.
    import yaml
    values = yaml.safe_load(Path("configs/inferops.yaml").read_text())["runtime"]
    Path(".run").mkdir(exist_ok=True)
    Path(".run/compose.env").write_text(f"INFEROPS_EXECUTION_MODE={values['mode']}\\nINFEROPS_API_PORT={values['api_port']}\\n")
elif args[0] == "compose":
    args = args[7:]  # project directory, root file, generated env file
    if args == ["config", "--environment"]:
        values = dict(line.split("=", 1) for line in Path(".run/compose.env").read_text().splitlines() if "=" in line)
        values.update({key: value for key, value in os.environ.items() if key.startswith("INFEROPS_")})
        print("\\n".join(f"{key}={value}" for key, value in values.items()))
    elif args == ["ps", "-a", "-q", "api"]:
        print(os.getenv("API_ID", ""))
    elif args and args[0] == "up" and os.getenv("STARTUP_FAILS"):
        print("simulated readiness failure", file=sys.stderr)
        sys.exit(1)
''')
    docker.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if not key.startswith(("INFEROPS_", "COMPOSE_"))}
    env.update(PATH=f"{binary}:{env['PATH']}", DOCKER_CALLS=str(tmp_path / "calls.jsonl"))

    def run(*arguments, **overrides):
        return subprocess.run(["bash", str(workspace / "run.sh"), *arguments], cwd=tmp_path,
                              env={**env, **overrides}, capture_output=True, text=True, timeout=10)

    def calls():
        path = Path(env["DOCKER_CALLS"])
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    return workspace, run, calls


def compose_actions(calls):
    return [call[7:] for call in calls if call[:2] == ["compose", "--project-directory"]]


def test_first_start_creates_settings_and_waits_for_readiness(launcher):
    workspace, run, calls = launcher
    result = run()
    assert result.returncode == 0, result.stderr
    assert (workspace / ".env").read_bytes() == (workspace / ".env.example").read_bytes()
    assert ["up", "-d", "--build", "--wait", "--wait-timeout", "600"] in compose_actions(calls())
    assert "http://localhost:55981/ui/" in result.stdout


def test_existing_settings_are_preserved_and_never_executed(launcher):
    workspace, run, _ = launcher
    marker = workspace / "unexpected-execution"
    settings = f"INFEROPS_EXECUTION_MODE=cpu\nINFEROPS_API_PORT=55981\nLABEL=$(touch '{marker}')\n"
    (workspace / ".env").write_text(settings)
    result = run("--mode", "auto", "--port", "55982", "--no-build")
    assert result.returncode == 0, result.stderr
    assert (workspace / ".env").read_text() == settings
    assert not marker.exists()
    assert "runtime=auto, API port=55982" in result.stdout


def test_detached_api_is_recreated_without_removing_other_services_or_volumes(launcher):
    _, run, calls = launcher
    result = run(API_ID="a" * 64, API_NETWORKS="0")
    assert result.returncode == 0, result.stderr
    actions = compose_actions(calls())
    assert ["stop", "-t", "60", "api"] in actions
    assert ["rm", "-f", "api"] in actions
    assert actions.index(["rm", "-f", "api"]) < next(i for i, action in enumerate(actions) if action[0] == "up")
    assert not any("down" in action or "-v" in action for action in actions)


def test_conflicting_port_names_owner_and_does_not_start_or_stop_services(launcher):
    _, run, calls = launcher
    result = run(PORT_OWNER="bbbbbbbbbbbb another-project 127.0.0.1:55981->8080/tcp")
    assert result.returncode == 1
    assert "another-project" in result.stderr and "--port" in result.stderr
    assert not any(action[0] in {"up", "stop", "rm"} for action in compose_actions(calls()))


def test_existing_api_can_keep_its_own_port(launcher):
    _, run, calls = launcher
    result = run(API_ID="a" * 64, PORT_OWNER=f"{'a' * 12} inferops-api-1 0.0.0.0:55981->8080/tcp")
    assert result.returncode == 0, result.stderr
    assert not any(action[0] == "rm" for action in compose_actions(calls()))


def test_startup_failure_prints_diagnostics_without_claiming_readiness(launcher):
    _, run, calls = launcher
    result = run(STARTUP_FAILS="1")
    assert result.returncode == 1
    assert "Startup failed" in result.stderr
    assert "InferOps is ready" not in result.stdout
    assert ["logs", "--tail=60", "api", "prepare", "postgres"] in compose_actions(calls())


def test_check_reports_detached_network_without_mutating_containers(launcher):
    _, run, calls = launcher
    result = run("check", API_ID="a" * 64, API_NETWORKS="0")
    assert result.returncode == 1 and "repair" in result.stderr
    assert not any(action[0] in {"up", "stop", "rm"} for action in compose_actions(calls()))


def test_repair_recreates_api_even_when_attached(launcher):
    _, run, calls = launcher
    result = run("repair", "--no-build", API_ID="a" * 64)
    assert result.returncode == 0, result.stderr
    assert ["rm", "-f", "api"] in compose_actions(calls())
    assert ["up", "-d", "--wait", "--wait-timeout", "600"] in compose_actions(calls())


@pytest.mark.parametrize("arguments", [("--port", "0"), ("--port", "65536"), ("--mode", "invalid"), ("--port",)])
def test_invalid_options_fail_before_docker_changes(launcher, arguments):
    _, run, calls = launcher
    assert run(*arguments).returncode == 1
    assert calls() == []


def test_docker_access_failure_is_actionable(launcher):
    _, run, calls = launcher
    result = run(DOCKER_UNAVAILABLE="1")
    assert result.returncode == 1 and "docker info" in result.stderr
    assert not compose_actions(calls())


def test_help_does_not_require_docker_or_create_env(launcher):
    workspace, run, calls = launcher
    assert run("--help").returncode == 0
    assert not (workspace / ".env").exists()
    assert calls() == []
