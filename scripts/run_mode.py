"""Run InferOps with one selected project block, natively or in Docker."""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
PID_FILE = APP / ".run/runtime.pid"


def stop_native_api(*, port: int | None = None) -> bool:
    """Stop only the registered API belonging to this workspace."""
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text())
        process = Path(f"/proc/{pid}")
        command = process.joinpath("cmdline").read_bytes().split(b"\0")
        owned = process.joinpath("cwd").resolve() == APP and b"inferops.api.main:app" in command
        if not owned:
            return False
        if port is not None and (b"--port" not in command or command[command.index(b"--port") + 1] != str(port).encode()):
            return False
        os.kill(pid, signal.SIGTERM)
        print(f"Stopped native InferOps API ({pid})")
        PID_FILE.unlink(missing_ok=True)
        return True
    except (ValueError, FileNotFoundError, ProcessLookupError):
        PID_FILE.unlink(missing_ok=True)
        return False


def ensure_container_port(port: int) -> None:
    """Hand off the native API port; diagnose unrelated listeners before startup."""
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", port)) != 0:
            return
    if stop_native_api(port=port):
        for _ in range(50):
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", port)) != 0:
                    return
            time.sleep(0.1)
    # An already running Compose API owns its port legitimately.
    output = subprocess.check_output(
        ["docker", "compose", "-f", str(APP / "compose.yaml"), "ps", "--format", "json", "api"], text=True,
    )
    entries = []
    for line in output.splitlines():
        if line.strip():
            entry = json.loads(line)
            entries.extend(entry if isinstance(entry, list) else [entry])
    if any(entry.get("State") == "running" and any(
        publisher.get("PublishedPort") == port for publisher in entry.get("Publishers") or []
    ) for entry in entries):
        return
    raise SystemExit(f"Port {port} is already in use by another process. Stop that process or choose API_PORT=<free port>.")


def runtime_environment(
    mode: str, port: int, *, domain: str | None = None, domain_root: str | None = None,
) -> dict[str, str]:
    """Resolve all project paths once; no project-specific names in the launcher."""
    sys.path.insert(0, str(APP))
    from inferops.core.config import ConfigStack, settings

    selected_root = domain_root or (None if domain else settings.domain_root)
    selected_name = domain or settings.active_domain
    stack = ConfigStack(APP / "configs", active_domain=selected_name, domain_root=selected_root)
    repository = stack.model_repository("model_repository").resolve()
    if stack.domain_root is None:
        repository = APP / "model_repository"
    env = dict(os.environ)
    env.update({
        "INFEROPS_EXECUTION_MODE": mode,
        "INFEROPS_CONFIGS_DIR": str(APP / "configs"),
        "INFEROPS_MODEL_REPOSITORY": str(repository),
        "INFEROPS_TRITON_MODEL_REPOSITORY": str(repository),
        "INFEROPS_TRANSPORT_KIND": "inproc",
        "INFEROPS_API_PORT": str(port),
    })
    # Relative plugins resolve against the selected folder's package.
    env.pop("INFEROPS_DOMAIN_PACKAGE", None)
    if stack.domain_root is not None:
        env["INFEROPS_ACTIVE_DOMAIN"] = stack.active_domain
        env["INFEROPS_DOMAIN_ROOT"] = str(stack.domain_root.resolve())
        env["INFEROPS_PROJECT_ROOT"] = str(stack.domain_root.resolve())
    else:
        for key in ("INFEROPS_ACTIVE_DOMAIN", "INFEROPS_DOMAIN_ROOT", "INFEROPS_PROJECT_ROOT"):
            env.pop(key, None)
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["auto", "cpu", "cuda", "triton", "deepstream"], default="cpu")
    parser.add_argument("--domain", "--project", help="Project block under projects/")
    parser.add_argument("--domain-root", help="Project block folder, including outside this repository")
    parser.add_argument("--container", action="store_true", help="Run the API in the reusable engine image")
    parser.add_argument("--build", action="store_true", help="Rebuild the engine image (not needed to swap projects)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-services", action="store_true", help="Use already running storage services")
    parser.add_argument("--stop", action="store_true", help="Stop this repository's native API")
    maintenance = parser.add_mutually_exclusive_group()
    maintenance.add_argument("--services-down", action="store_true", help="Stop Docker services; preserve volumes")
    maintenance.add_argument("--logs", action="store_true", help="Follow Docker service logs")
    maintenance.add_argument("--status", action="store_true", help="Show Docker service status")
    maintenance.add_argument("--compose-check", action="store_true", help="Validate Compose without starting services")
    args = parser.parse_args()
    if args.stop:
        stop_native_api()
        return
    env = runtime_environment(args.mode, args.port, domain=args.domain, domain_root=args.domain_root)
    if (args.container or args.mode == "deepstream") and "INFEROPS_TRANSPORT_KIND" not in os.environ:
        env.pop("INFEROPS_TRANSPORT_KIND", None)
    compose = ["docker", "compose", "-f", str(APP / "compose.yaml")]
    compose += ["--profile", args.mode]
    action = (["stop"] if args.services_down else ["logs", "-f", "--tail=200"] if args.logs else
              ["ps"] if args.status else ["config", "--quiet"] if args.compose_check else None)
    if action:
        subprocess.run([*compose, *action], env=env, cwd=APP, check=True)
        return
    if not args.container and args.mode != "deepstream":
        monitoring = APP / ".run/prometheus-native.yml"
        monitoring.parent.mkdir(parents=True, exist_ok=True)
        monitoring.write_text((APP / "deploy/observability/prometheus-native.yml").read_text().replace(
            "host.docker.internal:8080", f"host.docker.internal:{args.port}",
        ))
        env.setdefault("INFEROPS_PROMETHEUS_CONFIG", str(monitoring))
    # Container preparation belongs in Docker, so host Python needs no model dependencies.
    if not args.container and args.mode != "deepstream":
        if "INFEROPS_PROJECT_ROOT" in env:
            subprocess.run([sys.executable, "-m", "inferops.cli", "project", "prepare"], cwd=APP, env=env, check=True)
        subprocess.run([sys.executable, "-m", "inferops.cli", "preflight"], cwd=APP, env=env, check=True)
    if args.mode in {"triton", "deepstream"} or (args.container and args.mode == "cuda"):
        runtimes = subprocess.check_output(
            ["docker", "info", "--format", "{{range $name, $runtime := .Runtimes}}{{$name}} {{end}}"], text=True
        ).split()
        if "nvidia" not in runtimes:
            raise SystemExit("NVIDIA Docker runtime is missing. Run make setup-container-runtime in a terminal with sudo access.")
    if args.mode == "deepstream":
        if "INFEROPS_PROJECT_ROOT" not in env:
            raise SystemExit("DeepStream requires a project: pass --domain or --domain-root")
        ensure_container_port(args.port)
        command = [*compose, "up", "-d", "--wait"]
        if args.build:
            command.append("--build")
        subprocess.run([*command, "api", "prometheus", "grafana"], env=env, cwd=APP, check=True)
        print(f"DeepStream workspace: http://localhost:{args.port}/ui/")
        return
    if args.container:
        ensure_container_port(args.port)
        command = [*compose, "up", "-d", "--wait"]
        if args.build:
            command.append("--build")
        if args.no_services:
            command.append("--no-deps")
        subprocess.run([*command, "api", "prometheus", "grafana"], env=env, cwd=APP, check=True)
        print(f"{args.mode.upper()} Docker workspace: http://localhost:{args.port}/ui/")
        return
    if not args.no_services:
        services = ["postgres", "redis", "minio", "prometheus", "grafana"]
        if args.mode == "triton":
            services.append("triton")
        subprocess.run([*compose, "up", "-d", "--wait", *services], env=env, cwd=APP, check=True)
        if args.port == 8080:
            subprocess.run([*compose, "stop", "api"], env=env, cwd=APP, check=True)
    os.chdir(APP)
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))
    print(f"{args.mode.upper()} workspace: http://localhost:{args.port}/ui/", flush=True)
    os.execvpe(sys.executable, [sys.executable, "-m", "uvicorn", "inferops.api.main:app",
                              "--host", "0.0.0.0", "--port", str(args.port)], env)


if __name__ == "__main__":
    main()
