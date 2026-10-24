"""The workspace's single editable configuration; extension YAMLs are packaged defaults.

This module uses only the standard library and PyYAML so the Docker-only launcher
can read configuration without installing inference dependencies on the host.
"""
from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from urllib.parse import quote

import yaml

from inferops.domain.errors import ConfigError

FILENAME = "inferops.yaml"
SECTIONS = {"runtime", "platform", "sources", "evidence", "media", "storage", "docker", "projects", "workers"}
_exported = {}


def find_config(configs_dir=None):
    explicit = os.getenv("INFEROPS_CONFIG_FILE")
    path = Path(explicit) if explicit else Path(configs_dir or os.getenv("INFEROPS_CONFIGS_DIR", "configs")) / FILENAME
    if explicit and not path.is_file():
        raise ConfigError(f"central configuration not found: {path}")
    return path.resolve() if path.is_file() else None


def read_config(path):
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError("Cannot read the central configuration") from error
    validate_config(data)
    return data


def validate_config(data):
    if not isinstance(data, dict) or set(data) - SECTIONS:
        raise ConfigError(f"central configuration sections: {', '.join(sorted(SECTIONS))}")
    for key, value in data.items():
        if not isinstance(value, dict):
            raise ConfigError(f"{key} must be a mapping")
    runtime = data.get("runtime", {})
    allowed_runtime = {"project", "mode", "api_port", "services", "max_active_sources", "max_sessions",
                       "idle_seconds", "ort_threads", "opencv_threads", "cudnn_max_workspace",
                       "data_path", "spool_path", "role", "worker_manifest"}
    if set(runtime) - allowed_runtime:
        raise ConfigError(f"unknown runtime settings: {sorted(set(runtime) - allowed_runtime)}")
    if runtime.get("mode", "cpu") not in {"cpu", "auto", "cuda", "triton", "tensorrt"}:
        raise ConfigError("runtime.mode must be cpu, cuda, triton or tensorrt (auto is CPU)")
    port = runtime.get("api_port", 8080)
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        raise ConfigError("runtime.api_port must be between 1 and 65535")
    services = runtime.get("services", [])
    if not isinstance(services, list) or any(not isinstance(item, str) or item not in {"storage", "broker", "monitoring"} for item in services):
        raise ConfigError("runtime.services may contain storage, broker and monitoring")
    for field in ("max_active_sources", "max_sessions", "ort_threads", "opencv_threads"):
        if field in runtime and (not isinstance(runtime[field], int) or isinstance(runtime[field], bool) or runtime[field] < 1):
            raise ConfigError(f"runtime.{field} must be a positive integer")
    if data.get("storage", {}).get("kind", "filesystem") not in {"filesystem", "s3"}:
        raise ConfigError("storage.kind must be filesystem or s3")
    if data.get("media", {}).get("router", "mediamtx") not in {"direct", "mediamtx"}:
        raise ConfigError("media.router must be direct or mediamtx")
    ports = data.get("docker", {}).get("ports", {})
    if not isinstance(ports, dict) or any(not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535
                                          for port in ports.values()):
        raise ConfigError("docker.ports must map port names to numbers between 1 and 65535")
    selected = runtime.get("project")
    projects = data.get("projects", {})
    if selected and selected not in projects:
        raise ConfigError("runtime.project must name an entry in projects")
    for name, project in projects.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", str(name)):
            raise ConfigError("project names must be identifiers")
        if not isinstance(project, dict) or not isinstance(project.get("root"), str):
            raise ConfigError(f"projects.{name} requires a root directory")
        if set(project) - {"root", "parameters", "configuration"}:
            raise ConfigError(f"unknown project configuration field in projects.{name}")
        if not isinstance(project.get("parameters", {}), dict) or not isinstance(project.get("configuration", {}), dict):
            raise ConfigError(f"projects.{name} parameters/configuration must be mappings")
        if set(project.get("configuration", {})) - {"project", "models", "pipelines", "fusion", "alerts", "platform"}:
            raise ConfigError(f"unknown configuration section in projects.{name}")
        for section in ("models", "pipelines", "fusion", "alerts"):
            entries = project.get("configuration", {}).get(section, {})
            if not isinstance(entries, dict) or any(
                not re.fullmatch(r"[A-Za-z0-9_-]+", str(key)) or not isinstance(value, dict)
                for key, value in entries.items()
            ):
                raise ConfigError(f"projects.{name}.configuration.{section} must map identifiers to documents")


def project_root(path, data, key=None):
    key = key or data.get("runtime", {}).get("project")
    if not key:
        return None
    root = Path(data["projects"][key]["root"]).expanduser()
    return root.resolve() if root.is_absolute() else (Path(path).parent.parent / root).resolve()


def project_key(path, data, root):
    if root is None:
        return None
    mounted = os.getenv("INFEROPS_CONFIG_PROJECT")
    if mounted in data.get("projects", {}) and Path(root).resolve() == Path("/project"):
        return mounted
    return next((key for key in data.get("projects", {}) if project_root(path, data, key) == Path(root).resolve()), None)


def _parameters(value, parameters):
    if isinstance(value, dict):
        return {key: copy.deepcopy(parameters[key]) if key in parameters else _parameters(item, parameters)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_parameters(item, parameters) for item in value]
    return value


def documents(path, data, root, configs_dir):
    """Project defaults plus workspace overrides, addressed by legacy virtual paths."""
    result = {(Path(configs_dir) / "platform.yaml").resolve(): data.get("platform", {})}
    key = project_key(path, data, root)
    if key is None:
        return result
    selected = data["projects"][key]
    parameters = selected.get("parameters", {})
    patches = selected.get("configuration", {})
    root = Path(root).resolve()
    for section in ("models", "pipelines", "fusion", "alerts"):
        directory = root / ("models" if section == "models" else f"configs/{section}")
        paths = sorted([*directory.glob("*.yaml"), *directory.glob("*/model.yaml")])
        base = {file.parent.name if file.name == "model.yaml" else file.stem: file for file in paths}
        for name in set(base) | set(patches.get(section, {})):
            file = base.get(name) or (directory / name / "model.yaml" if section == "models" else directory / f"{name}.yaml")
            default = yaml.safe_load(file.read_text()) if file.is_file() else {}
            default = _parameters(default or {}, parameters)
            # A saved document is complete: removing a stage/key in the editor
            # must also remove it at runtime. Parameters customize packaged defaults.
            result[file.resolve()] = copy.deepcopy(patches.get(section, {}).get(name, default))
    manifest = root / ("project.yaml" if (root / "project.yaml").is_file() else "domain.yaml")
    if patches.get("project") is not None:
        result[manifest] = copy.deepcopy(patches["project"])
    if patches.get("platform") is not None:
        result[root / "configs/platform.yaml"] = patches["platform"]
    return result


def document_pointer(path, data, root, configs_dir, document):
    document = Path(document).resolve()
    if document == Path(path).resolve():
        return ()
    if document == (Path(configs_dir) / "platform.yaml").resolve():
        return ("platform",)
    key = project_key(path, data, root)
    if key is None:
        return None
    root = Path(root).resolve()
    prefix = ("projects", key, "configuration")
    if document in {root / "project.yaml", root / "domain.yaml"}:
        return (*prefix, "project")
    if document == root / "configs/platform.yaml":
        return (*prefix, "platform")
    for section in ("models", "pipelines", "fusion", "alerts"):
        directory = root / ("models" if section == "models" else f"configs/{section}")
        if document.parent == directory or (section == "models" and document.name == "model.yaml" and document.parent.parent == directory):
            name = document.parent.name if document.name == "model.yaml" else document.stem
            return (*prefix, section, name)
    return None


def replace_document(data, pointer, document):
    if not pointer:
        return copy.deepcopy(document)
    result = copy.deepcopy(data)
    parent = result
    for name in pointer[:-1]:
        parent = parent.setdefault(name, {})
    parent[pointer[-1]] = copy.deepcopy(document)
    return result


def _string(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def environment(path, data=None, *, context="native"):
    data = data or read_config(path)
    result = {}
    runtime = data.get("runtime", {})
    result["INFEROPS_EXECUTION_MODE"] = runtime.get("mode", "cpu")
    result["INFEROPS_API_PORT"] = str(runtime.get("api_port", 8080))
    result["INFEROPS_MONITORING_ENABLED"] = _string("monitoring" in runtime.get("services", []))
    selected = runtime.get("project")
    if selected:
        result["INFEROPS_PROJECT_ROOT"] = str(project_root(path, data))
        result["INFEROPS_CONFIG_PROJECT"] = selected
    for section, prefix in (("evidence", "INFEROPS_EVIDENCE_"),):
        for key, value in data.get(section, {}).items():
            result[prefix + key.upper()] = _string(value)
    if data.get("sources"):
        result["INFEROPS_SOURCE_DEFAULTS"] = json.dumps(data["sources"], separators=(",", ":"))
    for section, prefix in (("transport", "TRANSPORT"), ("pg", "PG"), ("s3", "S3"),
                            ("triton", "TRITON"), ("api", "API"), ("obs", "OBS")):
        for key, value in data.get("platform", {}).get(section, {}).items():
            result[f"INFEROPS_{prefix}_{key.upper()}"] = _string(value)
    admission = data.get("platform", {}).get("runtime", {}).get("admission", {})
    for key, value in admission.items():
        result[f"INFEROPS_ADMISSION_{key.upper()}"] = _string(value)
    if "local_device_capacity" in admission:
        result["INFEROPS_MODEL_CONCURRENCY"] = str(admission["local_device_capacity"])
    if "remote_outstanding" in admission:
        result["INFEROPS_TRITON_CONCURRENCY"] = str(admission["remote_outstanding"])
    for key, variable in {"max_sessions": "MAX_SESSIONS", "max_active_sources": "MAX_ACTIVE_SOURCES",
                          "idle_seconds": "SESSION_IDLE_SECONDS", "ort_threads": "ORT_THREADS",
                          "opencv_threads": "OPENCV_THREADS", "cudnn_max_workspace": "CUDNN_MAX_WORKSPACE",
                          "data_path": "DATA_PATH", "spool_path": "SPOOL_PATH",
                          "role": "ROLE", "worker_manifest": "WORKER_MANIFEST"}.items():
        if key in runtime:
            result[f"INFEROPS_{variable}"] = _string(runtime[key])
    result["INFEROPS_STORAGE_KIND"] = data.get("storage", {}).get("kind", "filesystem")
    media = data.get("media", {})
    result["INFEROPS_MEDIA_ROUTER"] = media.get("router", "mediamtx")
    for key, value in media.items():
        if key != "router":
            result[f"INFEROPS_MEDIAMTX_{key.upper()}"] = _string(value)
    docker = data.get("docker", {})
    for key, value in docker.get("ports", {}).items():
        prefix = "MEDIAMTX_" if key in {"api", "rtsp", "playback", "webrtc", "hls"} else ""
        result[f"INFEROPS_{prefix}{key.upper()}_PORT"] = str(value)
    if context == "native":
        ports = docker.get("ports", {})
        user = quote(os.getenv("INFEROPS_PG_USER", "postgres"), safe="")
        password = quote(os.getenv("INFEROPS_PG_PASSWORD", "postgres"), safe="")
        database = quote(os.getenv("INFEROPS_PG_DATABASE", "inferops"), safe="")
        endpoints = {
            "INFEROPS_PG_URL": f"postgresql+asyncpg://{user}:{password}@127.0.0.1:{ports.get('pg', 5432)}/{database}",
            "INFEROPS_S3_ENDPOINT": f"http://127.0.0.1:{ports.get('s3', 9000)}",
            "INFEROPS_TRANSPORT_BOOTSTRAP": f"127.0.0.1:{ports.get('kafka', 19092)}",
            "INFEROPS_TRITON_URL": f"127.0.0.1:{ports.get('triton_grpc', 8001)}",
            "INFEROPS_TRITON_HTTP_URL": f"http://127.0.0.1:{ports.get('triton_http', 8000)}",
            "INFEROPS_TRITON_METRICS_URL": f"http://127.0.0.1:{ports.get('triton_metrics', 8002)}",
            "INFEROPS_MEDIAMTX_API_URL": f"http://127.0.0.1:{ports.get('api', 9997)}",
            "INFEROPS_MEDIAMTX_RTSP_URL": f"rtsp://127.0.0.1:{ports.get('rtsp', 8554)}",
            "INFEROPS_MEDIAMTX_PLAYBACK_URL": f"http://127.0.0.1:{ports.get('playback', 9996)}",
        }
        for key, value in endpoints.items():
            result.setdefault(key, value)
    # An optional integration can expose additional adapter settings here without
    # adding another operator configuration file or another environment template.
    for key, value in data.get("platform", {}).get("environment", {}).items():
        if not re.fullmatch(r"INFEROPS_[A-Z0-9_]+", key):
            raise ConfigError("platform.environment keys must start with INFEROPS_")
        result[key] = _string(value)
    if context == "docker":
        if runtime.get("mode") == "tensorrt":
            raise ConfigError("Docker runtime.mode must use cpu, cuda or triton")
        mode = runtime.get("mode", "cpu")
        result["COMPOSE_PROFILES"] = ",".join([mode, *runtime.get("services", [])])
        for key in ("image", "triton_image"):
            if docker.get(key):
                result[f"INFEROPS_{key.upper()}"] = str(docker[key])
        if docker.get("media_bind_address"):
            result["INFEROPS_MEDIAMTX_BIND_ADDRESS"] = docker["media_bind_address"]
        if docker.get("media_public_host"):
            result["INFEROPS_MEDIAMTX_PUBLIC_HOST"] = docker["media_public_host"]
    result["INFEROPS_CONFIG_CONTEXT"] = context
    if data.get("workers"):
        result["INFEROPS_WORKER_MANIFEST"] = "/app/configs/inferops.yaml" if context == "docker" else str(path)
        result.setdefault("INFEROPS_ROLE", "api")
    return {key: str(value) for key, value in result.items()}


def apply_environment(path, data=None):
    """Export workspace defaults at server construction, retaining explicit host overrides."""
    global _exported
    inherited = json.loads(os.getenv("INFEROPS_CONFIG_VALUES", "{}"))
    owned = {**inherited, **_exported}
    values = environment(path, data, context=os.getenv("INFEROPS_CONFIG_CONTEXT", "native"))
    for key, old in owned.items():
        if key not in values and os.environ.get(key) == old:
            os.environ.pop(key, None)
            _exported.pop(key, None)
    for key, value in values.items():
        if key not in os.environ or (key in owned and os.environ[key] == owned[key]):
            os.environ[key] = value
            _exported[key] = value
    os.environ["INFEROPS_CONFIG_VALUES"] = json.dumps(
        {key: value for key, value in {**inherited, **_exported}.items()
         if key in values and os.getenv(key) == value}, separators=(",", ":"))
    return values


def environment_override(key):
    """Distinguish real host overrides from values generated from the YAML."""
    value = os.getenv(key)
    owned = {**json.loads(os.getenv("INFEROPS_CONFIG_VALUES", "{}")), **_exported}
    return value if value is not None and owned.get(key) != value else None
