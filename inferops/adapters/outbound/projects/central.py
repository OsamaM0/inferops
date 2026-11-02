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
DOCUMENT_SECTIONS = ("models", "pipelines", "fusion", "alerts")
OVERRIDE_SECTIONS = {"project", *DOCUMENT_SECTIONS, "rules", "platform"}
RULES_FILE = "rules.yaml"
# Deployment values every workspace needs; inferops.yaml lists only changes.
DEFAULTS = {
    "media": {"record": True, "record_retention": "24h", "record_segment": "1m"},
    "docker": {
        "image": "inferops-engine:local",
        "triton_image": "nvcr.io/nvidia/tritonserver:26.09-py3",
        "media_bind_address": "127.0.0.1",
        "media_public_host": "127.0.0.1",
        "ports": {"pg": 5432, "s3": 9000, "minio_console": 9001, "kafka": 19092, "prometheus": 9090,
                  "grafana": 3000, "triton_http": 8000, "triton_grpc": 8001, "triton_metrics": 8002,
                  "api": 9997, "rtsp": 8554, "playback": 9996, "webrtc": 8889, "hls": 8888},
    },
}
_SETTING_RE = re.compile(r"\$\{settings\.([A-Za-z_][A-Za-z0-9_]*)\}")
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
                       "idle_seconds", "ort_threads", "ort_gpu_threads", "opencv_threads",
                       "cudnn_max_workspace", "data_path", "spool_path", "role", "worker_manifest"}
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
    for field in ("max_active_sources", "max_sessions", "ort_threads", "ort_gpu_threads", "opencv_threads"):
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
        if set(project) - {"root", "models", "settings", "overrides"}:
            raise ConfigError(f"unknown field in projects.{name}; use root, models, settings and overrides")
        if "models" in project and (not isinstance(project["models"], list) or not project["models"]
                                    or any(not isinstance(item, str) for item in project["models"])):
            raise ConfigError(f"projects.{name}.models must list model names")
        if not isinstance(project.get("settings", {}), dict) or not isinstance(project.get("overrides", {}), dict):
            raise ConfigError(f"projects.{name} settings/overrides must be mappings")
        if set(project.get("overrides", {})) - OVERRIDE_SECTIONS:
            raise ConfigError(f"unknown override section in projects.{name}: use {', '.join(sorted(OVERRIDE_SECTIONS))}")
        for section in DOCUMENT_SECTIONS:
            entries = project.get("overrides", {}).get(section, {})
            if not isinstance(entries, dict) or any(
                not re.fullmatch(r"[A-Za-z0-9_-]+", str(key)) or not isinstance(value, dict)
                for key, value in entries.items()
            ):
                raise ConfigError(f"projects.{name}.overrides.{section} must map document IDs to changes")


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


def _named(items):
    return isinstance(items, list) and all(isinstance(item, dict) and "name" in item for item in items)


def merge_patch(base, patch):
    """Apply a saved change set: mappings merge, `null` deletes a key, and lists of
    named items merge by name (`$remove: true` drops one). Other lists replace."""
    if isinstance(base, dict) and isinstance(patch, dict):
        result = copy.deepcopy(base)
        for key, value in patch.items():
            if value is None:
                result.pop(key, None)
            else:
                result[key] = merge_patch(result[key], value) if key in result else copy.deepcopy(value)
        return result
    if _named(base) and _named(patch):
        removed = {item["name"] for item in patch if item.get("$remove")}
        changes = {item["name"]: item for item in patch if not item.get("$remove")}
        current = {item["name"]: item for item in base if item["name"] not in removed}
        # A patch naming every remaining item also fixes their order.
        order = list(changes) if set(current) <= set(changes) else [*current, *(n for n in changes if n not in current)]
        return [merge_patch(current[name], changes[name]) if name in current and name in changes
                else copy.deepcopy(current.get(name) or changes[name]) for name in order]
    return copy.deepcopy(patch)


def diff_patch(base, new):
    """The smallest change set that turns `base` into `new` under merge_patch."""
    if isinstance(base, dict) and isinstance(new, dict):
        patch = {key: None for key in base if key not in new}
        for key, value in new.items():
            if key not in base:
                patch[key] = copy.deepcopy(value)
            elif base[key] != value:
                patch[key] = diff_patch(base[key], value)
        return patch
    if _named(base) and _named(new) and base != new:
        old = {item["name"]: item for item in base}
        names = [item["name"] for item in new]
        kept = [name for name in old if name in names]
        appended = names[len(kept):]
        removals = [{"name": name, "$remove": True} for name in old if name not in names]
        if names[:len(kept)] == kept and not set(appended) & set(old):
            changed = [{"name": item["name"], **diff_patch(old[item["name"]], item)} for item in new
                       if item["name"] in old and old[item["name"]] != item]
            return [*changed, *(copy.deepcopy(item) for item in new if item["name"] not in old), *removals]
        return [{"name": item["name"], **diff_patch(old[item["name"]], item)} if item["name"] in old
                else copy.deepcopy(item) for item in new] + removals
    return copy.deepcopy(new)


def apply_settings(value, settings):
    """Replace ${settings.<name>}; a value that is only a reference keeps its type."""
    if isinstance(value, dict):
        return {key: apply_settings(item, settings) for key, item in value.items()}
    if isinstance(value, list):
        return [apply_settings(item, settings) for item in value]
    if not isinstance(value, str) or "${settings." not in value:
        return value
    def lookup(name):
        if name not in settings:
            raise ConfigError(f"undeclared project setting {name!r}; declare it under settings in project.yaml")
        return settings[name]
    whole = _SETTING_RE.fullmatch(value)
    return copy.deepcopy(lookup(whole.group(1))) if whole else _SETTING_RE.sub(lambda m: str(lookup(m.group(1))), value)


def apply_model_settings(model, settings):
    """A model resolves its own declared settings, at their defaults when not selected."""
    own = {name: item.get("default") for name, item in (model.get("settings") or {}).items()
           if isinstance(item, dict)} if isinstance(model, dict) else {}
    return apply_settings(model, {**own, **settings})


def _manifest_file(root):
    return root / ("project.yaml" if (root / "project.yaml").is_file() else "domain.yaml")


def models_library(configs_dir):
    """The shared model library: INFEROPS_MODELS_DIR, else models/ beside configs/."""
    explicit = os.getenv("INFEROPS_MODELS_DIR")
    return Path(explicit).expanduser().resolve() if explicit else Path(configs_dir).resolve().parent / "models"


def models_dir(root, configs_dir):
    """A project's own models/ (or legacy configs/models/) wins, then the shared library.

    Without either, the project owns an (empty) models/ folder; nothing is inherited.
    """
    for candidate in (Path(root) / "models", Path(root) / "configs/models", models_library(configs_dir)):
        if candidate.is_dir():
            return candidate.resolve()
    return (Path(root) / "models").resolve()


def section_dir(root, section, configs_dir):
    return models_dir(root, configs_dir) if section == "models" else Path(root).resolve() / f"configs/{section}"


def _section_files(root, section, configs_dir):
    directory = section_dir(root, section, configs_dir)
    paths = sorted([*directory.glob("*.yaml"), *directory.glob("*/model.yaml")])
    return directory, {file.parent.name if file.name == "model.yaml" else file.stem: file for file in paths}


def with_model_settings(manifest, directory):
    """Settings declared by the selected models, under the project's own declarations."""
    selected = manifest.get("models")
    declared = {}
    for file in sorted([*Path(directory).glob("*.yaml"), *Path(directory).glob("*/model.yaml")]):
        model = _raw(file)
        if isinstance(model, dict) and (selected is None or model.get("name") in selected):
            declared.update(model.get("settings") or {})
    return {**manifest, "settings": {**declared, **(manifest.get("settings") or {})}} if declared else manifest


def setting_values(manifest, workspace_settings=None):
    """Validated project setting values: declared defaults plus workspace choices."""
    from inferops.adapters.outbound.projects.manifest import DomainManifest
    try:
        return DomainManifest(**manifest).setting_values(workspace_settings)
    except ValueError as error:
        raise ConfigError(f"project settings: {error}") from error


def _project(path, data, root):
    """(key, overrides, setting values, manifest document) for the configured project."""
    key = project_key(path, data, root)
    if key is None:
        return None
    selected = data["projects"][key]
    overrides = selected.get("overrides", {})
    manifest_file = _manifest_file(Path(root).resolve())
    manifest = _raw(manifest_file) or {"name": Path(root).name}
    if overrides.get("project") is not None:
        manifest = merge_patch(manifest, overrides["project"])
    if "models" in selected:
        manifest = {**manifest, "models": selected["models"]}
    declared = with_model_settings(manifest, models_dir(root, Path(path).parent))
    return key, overrides, setting_values(declared, selected.get("settings")), manifest


def _raw(file):
    return (yaml.safe_load(file.read_text()) or {}) if file.is_file() else {}


def documents(path, data, root, configs_dir):
    """Effective project documents (packaged defaults + saved changes), by file path."""
    result = {(Path(configs_dir) / "platform.yaml").resolve(): data.get("platform", {})}
    project = _project(path, data, root)
    if project is None:
        return result
    _key, overrides, settings, manifest = project
    root = Path(root).resolve()
    for section in DOCUMENT_SECTIONS:
        directory, base = _section_files(root, section, Path(path).parent)
        for name in set(base) | set(overrides.get(section, {})):
            file = base.get(name) or (directory / name / "model.yaml" if section == "models" else directory / f"{name}.yaml")
            patch = overrides.get(section, {}).get(name, {})
            # Changes apply before settings, so untouched values still follow them.
            document = merge_patch(_raw(file), patch)
            result[file.resolve()] = (apply_model_settings(document, settings) if section == "models"
                                      else apply_settings(document, settings))
    rules = root / RULES_FILE
    if rules.is_file() or overrides.get("rules") is not None:
        result[rules] = apply_settings(merge_patch(_raw(rules), overrides.get("rules", {})), settings)
    if overrides.get("project") is not None or "models" in data["projects"][_key]:
        result[_manifest_file(root)] = manifest
    if overrides.get("platform") is not None:
        platform = root / "configs/platform.yaml"
        result[platform] = merge_patch(_raw(platform), overrides["platform"])
    return result


def settings_path(path):
    """Editor address of the selected project's settings; not a file on disk."""
    return Path(path).resolve().parent / "project-settings.yaml"


def settings_document(path, data, root):
    """The project's selected models (when it composes from the library) and setting values."""
    project = _project(path, data, root)
    if project is None:
        return {}
    _key, _overrides, settings, manifest = project
    return {"models": manifest["models"], **settings} if manifest.get("models") is not None else settings


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
    if document == settings_path(path):
        return ("projects", key, "settings")
    if document == root / RULES_FILE:
        return ("projects", key, "overrides", "rules")
    prefix = ("projects", key, "overrides")
    if document in {root / "project.yaml", root / "domain.yaml"}:
        return (*prefix, "project")
    if document == root / "configs/platform.yaml":
        return (*prefix, "platform")
    for section in DOCUMENT_SECTIONS:
        directory = section_dir(root, section, Path(path).parent)
        if document.parent == directory or (section == "models" and document.name == "model.yaml" and document.parent.parent == directory):
            name = document.parent.name if document.name == "model.yaml" else document.stem
            return (*prefix, section, name)
    return None


def _store_settings(path, data, root, key, overrides, document):
    """Keep the model choice and setting values only where they differ from the package."""
    document = dict(document)
    models = document.pop("models", None)
    result = copy.deepcopy(data)
    project = result["projects"][key]
    packaged = merge_patch(_raw(_manifest_file(Path(root).resolve())), overrides.get("project", {}))
    if models is None or models == packaged.get("models"):
        project.pop("models", None)
    else:
        project["models"] = list(models)
    if models:
        packaged = {**packaged, "models": models}
    declared = with_model_settings(packaged, models_dir(root, Path(path).parent)).get("settings", {})
    defaults = {name: item.get("default") for name, item in declared.items()}
    # Settings of deselected models no longer apply and are dropped with them.
    changed = {name: value for name, value in document.items() if name in defaults and defaults[name] != value}
    if changed:
        project["settings"] = changed
    else:
        project.pop("settings", None)
    return result


def store_document(path, data, root, configs_dir, document_path, document):
    """Save an edited document; project documents keep only their changes."""
    pointer = document_pointer(path, data, root, configs_dir, document_path)
    if pointer is None or pointer[:1] != ("projects",):
        return replace_document(data, pointer, document) if pointer is not None else None
    document_path = Path(document_path).resolve()
    _key, overrides, settings, _manifest = _project(path, data, root)
    if pointer[-1] == "settings":
        return _store_settings(path, data, root, pointer[1], overrides, document)
    default = _raw(document_path)
    if pointer[3] == "models":  # Shown with settings applied.
        default = apply_model_settings(default, settings)
    elif pointer[3] not in {"project", "platform"}:
        default = apply_settings(default, settings)
    patch = diff_patch(default, document)
    result = copy.deepcopy(data)
    parent = result
    for name in pointer[:-1]:
        parent = parent.setdefault(name, {})
    if patch:
        parent[pointer[-1]] = patch
    else:
        parent.pop(pointer[-1], None)
    # Drop emptied containers so an undone edit leaves no trace in the file.
    for depth in range(len(pointer) - 1, 2, -1):
        container = result
        for name in pointer[:depth - 1]:
            container = container[name]
        if not container[pointer[depth - 1]]:
            container.pop(pointer[depth - 1])
    return result


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


def _with_defaults(base, data):
    result = dict(base)
    for key, value in data.items():
        result[key] = _with_defaults(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def environment(path, data=None, *, context="native"):
    data = _with_defaults(DEFAULTS, data or read_config(path))
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
                          "ort_gpu_threads": "ORT_GPU_THREADS", "opencv_threads": "OPENCV_THREADS",
                          "cudnn_max_workspace": "CUDNN_MAX_WORKSPACE",
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
