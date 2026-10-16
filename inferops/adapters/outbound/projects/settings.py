"""Layered Pydantic settings."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import find_dotenv, load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env into os.environ so that _env_overrides_for() and nested settings
# classes (which read from os.environ directly) pick up local overrides.
_ENV_FILE = ""


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries without mutating either input."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _coerce_env(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        return value


def _env_overrides_for(prefix: str, cls: type[BaseSettings]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in cls.model_fields:
        env_name = f"{prefix}{field.upper()}"
        if env_name in os.environ:
            out[field] = _coerce_env(os.environ[env_name])
    return out


def resolve_domain_root(
    configs_dir: str | Path = "configs",
    *,
    active_domain: str | None = None,
    domain_root: str | Path | None = None,
) -> Path | None:
    if domain_root:
        return Path(domain_root)
    if not active_domain:
        return None

    from inferops.adapters.outbound.projects.manifest import project_directories

    candidates = [root / active_domain for root in project_directories(configs_dir)]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    from inferops.adapters.outbound.projects.packages import adapter_root, installed_adapters

    if active_domain in installed_adapters():
        return adapter_root(active_domain)
    return candidates[0]


class ConfigStack:
    """Merge platform, environment, model/pipeline, and tenant config layers."""

    def __init__(
        self,
        configs_dir: str | Path = "configs",
        *,
        active_domain: str | None = None,
        domain_root: str | Path | None = None,
    ) -> None:
        self.configs_dir = Path(configs_dir)
        self.domain_root = resolve_domain_root(
            self.configs_dir,
            active_domain=active_domain,
            domain_root=domain_root,
        )
        if self.domain_root is not None and not self.domain_root.is_dir():
            raise ValueError(f"project block does not exist: {self.domain_root}")
        self.active_domain = active_domain
        from inferops.adapters.outbound.projects.manifest import domain_package_name
        self.domain_package = domain_package_name(self.domain_root) if self.domain_root else None
        if self.domain_root is not None:
            self.active_domain = self.manifest().name

    @property
    def domain_configs_dir(self) -> Path | None:
        if self.domain_root is None:
            return None
        return self.domain_root / "configs"

    def manifest(self):
        from inferops.adapters.outbound.projects.manifest import load_domain_manifest

        return load_domain_manifest(self.domain_root)

    def platform(self) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        paths = [self.configs_dir / "platform.yaml"]
        if self.domain_configs_dir is not None:
            paths.append(self.domain_configs_dir / "platform.yaml")
        for path in paths:
            merged = deep_merge(merged, _load_yaml(path))
        manifest = self.manifest()
        if manifest is not None:
            merged = deep_merge(merged, manifest.platform)
        return merged

    def enabled_pipeline(self, name: str) -> bool:
        manifest = self.manifest()
        return manifest is None or manifest.pipelines is None or name in manifest.pipelines

    def spec_dir(self, section: str) -> Path:
        if self.domain_configs_dir is not None:
            ports = self.domain_root / "models"
            if section == "models" and ports.is_dir():
                return ports
            # A project owns its entire surface. An absent section means empty,
            # rather than inheriting another project's models or alert actions.
            return self.domain_configs_dir / section
        return self.configs_dir / section

    def model_repository(self, default_repository: str | Path = "model_repository") -> Path:
        repository = Path(default_repository)
        if self.domain_root is not None and not repository.is_absolute():
            return self.domain_root / repository
        return repository

    def tenant(self, tenant_id: str | None) -> dict[str, Any]:
        if not tenant_id:
            return {}
        return _load_yaml(self.configs_dir / "tenants" / f"{tenant_id}.yaml")

    def merged(self, *layers: dict[str, Any], tenant_id: str | None = None) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for layer in (self.platform(), *layers, self.tenant(tenant_id)):
            merged = deep_merge(merged, layer)
        return merged


class TransportSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFEROPS_TRANSPORT_", extra="ignore")
    kind: Literal["redpanda", "inproc"] = "redpanda"
    bootstrap: str = "localhost:19092"
    client_id: str = "inferops"
    max_value_bytes: int = 64 * 1024
    dlq_prefix: str = "inferops.dlq"


class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFEROPS_REDIS_", extra="ignore")
    url: str = "redis://localhost:6379/0"
    frame_ttl_s: int = 180


class S3Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFEROPS_S3_", extra="ignore")
    endpoint: str | None = "http://localhost:9000"
    # Public URL used to generate presigned URLs accessible from the browser.
    # When running behind a proxy, set this to the external-facing S3 URL,
    # e.g. https://mypod-9000.proxy.runpod.net
    public_url: str | None = None
    region: str = "us-east-1"
    access_key: str | None = "minio"
    secret_key: str | None = "minio12345"
    bucket: str = "inferops"


class PostgresSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFEROPS_PG_", extra="ignore")
    url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/inferops"


class TritonSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFEROPS_TRITON_", extra="ignore")
    url: str = "localhost:8001"
    http_url: str = "http://localhost:8000"


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFEROPS_API_", extra="ignore")
    host: str = "0.0.0.0"
    port: int = 8080
    cors_origins: str = "*"


class ObsSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFEROPS_OBS_", extra="ignore")
    log_level: str = "INFO"
    log_json: bool = True
    otel_endpoint: str | None = None
    prometheus_port: int = 9100
    service_name: str = "inferops"


class Settings(BaseSettings):
    """Top-level settings; sub-blocks loaded independently from env."""
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True,
    )

    env: Literal["dev", "staging", "prod"] = Field(default="dev", validation_alias="APP_ENV")
    configs_dir: str = Field(default="configs", validation_alias="INFEROPS_CONFIGS_DIR")
    active_domain: str | None = Field(default=None, validation_alias="INFEROPS_ACTIVE_DOMAIN")
    domain_root: str | None = Field(default=None, validation_alias="INFEROPS_DOMAIN_ROOT")
    model_repository: str = Field(default="model_repository", validation_alias="INFEROPS_MODEL_REPOSITORY")
    execution_mode: Literal["auto", "cpu", "cuda", "tensorrt", "triton", "deepstream"] = Field(
        default="cpu", validation_alias="INFEROPS_EXECUTION_MODE"
    )

    transport: TransportSettings = Field(default_factory=TransportSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    s3: S3Settings = Field(default_factory=S3Settings)
    pg: PostgresSettings = Field(default_factory=PostgresSettings)
    triton: TritonSettings = Field(default_factory=TritonSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    obs: ObsSettings = Field(default_factory=ObsSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def load_settings() -> Settings:
    global _ENV_FILE
    _ENV_FILE = find_dotenv()
    load_dotenv(_ENV_FILE, override=False)
    configs_dir = os.environ.get("INFEROPS_CONFIGS_DIR", "configs")
    active_domain = os.environ.get("INFEROPS_ACTIVE_DOMAIN")
    domain_root = os.environ.get("INFEROPS_DOMAIN_ROOT")
    if not domain_root and os.environ.get("INFEROPS_PROJECT_ROOT"):
        project_root = Path(os.environ["INFEROPS_PROJECT_ROOT"])
        if not project_root.is_absolute() and _ENV_FILE:
            project_root = Path(_ENV_FILE).parent / project_root
        domain_root = str(project_root.resolve())
    stack = ConfigStack(
        configs_dir,
        active_domain=active_domain,
        domain_root=domain_root,
    )
    overrides = stack.platform()
    if stack.active_domain:
        overrides["active_domain"] = stack.active_domain
    if stack.domain_root is not None:
        overrides["domain_root"] = str(stack.domain_root.resolve())

    top_env: dict[str, Any] = {}
    if "APP_ENV" in os.environ:
        top_env["env"] = os.environ["APP_ENV"]
    if "INFEROPS_CONFIGS_DIR" in os.environ:
        top_env["configs_dir"] = os.environ["INFEROPS_CONFIGS_DIR"]
    if "INFEROPS_DOMAIN_ROOT" in os.environ:
        top_env["domain_root"] = os.environ["INFEROPS_DOMAIN_ROOT"]
    if "INFEROPS_MODEL_REPOSITORY" in os.environ:
        top_env["model_repository"] = os.environ["INFEROPS_MODEL_REPOSITORY"]
    if "INFEROPS_EXECUTION_MODE" in os.environ:
        top_env["execution_mode"] = os.environ["INFEROPS_EXECUTION_MODE"]
    overrides = deep_merge(overrides, top_env)

    nested = {
        "transport": ("INFEROPS_TRANSPORT_", TransportSettings),
        "redis": ("INFEROPS_REDIS_", RedisSettings),
        "s3": ("INFEROPS_S3_", S3Settings),
        "pg": ("INFEROPS_PG_", PostgresSettings),
        "triton": ("INFEROPS_TRITON_", TritonSettings),
        "api": ("INFEROPS_API_", ApiSettings),
        "obs": ("INFEROPS_OBS_", ObsSettings),
    }
    for section, (prefix, cls) in nested.items():
        env_section = _env_overrides_for(prefix, cls)
        if env_section:
            overrides[section] = deep_merge(overrides.get(section, {}), env_section)
    return Settings(**overrides)


class _LegacySettings:
    """Lazy compatibility proxy; new application services use explicit settings."""

    def __getattr__(self, name):
        return getattr(get_settings(), name)

    def __setattr__(self, name, value):
        setattr(get_settings(), name, value)


settings = _LegacySettings()


def get_config_stack(current_settings: Settings | None = None) -> ConfigStack:
    current_settings = current_settings or settings
    return ConfigStack(
        current_settings.configs_dir,
        active_domain=current_settings.active_domain,
        domain_root=current_settings.domain_root,
    )


def get_spec_dir(section: str, current_settings: Settings | None = None) -> Path:
    return get_config_stack(current_settings).spec_dir(section)


def get_model_repository_root(current_settings: Settings | None = None) -> Path:
    current_settings = current_settings or settings
    return get_config_stack(current_settings).model_repository(current_settings.model_repository)
