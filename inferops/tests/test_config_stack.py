"""Layered configuration semantics."""
from pathlib import Path

from inferops.core.config import ConfigStack, get_settings, resolve_domain_root


def test_config_stack_merges_platform_and_tenant(tmp_path: Path):
    cfg = tmp_path / "configs"
    (cfg / "tenants").mkdir(parents=True)
    (cfg / "platform.yaml").write_text(
        "transport:\n  kind: inproc\n  max_value_bytes: 10\nfeatures:\n  a: 1\n",
        encoding="utf-8",
    )
    (cfg / "tenants" / "t1.yaml").write_text(
        "features:\n  b: 2\ntransport:\n  max_value_bytes: 20\n",
        encoding="utf-8",
    )

    merged = ConfigStack(cfg).merged({"features": {"c": 3}}, tenant_id="t1")

    assert merged["transport"]["kind"] == "inproc"
    assert merged["transport"]["max_value_bytes"] == 20
    assert merged["features"] == {"a": 1, "c": 3, "b": 2}


def test_settings_platform_yaml_and_env_override(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "configs"
    cfg.mkdir()
    (cfg / "platform.yaml").write_text(
        "transport:\n  kind: inproc\n  max_value_bytes: 10\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("INFEROPS_CONFIGS_DIR", str(cfg))
    monkeypatch.setenv("INFEROPS_TRANSPORT_MAX_VALUE_BYTES", "99")
    get_settings.cache_clear()
    try:
        s = get_settings()
        assert s.transport.kind == "inproc"
        assert s.transport.max_value_bytes == 99
    finally:
        get_settings.cache_clear()


def test_config_stack_prefers_active_domain_configs_and_merges_platform(tmp_path: Path):
    repo = tmp_path / "repo"
    cfg = repo / "configs"
    domain = repo / "examples" / "driver_behaviour" / "configs"
    (cfg / "pipelines").mkdir(parents=True)
    domain.mkdir(parents=True)
    (domain / "pipelines").mkdir()

    (cfg / "platform.yaml").write_text(
        "transport:\n  kind: inproc\nfeatures:\n  core: true\n",
        encoding="utf-8",
    )
    (domain / "platform.yaml").write_text(
        "features:\n  domain: true\ntransport:\n  max_value_bytes: 20\n",
        encoding="utf-8",
    )

    stack = ConfigStack(cfg, active_domain="driver_behaviour")

    assert stack.domain_root == repo / "examples" / "driver_behaviour"
    assert stack.spec_dir("pipelines") == domain / "pipelines"
    assert stack.platform()["features"] == {"core": True, "domain": True}
    assert stack.platform()["transport"] == {"kind": "inproc", "max_value_bytes": 20}


def test_config_stack_without_active_domain_uses_core_configs(tmp_path: Path):
    cfg = tmp_path / "configs"
    (cfg / "pipelines").mkdir(parents=True)

    stack = ConfigStack(cfg)

    assert stack.domain_root is None
    assert stack.domain_configs_dir is None
    assert stack.spec_dir("pipelines") == cfg / "pipelines"


def test_resolve_domain_root_prefers_explicit_path(tmp_path: Path):
    cfg = tmp_path / "configs"
    explicit = tmp_path / "custom-domain"
    cfg.mkdir()
    explicit.mkdir()

    assert resolve_domain_root(cfg, active_domain="driver_behaviour", domain_root=explicit) == explicit
