"""YAML env interpolation."""

from inferops.core.yaml_loader import _interp


def test_env_default(monkeypatch):
    monkeypatch.delenv("XYZ_NOT_SET", raising=False)
    assert _interp("${XYZ_NOT_SET:-fallback}") == "fallback"


def test_env_set(monkeypatch):
    monkeypatch.setenv("XYZ_TEST_VAR", "real")
    assert _interp({"a": "${XYZ_TEST_VAR}"}) == {"a": "real"}
