from types import SimpleNamespace

import pytest

from inferops.application.projects import validate_server_revision
from inferops.domain.errors import ConfigError
from inferops.domain.incident_spec import AlertSpec
from inferops.domain.pipeline import PipelineSpec


def test_explicit_admission_environment_overrides_legacy_limits(monkeypatch):
    from inferops.adapters.outbound.projects.settings import load_settings

    monkeypatch.setenv("INFEROPS_MODEL_CONCURRENCY", "2")
    monkeypatch.setenv("INFEROPS_TRITON_CONCURRENCY", "12")
    monkeypatch.setenv("INFEROPS_ADMISSION_LOCAL_DEVICE_CAPACITY", "3")
    monkeypatch.setenv("INFEROPS_ADMISSION_REMOTE_OUTSTANDING", "24")
    admission = load_settings().admission
    assert admission.local_device_capacity == 3
    assert admission.remote_outstanding == 24


def test_old_rest_sink_migrates_only_the_supported_url_contract():
    with pytest.warns(FutureWarning, match="sink_rest"):
        pipeline = PipelineSpec(name="test", sinks=[{"name": "hook", "stage": "sink_rest", "params": {"url": "https://example.test"}}])
    assert pipeline.sinks[0].stage == "sink_webhook"
    with pytest.raises(ValueError, match="migration"):
        PipelineSpec(name="test", sinks=[{"name": "hook", "stage": "sink_rest", "params": {"path": "/run"}}])


def test_durable_server_configuration_cannot_disable_persistence_or_silently_discard_kafka():
    application = SimpleNamespace(require_durable_configuration=True, broker_enabled=False)
    spec = AlertSpec(name="test", severity={"levels": [{"name": "low", "rank": 1}]}, persistence_enabled=False)
    definition = SimpleNamespace(alerts={"test": spec}, fusion={}, pipelines={})
    with pytest.raises(ConfigError, match="persistence"):
        validate_server_revision(application, definition)
    spec.persistence_enabled = True
    definition.pipelines["test"] = PipelineSpec(name="test", sinks=[{"name": "kafka", "stage": "sink_kafka", "params": {"topic": "required"}}])
    with pytest.raises(ConfigError, match="broker"):
        validate_server_revision(application, definition)
    application.broker_enabled = True
    validate_server_revision(application, definition)
