"""Shared dependencies."""
from __future__ import annotations

from inferops.alerts import get_alert_store
from inferops.fusion import get_fusion_store
from inferops.pipeline import get_pipeline_store
from inferops.registry import get_registry


def registry(): return get_registry()
def pipelines(): return get_pipeline_store()
def fusion(): return get_fusion_store()
def alerts(): return get_alert_store()
