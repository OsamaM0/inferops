"""Shared worker bootstrap."""
from __future__ import annotations

from inferops.alerts.store import autoload as autoload_alerts
from inferops.fusion.store import autoload as autoload_fusion
from inferops.pipeline.store import autoload as autoload_pipelines
from inferops.registry.store import autoload as autoload_models


async def autoload_all() -> None:
    await autoload_models()
    await autoload_pipelines()
    await autoload_fusion()
    await autoload_alerts()
