"""Shared worker bootstrap."""
from __future__ import annotations

from inferops.alerts.store import autoload as autoload_alerts
from inferops.core.config import get_config_stack, settings
from inferops.core.preparation import selected_surface
from inferops.fusion.store import autoload as autoload_fusion
from inferops.pipeline.store import autoload as autoload_pipelines
from inferops.registry.store import autoload as autoload_models


async def autoload_all() -> None:
    selected_surface(get_config_stack(), settings.execution_mode)
    await autoload_models()
    await autoload_pipelines()
    await autoload_fusion()
    await autoload_alerts()
