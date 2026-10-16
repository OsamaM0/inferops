"""Activate a compiled revision while existing sessions retain their resources."""
from __future__ import annotations


async def activate_revision(application, *, definition, models, stage_factory, annotators):
    previous = application.models
    application.definition = definition
    application.models = models
    application.stage_factory = stage_factory
    application.annotators = annotators
    if application.delivery is not None and hasattr(application.delivery, "definitions"):
        application.delivery.definitions = definition
    application.retired_models.append(previous)
    await application.collect_retired_models()
