"""Pure fusion algorithms and schema exports."""
from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    'FusionEngine': 'inferops.domain.fusion.engine',
    'FusionEvent': 'inferops.domain.fusion.signals',
    'Signal': 'inferops.domain.fusion.signals',
    'FusionSpec': 'inferops.domain.fusion.spec',
    'RuleSpec': 'inferops.domain.fusion.spec',
    'StateMachineSpec': 'inferops.domain.fusion.spec',
    'TransitionSpec': 'inferops.domain.fusion.spec',
    'WindowSpec': 'inferops.domain.fusion.spec',
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value
