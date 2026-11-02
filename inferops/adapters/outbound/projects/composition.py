"""Compose a project's one pipeline, fusion and alerts from its models and rules.yaml.

A model in the shared library declares how it runs (`stage`) and which fusion
signals it produces (`signals`). A project selects any combination of models;
rules that need a signal of an unselected model are switched off, and states
see that signal as absent (an empty window).
"""
from __future__ import annotations

import ast
from typing import Any

from inferops.domain.errors import ConfigError

RULES_FILE = "rules.yaml"
# Model fields used for composition, not by the inference backend.
COMPOSITION_FIELDS = frozenset({"settings", "signals", "stage"})


def model_signals(document: dict[str, Any]) -> dict[str, str]:
    """`signals` as a list reads each name from the item payload."""
    signals = document.get("signals") or {}
    return {name: f"signal('{name}')" for name in signals} if isinstance(signals, list) else dict(signals)


def _names(expression: str) -> set[str]:
    try:
        return {node.id for node in ast.walk(ast.parse(str(expression), mode="eval")) if isinstance(node, ast.Name)}
    except SyntaxError:
        return set()  # Reported by fusion validation.


def _order(selected: list[str], models: dict[str, dict]) -> list[str]:
    """Selection order, except a model runs after the selected models it names in `after`."""
    ordered: list[str] = []
    pending = list(selected)
    while pending:
        ready = next((name for name in pending if all(
            dependency in ordered or dependency not in selected
            for dependency in (models[name].get("stage") or {}).get("after", []))), None)
        if ready is None:
            raise ConfigError(f"models have a circular `after` order: {pending}")
        ordered.append(ready)
        pending.remove(ready)
    return ordered


def compose(name: str, manifest: dict[str, Any], models: dict[str, dict], rules: dict[str, Any]) -> dict[str, Any]:
    """Pipeline, fusion and alert documents plus the rules switched off by model choice.

    `models` holds every library model document by name; `manifest["models"]`
    selects the ones to run (all when omitted).
    """
    selected = list(manifest.get("models") or models)
    unknown = [model for model in selected if model not in models]
    if unknown:
        raise ConfigError(f"project {name!r} selects unknown models {unknown}; available: {sorted(models)}")
    helpers = {model for owner in selected for model in ((models[owner].get("stage") or {}).get("uses") or {}).values()
               if model in selected}
    stages: list[dict[str, Any]] = [{"name": "decode", "stage": "decode_any"}]
    tracked = []
    for model in _order(selected, models):
        stage = models[model].get("stage")
        if stage:
            uses = {key: value for key, value in (stage.get("uses") or {}).items() if value in selected}
            stages.append({"name": model, "stage": stage["plugin"],
                           "params": {"model": model, **uses, **(stage.get("params") or {})}})
        elif model not in helpers:
            stages.append({"name": model, "stage": "infer", "params": {"model": model}})
            tracked.append(model)
    stages += [{"name": "fuse", "stage": "fuse", "params": {"fusion": name}},
               {"name": "alert", "stage": "alert", "params": {"alerts": name}}]
    pipeline: dict[str, Any] = {
        "name": name, "source": {"name": "src", "stage": "source_rest"}, "stages": stages,
        "sinks": list(manifest.get("sinks") or []),
        "tags": {"label": manifest.get("label") or name.replace("_", " ").title(),
                 "description": f"Models: {', '.join(selected)}."},
    }
    if manifest.get("tracking"):
        pipeline["tracking"] = {"model_filter": tracked, **manifest["tracking"]}

    extract = {signal: expression for model in selected for signal, expression in model_signals(models[model]).items()}
    extract.update(model_signals(rules))
    absent = {signal: model for model in models if model not in selected
              for signal in model_signals(models[model]) if signal not in extract}
    sources = {signal: {signal} for signal in [*extract, *absent]}
    for derived, expression in (rules.get("derived") or {}).items():
        sources[derived] = set().union(*(sources.get(item, set()) for item in _names(expression)))
    active, inactive = [], {}
    for rule in rules.get("rules") or []:
        needed = set().union(*(sources.get(item, set()) for item in _names(rule.get("when", ""))))
        missing = sorted({absent[signal] for signal in needed if signal in absent})
        if missing:
            inactive[rule["name"]] = missing
        else:
            active.append(rule)
    fusion = {key: value for key, value in rules.items() if key not in {"signals", "alerts"}}
    fusion.update(name=name, signals=[*extract, *absent], extract=extract, rules=active, inactive_rules=inactive)
    return {"pipeline": pipeline, "fusion": fusion, "alerts": {**(rules.get("alerts") or {}), "name": name},
            "models": selected}
