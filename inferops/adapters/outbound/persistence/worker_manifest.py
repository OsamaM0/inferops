"""Resolve ${ENVIRONMENT_VARIABLE} source addresses before domain validation."""
import os
import re
from pathlib import Path

import yaml

from inferops.domain.workers import WorkerManifest


def load_worker_manifest(path, *, control_only=False):
    data = yaml.safe_load(Path(path).read_text())
    if isinstance(data.get("workers"), dict):
        data = data["workers"]  # Optional section in the central workspace YAML.
    if control_only:
        data["cameras"] = []  # The API needs endpoints, not camera credentials.
    for entry in data.get("cameras", []):
        uri = entry["source"]["uri"]
        if uri.startswith("${") and uri.endswith("}"):
            key = uri[2:-1]
            if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key) or not os.environ.get(key):
                raise ValueError(f"worker source URI environment reference {key!r} is missing")
            entry["source"]["uri"] = os.environ[key]
    return WorkerManifest.model_validate(data)
