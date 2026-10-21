"""Resolve logical model ports and explicit runtime names in one place."""
from __future__ import annotations

from pathlib import Path

from inferops.adapters.outbound.projects.models import expand_model_config, model_config_files
from inferops.adapters.outbound.projects.yaml import load_yaml
from inferops.domain.errors import ConfigError
from inferops.domain.models import canonical_mode


class ModelResolver:
    def __init__(self, directory: str | Path, *, overrides: dict[Path, dict] | None = None):
        self.names: set[str] = set()
        self.ports: dict[str, dict[str, str]] = {}
        self.variants: dict[str, list[str]] = {}
        root = Path(directory)
        files = set(model_config_files(root))
        files.update(path for path in (overrides or {}) if path.suffix == ".yaml"
                     and (path.parent == root or (path.name == "model.yaml" and path.parent.parent == root)))
        for path in sorted(files):
            data = overrides[path] if overrides and path in overrides else load_yaml(path)
            specs = expand_model_config(data)
            for spec in specs:
                if spec.name in self.names:
                    raise ConfigError(f"duplicate model name {spec.name!r}")
                self.names.add(spec.name)
            if "variants" not in data:
                continue
            name = data["name"]
            if name in self.ports:
                raise ConfigError(f"duplicate model port {name!r}")
            variants = dict(zip(data["variants"], specs, strict=True))
            self.variants[name] = [spec.name for spec in specs]
            choices = {}
            for mode, key, backend in (("cpu", "onnx", "onnxruntime"), ("cuda", "onnx", "onnxruntime"),
                                       ("triton", "triton", "triton"),
                                       ("tensorrt", "tensorrt", "tensorrt")):
                chosen = variants.get(key)
                candidates = [spec for spec in specs if spec.backend == backend]
                if chosen is None and len(candidates) == 1:
                    chosen = candidates[0]
                if chosen is not None:
                    choices[mode] = chosen.name
            self.ports[name] = choices

    def resolve(self, name: str, mode: str) -> str:
        mode = canonical_mode(mode)
        if name in self.ports:
            choice = self.ports[name].get(mode)
            if choice is None:
                raise ConfigError(f"model {name!r}: no unambiguous variant for {mode!r}; choose an explicit variant")
            return choice
        if name not in self.names:
            raise ConfigError(f"unknown model reference {name!r}")
        return name

    def contains(self, name: str) -> bool:
        return name in self.names or name in self.ports
