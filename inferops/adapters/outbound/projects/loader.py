"""Load a project without activating process settings or a global project package."""
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.abc
import importlib.util
import inspect
import sys
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from inferops.adapters.outbound.projects.configuration import load_platform, platform_paths
from inferops.adapters.outbound.projects.manifest import DomainManifest, load_domain_manifest, manifest_path
from inferops.adapters.outbound.projects.models import expand_model_config, model_config_files
from inferops.adapters.outbound.projects.resolver import ModelResolver, canonical_mode
from inferops.adapters.outbound.projects.yaml import load_yaml
from inferops.domain.errors import BackendError, ConfigError
from inferops.domain.fusion.spec import FusionSpec
from inferops.domain.incident_spec import AlertSpec
from inferops.domain.models import LabelsSpec, ModelSpec
from inferops.domain.pipeline import PipelineSpec, StageDecl
from inferops.ports.projects import ProjectDefinition

_PLUGIN_FIELDS = {"plugin", "stage", "backend", "annotator", "installer", "trigger_flags", "target"}
_GENERATED_DIRECTORIES = {"__pycache__", "build", "dist", "node_modules", "venv"}


def _python_sources(root: Path | None) -> dict[Path, bytes]:
    if root is None:
        return {}
    return {path: path.read_bytes() for path in sorted(root.rglob("*.py"))
            if not any(part.startswith(".") or part in _GENERATED_DIRECTORIES
                       for part in path.relative_to(root).parts[:-1])}


class _SnapshotImports(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Import a project revision from captured bytes, bypassing stale pyc files.

    The finder is restricted to one generated namespace and never intercepts
    ordinary imports. Old sessions can lazily import their original plugins
    after an on-disk configuration or plugin revision has been activated.
    """
    def __init__(self, root: Path, package: str, sources: dict[Path, bytes]) -> None:
        self.root, self.package = root, package
        self.sources: dict[str, tuple[Path, bytes, bool]] = {}
        self.packages = {package}
        for path, source in sources.items():
            parts = list(path.relative_to(root).with_suffix("").parts)
            is_package = parts[-1] == "__init__"
            if is_package:
                parts.pop()
            name = ".".join([package, *parts])
            self.sources[name] = path, source, is_package
            for length in range(len(parts)):
                self.packages.add(".".join([package, *parts[:length]]))
            if is_package:
                self.packages.add(name)

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.package and not fullname.startswith(f"{self.package}."):
            return None
        if fullname not in self.sources and fullname not in self.packages:
            raise ModuleNotFoundError(f"{fullname!r} is absent from this project revision")
        entry = self.sources.get(fullname)
        is_package = entry[2] if entry else True
        return importlib.util.spec_from_loader(fullname, self, is_package=is_package)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        entry = self.sources.get(module.__name__)
        if entry:
            path, source, is_package = entry
            module.__file__ = str(path)
        else:
            parts = module.__name__.split(".")[1:]
            path, source, is_package = self.root.joinpath(*parts), None, True
        if is_package:
            module.__path__ = [str(path.parent if entry else path)]
        if source is not None:
            exec(compile(source, str(path), "exec"), module.__dict__)


class ProjectPluginResolver:
    """Resolve local imports into a namespace unique to a project path/revision.

    Packages are registered in Python's import cache only on explicit plugin use.
    Neither sys.path nor the compatibility active-project state is modified.
    """
    def __init__(self, root: Path | None, revision: str,
                 sources: dict[Path, bytes] | None = None) -> None:
        self.root = root
        digest = hashlib.sha256(f"{root}:{revision}".encode()).hexdigest()[:24]
        self.package = f"_inferops_project_{digest}"
        self._imports = (_SnapshotImports(root, self.package,
                                         sources if sources is not None else _python_sources(root))
                         if root is not None else None)

    def resolve(self, reference: str) -> str:
        if reference.startswith("."):
            if self.root is None:
                raise ConfigError(f"relative plugin {reference!r} requires a project root")
            return f"{self.package}{reference}"
        if self.root is not None and reference.startswith(f"{self.root.name}."):
            return f"{self.package}{reference[len(self.root.name):]}"
        return reference

    def _ensure_package(self) -> None:
        if self.root is None or self.package in sys.modules:
            return
        sys.meta_path.insert(0, self._imports)
        try:
            importlib.import_module(self.package)
        except BaseException:
            sys.modules.pop(self.package, None)
            sys.meta_path.remove(self._imports)
            raise

    def load_object(self, reference: str) -> Any:
        reference = self.resolve(reference)
        if ":" in reference:
            module_name, attribute = reference.split(":", 1)
        else:
            module_name, _separator, attribute = reference.rpartition(".")
        if not module_name or not attribute:
            raise ConfigError(f"invalid plugin reference {reference!r}")
        if module_name == self.package or module_name.startswith(f"{self.package}."):
            self._ensure_package()
        try:
            obj = importlib.import_module(module_name)
            for component in attribute.split("."):
                obj = getattr(obj, component)
            return obj
        except (ImportError, AttributeError) as exc:
            raise ConfigError(f"cannot load plugin {reference!r}: {exc}") from exc

    def normalize(self, value: Any, field: str | None = None) -> Any:
        if isinstance(value, dict):
            return {key: self.normalize(child, key) for key, child in value.items()}
        if isinstance(value, list):
            return [self.normalize(child, field) for child in value]
        if isinstance(value, str) and field in _PLUGIN_FIELDS:
            return self.resolve(value)
        return value


class ProjectLoader:
    def __init__(self, root: str | Path | None, *, configs_dir: str | Path = "configs",
                 mode: str = "cpu", triton_url: str = "localhost:8001",
                 backend_options: dict[str, Any] | None = None, artifact_root: str | Path | None = None) -> None:
        if root is not None:
            selected = Path(root).expanduser()
            if not selected.is_dir():
                from inferops.adapters.outbound.projects.packages import adapter_root
                selected = adapter_root(str(root))
            self.root = selected.resolve()
        else:
            self.root = None
        self.configs_dir = Path(configs_dir).expanduser().resolve()
        self.mode = canonical_mode(mode)
        if self.mode not in {"cpu", "cuda", "tensorrt", "triton"}:
            raise ConfigError(f"unknown execution mode {mode!r}")
        self.triton_url = triton_url
        self.backend_options = dict(backend_options or {})
        self.artifact_root = Path(artifact_root).resolve() if artifact_root else None

    def _directories(self) -> dict[str, Path]:
        config_root = self.root / "configs" if self.root else self.configs_dir
        directories = {section: config_root / section for section in ("models", "pipelines", "fusion", "alerts")}
        if self.root and (self.root / "models").is_dir():
            directories["models"] = self.root / "models"
        return directories

    def _revision(self, directories: dict[str, Path], sources: dict[Path, bytes],
                  overrides: Mapping[Path, dict]) -> str:
        digest = hashlib.sha256(f"{self.root}:{self.mode}".encode())
        files: set[Path] = set()
        for directory in directories.values():
            if directory.exists():
                files.update(directory.rglob("*.yaml"))
        if self.root:
            files.update(path for path in [self.root / "project.yaml", self.root / "domain.yaml",
                                          self.root / "configs" / "platform.yaml", self.root / "__init__.py"]
                         if path.is_file())
            files.update(sources)
        else:
            platform = self.configs_dir / "platform.yaml"
            if platform.is_file():
                files.add(platform)
        files.update(overrides)
        files.update(path for path in platform_paths(self.configs_dir, self.root) if path.is_file())
        for path in sorted(files):
            digest.update(str(path).encode())
            if path in overrides:
                digest.update(yaml.safe_dump(overrides[path], sort_keys=True).encode())
            elif path.suffix == ".yaml":
                digest.update(yaml.safe_dump(load_yaml(path), sort_keys=True).encode())
            else:
                digest.update(sources[path] if path in sources else path.read_bytes())
        return digest.hexdigest()

    def load(self, overrides: Mapping[Path, dict] | None = None) -> ProjectDefinition:
        """Compile a candidate configuration without writing files or activating it.

        Override keys are absolute paths (relative keys resolve from the CWD).
        Existing definitions, files, and runtime resources remain untouched.
        """
        overrides = {Path(path).resolve(): value for path, value in (overrides or {}).items()}
        directories = self._directories()
        platform_file = (self.root / "configs" if self.root else self.configs_dir) / "platform.yaml"
        special_paths = {platform_file, *platform_paths(self.configs_dir, self.root)}
        if self.root:
            special_paths.update({self.root / "project.yaml", self.root / "domain.yaml"})
        for path, value in overrides.items():
            section_path = any(path.parent == directory or (
                section == "models" and path.name == "model.yaml" and path.parent.parent == directory)
                               for section, directory in directories.items())
            if path.suffix != ".yaml" or not (section_path or path in special_paths):
                raise ConfigError(f"configuration override is outside this project: {path}")
            if not isinstance(value, dict):
                raise ConfigError(f"configuration override must be a mapping: {path}")

        def read(path):
            return overrides[path] if path in overrides else load_yaml(path)

        def files(section):
            directory = directories[section]
            found = set(model_config_files(directory) if section == "models" else directory.glob("*.yaml"))
            found.update(path for path in overrides if path.parent == directory
                         or (section == "models" and path.name == "model.yaml" and path.parent.parent == directory))
            return sorted(found)

        sources = _python_sources(self.root)
        revision = self._revision(directories, sources, overrides)
        plugins = ProjectPluginResolver(self.root, revision, sources)
        manifest = (DomainManifest(**read(manifest_path(self.root)))
                    if self.root and manifest_path(self.root) in overrides else load_domain_manifest(self.root))
        resolver = ModelResolver(directories["models"], overrides=overrides)
        models: dict[str, ModelSpec] = {}
        source_dirs: dict[str, Path] = dict(directories)
        source_dirs["platform_configs"] = self.configs_dir
        if self.artifact_root is not None:
            source_dirs["artifact_root"] = self.artifact_root
        for path, raw in ((source, spec) for source in files("models") for spec in expand_model_config(read(source))):
            spec = ModelSpec(**plugins.normalize(raw.model_dump()))
            if spec.backend == "onnxruntime" and self.mode in {"cpu", "cuda"}:
                spec.device.preferred = "gpu" if self.mode == "cuda" else "cpu"
            if spec.artifact.path:
                artifact = Path(spec.artifact.path).expanduser()
                if not artifact.is_absolute():
                    candidates = [path.parent / artifact, path.parent.parent / artifact]
                    if self.root:
                        candidates.extend([self.root / artifact, self.root / "model_repository" / artifact])
                    elif self.artifact_root:
                        candidates.extend([self.artifact_root / artifact, self.artifact_root.parent / artifact])
                    # A project does not inherit artifacts from the process CWD.
                    found = next((candidate for candidate in candidates if candidate.is_file()), None)
                    artifact = found or ((self.root or path.parent) / artifact)
                spec.artifact.path = str(artifact.resolve())
            if isinstance(spec.labels, LabelsSpec) and spec.labels.from_file:
                labels = Path(spec.labels.from_file)
                if not labels.is_absolute():
                    spec.labels.from_file = str((path.parent / labels).resolve())
            models[spec.name] = spec
            source_dirs[spec.name] = path.parent

        def load_section(section: str, schema):
            values = {}
            for path in files(section):
                value = schema(**plugins.normalize(read(path)))
                if value.name in values:
                    raise ConfigError(f"duplicate {section} name {value.name!r}")
                values[value.name] = value
            return values

        declared_pipelines = load_section("pipelines", PipelineSpec)
        fusion = load_section("fusion", FusionSpec)
        alerts = load_section("alerts", AlertSpec)
        self._validate(declared_pipelines, fusion, alerts, resolver, manifest)
        pipelines = {}
        for name, pipeline in declared_pipelines.items():
            if manifest and manifest.pipelines is not None and name not in manifest.pipelines:
                continue
            if pipeline.edges:
                linear = ([pipeline.source.name] if pipeline.source else []) + [stage.name for stage in pipeline.stages]
                if pipeline.edges != [[left, right] for left, right in pairwise(linear)]:
                    raise ConfigError(f"pipeline {name!r}: explicit DAG edges are not supported by ordered execution")
            pipeline = pipeline.for_mode(self.mode, resolver)
            for stage in pipeline.stages:
                if stage.stage == "model_group":
                    stage.params["models"] = [resolver.resolve(model, self.mode)
                                              for model in stage.params.get("models", [])]
            self._inject_tracking(pipeline)
            pipelines[name] = pipeline
        platform = load_platform(self.configs_dir, self.root, manifest, overrides=overrides)
        # Legacy extension instantiation receives normalized absolute refs.
        # Register this explicit instance namespace before those adapters use
        # importlib; never register it under the external project name.
        plugins._ensure_package()
        factory = BackendFactory(plugins, self.mode, self.triton_url, self.backend_options)
        return ProjectDefinition(root=self.root, name=manifest.name if manifest else "inferops",
                                 mode=self.mode, revision=revision, models=MappingProxyType(models),
                                 pipelines=MappingProxyType(pipelines), fusion=MappingProxyType(fusion),
                                 alerts=MappingProxyType(alerts), source_dirs=MappingProxyType(source_dirs),
                                 plugin_resolver=plugins, object_loader=plugins.load_object,
                                 backend_factory=factory, resolver=resolver, manifest=manifest,
                                 platform=MappingProxyType(platform))

    @staticmethod
    def _validate(pipelines, fusion, alerts, resolver, manifest) -> None:
        """Reject broken names and builtin parameter contracts before startup.

        Do not import or instantiate runtime classes here: optional clients and
        project resource construction remain lazy and explicitly engine-owned.
        """
        from inferops.adapters.outbound.stages.registry import (
            APPLICATION_STAGES,
            BUILTINS,
            required_stage_parameters,
        )
        errors = []
        required = required_stage_parameters()

        def reference(section, name, location):
            if not isinstance(name, str) or not name:
                errors.append(f"{location}: {section} reference must be a nonempty name")
                return
            exists = resolver.contains(name) if section == "models" else name in {
                "pipelines": pipelines, "fusion": fusion, "alerts": alerts}[section]
            if not exists:
                errors.append(f"{location}: unknown {section} reference {name!r}")

        for pipeline in pipelines.values():
            declarations = ([pipeline.source] if pipeline.source else []) + pipeline.stages + pipeline.sinks
            names = [decl.name for decl in declarations]
            if len(names) != len(set(names)):
                errors.append(f"pipeline {pipeline.name!r}: stage names must be unique")
            if pipeline.limits.max_concurrency < 1 or pipeline.limits.queue_size < 1:
                errors.append(f"pipeline {pipeline.name!r}: concurrency and queue size must be positive")
            if pipeline.limits.timeout_ms is not None and pipeline.limits.timeout_ms <= 0:
                errors.append(f"pipeline {pipeline.name!r}: timeout_ms must be positive")
            for declaration in declarations:
                location = f"pipeline {pipeline.name!r}, stage {declaration.name!r}"
                alias, params = declaration.stage, declaration.params
                if alias not in BUILTINS and alias not in APPLICATION_STAGES and ":" not in alias and "." not in alias:
                    errors.append(f"{location}: unknown builtin stage {alias!r}")
                selected_from_modes = alias == "infer" and bool(params.get("model_modes"))
                for key in required.get(alias, []):
                    if not selected_from_modes and (not isinstance(params.get(key), str) or not params[key].strip()):
                        errors.append(f"{location}: {key} is required")
                for key, section in (("model", "models"), ("fallback_model", "models"),
                                     ("fusion", "fusion"), ("alerts", "alerts")):
                    if params.get(key) is not None:
                        reference(section, params[key], location)
                modes = params.get("model_modes", {})
                if not isinstance(modes, dict):
                    errors.append(f"{location}: model_modes must be a mapping")
                else:
                    for mode, model in modes.items():
                        if mode not in {"auto", "cpu", "cuda", "tensorrt", "triton"}:
                            errors.append(f"{location}: unknown execution mode {mode!r}")
                        reference("models", model, location)
                if alias == "model_group":
                    group = params.get("models")
                    if not isinstance(group, list) or not group:
                        errors.append(f"{location}: model_group requires a nonempty models list")
                    else:
                        for model in group:
                            reference("models", model, location)
                if alias in {"decode_any", "decode_video"}:
                    every_n = params.get("every_n", 1)
                    if not isinstance(every_n, int) or isinstance(every_n, bool) or every_n < 1:
                        errors.append(f"{location}: every_n must be a positive integer")
                if alias == "route_by":
                    try:
                        ast.parse(params.get("when", "True"), mode="eval")
                    except (SyntaxError, TypeError) as exc:
                        errors.append(f"{location}: invalid route expression: {exc}")
            if pipeline.tracking:
                for model in pipeline.tracking.model_filter:
                    reference("models", model, f"pipeline {pipeline.name!r} tracking")
        if manifest:
            for name in manifest.pipelines or []:
                reference("pipelines", name, "project pipelines")
            for name in manifest.prewarm_pipelines:
                reference("pipelines", name, "project prewarm_pipelines")
                if manifest.pipelines is not None and name not in manifest.pipelines:
                    errors.append(f"prewarm pipeline {name!r} is not enabled")
            for name in manifest.prewarm_models:
                reference("models", name, "project prewarm_models")
        for spec in fusion.values():
            for signal in [*spec.windows, *spec.extract]:
                if signal not in spec.signals:
                    errors.append(f"fusion {spec.name!r}: undeclared signal {signal!r}")
            for expression in [*spec.extract.values(), *spec.derived.values(), *(rule.when for rule in spec.rules)]:
                try:
                    ast.parse(expression, mode="eval")
                except (SyntaxError, TypeError) as exc:
                    errors.append(f"fusion {spec.name!r}: invalid expression: {exc}")
        for spec in alerts.values():
            levels = {level.name for level in spec.severity.levels}
            if spec.severity.default not in levels or any(level not in levels for level in spec.severity.mapping.values()):
                errors.append(f"alerts {spec.name!r}: severity mappings/default must name declared levels")
            for sink in spec.sinks:
                if sink.min_severity is not None and sink.min_severity not in levels:
                    errors.append(f"alerts {spec.name!r}, sink {sink.name!r}: unknown minimum severity")
                key = "target" if sink.kind == "plugin" else "url" if sink.kind in {"webhook", "slack"} else None
                if key and (not isinstance(sink.params.get(key), str) or not sink.params[key].strip()):
                    errors.append(f"alerts {spec.name!r}, sink {sink.name!r}: {key} is required")
        if errors:
            raise ConfigError("Invalid project configuration:\n" + "\n".join(errors))

    @staticmethod
    def _inject_tracking(pipeline: PipelineSpec) -> None:
        tracking = pipeline.tracking
        if not tracking or not tracking.enabled or any(stage.stage == "track" for stage in pipeline.stages):
            return
        anchor = tracking.insert_after or next((stage.name for stage in pipeline.stages
                                                if stage.stage == "infer"), "")
        if not anchor:
            return
        index = next((index for index, stage in enumerate(pipeline.stages) if stage.name == anchor), None)
        if index is None:
            raise ConfigError(f"tracking anchor {anchor!r} not found in pipeline {pipeline.name!r}")
        params = tracking.model_dump(exclude={"enabled", "insert_after"})
        pipeline.stages.insert(index + 1, StageDecl(name="auto_track", stage="track", params=params))


class BackendFactory:
    def __init__(self, plugins: ProjectPluginResolver, mode: str, triton_url: str,
                 options: dict[str, Any]) -> None:
        self.plugins, self.mode, self.triton_url, self.options = plugins, mode, triton_url, options

    def __call__(self, spec: ModelSpec, base_dir: Path | None = None):
        from inferops.adapters.outbound.inference import BUILTIN_BACKENDS
        reference = BUILTIN_BACKENDS.get(spec.backend, spec.backend)
        cls = self.plugins.load_object(reference)
        if not inspect.isclass(cls):
            raise BackendError(f"backend {spec.backend!r} is not a class")
        for method in ("load", "infer", "close"):
            if not callable(getattr(cls, method, None)):
                raise BackendError(f"backend {spec.backend!r} lacks {method}()")
        options = dict(self.options)
        options.update(base_dir=base_dir)
        if spec.backend in {"onnxruntime", "tensorrt"}:
            options.update(execution_mode=self.mode, respect_env=False,
                           threads=int(self.options.get("threads", 4)))
        if spec.backend == "triton":
            options.update(url=self.triton_url)
        parameters = inspect.signature(cls).parameters
        if not any(parameter.kind == parameter.VAR_KEYWORD for parameter in parameters.values()):
            options = {key: value for key, value in options.items() if key in parameters}
        return cls(**options)
