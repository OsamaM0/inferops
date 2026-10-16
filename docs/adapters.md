# Author an InferOps adapter

An adapter is an ordinary independently versioned Python distribution. Its
package contains `project.yaml`, `models/`, `configs/`, and optional plugins.
The distribution name may use hyphens; its import package must be a valid Python
identifier. Keep application labels and domain rules in this package.

A standard layout is:

```text
my-adapter/
  pyproject.toml
  src/my_adapter/
    __init__.py
    project.yaml
    models/detector/model.yaml
    configs/pipelines/detection.yaml
    plugins/__init__.py
```

Minimal package metadata:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "inferops-my-adapter"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["inferops>=0.1.0,<0.2"]

[project.entry-points."inferops.projects"]
my_adapter = "my_adapter"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
my_adapter = ["project.yaml", "configs/**/*.yaml", "models/*/model.yaml"]
```

The entry point refers to the package module, not a function. Discovery validates
that its directory contains `project.yaml`. Include all configuration assets in
the wheel; keep large weights outside it and declare artifact sources in model
configuration. List extra Python dependencies needed by custom plugins in the
adapter's `dependencies`, leaving runtime selection to the user. Avoid requiring
the CPU extra in an adapter because consumers may choose CUDA instead.

Use relative plugin references such as `.models.detector.adapter:Postprocess`.
Workspace projects and installed adapter packages use the same config contracts.
After installing, use `inferops adapters init my_adapter ./my_workspace` to make
a writable copy. Select it with `INFEROPS_PROJECT_ROOT`, or pass it to
`InferenceSession.from_project`. Installed adapters can also be selected by name
with `INFEROPS_ACTIVE_DOMAIN` or `InferenceSession.from_project('my_adapter')`;
prepare their artifacts in a workspace first.

A package containing only a model integration can use a minimal project manifest
with `pipelines: []` and one model port. Another project can import its public
plugin classes using absolute `module:Class` references. It does not need to
modify the InferOps backend factory: custom backends subclass
`inferops.ports.inference.InferenceBackend` and set `backend: 'module:Class'`.

Build and verify separately:

```bash
python -m build
pip install dist/inferops_my_adapter-0.1.0-py3-none-any.whl
inferops adapters list
inferops adapters init my_adapter ./check_workspace
inferops validate --domain-root ./check_workspace
```

Driver behaviour demonstrates this contract under
[`projects/driver_behaviour`](../projects/driver_behaviour/README.md). The core
wheel excludes that directory, its weights and its domain-specific classes.
