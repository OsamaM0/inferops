# Projects

Each folder is a runnable project. Open **Configuration** in the UI to edit
project settings, models, pipelines, rules and alerts through forms. The editor
discovers each project's schema and logical model variants automatically.
Use **Add configuration**, **Validate**, then **Save changes** to extend it.

For file-based workflows, start with `project.yaml`, then edit model
configuration in `models/<name>/model.yaml` and pipeline/rule/action configuration
in `configs/`. Real weights are prepared into that project's `model_repository/`.

Select a project with `INFEROPS_PROJECT_ROOT=./projects/<name>` in the project `.env`.
Run `docker compose up -d --build` from the project root.

Create an empty project with `make new-project NAME=<name>` and add a real model
with `make new-model PROJECT=<name> NAME=<model>`. See the
[project guide](../docs/project-blocks.md) for all configuration controls.

Projects are optional **adapters**, excluded from the core InferOps wheel. An
adapter can be pip-installed independently; driver behaviour includes its own
`pyproject.toml`. Run `pip install -e projects/driver_behaviour` from the engine
root, then `inferops adapters list` or
`inferops adapters init driver_behaviour ./driver_workspace`. See
[adapter authoring](../docs/adapters.md) and the
[Python library guide](../docs/python-library.md).
