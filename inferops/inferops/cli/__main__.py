"""inferops CLI."""
from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import typer
from rich import print as rprint
from rich.table import Table

from inferops import __version__
from inferops.core.config import ConfigStack, settings

app = typer.Typer(help="InferOps CLI")
models_app = typer.Typer(help="Model registry")
pipelines_app = typer.Typer(help="Pipelines")
worker_app = typer.Typer(help="Workers")
app.add_typer(models_app, name="models")
app.add_typer(pipelines_app, name="pipelines")
app.add_typer(worker_app, name="worker")


_SPEC_TYPES = ("models", "pipelines", "fusion", "alerts")


def _stack_for(
    configs_dir: str | None,
    domain: str | None,
    domain_root: str | None,
) -> ConfigStack:
    return ConfigStack(
        configs_dir or settings.configs_dir,
        active_domain=domain or settings.active_domain,
        domain_root=domain_root or settings.domain_root,
    )


def _validate_specs(stack: ConfigStack) -> int:
    from inferops.alerts.spec import AlertSpec
    from inferops.core.yaml_loader import load_yaml
    from inferops.fusion.spec import FusionSpec
    from inferops.pipeline.spec import PipelineSpec
    from inferops.registry.model_spec import ModelSpec

    errors = 0
    for kind, cls in [("models", ModelSpec), ("pipelines", PipelineSpec),
                      ("fusion", FusionSpec), ("alerts", AlertSpec)]:
        d = stack.spec_dir(kind)
        if not d.exists():
            continue
        for p in sorted(d.glob("*.yaml")):
            try:
                cls(**load_yaml(p))
                rprint(f"[green]OK[/green] {p}")
            except Exception as e:
                errors += 1
                rprint(f"[red]FAIL[/red] {p}: {e}")
    return errors


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _artifact_candidates(path: str, *, spec_dir: Path, model_repository: Path) -> list[Path]:
    """Delegate to the shared resolver so preflight and runtime always agree.

    This used to be a hand-rolled list that omitted ``settings.domain_root`` and
    the configs-dir sibling, so ``inferops preflight`` could report a model as
    missing that OnnxRuntimeBackend resolved without trouble.
    """
    from inferops.core.artifacts import artifact_candidates

    return artifact_candidates(
        path,
        base_dir=spec_dir,
        model_repository=model_repository,
        domain_root=settings.domain_root,
        configs_dir=settings.configs_dir,
    )


@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context, version: bool = typer.Option(False, "--version")):
    if version:
        rprint(f"inferops {__version__}")
        raise typer.Exit()


@app.command()
def validate(
    configs_dir: str = typer.Option(None, help="Override core configs dir"),
    domain: str | None = typer.Option(None, "--domain", help="Active domain name to validate"),
    domain_root: str | None = typer.Option(None, help="Explicit active domain root"),
):
    """Validate every YAML in the active InferOps config surface."""
    stack = _stack_for(configs_dir, domain, domain_root)
    errors = _validate_specs(stack)
    raise typer.Exit(1 if errors else 0)


@app.command()
def preflight(
    configs_dir: str = typer.Option(None, help="Override core configs dir"),
    domain: str | None = typer.Option(None, "--domain", help="Active domain name to check"),
    domain_root: str | None = typer.Option(None, help="Explicit active domain root"),
    require_artifacts: bool = typer.Option(True, help="Require local model artifacts to exist"),
):
    """Check the selected config surface before serving or running inference."""
    from inferops.core.plugins import load_object
    from inferops.core.yaml_loader import load_dir
    from inferops.pipeline.registry import resolve_stage
    from inferops.pipeline.spec import PipelineSpec
    from inferops.registry.model_spec import ModelSpec

    stack = _stack_for(configs_dir, domain, domain_root)
    errors = _validate_specs(stack)
    model_repository = stack.model_repository(settings.model_repository)
    if stack.active_domain:
        rprint(f"[cyan]Domain[/cyan] {stack.active_domain}: {stack.domain_root}")
    else:
        rprint("[cyan]Domain[/cyan] core-only")

    models_dir = stack.spec_dir("models")
    model_specs: list[tuple[Path, ModelSpec]] = []
    for path, data in load_dir(models_dir):
        spec = ModelSpec(**data)
        model_specs.append((path, spec))

        backend_modules = {
            "onnxruntime": "onnxruntime",
            "triton": "tritonclient",
            "tensorrt": "tensorrt",
        }
        module = backend_modules.get(spec.backend)
        if module and not _module_available(module):
            errors += 1
            rprint(f"[red]FAIL[/red] {path}: backend dependency {module!r} is not installed")

        for plugin in [spec.preprocess, spec.postprocess]:
            if plugin is None:
                continue
            try:
                load_object(plugin.plugin)
            except Exception as exc:
                errors += 1
                rprint(f"[red]FAIL[/red] {path}: cannot import plugin {plugin.plugin}: {exc}")

        if not require_artifacts:
            continue
        if spec.backend == "onnxruntime" and spec.artifact.path:
            candidates = _artifact_candidates(spec.artifact.path, spec_dir=path.parent, model_repository=model_repository)
            if not any(candidate.exists() for candidate in candidates):
                errors += 1
                rprint(f"[red]FAIL[/red] {path}: model artifact not found; tried {[str(c) for c in candidates]}")
        elif spec.backend == "triton" and spec.artifact.triton_model_name:
            artifact = model_repository / spec.artifact.triton_model_name / "1" / "model.onnx"
            if not artifact.exists():
                errors += 1
                rprint(f"[red]FAIL[/red] {path}: Triton model artifact not found at {artifact}")

    pipelines_dir = stack.spec_dir("pipelines")
    for path, data in load_dir(pipelines_dir):
        spec = PipelineSpec(**data)
        for stage in ([spec.source] if spec.source else []) + spec.stages + spec.sinks:
            try:
                resolve_stage(stage.stage)
            except Exception as exc:
                errors += 1
                rprint(f"[red]FAIL[/red] {path}: cannot resolve stage {stage.stage}: {exc}")

    if errors:
        raise typer.Exit(1)
    rprint("[green]Preflight OK[/green]")


@models_app.command("list")
def models_list():
    async def _go():
        from inferops.registry import autoload, get_registry
        await autoload()
        t = Table("name", "task", "backend", "version")
        for m in get_registry().list_models():
            t.add_row(m.name, m.task, m.backend, str(m.version))
        rprint(t)
    asyncio.run(_go())


@models_app.command("show")
def models_show(name: str):
    async def _go():
        from inferops.registry import autoload, get_registry
        await autoload()
        rprint(json.dumps(get_registry().get_spec(name).model_dump(mode="json"), indent=2))
    asyncio.run(_go())


@pipelines_app.command("list")
def pipelines_list():
    async def _go():
        from inferops.pipeline import autoload, get_pipeline_store
        await autoload()
        t = Table("name", "stages", "sinks")
        for p in get_pipeline_store().list():
            t.add_row(p.name, str(len(p.stages)), str(len(p.sinks)))
        rprint(t)
    asyncio.run(_go())


@pipelines_app.command("run")
def pipelines_run(name: str, input_path: str = typer.Argument(..., help="Image file")):
    async def _go():
        from inferops.core.types import StageItem, now_ms
        from inferops.pipeline import autoload, get_pipeline_store
        from inferops.registry import autoload as autoload_models
        await autoload_models()
        await autoload()
        runner = await get_pipeline_store().get_runner(name)
        item = StageItem(stream_id="cli", ts_ms=now_ms())
        item.payload["image_bytes"] = Path(input_path).read_bytes()
        outs = await runner.run_one(item)
        for o in outs:
            rprint(json.dumps(o.model_dump(mode="json"), indent=2))
    asyncio.run(_go())


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8080, reload: bool = False):
    """Start the API server."""
    import uvicorn
    uvicorn.run("inferops.api.main:app", host=host, port=port, reload=reload)


@worker_app.command("pipeline")
def worker_pipeline(name: str, max_concurrency: int | None = typer.Option(None)):
    """Run a source-driven pipeline worker."""
    from inferops.workers.pipeline_worker import run_pipeline_worker
    asyncio.run(run_pipeline_worker(name, max_concurrency=max_concurrency))


@worker_app.command("topic")
def worker_topic(
    name: str,
    input_topic: str = typer.Option(...),
    output_topic: str = typer.Option(...),
    group: str | None = typer.Option(None),
):
    """Run a topic-to-topic pipeline worker."""
    from inferops.workers.topic_worker import run_topic_worker
    asyncio.run(run_topic_worker(name, input_topic=input_topic, output_topic=output_topic, group=group))


if __name__ == "__main__":
    app()
