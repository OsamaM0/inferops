"""inferops CLI."""
from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from typing import Annotated

import typer
from rich import print as rprint
from rich.table import Table

from inferops import __version__
from inferops.adapters.outbound.projects.settings import ConfigStack, settings

app = typer.Typer(help="InferOps CLI")
models_app = typer.Typer(help="Model registry")
pipelines_app = typer.Typer(help="Pipelines")
worker_app = typer.Typer(help="Workers")
domain_app = typer.Typer(help="Domain packs (the per-project layer: configs + plugins + assets)")
app.add_typer(models_app, name="models")
app.add_typer(pipelines_app, name="pipelines")
app.add_typer(worker_app, name="worker")
app.add_typer(domain_app, name="domain")
app.add_typer(domain_app, name="project")
adapters_app = typer.Typer(help="Independently installed project adapters")
app.add_typer(adapters_app, name="adapters")
ownership_app = typer.Typer(help="Explicit distributed ownership migrations")
app.add_typer(ownership_app, name="ownership")
evidence_app = typer.Typer(help="Evidence retention and recovery")
app.add_typer(evidence_app, name="evidence")


@ownership_app.command("adopt-legacy")
def ownership_adopt_legacy(scope: str = typer.Option(...), owners_stopped: bool = False):
    """Assign legacy durable records after stopping old owners and backing up state."""
    import os

    from inferops.adapters.outbound.persistence.ownership import adopt_legacy
    from inferops.adapters.outbound.persistence.sql import SqlRepository
    url = os.environ.get("INFEROPS_DATABASE_URL", "")
    if not url.startswith("postgresql") or not owners_stopped:
        raise typer.BadParameter("PostgreSQL INFEROPS_DATABASE_URL and --owners-stopped are required")
    async def execute():
        repository = SqlRepository(url, maintenance=True)
        await repository.start()
        try:
            rprint(await adopt_legacy(repository, scope, owners_stopped=owners_stopped))
        finally:
            await repository.close()
    asyncio.run(execute())


@evidence_app.command("retention")
def evidence_retention(before: str = typer.Option(...), url: str = "http://127.0.0.1:8080",
                       worker_id: str | None = None, apply: bool = False, limit: int = 100):
    """List a bounded cleanup plan; --apply uses the owner's tombstone workflow."""
    import httpx
    params = {"before": before, "limit": limit}
    if worker_id:
        params["worker_id"] = worker_id
    with httpx.Client(base_url=url, timeout=30) as client:
        response = client.get("/v1/retention/evidence", params=params)
        response.raise_for_status()
        plan = response.json()
        if apply:
            plan["results"] = []
            for row in plan["candidates"]:
                result = client.post(f"/v1/retention/evidence/{row['id']}", params={"before": before})
                result.raise_for_status()
                plan["results"].extend(result.json())
            plan["dry_run"] = False
        print(json.dumps(plan, indent=2, default=str))


@evidence_app.command("prune-outbox")
def prune_outbox(before: str = typer.Option(...), url: str = "http://127.0.0.1:8080", apply: bool = False):
    """Dry-run delivered-message cleanup by delivery age; inbox retention stays disabled."""
    import httpx
    response = httpx.request("POST" if apply else "GET", f"{url}/v1/retention/outbox",
                            params={"before": before}, timeout=30)
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2))


@adapters_app.command("list")
def adapters_list():
    from inferops.adapters.outbound.projects.packages import installed_adapters
    for name, entry in sorted(installed_adapters().items()):
        rprint(f"{name}: {entry.value}")


@adapters_app.command("init")
def adapters_init(name: str, destination: Path):
    """Create a writable project from an installed adapter."""
    from inferops.adapters.outbound.projects.packages import create_project
    try:
        root = create_project(name, destination)
    except (ValueError, FileExistsError) as exc:
        rprint(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    rprint(f"Created {root}; select it with INFEROPS_PROJECT_ROOT={root}")


def _stack_for(
    configs_dir: str | None,
    domain: str | None,
    domain_root: str | None,
) -> ConfigStack:
    return ConfigStack(
        configs_dir or settings.configs_dir,
        active_domain=domain or settings.active_domain,
        domain_root=domain_root or (None if domain else settings.domain_root),
    )


def _validate_specs(stack: ConfigStack) -> int:
    from inferops.adapters.outbound.projects.models import load_model_variants, model_config_files
    from inferops.adapters.outbound.projects.yaml import load_yaml
    from inferops.domain.fusion.spec import FusionSpec
    from inferops.domain.incident_spec import AlertSpec
    from inferops.domain.models import ModelSpec
    from inferops.domain.pipeline import PipelineSpec

    errors = 0
    if stack.domain_root is not None:
        try:
            manifest = stack.manifest()
        except Exception as e:
            errors += 1
            from inferops.adapters.outbound.projects.manifest import manifest_path

            rprint(f"[red]FAIL[/red] {manifest_path(stack.domain_root)}: {e}")
        else:
            if manifest is None:
                errors += 1
                rprint(f"[red]FAIL[/red] domain root does not exist: {stack.domain_root}")
            else:
                rprint(f"[green]OK[/green] domain {manifest.name} ({stack.domain_root})")
    for kind, cls in [("models", ModelSpec), ("pipelines", PipelineSpec),
                      ("fusion", FusionSpec), ("alerts", AlertSpec)]:
        d = stack.spec_dir(kind)
        if not d.exists():
            continue
        paths = model_config_files(d) if kind == "models" else sorted(d.glob("*.yaml"))
        for p in paths:
            try:
                if kind == "models":
                    load_model_variants(p)
                else:
                    cls(**load_yaml(p))
                rprint(f"[green]OK[/green] {p}")
            except Exception as e:
                errors += 1
                rprint(f"[red]FAIL[/red] {p}: {e}")
    if not errors:
        from inferops.adapters.outbound.projects.validation import validate_config_stack

        try:
            validate_config_stack(stack)
        except Exception as exc:
            errors += 1
            rprint(f"[red]FAIL[/red] {exc}")
    return errors


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


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
    from inferops.adapters.outbound.inference import BACKEND_DEPENDENCIES
    from inferops.adapters.outbound.projects.loader import ProjectPluginResolver
    from inferops.adapters.outbound.projects.yaml import load_dir
    from inferops.adapters.outbound.stages.registry import APPLICATION_STAGES, resolve_stage
    from inferops.domain.incident_spec import AlertSpec

    stack = _stack_for(configs_dir, domain, domain_root)
    plugins = ProjectPluginResolver(stack.domain_root.resolve() if stack.domain_root else None, 'preflight')
    load_object = plugins.load_object
    errors = _validate_specs(stack)
    if errors:
        raise typer.Exit(1)
    if stack.active_domain:
        rprint(f"[cyan]Domain[/cyan] {stack.active_domain}: {stack.domain_root}")
    else:
        rprint("[cyan]Domain[/cyan] core-only")

    from inferops.adapters.outbound.projects.preparation import (
        artifact_target,
        selected_surface,
        validate_artifact,
    )

    pipelines, models = selected_surface(stack, settings.execution_mode)
    pipeline_specs = [(stack.spec_dir("pipelines") / f"{spec.name}.yaml", spec) for spec in pipelines]
    for path, spec in models:
        module = BACKEND_DEPENDENCIES.get(spec.backend)
        if module and not _module_available(module):
            errors += 1
            rprint(f"[red]FAIL[/red] {path}: backend dependency {module!r} is not installed")
        elif module is None:
            try:
                from inferops.ports.inference import InferenceBackend

                cls = load_object(spec.backend)
                if not isinstance(cls, type) or not issubclass(cls, InferenceBackend):
                    raise ValueError('custom backend must extend InferenceBackend')
            except Exception as exc:
                errors += 1
                rprint(f"[red]FAIL[/red] {path}: {exc}")

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
        if spec.backend in {"onnxruntime", "triton"} or spec.artifact.path:
            try:
                validate_artifact(artifact_target(spec, path, stack), spec)
            except Exception as exc:
                errors += 1
                rprint(f"[red]FAIL[/red] {exc}")

    for path, spec in pipeline_specs:
        for stage in ([spec.source] if spec.source else []) + spec.stages + spec.sinks:
            try:
                if stage.stage not in APPLICATION_STAGES:
                    if ':' in stage.stage or stage.stage.startswith('.'):
                        load_object(stage.stage)
                    else:
                        resolve_stage(stage.stage)
            except Exception as exc:
                errors += 1
                rprint(f"[red]FAIL[/red] {path}: cannot resolve stage {stage.stage}: {exc}")
            for key in ("annotator", "trigger_flags"):
                ref = (stage.params or {}).get(key)
                if isinstance(ref, str):
                    try:
                        load_object(ref)
                    except Exception as exc:
                        errors += 1
                        rprint(f"[red]FAIL[/red] {path}: cannot import {key} {ref}: {exc}")

    for path, data in load_dir(stack.spec_dir("alerts")):
        for sink in AlertSpec(**data).sinks:
            if sink.kind == "plugin":
                try:
                    from inferops.ports.extensions import AlertSink

                    cls = load_object(sink.params["target"])
                    if not isinstance(cls, type) or not issubclass(cls, AlertSink):
                        raise ValueError("custom alert action must extend AlertSink")
                except Exception as exc:
                    errors += 1
                    rprint(f"[red]FAIL[/red] {path}: action {sink.name}: {exc}")

    if errors:
        raise typer.Exit(1)
    rprint("[green]Preflight OK[/green]")


# ── domain packs ─────────────────────────────────────────────────────────────

def _domain_stack(domain: str | None, domain_root: str | None) -> ConfigStack:
    stack = _stack_for(None, domain, domain_root)
    if stack.domain_root is None:
        rprint("[red]no domain selected[/red]: pass --domain <name> or set INFEROPS_ACTIVE_DOMAIN")
        raise typer.Exit(2)
    if not stack.domain_root.is_dir():
        rprint(f"[red]domain root not found[/red]: {stack.domain_root}")
        raise typer.Exit(2)
    return stack


@domain_app.command("list")
def domain_list():
    """List project blocks under projects/ (and legacy examples/ or domains/)."""
    from inferops.adapters.outbound.projects.manifest import discover_domains, load_domain_manifest

    t = Table("name", "label", "triton", "pipelines", "root")
    for name, root in discover_domains(settings.configs_dir):
        manifest = load_domain_manifest(root)
        pipelines = sorted(p.stem for p in (root / "configs" / "pipelines").glob("*.yaml"))
        t.add_row(name, manifest.display_label if manifest else "", "yes" if manifest and manifest.triton else "no",
                  ", ".join(pipelines), str(root))
    rprint(t)


@domain_app.command("info")
def domain_info(
    domain: str | None = typer.Option(None, "--domain"),
    domain_root: str | None = typer.Option(None),
):
    """Show the resolved manifest of one domain pack as JSON."""
    stack = _domain_stack(domain, domain_root)
    manifest = stack.manifest()
    data = manifest.model_dump(mode="json")
    data["root"] = str(stack.domain_root.resolve())
    data["package"] = stack.domain_package
    data["model_repository"] = str(stack.model_repository(settings.model_repository).resolve())
    rprint(json.dumps(data, indent=2))


@domain_app.command("env")
def domain_env(
    domain: str | None = typer.Option(None, "--domain"),
    domain_root: str | None = typer.Option(None),
):
    """Print KEY=VALUE lines the Makefile uses to drive serve/setup for a domain."""
    from inferops.adapters.outbound.projects.manifest import manifest_env

    stack = _domain_stack(domain, domain_root)
    env = manifest_env(stack.manifest(), stack.domain_root, stack.model_repository(settings.model_repository))
    for key, value in env.items():
        print(f"{key}={value}")


@domain_app.command("prepare")
@domain_app.command("setup-assets")
def domain_setup_assets(
    domain: str | None = typer.Option(None, "--domain", "--project"),
    domain_root: str | None = typer.Option(None),
    force: bool = typer.Option(False, help="Refresh configured model weights"),
):
    """Validate the project and prepare only its selected real model weights."""
    from inferops.adapters.outbound.projects.preparation import prepare_project

    try:
        count = prepare_project(_domain_stack(domain, domain_root), settings.execution_mode, force=force)
    except Exception as exc:
        rprint(f"[red]Preparation failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    rprint(f"[green]Project prepared[/green]: {count} model artifacts")


@domain_app.command("init")
def domain_init(
    name: str = typer.Argument(..., help="Domain name (Python identifier), e.g. ppe_detection"),
    parent: str = typer.Option("projects", help="Directory to create the project in"),
    labels: str = typer.Option("person,helmet,vest", help="Comma-separated detection classes"),
    label: str | None = typer.Option(None, help="Human-readable label"),
):
    """Create an empty project ready for real model configuration."""
    from inferops.adapters.outbound.projects.manifest import project_directories, scaffold_domain

    base = project_directories(settings.configs_dir)[0] if parent == "projects" else Path(settings.configs_dir).resolve().parent / parent
    root = (base / name).resolve()
    classes = [c.strip() for c in labels.split(",") if c.strip()]
    created = scaffold_domain(root, name, classes, label=label)
    for path in created:
        rprint(f"[green]created[/green] {path}")
    rprint(f"\nNext: [cyan]make validate PROJECT={name}[/cyan] then [cyan]make run PROJECT={name} MODE=cpu[/cyan]")


@models_app.command("init")
def model_init(
    name: str = typer.Argument(..., help="Model port name, e.g. helmet_detector"),
    domain: str | None = typer.Option(None, "--domain", "--project"),
    domain_root: str | None = typer.Option(None),
    labels: str = typer.Option("person,helmet,vest", help="Comma-separated detection classes"),
    format: str = typer.Option("yolo", help="Export format: yolo, detr, or rfdetr"),
):
    """Create a model folder containing its adapter and runtime configuration."""
    from inferops.adapters.outbound.projects.model_scaffold import scaffold_model_port

    stack = _domain_stack(domain, domain_root)
    try:
        created = scaffold_model_port(stack.domain_root, name,
                                      [c.strip() for c in labels.split(",") if c.strip()], format=format)
    except (ValueError, FileExistsError) as exc:
        rprint(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    for path in created:
        rprint(f"[green]created[/green] {path}")
    rprint(f"Configure real weights, add [cyan]model: {name}[/cyan] to a pipeline, and enable it in project.yaml.")


def _project_definition():
    from inferops.adapters.outbound.projects.loader import ProjectLoader
    return ProjectLoader(settings.domain_root, configs_dir=settings.configs_dir,
                         mode=settings.execution_mode, triton_url=settings.triton.url,
                         artifact_root=Path(settings.model_repository).expanduser().resolve()).load()


@models_app.command("list")
def models_list():
    t = Table("name", "task", "backend", "version")
    for model in _project_definition().models.values():
        t.add_row(model.name, model.task, model.backend, str(model.version))
    rprint(t)


@models_app.command("show")
def models_show(name: str):
    definition = _project_definition()
    rprint(json.dumps(definition.models[definition.resolve_model(name)].model_dump(mode="json"), indent=2))


@pipelines_app.command("list")
def pipelines_list():
    t = Table("name", "stages", "sinks")
    for pipeline in _project_definition().pipelines.values():
        t.add_row(pipeline.name, str(len(pipeline.stages)), str(len(pipeline.sinks)))
    rprint(t)


@pipelines_app.command("run")
def pipelines_run(name: str, input_path: str = typer.Argument(..., help="Image or video file")):
    async def execute():
        from inferops.adapters.inbound.workers.bootstrap import server_engine
        async with server_engine() as engine:
            summary = await engine.run(name, Path(input_path), stream_name="cli", owner_kind="cli")
            rprint(json.dumps(summary, indent=2, default=str))
    asyncio.run(execute())


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8080, reload: bool = False):
    """Start the API server."""
    import uvicorn

    # The new application's lifespan compiles its explicit project definition
    # and prewarms required models. Do not activate legacy global stores first.
    uvicorn.run("inferops.adapters.inbound.http.main:app", host=host, port=port, reload=reload)


@worker_app.command("pipeline")
def worker_pipeline(name: str, max_concurrency: int | None = typer.Option(None)):
    """Run a source-driven pipeline worker."""
    from inferops.adapters.inbound.workers.pipeline_worker import run_pipeline_worker
    asyncio.run(run_pipeline_worker(name, max_concurrency=max_concurrency))


@worker_app.command("cameras")
def worker_cameras(manifest: Annotated[Path, typer.Option()], worker_id: Annotated[str, typer.Option()],
                   host: str = "0.0.0.0", port: int = 8080):
    """Run assigned sources in a fenced shard with authenticated owner controls."""
    from inferops.adapters.inbound.workers.camera_worker import run_camera_worker
    run_camera_worker(manifest, worker_id, host=host, port=port)


@worker_app.command("topic")
def worker_topic(
    name: str,
    input_topic: str = typer.Option(...),
    output_topic: str = typer.Option(...),
    group: str | None = typer.Option(None),
):
    """Run a topic-to-topic pipeline worker."""
    from inferops.adapters.inbound.workers.topic_worker import run_topic_worker
    asyncio.run(run_topic_worker(name, input_topic=input_topic, output_topic=output_topic, group=group))


if __name__ == "__main__":
    app()
