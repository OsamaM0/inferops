"""inferops CLI."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import subprocess
import sys
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
domain_app = typer.Typer(help="Domain packs (the per-project layer: configs + plugins + assets)")
app.add_typer(models_app, name="models")
app.add_typer(pipelines_app, name="pipelines")
app.add_typer(worker_app, name="worker")
app.add_typer(domain_app, name="domain")


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
    if stack.domain_root is not None:
        try:
            manifest = stack.manifest()
        except Exception as e:
            errors += 1
            rprint(f"[red]FAIL[/red] {stack.domain_root / 'domain.yaml'}: {e}")
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


def _artifact_candidates(path: str, *, spec_dir: Path, model_repository: Path, stack: ConfigStack) -> list[Path]:
    """Delegate to the shared resolver so preflight and runtime always agree."""
    from inferops.core.artifacts import artifact_candidates

    return artifact_candidates(
        path,
        base_dir=spec_dir,
        model_repository=model_repository,
        domain_root=stack.domain_root,
        configs_dir=stack.configs_dir,
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
    from inferops.runtimes import BACKEND_DEPENDENCIES, resolve_backend_class

    stack = _stack_for(configs_dir, domain, domain_root)
    errors = _validate_specs(stack)
    model_repository = stack.model_repository(settings.model_repository)
    if stack.active_domain:
        rprint(f"[cyan]Domain[/cyan] {stack.active_domain}: {stack.domain_root}")
    else:
        rprint("[cyan]Domain[/cyan] core-only")

    # A primary model whose pipeline names a fallback_model may legitimately
    # have no local weights yet (fresh scaffold, Triton-only host): report it
    # as a warning, not a failure, so a new pack passes preflight out of the box.
    pipelines_dir = stack.spec_dir("pipelines")
    pipeline_specs = [(path, PipelineSpec(**data)) for path, data in load_dir(pipelines_dir)]
    fallbacks: dict[str, str] = {}
    for _path, pspec in pipeline_specs:
        for stage in pspec.stages:
            params = stage.params or {}
            if isinstance(params.get("model"), str) and isinstance(params.get("fallback_model"), str):
                fallbacks.setdefault(params["model"], params["fallback_model"])

    def _artifact_missing(path: Path, spec: ModelSpec, where: str) -> int:
        if spec.name in fallbacks:
            rprint(f"[yellow]WARN[/yellow] {path}: model artifact not found ({where}); "
                   f"pipelines fall back to {fallbacks[spec.name]!r}")
            return 0
        rprint(f"[red]FAIL[/red] {path}: model artifact not found; {where}")
        return 1

    models_dir = stack.spec_dir("models")
    model_specs: list[tuple[Path, ModelSpec]] = []
    for path, data in load_dir(models_dir):
        spec = ModelSpec(**data)
        model_specs.append((path, spec))

        module = BACKEND_DEPENDENCIES.get(spec.backend)
        if module and not _module_available(module):
            errors += 1
            rprint(f"[red]FAIL[/red] {path}: backend dependency {module!r} is not installed")
        elif module is None:
            try:
                resolve_backend_class(spec.backend)
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
        if spec.backend == "onnxruntime" and spec.artifact.path:
            candidates = _artifact_candidates(spec.artifact.path, spec_dir=path.parent,
                                              model_repository=model_repository, stack=stack)
            if not any(candidate.exists() for candidate in candidates):
                errors += _artifact_missing(path, spec, f"tried {[str(c) for c in candidates]}")
        elif spec.backend == "triton" and spec.artifact.triton_model_name:
            artifact = model_repository / spec.artifact.triton_model_name / "1" / "model.onnx"
            if not artifact.exists():
                errors += _artifact_missing(path, spec, f"expected Triton artifact at {artifact}")

    for path, spec in pipeline_specs:
        for stage in ([spec.source] if spec.source else []) + spec.stages + spec.sinks:
            try:
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
    """List domain packs discovered under examples/ and domains/."""
    from inferops.core.domain import discover_domains, load_domain_manifest

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
    from inferops.core.domain import manifest_env

    stack = _domain_stack(domain, domain_root)
    env = manifest_env(stack.manifest(), stack.domain_root, stack.model_repository(settings.model_repository))
    for key, value in env.items():
        print(f"{key}={value}")


@domain_app.command("setup-assets")
def domain_setup_assets(
    domain: str | None = typer.Option(None, "--domain"),
    domain_root: str | None = typer.Option(None),
    force: bool = typer.Option(False, help="Run installers even when the asset exists"),
):
    """Run the manifest's asset installers for artifacts that are missing."""
    from inferops.core.plugins import load_object

    stack = _domain_stack(domain, domain_root)
    manifest = stack.manifest()
    root = stack.domain_root
    failures = 0
    if not manifest.assets:
        rprint(f"[cyan]{manifest.name}[/cyan]: no assets declared in domain.yaml")
    for asset in manifest.assets:
        target = root / asset.path
        if target.exists() and not force:
            rprint(f"[green]present[/green] {target}")
            continue
        if not asset.installer:
            level = "red" if asset.required else "yellow"
            rprint(f"[{level}]missing[/{level}] {target} (no installer declared)")
            failures += int(asset.required)
            continue
        rprint(f"[cyan]installing[/cyan] {asset.description or asset.path}")
        try:
            if ":" in asset.installer and not asset.installer.endswith(".py"):
                load_object(asset.installer)(target)
            else:
                subprocess.check_call([sys.executable, str(root / asset.installer)], cwd=root)
        except Exception as exc:
            level = "red" if asset.required else "yellow"
            rprint(f"[{level}]failed[/{level}] {asset.path}: {exc}")
            failures += int(asset.required)
    raise typer.Exit(1 if failures else 0)


@domain_app.command("init")
def domain_init(
    name: str = typer.Argument(..., help="Domain name (Python identifier), e.g. ppe_detection"),
    parent: str = typer.Option("examples", help="Directory to create the pack in (examples/ or domains/)"),
    labels: str = typer.Option("person,helmet,vest", help="Comma-separated detection classes"),
    label: str | None = typer.Option(None, help="Human-readable label"),
):
    """Scaffold a runnable domain pack (synthetic detector, fusion, alerts, evidence)."""
    from inferops.core.domain import scaffold_domain

    root = Path(settings.configs_dir).resolve().parent / parent / name
    classes = [c.strip() for c in labels.split(",") if c.strip()]
    created = scaffold_domain(root, name, classes, label=label)
    for path in created:
        rprint(f"[green]created[/green] {path}")
    rprint(f"\nNext: [cyan]make validate DOMAIN={name}[/cyan] then [cyan]make serve-mock DOMAIN={name}[/cyan]")


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
def pipelines_run(name: str, input_path: str = typer.Argument(..., help="Image or video file")):
    """Run a pipeline once on an image (or a video, by extension) and print the results."""
    async def _go():
        from inferops.alerts import autoload as autoload_alerts
        from inferops.core.types import StageItem, now_ms
        from inferops.fusion import autoload as autoload_fusion
        from inferops.pipeline import autoload, get_pipeline_store
        from inferops.registry import autoload as autoload_models
        await autoload_models()
        await autoload()
        await autoload_fusion()
        await autoload_alerts()
        runner = await get_pipeline_store().get_runner(name)
        item = StageItem(stream_id="cli", ts_ms=now_ms())
        src = Path(input_path)
        if src.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            item.payload["video_path"] = str(src)
        else:
            item.payload["image_bytes"] = src.read_bytes()
        outs = await runner.run_one(item)
        for o in outs:
            d = o.model_dump(mode="json", exclude={"payload"})
            d["payload"] = {k: v for k, v in o.payload.items()
                            if k not in {"image_bytes", "image_np", "feeds", "frame_meta"}}
            rprint(json.dumps(d, indent=2, default=str))
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
