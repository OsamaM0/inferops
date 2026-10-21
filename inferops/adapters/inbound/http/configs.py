"""Project editing at the filesystem boundary; sessions retain compiled revisions."""
from __future__ import annotations

import asyncio
import hashlib
import os
import re
import tempfile
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, model_validator

from inferops.adapters.outbound.projects.configuration import (
    SECTIONS,
    configuration_paths,
    configuration_schema,
    validate_document,
)
from inferops.adapters.outbound.projects.loader import ProjectLoader
from inferops.adapters.outbound.projects.yaml import parse_yaml
from inferops.adapters.outbound.stages.factory import stage_factory
from inferops.application.models import ModelPool

router = APIRouter()


class Update(BaseModel):
    content: str | None = None
    data: dict | None = None
    reload: bool = True
    version: str | None = None
    validate_only: bool = False

    @model_validator(mode="after")
    def one_document(self):
        if (self.content is None) == (self.data is None):
            raise ValueError("provide exactly one of content or data")
        return self


def _loader(engine):
    definition = engine.application.definition
    factory = definition.backend_factory
    return ProjectLoader(definition.root, configs_dir=definition.source_dirs.get(
                         'platform_configs', definition.source_dirs['pipelines'].parent),
                         mode=definition.mode, triton_url=getattr(factory, 'triton_url', 'localhost:8001'),
                         backend_options=getattr(factory, 'options', {}),
                         artifact_root=definition.source_dirs.get('artifact_root'))


def _path(engine, section, name, *, create=False):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in {'.','..'}:
        raise HTTPException(400, 'invalid configuration name')
    definition = engine.application.definition
    paths = configuration_paths(definition)
    if (section, name) in paths:
        if create:
            raise HTTPException(409, 'configuration already exists')
        path = paths[section, name]
    elif create and section in {'models','pipelines','fusion','alerts'}:
        base = definition.source_dirs[section]
        path = base / name / 'model.yaml' if section == 'models' and definition.root else base / f'{name}.yaml'
    else:
        raise HTTPException(404, 'configuration not found')
    # Never let a project symlink redirect an editor write outside its config root.
    base = definition.root if section == 'project' else (
        path.parent if section == 'platform' else definition.source_dirs[section])
    if not path.resolve().is_relative_to(base.resolve()):
        raise HTTPException(400, 'configuration is outside its section')
    return path


def _entry(path, section, *, editor=False):
    content = path.read_text(encoding="utf-8")
    # Keep ${ENV} references literal in forms and on disk.
    data = parse_yaml(content, interpolate=False)
    if not isinstance(data, dict):
        raise HTTPException(400, 'configuration must be a mapping')
    name = path.parent.name if path.name == 'model.yaml' else path.stem
    behavior = 'live' if section in {'pipelines', 'fusion', 'alerts'} else 'conditional' if section == 'models' else 'restart'
    result = {'section':section,'name':name,'path':str(path),
              'title':data.get('label') or data.get('tags', {}).get('label') or data.get('name', name),
              'description':data.get('description') or data.get('tags', {}).get('description', ''),
              'logical_name':data.get('name', name), 'group': SECTIONS[section],
              'variants':list(data.get('variants', {})),
              'content':content, 'reload_behavior': behavior,
              'version':hashlib.sha256(content.encode()).hexdigest()}
    if editor:
        result.update(data=data, schema=configuration_schema(section, data))
    return result


def _entries(definition):
    entries = []
    for (section, name), path in configuration_paths(definition).items():
        entry = _entry(path, section)
        entry["name"] = name
        if section == "platform":
            entry["title"] = "Platform defaults" if name == "platform" else "Project platform overrides"
        entries.append(entry)
    return entries


@router.get('/configs')
async def list_configs(request: Request):
    engine = request.app.state.engine
    definition = engine.application.definition
    files = _entries(definition)
    by_logical = {(entry["section"], entry["logical_name"]): entry for entry in files}
    def linked(section, logical):
        entry = by_logical.get((section, logical))
        return {key: entry[key] for key in ("section", "name", "title")} if entry else None
    domains = []
    for pipeline in definition.pipelines.values():
        references = {}
        links = [linked("pipelines", pipeline.name)]
        for stage in ([pipeline.source] if pipeline.source else []) + pipeline.stages + pipeline.sinks:
            for key in ['model','fusion','alerts']:
                values = stage.params.get("models", []) if key == "model" else []
                if stage.params.get(key):
                    values = [stage.params[key], *values]
                for value in values:
                    logical = next((name for name, variants in definition.resolver.variants.items()
                                    if value in variants), value) if key == "model" else value
                    references.setdefault(key, logical)
                    item = linked("models" if key == "model" else key, logical)
                    if item and item not in links:
                        links.append(item)
        pipeline_link = linked("pipelines", pipeline.name)
        domains.append({'id':pipeline.name,'label':pipeline.tags.get('label',pipeline.name.replace('_',' ').title()),
                        'description':pipeline.tags.get('description', ''),
                        'pipeline':pipeline.name,'source_stage':pipeline.source.stage if pipeline.source else None,
                        'config_names':{'pipeline':pipeline_link["name"] if pipeline_link else pipeline.name,**references},
                        'config_links':[link for link in links if link],
                        'capabilities':['camera','image','video','rtsp'],
                        'stages':[s.model_dump(mode='json') for s in pipeline.stages]})
    return {'configs_dir':str(definition.source_dirs['pipelines'].parent),'domain_root':str(definition.root) if definition.root else None,
            'domain_pack':definition.manifest.model_dump(mode='json') if definition.manifest else None,
            'active_domain':definition.name if definition.root else None,'files':files,'domains':domains,
            'sections':['platform','models','pipelines','fusion','alerts'] + (['project'] if definition.root else []),
            'groups':SECTIONS,
            'precedence':['Platform defaults', 'Project platform overrides', 'Project settings', 'Environment variables'],
            'environment_overrides':sorted(key for key in os.environ if key.startswith('INFEROPS_') or key == 'APP_ENV'),
            'revision':definition.revision}


@router.get('/configs/schema/{section}')
async def config_schema(section: str, model_port: bool = False):
    if section not in SECTIONS:
        raise HTTPException(404, 'unknown configuration section')
    return configuration_schema(section, {"variants": {}} if model_port else {})


@router.get('/configs/{section}/{name}')
async def get_config(request: Request, section: str, name: str):
    entry = _entry(_path(request.app.state.engine,section,name),section,editor=True)
    entry["name"] = name
    return entry


async def _activate(engine, definition):
    from inferops.bootstrap import build_annotators
    application = engine.application
    previous = application.models
    models = ModelPool(definition, max_concurrency=previous.max_concurrency,
                       triton_concurrency=previous.triton_concurrency,
                       admission_pending=previous.admission_pending,
                       admission_queue_ms=previous.admission_queue_ms)
    annotators = build_annotators(definition)
    from inferops.application.projects import activate_revision
    await activate_revision(application, definition=definition, models=models,
                            stage_factory=stage_factory(definition), annotators=annotators)


def _restart_status(previous, candidate, section=None):
    def startup_contract(definition):
        return sorted((spec.name, spec.backend, spec.artifact.model_dump(),
                       [tensor.model_dump() for tensor in spec.inputs],
                       [tensor.model_dump() for tensor in spec.outputs])
                      for spec in definition.models.values())
    preparation = section == 'project' or startup_contract(previous) != startup_contract(candidate)
    restart = preparation or section == 'platform' or dict(previous.platform) != dict(candidate.platform)
    return preparation, restart or previous.manifest != candidate.manifest


@router.put('/configs/{section}/{name}')
@router.post('/configs/{section}/{name}', status_code=201)
async def update_config(request: Request, section: str, name: str, payload: Update):
    engine = request.app.state.engine
    lock = getattr(request.app.state,'config_lock',None)
    if lock is None:
        lock = request.app.state.config_lock = asyncio.Lock()
    async with lock:
        create = request.method == 'POST'
        path = _path(engine,section,name,create=create)
        if payload.version is not None and not create:
            version = hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()
            if version != payload.version:
                raise HTTPException(409, 'Configuration changed since it was opened. Reopen it before saving.')
        content = payload.content if payload.content is not None else yaml.safe_dump(
            payload.data, sort_keys=False, allow_unicode=True)
        data = parse_yaml(content)
        if not isinstance(data,dict):
            raise HTTPException(400,'configuration must be a mapping')
        validate_document(section, data)
        candidate = _loader(engine).load(overrides={path:data})
        from inferops.application.projects import validate_server_revision
        validate_server_revision(engine.application, candidate)
        previous = engine.application.definition
        preparation_required, restart_required = _restart_status(previous, candidate, section)
        if payload.validate_only:
            return {'valid':True, 'saved':False, 'restart_required':restart_required,
                    'preparation_required':preparation_required}
        def commit():
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False,encoding='utf-8') as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
                temporary = output.name
            try:
                if create:
                    try:
                        os.link(temporary,path)  # Fail rather than overwrite an external creator.
                    except FileExistsError as exc:
                        raise HTTPException(409, 'configuration already exists') from exc
                else:
                    os.replace(temporary,path)
                descriptor = os.open(path.parent,os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            finally:
                Path(temporary).unlink(missing_ok=True)
        await asyncio.to_thread(commit)
        activated = payload.reload and not restart_required
        if activated:
            await _activate(engine,candidate)
    count = len(candidate.models) if section == 'models' else len(getattr(candidate, section, {})) if section in {'pipelines','fusion','alerts'} else 0
    result = {'section':section,'reloaded':count if activated else 0,
              'restart_required':restart_required}
    return {**_entry(path,section,editor=True),'name':name,'saved':True,'revision':candidate.revision,
            'enabled':data.get('name') in candidate.pipelines if section == 'pipelines' else None,
            'restart_required':result['restart_required'],'preparation_required':preparation_required,
            'reload':result if payload.reload else None}


@router.post('/configs/reload')
@router.post('/models/reload')
@router.post('/pipelines/reload')
async def reload_configs(request: Request):
    engine = request.app.state.engine
    lock = getattr(request.app.state, 'config_lock', None)
    if lock is None:
        lock = request.app.state.config_lock = asyncio.Lock()
    async with lock:
        candidate = _loader(engine).load()
        preparation, restart = _restart_status(engine.application.definition, candidate)
        if not restart:
            await _activate(engine,candidate)
        return {'revision':candidate.revision,'reloaded':0 if restart else len(candidate.pipelines),
                'restart_required':restart, 'preparation_required':preparation}
