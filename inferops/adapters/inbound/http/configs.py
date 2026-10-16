"""Project editing at the filesystem boundary; sessions retain compiled revisions."""
from __future__ import annotations

import asyncio
import os
import re
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from inferops.adapters.outbound.projects.loader import ProjectLoader
from inferops.adapters.outbound.projects.yaml import parse_yaml
from inferops.adapters.outbound.stages.factory import stage_factory
from inferops.application.models import ModelPool

router = APIRouter()


class Update(BaseModel):
    content: str
    reload: bool = True


def _loader(engine):
    definition = engine.application.definition
    factory = definition.backend_factory
    return ProjectLoader(definition.root, configs_dir=definition.source_dirs['pipelines'].parent,
                         mode=definition.mode, triton_url=getattr(factory, 'triton_url', 'localhost:8001'),
                         artifact_root=definition.source_dirs.get('artifact_root'))


def _path(engine, section, name):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in {'.','..'}:
        raise HTTPException(400, 'invalid configuration name')
    definition = engine.application.definition
    if section == 'project' and definition.root:
        path = definition.root / 'project.yaml'
    elif section == 'platform':
        path = definition.source_dirs['pipelines'].parent / 'platform.yaml'
    elif section in {'models','pipelines','fusion','alerts'}:
        base = definition.source_dirs[section]
        nested = base / name / 'model.yaml'
        path = nested if section == 'models' and nested.is_file() else base / f'{name}.yaml'
    else:
        raise HTTPException(400, 'unknown configuration section')
    if not path.is_file():
        raise HTTPException(404, 'configuration not found')
    return path


def _entry(path, section):
    data = parse_yaml(path.read_text())
    name = path.parent.name if path.name == 'model.yaml' else path.stem
    return {'section':section,'name':name,'path':str(path),'title':data.get('name',name),'content':path.read_text()}


@router.get('/configs')
async def list_configs(request: Request):
    engine = request.app.state.engine
    definition = engine.application.definition
    files = []
    for section in ['models','pipelines','fusion','alerts']:
        folder = definition.source_dirs[section]
        paths = sorted(folder.glob('*.yaml'))
        if section == 'models':
            paths += sorted(folder.glob('*/model.yaml'))
        files += [_entry(path,section) for path in paths]
    if definition.root and (definition.root / 'project.yaml').is_file():
        files.append(_entry(definition.root / 'project.yaml','project'))
    platform = definition.source_dirs['pipelines'].parent / 'platform.yaml'
    if platform.is_file():
        files.append(_entry(platform,'platform'))
    domains = []
    for pipeline in definition.pipelines.values():
        references = {}
        for stage in pipeline.stages:
            for key in ['model','fusion','alerts']:
                if key in stage.params:
                    references[key] = stage.params[key]
        # Runtime names are resolved in plans; config editing uses logical model folders.
        for logical, variants in definition.resolver.variants.items():
            if references.get('model') in variants:
                references['model'] = logical
        domains.append({'id':pipeline.name,'label':pipeline.tags.get('label',pipeline.name.replace('_',' ').title()),
                        'pipeline':pipeline.name,'source_stage':pipeline.source.stage if pipeline.source else None,
                        'config_names':{'pipeline':pipeline.name,**references},
                        'capabilities':['camera','image','video','rtsp'],
                        'stages':[s.model_dump(mode='json') for s in pipeline.stages]})
    return {'configs_dir':str(definition.source_dirs['pipelines'].parent),'domain_root':str(definition.root) if definition.root else None,
            'domain_pack':definition.manifest.model_dump(mode='json') if definition.manifest else None,
            'active_domain':definition.name if definition.root else None,'files':files,'domains':domains,
            'sections':['platform','models','pipelines','fusion','alerts'] + (['project'] if definition.root else []),'revision':definition.revision}


@router.get('/configs/{section}/{name}')
async def get_config(request: Request, section: str, name: str):
    return _entry(_path(request.app.state.engine,section,name),section)


async def _activate(engine, definition):
    from inferops.bootstrap import build_annotators
    application = engine.application
    previous = application.models
    models = ModelPool(definition, max_concurrency=previous.max_concurrency)
    annotators = build_annotators(definition)
    from inferops.application.projects import activate_revision
    await activate_revision(application, definition=definition, models=models,
                            stage_factory=stage_factory(definition), annotators=annotators)


@router.put('/configs/{section}/{name}')
async def update_config(request: Request, section: str, name: str, payload: Update):
    engine = request.app.state.engine
    lock = getattr(request.app.state,'config_lock',None)
    if lock is None:
        lock = request.app.state.config_lock = asyncio.Lock()
    async with lock:
        path = _path(engine,section,name)
        data = parse_yaml(payload.content)
        if not isinstance(data,dict):
            raise HTTPException(400,'configuration must be a mapping')
        candidate = _loader(engine).load(overrides={path:data})
        previous = engine.application.definition
        def startup_contract(definition):
            return sorted((spec.name, spec.backend, spec.artifact.model_dump(),
                           [tensor.model_dump() for tensor in spec.inputs],
                           [tensor.model_dump() for tensor in spec.outputs])
                          for spec in definition.models.values())
        preparation_required = section == 'project' or (section == 'models' and startup_contract(previous) != startup_contract(candidate))
        def commit():
            with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False,encoding='utf-8') as output:
                output.write(payload.content)
                output.flush()
                os.fsync(output.fileno())
                temporary = output.name
            try:
                os.replace(temporary,path)
                descriptor = os.open(path.parent,os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            finally:
                Path(temporary).unlink(missing_ok=True)
        await asyncio.to_thread(commit)
        if payload.reload and not preparation_required and section != 'platform':
            await _activate(engine,candidate)
    count = len(candidate.models) if section == 'models' else len(getattr(candidate, section, {})) if section in {'pipelines','fusion','alerts'} else 0
    result = {'section':section,'reloaded':0 if preparation_required else count,
              'restart_required':preparation_required or section=='platform'}
    return {**_entry(path,section),'saved':True,'revision':candidate.revision,
            'restart_required':result['restart_required'],'preparation_required':preparation_required,
            'reload':result if payload.reload else None}


@router.post('/configs/reload')
@router.post('/models/reload')
@router.post('/pipelines/reload')
async def reload_configs(request: Request):
    engine = request.app.state.engine
    candidate = _loader(engine).load()
    await _activate(engine,candidate)
    return {'revision':candidate.revision,'reloaded':len(candidate.pipelines),'restart_required':False}
