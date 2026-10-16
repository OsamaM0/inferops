"""Stable Python facade. Importing this module opens no resources."""
from __future__ import annotations

from inferops.application.engine import PipelineApplication
from inferops.domain.processing import RunSummary


class Engine:
    def __init__(self, application: PipelineApplication):
        self.application = application

    @classmethod
    def from_project(cls, root, *, runtime="cpu", repository=None, storage=None,
                     evidence=False, spool_path=None, evidence_config=None, **options):
        from inferops.bootstrap import build_engine
        return build_engine(root, runtime=runtime, repository=repository, storage=storage,
                            evidence=evidence, spool_path=spool_path, evidence_config=evidence_config, **options)

    async def predict(self, model, image):
        return await self.application.models.predict(model, image)

    async def open_session(self, pipeline, *, stream_name="", owner_kind="python"):
        return await self.application.open_session(pipeline, stream_name=stream_name, owner_kind=owner_kind)

    async def run(self, pipeline, source, *, stream_name="", owner_kind="python"):
        session = await self.open_session(pipeline, stream_name=stream_name, owner_kind=owner_kind)
        summary = RunSummary(pipeline, session.id)
        try:
            decoder = next((d for d in session.spec.stages if d.stage in {"decode_any", "decode_video"}), None)
            every_n = decoder.params.get("every_n", 1) if decoder else 1
            async for frame in self.application.source_factory(source, every_n=every_n):
                outputs = await session.process(frame)
                summary.add(outputs)
        finally:
            await session.close()
        if self.application.evidence:
            await self.application.evidence.wait_idle()
        result = summary.asdict()
        records = await self.application.repository.list_evidence(session_id=session.id)
        result["evidence"] = records
        result["evidence_uris"] = sorted({asset["uri"] if isinstance(asset, dict) else asset
                                         for record in records for asset in record.get("artifacts", {}).values()})
        result["execution_mode"] = getattr(self.application.definition, "mode", "cpu")
        result["alerts"] = [a for a in await self.application.repository.list_incidents()
                            if a.get("payload", {}).get("session_id") == session.id]
        result["fusion_events"] = [e for e in await self.application.repository.list_events(pipeline)
                                   if e.get("payload", {}).get("session_id") == session.id]
        return result

    async def start(self):
        await self.application.start()
        return self

    async def close(self):
        await self.application.close()

    async def __aenter__(self):
        return await self.start()

    async def __aexit__(self, *exc):
        await self.close()


class Model:
    """Public model handle backed by the shared application model pool."""
    def __init__(self, spec, *, base_dir=None, backend_factory=None, object_loader=None):
        from inferops.bootstrap import build_model_pool
        self._pool = build_model_pool(spec, base_dir=base_dir, backend_factory=backend_factory,
                                      object_loader=object_loader)
        self.spec = self._pool.get_spec(spec.name)
        self.base_dir = self._pool.source_dir(spec.name)
        self._owns_pool = True

    @classmethod
    def _from_pool(cls, pool, name):
        handle = cls.__new__(cls)
        handle._pool, handle._owns_pool = pool, False
        handle.spec, handle.base_dir = pool.get_spec(name), pool.source_dir(name)
        return handle

    @classmethod
    def from_yaml(cls, path):
        from pathlib import Path

        from inferops.adapters.outbound.projects.models import load_model_file
        path = Path(path).resolve()
        return cls(load_model_file(path), base_dir=path.parent)

    async def load(self):
        if self._owns_pool and self._pool.closed:
            from inferops.application.models import ModelPool
            self._pool = ModelPool(self._pool.definition)
        await self._pool.get_backend(self.spec.name)
        return self

    async def predict(self, image, *, stream_id="local"):
        from inferops.domain.types import now_ms
        await self.load()
        return await self._pool.predict(self.spec.name, image, stream_id=stream_id, ts_ms=now_ms())

    async def infer(self, inputs):
        await self.load()
        return await self._pool.infer(self.spec.name, inputs)

    async def close(self):
        """Close an independent pool; project sessions own their shared pool."""
        if self._owns_pool:
            await self._pool.close()

    async def __aenter__(self):
        return await self.load()

    async def __aexit__(self, *exc):
        await self.close()


class InferenceSession:
    """Public collection of model handles with shared application resources."""
    def __init__(self, models):
        if not models:
            raise ValueError("an inference session requires at least one model")
        self.models = dict(models)
        self._pool = None

    @classmethod
    def from_project(cls, root, *, models=None, runtime="cpu"):
        from inferops.bootstrap import build_model_session
        pool, names = build_model_session(root, models=models, runtime=runtime)
        session = cls({name: Model._from_pool(pool, resolved) for name, resolved in names.items()})
        session._pool = pool
        return session

    async def predict(self, image, *, stream_id="local", parallel=False):
        import asyncio
        if parallel:
            results = await asyncio.gather(*(model.predict(image, stream_id=stream_id)
                                             for model in self.models.values()), return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            return dict(zip(self.models, results, strict=True))
        return {name: await model.predict(image, stream_id=stream_id) for name, model in self.models.items()}

    async def close(self):
        import asyncio
        if self._pool is not None:
            await self._pool.close()
        else:
            results = await asyncio.gather(*(model.close() for model in self.models.values()), return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    raise result

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()
