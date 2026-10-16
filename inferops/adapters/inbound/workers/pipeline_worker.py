"""Run a configured pipeline as a long-running worker."""
from __future__ import annotations

import argparse
import asyncio
from types import SimpleNamespace

from inferops.adapters.inbound.workers.bootstrap import server_engine
from inferops.adapters.outbound.observability import get_logger

log = get_logger(__name__)


async def run_pipeline_worker(pipeline: str, *, max_concurrency: int | None = None, engine=None) -> None:
    from inferops.adapters.inbound.workers.execution import process_source
    engine = engine or server_engine()
    if max_concurrency is not None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        # Backend capacity may grow; each stateful session still processes in order.
        engine.application.models.max_concurrency = max_concurrency
    async with engine:
        spec = engine.application.definition.pipelines[pipeline]
        if spec.source is None or spec.source.stage == "source_rest":
            raise ValueError("pipeline worker requires a configured streaming source")
        if spec.source.stage == "source_kafka":
            from inferops.adapters.inbound.workers.topic_worker import run_topic_worker
            params = spec.source.params
            await run_topic_worker(pipeline, input_topic=params["topic"],
                                   output_topic=params.get("output_topic"), group=params.get("group"),
                                   engine=engine, manage_engine=False)
            return
        source = engine.application.stage_factory(spec.source)
        await source.setup(SimpleNamespace(pipeline=pipeline, name=spec.source.name, params=spec.source.params,
                                            registry=engine.application.models, transport=engine.application.transport,
                                            extra={"owner": engine.application.owner}))
        log.info("pipeline_worker_starting", pipeline=pipeline)
        try:
            await process_source(engine, pipeline, source)
        finally:
            await source.finalize()
            await source.teardown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an InferOps pipeline worker")
    parser.add_argument("pipeline")
    parser.add_argument("--max-concurrency", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(run_pipeline_worker(args.pipeline, max_concurrency=args.max_concurrency))


if __name__ == "__main__":
    main()
