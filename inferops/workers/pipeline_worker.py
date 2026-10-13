"""Run a configured pipeline as a long-running worker."""
from __future__ import annotations

import argparse
import asyncio

from inferops.obs import get_logger
from inferops.pipeline.store import get_pipeline_store
from inferops.workers.bootstrap import autoload_all

log = get_logger(__name__)


async def run_pipeline_worker(pipeline: str, *, max_concurrency: int | None = None) -> None:
    await autoload_all()
    runner = await get_pipeline_store().get_runner(pipeline)
    log.info("pipeline_worker_starting", pipeline=pipeline)
    await runner.start(max_concurrency=max_concurrency)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an InferOps pipeline worker")
    parser.add_argument("pipeline")
    parser.add_argument("--max-concurrency", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(run_pipeline_worker(args.pipeline, max_concurrency=args.max_concurrency))


if __name__ == "__main__":
    main()
