"""Consume StageItems from one topic, run a pipeline, publish output items."""
from __future__ import annotations

import argparse
import asyncio

import orjson

from inferops.adapters.transport import get_transport
from inferops.core.types import StageItem
from inferops.obs import get_logger
from inferops.pipeline.store import get_pipeline_store
from inferops.workers.bootstrap import autoload_all

log = get_logger(__name__)


async def run_topic_worker(
    pipeline: str,
    *,
    input_topic: str,
    output_topic: str,
    group: str | None = None,
) -> None:
    await autoload_all()
    transport = get_transport()
    await transport.admin_ensure_topic(input_topic)
    await transport.admin_ensure_topic(output_topic)
    consumer = await transport.consumer(group or f"inferops.{pipeline}.worker")
    producer = await transport.producer(f"topic-worker-{pipeline}")
    await consumer.subscribe([input_topic], group or f"inferops.{pipeline}.worker")
    runner = await get_pipeline_store().get_runner(pipeline)
    log.info("topic_worker_starting", pipeline=pipeline, input=input_topic, output=output_topic)
    try:
        async for msg in consumer.stream():
            try:
                payload = orjson.loads(msg.value)
                item = StageItem(**payload) if isinstance(payload, dict) and "id" in payload else StageItem(payload=payload)
                outs = await runner.run_one(item)
                for out in outs:
                    key = (out.stream_id or item.stream_id or "").encode() or None
                    await producer.send(output_topic, orjson.dumps(out.model_dump(mode="json")), key=key)
                await consumer.commit(msg)
            except Exception as exc:
                log.exception("topic_worker_item_failed", err=str(exc))
    finally:
        await consumer.close()
        await producer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an InferOps topic worker")
    parser.add_argument("pipeline")
    parser.add_argument("--input-topic", required=True)
    parser.add_argument("--output-topic", required=True)
    parser.add_argument("--group", default=None)
    args = parser.parse_args()
    asyncio.run(
        run_topic_worker(
            args.pipeline,
            input_topic=args.input_topic,
            output_topic=args.output_topic,
            group=args.group,
        )
    )


if __name__ == "__main__":
    main()
