"""Consume StageItems from one topic, run a pipeline, publish output items."""
from __future__ import annotations

import argparse
import asyncio
import os
import time
from contextlib import asynccontextmanager

import orjson

from inferops.adapters.inbound.workers.bootstrap import server_engine
from inferops.adapters.outbound.observability import get_logger
from inferops.domain.types import StageItem

log = get_logger(__name__)


async def run_topic_worker(
    pipeline: str,
    *,
    input_topic: str,
    output_topic: str | None,
    group: str | None = None,
    engine=None,
    manage_engine: bool = True,
) -> None:
    from inferops.adapters.inbound.workers.execution import item_frames, require_durable_processing

    engine = engine or server_engine()
    group = group or f"inferops.{pipeline}.worker"
    max_sessions = int(os.environ.get("INFEROPS_WORKER_MAX_SESSIONS", "64"))
    idle_s = float(os.environ.get("INFEROPS_WORKER_SESSION_IDLE_S", "300"))
    if max_sessions < 1 or idle_s <= 0:
        raise ValueError("worker session limits must be positive")

    @asynccontextmanager
    async def lifetime():
        if manage_engine:
            async with engine:
                yield
        else:
            yield

    async with lifetime():
        transport = engine.application.transport
        if transport is None:
            raise ValueError("topic worker requires an explicitly configured transport")
        await transport.admin_ensure_topic(input_topic)
        if output_topic:
            await transport.admin_ensure_topic(output_topic)
        consumer = await transport.consumer(group)
        sessions = {}
        try:
            await consumer.subscribe([input_topic], group)
            log.info("topic_worker_starting", pipeline=pipeline, input=input_topic, output=output_topic)
            async for msg in consumer.stream():
                data = orjson.loads(msg.value)
                if not isinstance(data, dict):
                    raise ValueError("broker input must be an object")
                item = StageItem(**data) if "payload" in data else StageItem(payload=data)
                if msg.partition is not None and msg.offset is not None:
                    event_id = f"{msg.topic}:{msg.partition}:{msg.offset}"
                else:
                    event_id = data.get("event_id") or data.get("id")
                    if not event_id:
                        raise ValueError("broker input requires partition/offset or a stable event_id")
                now = time.monotonic()
                for key, (session, _sequence) in list(sessions.items()):
                    if now - session.last_active > idle_s:
                        await session.close("idle_timeout")
                        sessions.pop(key)
                external_key = msg.key.hex() if msg.key is not None else item.stream_id
                identity = (msg.topic, msg.partition, external_key)
                if identity not in sessions:
                    if len(sessions) >= max_sessions:
                        raise RuntimeError("broker session capacity exhausted")
                    session = await engine.open_session(pipeline, stream_name=item.stream_id or external_key,
                                                        owner_kind="broker")
                    sessions[identity] = session, 0
                session, sequence = sessions[identity]
                context = {"consumer": group, "event_id": str(event_id), "output_topic": output_topic,
                           "output_key": msg.key.decode("utf-8", errors="replace")
                           if msg.key is not None else item.stream_id}
                count = 0
                async for frame in item_frames(engine, item, sequence=sequence, broker_event=context):
                    outputs = await session.process(frame)
                    require_durable_processing(outputs)
                    count += 1
                if count != 1:
                    raise ValueError("broker message must produce exactly one frame")
                sessions[identity] = session, sequence + 1
                # Only this message's partition/offset is acknowledged. The
                # application has committed inbox effects and output outbox first.
                await consumer.commit(msg)
        except Exception as exc:
            # Stop instead of allowing a later offset to skip a failed message.
            log.exception("topic_worker_item_failed", err=str(exc))
            raise
        finally:
            await consumer.close()
            await asyncio.gather(*(session.close("worker_shutdown")
                                   for session, _sequence in sessions.values()))


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
