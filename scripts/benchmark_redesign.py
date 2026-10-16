"""Bounded CPU live replay with real models and evidence capture enabled.

Run with explicit INFEROPS_TEST_PG_URL / INFEROPS_TEST_S3_ENDPOINT. Only a
uniquely named benchmark schema/bucket are removed afterwards. No model or
project files are changed. Reply latency excludes HTTP/network transport.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import resource
import tempfile
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from uuid import uuid4

import av
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from inferops import Engine
from inferops.adapters.inbound.sources import process_live_frames, source_frames
from inferops.adapters.outbound.messaging.delivery import DeliveryHub, ServerDelivery
from inferops.adapters.outbound.messaging.inproc import InProcTransport
from inferops.adapters.outbound.persistence.sql import SqlRepository
from inferops.adapters.outbound.stages.base import Stage
from inferops.adapters.outbound.storage.s3 import S3Storage
from inferops.domain.evidence import EvidenceConfig
from inferops.domain.pipeline import StageDecl


def percentiles(values):
    ordered = sorted(values)
    return {f"p{p}": ordered[min(len(ordered) - 1, int((len(ordered) - 1) * p / 100))]
            if ordered else None for p in (50, 95, 99)}


class IncidentBurst(Stage):
    async def process(self, item):
        # Separate labels keep synthetic load distinguishable from real rules.
        second = item.payload["media_time_ns"] / 1e9
        if 5 <= second < 6:
            item.payload.setdefault("alert_candidates", []).append(
                {"type": f"benchmark_burst_{item.seq}", "ts_ms": item.ts_ms,
                 "payload": {"benchmark_injected": True}})
        return item


async def benchmark(args):
    url = os.environ["INFEROPS_TEST_PG_URL"]
    endpoint = os.environ["INFEROPS_TEST_S3_ENDPOINT"]
    if not url.startswith("postgresql"):
        raise ValueError("benchmark requires an explicit PostgreSQL test endpoint")
    unique = uuid4().hex
    schema, bucket = f"inferops_bench_{unique}", f"inferops-bench-{unique}"
    repository = SqlRepository(url)
    for table in repository.metadata.tables.values():
        table.schema = schema
    storage = S3Storage(endpoint=endpoint, bucket=bucket,
                        access_key=os.environ.get("INFEROPS_TEST_S3_ACCESS_KEY", "minio"),
                        secret_key=os.environ.get("INFEROPS_TEST_S3_SECRET_KEY", "minio12345"))
    admin = create_async_engine(url.replace("postgresql://", "postgresql+asyncpg://", 1))
    config = EvidenceConfig(disk_reserve_bytes=0)
    latencies, completions, accepted = [], {}, {}
    peaks = {"spool_bytes": 0, "ring_bytes": 0, "pending_outbox": 0, "outbox_lag_seconds": 0.0}
    report = {"capture_enabled": True, "purpose": "bounded smoke benchmark; not production capacity acceptance",
              "baseline_comparison": "not measured; run the compatibility release on the same host before cutover",
              "delivery_transport": "in-process transport; PostgreSQL and MinIO are real services",
              "runtime": "cpu", "inference_threads": args.threads, "concurrency": args.concurrency,
              "replay_fps_per_stream": args.fps, "duration_seconds": args.seconds,
              "latency_scope": "decoded frame arrival through shared application reply; excludes network",
              "synthetic_incidents": "additional benchmark_burst_<sequence> candidates at media seconds 5-6",
              "configuration_revision": None, "evidence_config": asdict(config),
              "host": {"platform": platform.platform(), "cpu_count": os.cpu_count()},
              "media": {"path": str(args.media), "sha256": hashlib.sha256(args.media.read_bytes()).hexdigest()}}
    async with admin.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    try:
        with tempfile.TemporaryDirectory(prefix="inferops-benchmark-") as scratch:
            selected = Engine.from_project(args.project, repository=repository, storage=storage, evidence=True,
                spool_path=Path(scratch) / "spool", evidence_config=config,
                backend_options={"threads": args.threads}, transport=InProcTransport())
            application = selected.application
            definition = application.definition
            report["configuration_revision"] = definition.revision
            report["models"] = {name: hashlib.sha256(Path(spec.artifact.path).read_bytes()).hexdigest()
                                for name, spec in definition.models.items()
                                if spec.artifact.path and Path(spec.artifact.path).is_file()}
            plan = definition.pipelines[args.pipeline].model_copy(deep=True)
            index = next(i for i, stage in enumerate(plan.stages) if stage.stage == "alert")
            plan.stages.insert(index, StageDecl(name="benchmark_burst", stage="benchmark_burst"))
            application.definition = replace(definition, pipelines={**definition.pipelines, args.pipeline: plan})
            factory = application.stage_factory
            application.stage_factory = lambda declaration: (IncidentBurst() if declaration.stage == "benchmark_burst"
                                                               else factory(declaration))
            application.delivery = ServerDelivery(DeliveryHub(), transport=application.transport, definitions=definition)
            async with selected:
                warm = await selected.open_session(args.pipeline, stream_name="warmup")
                async for frame in source_frames(args.media):
                    await warm.process(frame)
                    break
                await warm.close()
                await application.evidence.wait_idle()
                with av.open(str(args.media)) as video:
                    rate = float(video.streams.video[0].average_rate)
                every_n = max(1, round(rate / args.fps))
                report["source_fps"] = rate
                report["sampled_source_fps"] = rate / every_n
                report["sample_every_n"] = every_n

                async def replay():
                    sequence, offset = 0, 0
                    origin = time.monotonic()
                    while True:
                        last_media = 0
                        async for frame in source_frames(args.media, every_n=every_n):
                            media = offset + frame.media_time_ns
                            if media >= args.seconds * 1e9:
                                return
                            await asyncio.sleep(max(0, origin + media / 1e9 - time.monotonic()))
                            last_media = frame.media_time_ns
                            yield replace(frame, sequence=sequence, media_time_ns=media)
                            sequence += 1
                        offset += last_media + int(1e9 * every_n / rate)

                async def completed(outputs, latency):
                    latencies.append(latency * 1000)
                    for output in outputs:
                        if output.payload.get("incident_error"):
                            raise RuntimeError(output.payload["incident_error"])
                        for record in output.payload.get("evidence", []):
                            accepted.setdefault(record["id"], time.monotonic())

                async def sample():
                    while True:
                        peaks["spool_bytes"] = max(peaks["spool_bytes"], await application.evidence.spool.usage_bytes())
                        peaks["ring_bytes"] = max(peaks["ring_bytes"], application.evidence.diagnostics()["ring_bytes"])
                        diagnostics = await repository.diagnostics()
                        peaks["pending_outbox"] = max(peaks["pending_outbox"], diagnostics["pending_outbox"])
                        peaks["outbox_lag_seconds"] = max(peaks["outbox_lag_seconds"], diagnostics["oldest_outbox_age_seconds"])
                        for record in await repository.list_evidence(limit=10000):
                            if record["state"] != "Pending" and record["id"] in accepted:
                                completions.setdefault(record["id"], time.monotonic() - accepted[record["id"]])
                        await asyncio.sleep(.1)

                observer = asyncio.create_task(sample())
                started = time.monotonic()
                sessions = [await selected.open_session(args.pipeline, stream_name="same-external-name")
                            for _ in range(args.concurrency)]
                try:
                    drops = await asyncio.gather(*(process_live_frames(session, replay(), on_result=completed)
                                                   for session in sessions))
                    for session in sessions:
                        await session.close()
                    await application.evidence.wait_idle()
                    await application.relay.flush()
                    await asyncio.sleep(.25)
                    session_ids = {session.id for session in sessions}
                    records = [record for record in await repository.list_evidence(limit=10000)
                               if record["session_id"] in session_ids]
                    types = {row["id"]: row["type"] for row in await repository.list_incidents(limit=10000)}
                    report.update(elapsed_seconds=time.monotonic() - started, frames_processed=len(latencies),
                                  frames_dropped=sum(drops), reply_latency_ms=percentiles(latencies),
                                  evidence_completion_seconds=percentiles(list(completions.values())),
                                  evidence_states=dict(Counter(record["state"] for record in records)),
                                  evidence_reasons=dict(Counter(reason for record in records
                                                                for reason in record.get("errors", []))),
                                  evidence_outcomes=[{**{key: record.get(key) for key in ("state", "media_time_ns", "request_start_ns",
                                      "request_end_ns", "actual_start_ns", "actual_end_ns", "missing_intervals", "errors")},
                                      "incident_type": types[record["incident_id"]]} for record in records],
                                  warmup_excluded_from_evidence_counts=True,
                                  peaks=peaks, peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                                  final_delivery=await repository.diagnostics())
                finally:
                    observer.cancel()
                    await asyncio.gather(observer, return_exceptions=True)
    finally:
        await repository.close()
        try:
            await storage.start()
            def cleanup():
                for page in storage.client.get_paginator("list_objects_v2").paginate(Bucket=bucket):
                    keys = [{"Key": row["Key"]} for row in page.get("Contents", [])]
                    if keys:
                        storage.client.delete_objects(Bucket=bucket, Delete={"Objects": keys})
                storage.client.delete_bucket(Bucket=bucket)
            await asyncio.to_thread(cleanup)
        finally:
            await storage.close()
            async with admin.begin() as connection:
                await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
            await admin.dispose()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("projects/driver_behaviour"))
    parser.add_argument("--pipeline", default="driver_behaviour")
    parser.add_argument("--media", type=Path, default=Path("research/driver_training/test.mp4"))
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--fps", type=float, default=5)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path(".run/verification/redesign-benchmark.json"))
    options = parser.parse_args()
    if min(options.seconds, options.fps, options.concurrency, options.threads) <= 0:
        parser.error("duration, FPS, concurrency and threads must be positive")
    asyncio.run(benchmark(options))
