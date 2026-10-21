"""Persistent, bounded export jobs; heavy optional dependencies stay out of the API."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path


class CheckpointJobs:
    def __init__(self, root, *, python=None, timeout=900):
        self.root = Path(root)
        self.python = python or os.getenv("INFEROPS_EXPORT_PYTHON")
        self.timeout = timeout
        self.tasks = {}
        self.records = {}
        self.lock = asyncio.Lock()

    def save(self, record):
        path = self.root / record["id"] / "job.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(path)

    async def start(self):
        self.root.mkdir(parents=True, exist_ok=True)
        for path in self.root.glob("*/job.json"):
            record = json.loads(path.read_text())
            if record["status"] in {"queued", "running"}:
                record.update(status="interrupted", error="Export interrupted by server restart; submit again")
                self.save(record)
            self.records[record["id"]] = record

    async def submit(self, checkpoint, *, variant="auto", resolution=576, max_batch=4):
        if not self.python or not Path(self.python).is_file():
            raise ValueError("Set INFEROPS_EXPORT_PYTHON to the isolated RF-DETR export environment")
        if variant not in {"auto", "nano", "small", "medium", "large", "base"}:
            raise ValueError("Unsupported RF-DETR architecture")
        if max_batch not in {2, 4, 8} or not 32 <= resolution <= 2048:
            raise ValueError("Invalid export batch/resolution")
        if len(self.tasks) >= 2:
            raise ValueError("Export queue is full; wait for the current jobs")
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_file() or checkpoint.suffix.lower() not in {".pt", ".pth"}:
            raise ValueError("Supply an existing .pth/.pt RF-DETR checkpoint")
        job_id = uuid.uuid4().hex
        folder = self.root / job_id
        folder.mkdir()
        record = dict(id=job_id, status="queued", checkpoint=checkpoint.name,
                      variant=variant, resolution=resolution, max_batch=max_batch)
        self.records[job_id] = record
        self.save(record)
        task = asyncio.create_task(self._run(record, checkpoint))
        self.tasks[job_id] = task
        task.add_done_callback(lambda _: self.tasks.pop(job_id, None))
        return record.copy()

    async def _run(self, record, checkpoint):
        process = None
        folder = self.root / record["id"]
        try:
            async with self.lock:
                record["status"] = "running"
                self.save(record)
                with (folder / "export.log").open("wb") as log:
                    launch = asyncio.create_task(asyncio.create_subprocess_exec(
                        str(Path(self.python).absolute()), "-m",
                        "inferops.adapters.outbound.inference.checkpoint_export",
                        "--checkpoint", str(checkpoint), "--output", str(folder.resolve()),
                        "--variant", record["variant"], "--resolution", str(record["resolution"]),
                        "--max-batch", str(record["max_batch"]), stdout=log,
                        stderr=asyncio.subprocess.STDOUT,
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"},
                    ))
                    try:
                        process = await asyncio.shield(launch)
                    except asyncio.CancelledError:
                        process = await launch
                        raise
                    code = await asyncio.wait_for(process.wait(), self.timeout)
                if code:
                    raise ValueError("Conversion failed; inspect the export log")
                report = json.loads((folder / "report.json").read_text())
                if report.get("status") != "validated" or not (folder / "model.onnx").is_file():
                    raise ValueError("Worker did not produce a validated artifact")
                record.update(status="validated", report=report,
                              artifact_path=str((folder / "model.onnx").resolve()))
        except asyncio.CancelledError:
            record.update(status="interrupted", error="Server stopped during export; submit again")
            raise
        except Exception as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
        finally:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
            self.save(record)

    async def close(self):
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

