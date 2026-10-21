"""Recoverable, bounded local capture spool.

A manifest references only fsynced files. Its atomic replacement is the commit
point for a generation; orphan frame/temp files never imply completed capture.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from inferops.domain.evidence import EncodedFrame


class SpoolCapacityError(RuntimeError):
    reason_code = "spool_capacity"


class SpoolCorruptError(RuntimeError):
    reason_code = "spool_corrupt"


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class CrashSafeSpool:
    def __init__(self, path: str | Path, *, max_bytes: int = 10 * 1024**3,
                 disk_reserve_bytes: int = 1024**3):
        self.path = Path(path)
        self.max_bytes = max_bytes
        self.disk_reserve_bytes = disk_reserve_bytes
        self._lock = asyncio.Lock()
        # One owner per spool directory. Rebuild after restart, then update only
        # the files touched by a commit; never walk the whole tree per frame.
        self._sizes: dict[Path, int] | None = None
        self._bytes = 0

    async def _run(self, operation, *args):
        async with self._lock:
            task = asyncio.create_task(asyncio.to_thread(operation, *args))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                # A native filesystem write must finish before another commit
                # can acquire the lock or the service starts shutting down.
                await asyncio.gather(task, return_exceptions=True)
                raise

    async def start(self) -> None:
        await self._run(self._start)

    def _start(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        _sync_directory(self.path)
        self._reconcile()

    def _reconcile(self) -> None:
        self._sizes = {path: path.stat().st_size for path in self.path.rglob("*") if path.is_file()}
        self._bytes = sum(self._sizes.values())

    def _refresh_size(self, path: Path) -> None:
        if self._sizes is None:
            self._reconcile()
        old = self._sizes.pop(path, 0)
        size = path.stat().st_size if path.exists() else 0
        if size:
            self._sizes[path] = size
        self._bytes += size - old

    def _directory(self, evidence_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", evidence_id):
            raise ValueError("Invalid evidence ID")
        result = self.path / evidence_id
        if result.is_symlink():
            raise SpoolCorruptError("Spool directory cannot be a symlink")
        return result

    def _usage(self) -> int:
        if self._sizes is None:
            self._reconcile()
        return self._bytes

    async def usage_bytes(self) -> int:
        return await self._run(self._usage)

    def _atomic_write(self, path: Path, data: bytes) -> None:
        temporary = path.with_name(path.name + ".tmp")
        if path.is_symlink() or temporary.is_symlink():
            raise SpoolCorruptError("Spool files cannot be symlinks")
        self._refresh_size(path)
        self._refresh_size(temporary)
        # Account for the temporary replacement too: no overwrite hides quota use.
        if self._usage() + len(data) > self.max_bytes:
            raise SpoolCapacityError("Capture spool quota exhausted")
        if shutil.disk_usage(self.path).free - len(data) < self.disk_reserve_bytes:
            raise SpoolCapacityError("Capture spool disk reserve reached")
        try:
            with temporary.open("wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            _sync_directory(path.parent)
        finally:
            # Include partial temp files even when write/fsync/replace fails.
            self._refresh_size(path)
            self._refresh_size(temporary)

    async def write(self, manifest: dict[str, Any], frames: list[EncodedFrame]) -> dict[str, Any]:
        return await self._run(self._write, manifest, frames)

    def _write(self, manifest: dict[str, Any], frames: list[EncodedFrame]) -> dict[str, Any]:
        directory = self._directory(str(manifest["evidence_id"]))
        if not directory.exists():
            directory.mkdir()
            _sync_directory(self.path)
        committed_generation = 0
        previous = directory / "manifest.json"
        if previous.exists():
            try:
                committed_generation = int(json.loads(previous.read_bytes())["generation"])
            except (OSError, KeyError, ValueError, TypeError) as error:
                raise SpoolCorruptError(f"Invalid committed capture manifest: {error}") from error
        entries = {entry["sequence"]: entry for entry in manifest.get("frames", [])}
        for frame in frames:
            if frame.timeline_id != manifest["timeline_id"]:
                raise ValueError("Mixed capture timelines")
            digest = hashlib.sha256(frame.jpeg).hexdigest()
            name = f"frame-{frame.sequence}-{digest[:16]}.jpg"
            path = directory / name
            if not path.exists():
                self._atomic_write(path, frame.jpeg)
            entries[frame.sequence] = {
                "file": name, "sha256": digest, "media_time_ns": frame.media_time_ns,
                "sequence": frame.sequence, "timeline_id": frame.timeline_id, "bytes": len(frame.jpeg),
            }
        result = {**manifest, "schema_version": 1,
                  "generation": max(int(manifest.get("generation") or 0), committed_generation + 1),
                  "frames": sorted(entries.values(), key=lambda entry: (entry["media_time_ns"], entry["sequence"]))}
        self._atomic_write(directory / "manifest.json", json.dumps(result, sort_keys=True).encode())
        return result

    async def read(self, evidence_id: str) -> tuple[dict[str, Any], list[EncodedFrame]]:
        return await self._run(self._read, evidence_id)

    def _read(self, evidence_id: str) -> tuple[dict[str, Any], list[EncodedFrame]]:
        directory = self._directory(evidence_id)
        try:
            manifest = json.loads((directory / "manifest.json").read_bytes())
            if manifest["evidence_id"] != evidence_id or manifest["schema_version"] != 1:
                raise ValueError("Unsupported or mismatched manifest")
            frames = []
            for entry in manifest["frames"]:
                name = entry["file"]
                if Path(name).name != name or (directory / name).is_symlink():
                    raise ValueError("Unsafe spool filename")
                data = (directory / name).read_bytes()
                if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                    raise ValueError("Frame checksum mismatch")
                if "bytes" in entry and len(data) != entry["bytes"]:
                    raise ValueError("Frame size mismatch")
                if entry["timeline_id"] != manifest["timeline_id"]:
                    raise ValueError("Mixed capture timelines")
                frames.append(EncodedFrame(entry["timeline_id"], int(entry["media_time_ns"]),
                                           int(entry["sequence"]), data))
            return manifest, frames
        except (OSError, KeyError, ValueError, TypeError) as error:
            raise SpoolCorruptError(f"Invalid capture manifest: {error}") from error

    async def list_ids(self) -> list[str]:
        return await asyncio.to_thread(lambda: sorted(
            path.name for path in self.path.iterdir()
            if path.is_dir() and not path.is_symlink()
        ))

    async def delete(self, evidence_id: str) -> None:
        await self._run(self._delete, evidence_id)

    def _delete(self, evidence_id: str) -> None:
        directory = self._directory(evidence_id)
        self._usage()
        if directory.exists():
            try:
                shutil.rmtree(directory)
                _sync_directory(self.path)
            finally:
                for path in list(self._sizes):
                    if path.parent == directory:
                        self._refresh_size(path)

    async def write_artifact(self, evidence_id: str, name: str, data: bytes) -> bytes:
        if name not in {"snapshot.jpg", "clip.mp4", "artifacts.json", "evidence-manifest.json"}:
            raise ValueError("Unsupported artifact name")
        await self._run(self._write_artifact, evidence_id, name, data)
        return data

    def _write_artifact(self, evidence_id: str, name: str, data: bytes) -> None:
        directory = self._directory(evidence_id)
        self._atomic_write(directory / name, data)
        if name == "artifacts.json":
            return
        path = directory / "artifacts.json"
        try:
            artifacts = json.loads(path.read_bytes()) if path.exists() else {}
        except (OSError, ValueError, TypeError) as error:
            raise SpoolCorruptError(f"Invalid artifact manifest: {error}") from error
        artifacts[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        self._atomic_write(path, json.dumps(artifacts, sort_keys=True).encode())

    async def read_artifact(self, evidence_id: str, name: str) -> bytes | None:
        if name not in {"snapshot.jpg", "clip.mp4", "artifacts.json", "evidence-manifest.json"}:
            raise ValueError("Unsupported artifact name")
        path = self._directory(evidence_id) / name
        return await self._run(self._read_artifact, path, name)

    @staticmethod
    def _read_artifact(path: Path, name: str) -> bytes | None:
        if path.is_symlink():
            raise SpoolCorruptError("Artifact cannot be a symlink")
        if not path.exists():
            return None
        data = path.read_bytes()
        if name == "artifacts.json":
            return data
        manifest = path.parent / "artifacts.json"
        # Artifact bytes committed before their checksum manifest are not yet
        # trusted. Rebuild them from the durable capture after a crash.
        if not manifest.exists():
            return None
        try:
            entry = json.loads(manifest.read_bytes()).get(name)
            if entry is None:
                return None
            if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError("Artifact checksum mismatch")
        except (OSError, KeyError, ValueError, TypeError) as error:
            raise SpoolCorruptError(f"Invalid artifact manifest: {error}") from error
        return data
