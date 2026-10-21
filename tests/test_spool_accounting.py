import asyncio
import threading
from pathlib import Path

import pytest

from inferops.adapters.outbound.media.spool import CrashSafeSpool, SpoolCapacityError
from inferops.domain.evidence import EncodedFrame


def actual(root):
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


async def test_quota_recovers_orphans_counts_replacements_and_releases_deleted_bytes(tmp_path):
    directory = tmp_path / "old"
    directory.mkdir()
    (directory / "clip.mp4.tmp").write_bytes(b"orphan" * 100)
    spool = CrashSafeSpool(tmp_path, disk_reserve_bytes=0)
    await spool.start()
    assert await spool.usage_bytes() == 600
    m = await spool.write({"evidence_id": "e1", "timeline_id": "t1"}, [EncodedFrame("t1", 0, 0, b"frame")])
    for data in (b"small", b"large" * 100, b"tiny"):
        await spool.write_artifact("e1", "snapshot.jpg", data)
        assert await spool.usage_bytes() == actual(tmp_path)
    spool.max_bytes = await spool.usage_bytes() + 3
    with pytest.raises(SpoolCapacityError):
        await spool.write(m, [EncodedFrame("t1", 1, 1, b"new-frame")])
    restarted = CrashSafeSpool(tmp_path, disk_reserve_bytes=0)
    await restarted.start()
    assert await restarted.usage_bytes() == actual(tmp_path)
    await restarted.delete("e1")
    assert await restarted.usage_bytes() == 600


async def test_burst_writes_do_not_rescan_unrelated_capture_directories(tmp_path, monkeypatch):
    for i in range(60):
        directory = tmp_path / str(i)
        directory.mkdir()
        (directory / "orphan.tmp").write_bytes(b"retained")
    spool = CrashSafeSpool(tmp_path, disk_reserve_bytes=0)
    await spool.start()

    def forbidden(*args, **kwargs):
        raise AssertionError("Per-frame full spool traversal")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "rglob", forbidden)
        manifest = {"evidence_id": "new", "timeline_id": "t1"}
        for seq in range(20):
            manifest = await spool.write(manifest, [EncodedFrame("t1", seq, seq, b"jpeg")])
    assert await spool.usage_bytes() == actual(tmp_path)


async def test_failed_fsync_keeps_temp_bytes_in_quota(tmp_path, monkeypatch):
    spool = CrashSafeSpool(tmp_path, disk_reserve_bytes=0)
    await spool.start()
    import os

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", lambda *_: (_ for _ in ()).throw(OSError("disk error")))
        (tmp_path / "e1").mkdir()
        with pytest.raises(OSError):
            await spool.write(
                {"evidence_id": "e1", "timeline_id": "t1"}, [EncodedFrame("t1", 0, 0, b"durable?")]
            )
    assert await spool.usage_bytes() == actual(tmp_path) == 8


async def test_cancelling_commit_does_not_release_lock_while_native_write_runs(tmp_path):
    spool = CrashSafeSpool(tmp_path, disk_reserve_bytes=0)
    await spool.start()
    entered, release = threading.Event(), threading.Event()

    def stalled():
        entered.set()
        release.wait(3)

    writer = asyncio.create_task(spool._run(stalled))
    await asyncio.to_thread(entered.wait, 2)
    writer.cancel()
    reader = asyncio.create_task(spool.usage_bytes())
    await asyncio.sleep(0.01)
    assert not writer.done() and not reader.done()
    release.set()
    await asyncio.gather(writer, reader, return_exceptions=True)
