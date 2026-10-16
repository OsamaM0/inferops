"""Durability and isolation acceptance tests without external services."""
from __future__ import annotations

import asyncio
import hashlib
import json

import pytest

from inferops.adapters.outbound.media.spool import CrashSafeSpool, SpoolCorruptError
from inferops.application.evidence import EvidenceService
from inferops.domain.evidence import EncodedFrame, EvidenceConfig


class Repository:
    def __init__(self):
        self.records = {}

    async def get_evidence(self, evidence_id):
        return dict(self.records[evidence_id]) if evidence_id in self.records else None

    async def list_evidence(self, session_id=None, limit=100):
        return [dict(value) for value in self.records.values()
                if session_id is None or value["session_id"] == session_id][:limit]

    async def update_evidence(self, evidence_id, **fields):
        self.records[evidence_id].update(fields)


class Storage:
    def __init__(self):
        self.objects = {}
        self.fail_clip = False
        self.fail_manifest = False
        self.put_calls = []

    async def put(self, key, data, content_type):
        self.put_calls.append(key)
        if key.endswith("clip.mp4") and self.fail_clip:
            raise OSError("Storage temporarily unavailable")
        if key.endswith("manifest.json") and self.fail_manifest:
            raise OSError("Manifest storage temporarily unavailable")
        self.objects[key] = data
        return key

    async def verify(self, uri, sha256=None):
        return uri in self.objects and (sha256 is None or hashlib.sha256(self.objects[uri]).hexdigest() == sha256)

    async def delete(self, uri):
        self.objects.pop(uri, None)


class Encoder:
    def snapshot(self, image, item=None, annotate=None):
        return image

    def clip(self, frames, fps):
        return b"mp4:" + b"|".join(frame.jpeg for frame in frames)


def record(repository, evidence_id="e1", session_id="s1"):
    value = {"id": evidence_id, "session_id": session_id, "timeline_id": "t1",
             "incident_id": "i1", "pipeline": "driver", "media_time_ns": 100_000_000,
             "request_start_ns": 0, "request_end_ns": 200_000_000, "state": "Pending",
             "artifacts": {}, "errors": []}
    repository.records[evidence_id] = value
    return value


def config(**overrides):
    values = dict(pre_seconds=0.1, post_seconds=0.1, sample_fps=10,
                  max_spool_bytes=10_000_000, disk_reserve_bytes=0, retry_delays=())
    return EvidenceConfig(**{**values, **overrides})


def service(tmp_path, repository=None, storage=None, **overrides):
    repository = repository or Repository()
    storage = storage or Storage()
    cfg = config(**overrides)
    spool = CrashSafeSpool(tmp_path, max_bytes=cfg.max_spool_bytes, disk_reserve_bytes=0)
    return EvidenceService(repository, storage, spool, cfg, encoder=Encoder()), repository, storage


async def capture(service, value):
    sid = value["session_id"]
    await service.observe(sid, "t1", 0, 0, b"pre")
    await service.observe(sid, "t1", 100_000_000, 1, b"trigger")
    await service.request(value)
    await service.observe(sid, "t1", 200_000_000, 2, b"post")
    await service.wait_idle()


async def test_capture_finishes_at_media_window_without_stream_close(tmp_path):
    evidence, repository, storage = service(tmp_path)
    await capture(evidence, record(repository))
    saved = repository.records["e1"]
    assert saved["state"] == "Saved"
    assert saved["actual_start_ns"] == 0 and saved["actual_end_ns"] == 200_000_000
    assert saved["missing_intervals"] == []
    assert storage.objects["evidence/e1/snapshot.jpg"] == b"trigger"
    completed = json.loads(storage.objects[saved["manifest_uri"]])
    assert completed["evidence_id"] == "e1" and completed["session_id"] == "s1"
    assert completed["requested_window"] == {"start_ns": 0, "end_ns": 200_000_000}
    assert completed["actual_window"] == completed["requested_window"]
    assert completed["artifacts"] == saved["artifacts"]
    assert hashlib.sha256(storage.objects[saved["manifest_uri"]]).hexdigest() == saved["manifest_sha256"]
    assert not (tmp_path / "e1").exists()  # cleanup follows durable completion
    await evidence.close()


async def test_upload_failure_retains_material_and_retry_identity(tmp_path):
    evidence, repository, storage = service(tmp_path)
    storage.fail_clip = True
    await capture(evidence, record(repository))
    assert repository.records["e1"]["state"] == "Incomplete"
    assert repository.records["e1"]["retry_available"]
    assert (tmp_path / "e1" / "manifest.json").exists()
    snapshot = storage.objects["evidence/e1/snapshot.jpg"]
    storage.fail_clip = False
    await evidence.retry("e1")
    await evidence.wait_idle()
    assert repository.records["e1"]["state"] == "Saved"
    assert storage.objects["evidence/e1/snapshot.jpg"] == snapshot
    assert set(storage.objects) == {"evidence/e1/snapshot.jpg", "evidence/e1/clip.mp4", "evidence/e1/manifest.json"}
    await evidence.close()


async def test_recovery_reports_interrupted_post_window(tmp_path):
    repository, storage = Repository(), Storage()
    value = record(repository)
    spool = CrashSafeSpool(tmp_path, disk_reserve_bytes=0)
    await spool.start()
    await spool.write({"evidence_id": "e1", "generation": 1, "session_id": "s1", "timeline_id": "t1",
                       "media_time_ns": value["media_time_ns"], "request_start_ns": 0,
                       "request_end_ns": 200_000_000, "closed": False, "errors": []},
                      [EncodedFrame("t1", 0, 0, b"pre"), EncodedFrame("t1", 100_000_000, 1, b"trigger")])
    recovered, _, _ = service(tmp_path, repository, storage)
    await recovered.recover()
    await recovered.wait_idle()
    assert repository.records["e1"]["state"] == "Incomplete"
    assert "process_interrupted" in repository.records["e1"]["errors"]
    assert repository.records["e1"]["actual_end_ns"] == 100_000_000
    await recovered.close()


async def test_missing_and_partial_manifest_cannot_be_saved(tmp_path):
    evidence, repository, _ = service(tmp_path)
    record(repository)
    directory = tmp_path / "e1"
    directory.mkdir()
    (directory / "manifest.json.tmp").write_text('{"closed": true')
    await evidence.recover()
    await evidence.wait_idle()
    assert repository.records["e1"]["state"] == "Failed"
    assert repository.records["e1"]["errors"] == ["spool_corrupt"]
    assert not repository.records["e1"]["retry_available"]
    await evidence.close()


async def test_spool_checksum_rejects_corrupt_frames(tmp_path):
    spool = CrashSafeSpool(tmp_path, disk_reserve_bytes=0)
    await spool.start()
    manifest = await spool.write({"evidence_id": "e1", "generation": 1, "timeline_id": "t1"},
                                [EncodedFrame("t1", 0, 0, b"good")])
    (tmp_path / "e1" / manifest["frames"][0]["file"]).write_bytes(b"corrupt")
    with pytest.raises(SpoolCorruptError, match="checksum"):
        await spool.read("e1")


async def test_capture_overload_is_visible_on_accepted_record(tmp_path):
    evidence, repository, _ = service(tmp_path, max_active_captures=1)
    await evidence.observe("s1", "t1", 100_000_000, 1, b"trigger")
    await evidence.request(record(repository))
    await evidence.request(record(repository, "e2"))
    assert repository.records["e2"]["state"] == "Failed"
    assert repository.records["e2"]["errors"] == ["capture_capacity"]
    await evidence.close()


async def test_quota_exhaustion_does_not_silently_drop_incident(tmp_path):
    evidence, repository, _ = service(tmp_path, max_spool_bytes=10)
    await evidence.observe("s1", "t1", 100_000_000, 1, b"trigger")
    await evidence.request(record(repository))
    assert repository.records["e1"]["state"] == "Failed"
    assert repository.records["e1"]["errors"] == ["spool_capacity"]
    await evidence.close()


async def test_capture_cannot_mix_timelines_or_sessions(tmp_path):
    evidence, repository, storage = service(tmp_path)
    await evidence.observe("s1", "t1", 100_000_000, 1, b"first-session")
    await evidence.observe("s2", "t1", 100_000_000, 1, b"second-session")
    await evidence.request(record(repository, "e1", "s1"))
    await evidence.observe("s1", "t2", 0, 2, b"new-timeline")
    await evidence.wait_idle()
    assert repository.records["e1"]["state"] == "Incomplete"
    assert storage.objects["evidence/e1/snapshot.jpg"] == b"first-session"
    assert b"second-session" not in storage.objects["evidence/e1/clip.mp4"]
    assert b"new-timeline" not in storage.objects["evidence/e1/clip.mp4"]
    await evidence.close()


async def test_atomic_manifest_references_only_committed_frames(tmp_path):
    spool = CrashSafeSpool(tmp_path, disk_reserve_bytes=0)
    await spool.start()
    manifest = await spool.write({"evidence_id": "e1", "generation": 1, "timeline_id": "t1"},
                                [EncodedFrame("t1", 0, 0, b"jpeg")])
    (tmp_path / "e1" / "orphan.jpg.tmp").write_bytes(b"partial")
    durable, frames = await spool.read("e1")
    assert durable == json.loads((tmp_path / "e1" / "manifest.json").read_text())
    assert len(frames) == 1 and frames[0].jpeg == b"jpeg"
    assert manifest["schema_version"] == 1


async def test_manifest_generations_increase_for_frame_and_close_commits(tmp_path):
    spool = CrashSafeSpool(tmp_path, disk_reserve_bytes=0)
    await spool.start()
    first = await spool.write({"evidence_id": "e1", "generation": 1, "timeline_id": "t1"},
                              [EncodedFrame("t1", 0, 0, b"jpeg")])
    second = await spool.write(first, [EncodedFrame("t1", 100_000_000, 1, b"next")])
    closed = await spool.write({**second, "closed": True}, [])
    assert [first["generation"], second["generation"], closed["generation"]] == [1, 2, 3]
    assert all(entry["bytes"] == len(frame.jpeg) for entry, frame in zip(closed["frames"],
                                                                       (await spool.read("e1"))[1], strict=True))


async def test_uncommitted_or_corrupt_artifact_is_never_trusted(tmp_path):
    spool = CrashSafeSpool(tmp_path, disk_reserve_bytes=0)
    await spool.start()
    await spool.write({"evidence_id": "e1", "generation": 1, "timeline_id": "t1"},
                      [EncodedFrame("t1", 0, 0, b"jpeg")])
    path = tmp_path / "e1" / "clip.mp4"
    path.write_bytes(b"uncommitted")
    assert await spool.read_artifact("e1", "clip.mp4") is None
    await spool.write_artifact("e1", "clip.mp4", b"committed")
    assert await spool.read_artifact("e1", "clip.mp4") == b"committed"
    path.write_bytes(b"corrupt")
    with pytest.raises(SpoolCorruptError, match="checksum"):
        await spool.read_artifact("e1", "clip.mp4")


async def test_post_window_frame_is_never_added_to_evidence(tmp_path):
    evidence, repository, storage = service(tmp_path)
    await evidence.observe("s1", "t1", 0, 0, b"pre")
    await evidence.observe("s1", "t1", 100_000_000, 1, b"trigger")
    await evidence.request(record(repository))
    await evidence.observe("s1", "t1", 400_000_000, 2, b"outside")
    await evidence.wait_idle()
    assert b"outside" not in storage.objects["evidence/e1/clip.mp4"]
    assert repository.records["e1"]["actual_end_ns"] == 100_000_000
    await evidence.close()


async def test_sampling_does_not_delay_capture_completion(tmp_path):
    evidence, repository, _ = service(tmp_path)
    value = record(repository)
    value["request_end_ns"] = 150_000_000
    await evidence.observe("s1", "t1", 0, 0, b"pre")
    await evidence.observe("s1", "t1", 100_000_000, 1, b"trigger")
    await evidence.request(value)
    await evidence.observe("s1", "t1", 150_000_000, 2, b"unsampled")
    await evidence.wait_idle()
    assert repository.records["e1"]["state"] == "Saved"
    await evidence.close()


async def test_repeated_request_cannot_replace_saved_evidence(tmp_path):
    evidence, repository, storage = service(tmp_path)
    value = record(repository)
    await capture(evidence, value)
    saved = dict(repository.records["e1"])
    objects = dict(storage.objects)
    await evidence.request(value)
    assert repository.records["e1"] == saved
    assert storage.objects == objects
    assert not evidence._captures
    await evidence.close()


async def test_restart_does_not_reset_exhausted_delivery_attempts(tmp_path):
    evidence, repository, storage = service(tmp_path)
    storage.fail_clip = True
    await capture(evidence, record(repository))
    await evidence.close()
    attempts = repository.records["e1"]["attempts"]
    recovered, _, _ = service(tmp_path, repository, storage)
    await recovered.recover()
    await recovered.wait_idle()
    assert repository.records["e1"]["state"] == "Incomplete"
    assert repository.records["e1"]["attempts"] == attempts
    await recovered.close()


async def test_annotation_cannot_exceed_ring_or_capture_bounds(tmp_path):
    evidence, repository, _ = service(tmp_path, per_session_bytes=10, global_bytes=10)

    def annotate(jpeg, item, callback=None):
        return b"annotation-over-limit"

    evidence.encoder.annotate = annotate
    await evidence.observe("s1", "t1", 100_000_000, 1, b"jpeg")
    await evidence.request(record(repository))
    await evidence.annotate_frame("s1", "t1", 100_000_000, {})
    await evidence.wait_idle()
    assert sum(len(frame.jpeg) for ring in evidence._rings.values() for frame in ring) <= 10
    assert "capture_material_capacity" in repository.records["e1"]["errors"]
    assert repository.records["e1"]["state"] == "Incomplete"
    await evidence.close()


async def test_manual_deletion_tombstone_survives_interrupted_storage_delete(tmp_path):
    evidence, repository, storage = service(tmp_path)
    await capture(evidence, record(repository))
    remove = storage.delete

    async def interrupted(uri):
        raise OSError("storage unavailable")

    storage.delete = interrupted
    with pytest.raises(OSError):
        await evidence.delete("e1")
    tombstone = repository.records["e1"]
    assert tombstone["deleted"] and tombstone["deletion_pending"]
    assert tombstone["artifacts"] == {} and tombstone["deleted_artifacts"]["clip"]
    assert tombstone["manifest_uri"] is None and tombstone["deleted_manifest_uri"] == "evidence/e1/manifest.json"
    with pytest.raises(ValueError, match="manually deleted"):
        await evidence.retry("e1")
    await evidence.close()
    storage.delete = remove
    recovered, _, _ = service(tmp_path, repository, storage)
    await recovered.recover()
    assert not storage.objects
    assert not repository.records["e1"]["deletion_pending"]
    assert repository.records["e1"]["deleted"]
    assert (await recovered.delete("e1"))["deleted"]
    await recovered.close()


async def test_manifest_failure_is_retryable_and_existing_artifacts_reused(tmp_path):
    evidence, repository, storage = service(tmp_path)
    storage.fail_manifest = True
    await capture(evidence, record(repository))
    assert repository.records["e1"]["state"] == "Incomplete"
    assert "manifest_uri" not in repository.records["e1"]
    assert set(repository.records["e1"]["artifacts"]) == {"snapshot", "clip"}
    storage.fail_manifest = False
    await evidence.retry("e1")
    await evidence.wait_idle()
    assert repository.records["e1"]["state"] == "Saved"
    assert storage.put_calls.count("evidence/e1/snapshot.jpg") == 1
    assert storage.put_calls.count("evidence/e1/clip.mp4") == 1
    assert storage.put_calls.count("evidence/e1/manifest.json") == 2
    await evidence.close()


async def test_frame_encoding_failure_preserves_inference_and_records_incompleteness(tmp_path):
    evidence, repository, storage = service(tmp_path)
    await evidence.observe("s1", "t1", 0, 0, b"pre")
    await evidence.observe("s1", "t1", 100_000_000, 1, b"trigger")
    await evidence.request(record(repository))

    def broken_snapshot(*args):
        raise ValueError("encoder unavailable")

    evidence.encoder.snapshot = broken_snapshot
    await evidence.observe("s1", "t1", 200_000_000, 2, b"post")
    await evidence.wait_idle()
    assert repository.records["e1"]["state"] == "Incomplete"
    assert "frame_encoding_failed" in repository.records["e1"]["errors"]
    assert evidence.dropped_frames == 1
    assert storage.objects["evidence/e1/snapshot.jpg"] == b"trigger"
    await evidence.close()


async def test_trigger_without_encoded_frames_fails_explicitly(tmp_path):
    evidence, repository, _ = service(tmp_path, post_seconds=0)

    def broken_snapshot(*args):
        raise ValueError("encoder unavailable")

    evidence.encoder.snapshot = broken_snapshot
    await evidence.observe("s1", "t1", 100_000_000, 1, b"trigger")
    await evidence.request(record(repository))
    await evidence.wait_idle()
    assert repository.records["e1"]["state"] == "Failed"
    assert "frame_encoding_failed" in repository.records["e1"]["errors"]
    assert "no_capture_material" in repository.records["e1"]["errors"]
    await evidence.close()


async def test_annotation_failure_before_incident_is_carried_into_evidence(tmp_path):
    evidence, repository, storage = service(tmp_path)
    await evidence.observe("s1", "t1", 0, 0, b"pre")
    await evidence.observe("s1", "t1", 100_000_000, 1, b"trigger")

    def broken_annotation(*args):
        raise RuntimeError("overlay unavailable")

    evidence.encoder.annotate = broken_annotation
    await evidence.annotate_frame("s1", "t1", 100_000_000, {})
    await evidence.request(record(repository))
    await evidence.observe("s1", "t1", 200_000_000, 2, b"post")
    await evidence.wait_idle()
    assert repository.records["e1"]["state"] == "Incomplete"
    assert "frame_annotation_failed" in repository.records["e1"]["errors"]
    assert storage.objects["evidence/e1/snapshot.jpg"] == b"trigger"
    await evidence.close()


async def test_bounded_shutdown_retains_pending_material_for_restart(tmp_path):
    evidence, repository, storage = service(tmp_path, shutdown_timeout_seconds=.03)
    put = storage.put
    entered = asyncio.Event()

    async def stalled_put(key, data, content_type):
        if key.endswith("clip.mp4"):
            entered.set()
            await asyncio.Event().wait()
        return await put(key, data, content_type)

    storage.put = stalled_put
    await evidence.observe("s1", "t1", 0, 0, b"pre")
    await evidence.observe("s1", "t1", 100_000_000, 1, b"trigger")
    await evidence.request(record(repository))
    await evidence.observe("s1", "t1", 200_000_000, 2, b"post")
    await asyncio.wait_for(entered.wait(), 1)
    await asyncio.wait_for(evidence.close(), 1)
    assert repository.records["e1"]["state"] == "Pending"
    assert (tmp_path / "e1" / "manifest.json").exists()
    storage.put = put
    recovered, _, _ = service(tmp_path, repository, storage)
    await recovered.recover()
    await recovered.wait_idle()
    assert repository.records["e1"]["state"] == "Saved"
    assert not (tmp_path / "e1").exists()
    await recovered.close()


@pytest.mark.parametrize("values", [{"shutdown_timeout_seconds": 0}, {"shutdown_timeout_seconds": float("inf")},
                                   {"sample_fps": float("nan")}, {"retry_delays": (float("inf"),)}])
def test_evidence_operational_time_settings_are_finite_and_positive(values):
    with pytest.raises(ValueError):
        config(**values)


async def test_delayed_decision_retains_pre_window_and_finalizes_passed_window(tmp_path):
    evidence, repository, storage = service(tmp_path, processing_lag_seconds=1)
    await evidence.start()
    for sequence in range(6):
        await evidence.observe("s1", "t1", sequence * 100_000_000, sequence, f"frame{sequence}".encode())
    value = record(repository)
    value.update(media_time_ns=150_000_000, request_end_ns=250_000_000)
    await evidence.request(value)
    await evidence.wait_idle()
    saved = repository.records["e1"]
    assert saved["state"] == "Saved"
    assert b"frame0" in storage.objects[saved["artifacts"]["clip"]["uri"]]
    assert b"frame5" not in storage.objects[saved["artifacts"]["clip"]["uri"]]
    await evidence.close()
