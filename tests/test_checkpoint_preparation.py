import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from inferops.adapters.outbound.inference.checkpoint_export import compare_detections, infer_variant
from inferops.adapters.outbound.inference.checkpoint_jobs import CheckpointJobs


def predictions(labels=(0, 1), boxes=None):
    logits = np.full((1, len(labels), 3), -8, dtype=np.float32)
    for i, label in enumerate(labels):
        logits[0, i, label] = 2
    return [np.array(boxes if boxes is not None else [[[.2, .3, .4, .5]] * len(labels)],
                     dtype=np.float32), logits]


def test_foreground_contract_matches_permutations_and_ignores_background():
    actual = predictions((0, 1, 2))
    expected = [x[:, [1, 0, 2]].copy() for x in actual]
    expected[0][:, 2] = 0
    assert compare_detections(actual, expected, classes=2) == 2
    expected[0][:, 0, 0] += .01
    with pytest.raises(ValueError, match="boxes"):
        compare_detections(actual, expected, classes=2)


def test_foreground_threshold_crossings_and_missing_detections_fail():
    original = predictions((0,))
    changed = predictions((2,))
    with pytest.raises(ValueError, match="counts"):
        compare_detections(original, changed, classes=2)
    changed = predictions((1,))
    with pytest.raises(ValueError, match="classes"):
        compare_detections(original, changed, classes=2)


def test_legacy_architecture_inference_requires_known_weight_signature():
    state = {
        "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight": SimpleNamespace(shape=(384, 3, 16, 16)),
        "backbone.0.encoder.encoder.embeddings.position_embeddings": SimpleNamespace(shape=(1, 1297, 384)),
        **{f"transformer.decoder.layers.{i}.weight": None for i in range(4)},
    }
    assert infer_variant(state) == "medium"
    state["backbone.0.encoder.encoder.embeddings.position_embeddings"].shape = (1, 100, 384)
    with pytest.raises(ValueError):
        infer_variant(state)


def worker(tmp_path, *, hang=False):
    script = tmp_path / "worker"
    script.write_text("#!/usr/bin/env python3\nimport sys,json,time\nfrom pathlib import Path\n"
                      "out=Path(sys.argv[sys.argv.index('--output')+1])\n"
                      + ("time.sleep(60)\n" if hang else
                         "(out/'model.onnx').write_bytes(b'tested-worker-artifact')\n"
                         "(out/'report.json').write_text(json.dumps({'status':'validated'}))\n"))
    script.chmod(0o755)
    return script


async def test_export_job_runs_process_persists_result_and_recovers_restart(tmp_path):
    checkpoint = tmp_path / "trained.pth"
    checkpoint.write_bytes(b"checkpoint")
    jobs = CheckpointJobs(tmp_path / "jobs", python=worker(tmp_path))
    await jobs.start()
    record = await jobs.submit(checkpoint)
    await asyncio.gather(*list(jobs.tasks.values()))
    assert jobs.records[record["id"]]["status"] == "validated"
    assert Path(jobs.records[record["id"]]["artifact_path"]).read_bytes() == b"tested-worker-artifact"
    recovered = CheckpointJobs(jobs.root)
    await recovered.start()
    assert recovered.records[record["id"]]["status"] == "validated"
    saved = recovered.records[record["id"]]
    saved["status"] = "running"
    recovered.save(saved)
    await recovered.start()
    assert recovered.records[record["id"]]["status"] == "interrupted"


@pytest.mark.parametrize("cancel", [False, True])
async def test_timeout_and_shutdown_interrupt_export_without_promoting_artifact(tmp_path, cancel):
    checkpoint = tmp_path / "trained.pt"
    checkpoint.touch()
    jobs = CheckpointJobs(tmp_path / "jobs", python=worker(tmp_path, hang=True), timeout=.05)
    await jobs.start()
    record = await jobs.submit(checkpoint)
    if cancel:
        await asyncio.sleep(.02)
        await jobs.close()
    else:
        await asyncio.gather(*list(jobs.tasks.values()))
    saved = json.loads((jobs.root / record["id"] / "job.json").read_text())
    assert saved["status"] == ("interrupted" if cancel else "failed")
    assert "artifact_path" not in saved


async def test_export_is_optional_and_requires_checkpoint(tmp_path):
    jobs = CheckpointJobs(tmp_path / "jobs", python="/does/not/exist")
    await jobs.start()
    with pytest.raises(ValueError, match="environment"):
        await jobs.submit(tmp_path / "missing.pth")
