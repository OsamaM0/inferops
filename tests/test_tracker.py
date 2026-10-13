"""Tests for the IoU + Kalman object tracker."""
from __future__ import annotations

from inferops.core.tracking import IoUKalmanTracker
from inferops.core.types import Detection


def _det(label: str, score: float, box):
    return Detection(label=label, score=score, box=box)


def test_track_ids_persist_across_frames():
    tracker = IoUKalmanTracker(iou_threshold=0.3, max_age=5, min_hits=1)
    # Two objects moving slowly to the right across 5 frames.
    track_a, track_b = None, None
    for i in range(5):
        dx = i * 0.02
        dets = [
            _det("driver", 0.9, (0.10 + dx, 0.10, 0.30 + dx, 0.30)),
            _det("driver", 0.8, (0.60 + dx, 0.50, 0.80 + dx, 0.70)),
        ]
        out = tracker.update(dets)
        ids = sorted(int(d.attrs["track_id"]) for d in out)
        if i == 0:
            track_a, track_b = ids
        else:
            assert sorted(ids) == [track_a, track_b], f"frame {i}: ids changed -> {ids}"


def test_track_survives_short_occlusion():
    tracker = IoUKalmanTracker(iou_threshold=0.3, max_age=8, min_hits=1)
    out = tracker.update([_det("p", 0.9, (0.10, 0.10, 0.30, 0.30))])
    original = int(out[0].attrs["track_id"])
    # Two empty frames simulate an occlusion.
    tracker.update([])
    tracker.update([])
    # Detection reappears near predicted location.
    out = tracker.update([_det("p", 0.9, (0.16, 0.10, 0.36, 0.30))])
    assert out, "tracker should have reassigned the lost track"
    assert int(out[0].attrs["track_id"]) == original


def test_per_class_isolates_classes():
    tracker = IoUKalmanTracker(iou_threshold=0.3, max_age=5, min_hits=1,
                               per_class=True)
    out1 = tracker.update([_det("face", 0.9, (0.1, 0.1, 0.2, 0.2))])
    fid = int(out1[0].attrs["track_id"])
    # Same box, different class -> must get a new id.
    out2 = tracker.update([
        _det("face", 0.9, (0.1, 0.1, 0.2, 0.2)),
        _det("hand", 0.9, (0.1, 0.1, 0.2, 0.2)),
    ])
    ids = {d.label: int(d.attrs["track_id"]) for d in out2}
    assert ids["face"] == fid
    assert ids["hand"] != fid
