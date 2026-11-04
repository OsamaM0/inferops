"""Tests for the IoU + Kalman object tracker."""
from __future__ import annotations

from inferops.domain.tracking import IoUKalmanTracker
from inferops.domain.types import Detection


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


# ── appearance, time-based and exit-aware aging (opt-in) ────────────────────

import numpy as np  # noqa: E402


def _vec(seed: int, noise: float = 0.0, noise_seed: int = 99) -> list[float]:
    """A stable random unit vector per identity, optionally perturbed."""
    v = np.random.default_rng(seed).normal(size=64)
    if noise:
        v = v + np.random.default_rng(noise_seed).normal(scale=noise, size=64) * np.linalg.norm(v) / 8.0
    return (v / np.linalg.norm(v)).tolist()


def _face(box, seed=None, score=0.9, **noise):
    attrs = {"embedding": _vec(seed, **noise)} if seed is not None else {}
    return Detection(label="face", score=score, box=box, attrs=attrs)


def test_appearance_relinks_a_lost_track_after_a_gap_at_a_new_position():
    for appearance, same in ((True, True), (False, False)):
        tracker = IoUKalmanTracker(min_hits=1, max_age=30, appearance=appearance, reid_threshold=0.5)
        for _ in range(4):
            first = tracker.update([_face((0.40, 0.40, 0.50, 0.50), seed=1)])
        original = first[0].attrs["track_id"]
        for _ in range(5):
            tracker.update([])
        # Re-appears one box width to the right: no IoU with the prediction.
        out = tracker.update([_face((0.52, 0.42, 0.62, 0.52), seed=1, noise=0.3)])
        assert (out[0].attrs["track_id"] == original) is same
        assert out[0].attrs.get("track_relinked", False) is same


def test_appearance_relink_respects_the_spatial_gate():
    tracker = IoUKalmanTracker(min_hits=1, appearance=True, reid_threshold=0.5, reid_max_distance=2.0)
    original = tracker.update([_face((0.10, 0.10, 0.20, 0.20), seed=1)])[0].attrs["track_id"]
    tracker.update([])
    far = tracker.update([_face((0.80, 0.80, 0.90, 0.90), seed=1)])  # 7 box sizes away
    assert far[0].attrs["track_id"] != original


def test_appearance_gate_prevents_an_id_swap_when_boxes_trade_places():
    tracker = IoUKalmanTracker(min_hits=1, appearance=True, appearance_gate=0.3, reid_threshold=0.5)
    box_a, box_b = (0.30, 0.40, 0.40, 0.50), (0.36, 0.40, 0.46, 0.50)
    for _ in range(3):
        out = tracker.update([_face(box_a, seed=1), _face(box_b, seed=2)])
    ids = [d.attrs["track_id"] for d in out]
    # Person 1 now stands where person 2 was and vice versa (they crossed).
    swapped = tracker.update([_face(box_b, seed=1), _face(box_a, seed=2)])
    assert [d.attrs["track_id"] for d in swapped] == ids

    plain = IoUKalmanTracker(min_hits=1)
    for _ in range(3):
        out = plain.update([_face(box_a, seed=1), _face(box_b, seed=2)])
    ids = [d.attrs["track_id"] for d in out]
    assert [d.attrs["track_id"] for d in plain.update([_face(box_b), _face(box_a)])] == ids[::-1]


def test_border_exit_closes_quickly_while_an_interior_loss_is_held():
    tracker = IoUKalmanTracker(min_hits=1, max_lost_ms=5000, exit_lost_ms=500, exit_margin=0.02)
    inside = (0.40, 0.40, 0.50, 0.50)
    at_edge = (0.89, 0.40, 0.995, 0.50)
    out = tracker.update([_face(inside), _face(at_edge)], ts_ms=0)
    inside_id, edge_id = (d.attrs["track_id"] for d in out)
    for t in range(100, 700, 100):
        tracker.update([], ts_ms=t)
    assert tracker.pop_closed_ids() == [edge_id]
    for t in range(700, 5100, 100):
        tracker.update([], ts_ms=t)
    assert tracker.pop_closed_ids() == []
    back = tracker.update([_face((0.41, 0.40, 0.51, 0.50))], ts_ms=5000)
    assert back[0].attrs["track_id"] == inside_id
    tracker.update([], ts_ms=10_100)
    assert tracker.pop_closed_ids() == [inside_id]


def test_time_based_hold_does_not_depend_on_the_frame_rate():
    tracker = IoUKalmanTracker(min_hits=1, max_age=2, max_lost_ms=3000)
    original = tracker.update([_face((0.40, 0.40, 0.50, 0.50))], ts_ms=0)[0].attrs["track_id"]
    for t in range(33, 2000, 33):  # 60 missed updates at 30 fps, 2 s
        tracker.update([], ts_ms=t)
    assert tracker.update([_face((0.40, 0.40, 0.50, 0.50))], ts_ms=2000)[0].attrs["track_id"] == original


def test_without_timestamps_the_frame_count_limit_still_applies():
    tracker = IoUKalmanTracker(min_hits=1, max_age=3, max_lost_ms=60_000, exit_lost_ms=1)
    track_id = tracker.update([_face((0.90, 0.40, 0.999, 0.50))])[0].attrs["track_id"]
    for _ in range(3):
        tracker.update([])
    assert tracker.pop_closed_ids() == []
    tracker.update([])
    assert tracker.pop_closed_ids() == [track_id]


def test_track_snapshot_reports_confirmation_and_visibility():
    tracker = IoUKalmanTracker(min_hits=2, appearance=True)
    tracker.update([_face((0.1, 0.1, 0.2, 0.2), seed=3)])
    (view,) = tracker.tracks()
    assert not view.confirmed and view.hits == 1 and view.time_since_update == 0 and view.has_embedding
    tracker.update([_face((0.1, 0.1, 0.2, 0.2))])
    (view,) = tracker.tracks()
    assert view.confirmed and view.hits == 2
    tracker.update([])
    assert tracker.tracks()[0].time_since_update == 1


def test_observation_recovery_matches_the_last_box_when_the_prediction_drifted():
    def centred(size):
        return (0.5 - size / 2, 0.5 - size / 2, 0.5 + size / 2, 0.5 + size / 2)

    for recovery in (False, True):
        tracker = IoUKalmanTracker(min_hits=1, observation_recovery=recovery)
        for size in (0.20, 0.17, 0.14, 0.11):           # shrinking box: the area velocity is negative
            out = tracker.update([_face(centred(size))])
        original = out[0].attrs["track_id"]
        for _ in range(4):                               # unobserved: the predicted box collapses
            tracker.update([])
        back = tracker.update([_face(centred(0.11))])    # reappears exactly where it was last seen
        assert (back[0].attrs["track_id"] == original) is recovery


def test_a_fast_jolt_inside_the_view_is_held_not_treated_as_an_exit():
    tracker = IoUKalmanTracker(min_hits=1, max_lost_ms=5000, exit_lost_ms=500)
    t = 0
    for x in (0.30, 0.35, 0.41, 0.46, 0.50):             # sudden lurch to the right, still inside
        out = tracker.update([_face((x, 0.55, x + 0.22, 0.75))], ts_ms=t)
        t += 80
    original = out[0].attrs["track_id"]
    for _ in range(20):                                  # 1.6 s unseen: the prediction leaves the frame
        tracker.update([], ts_ms=t)
        t += 80
    assert tracker.pop_closed_ids() == []
    assert tracker.tracks()[0].track_id == original


def test_nearby_recovery_relinks_a_small_fast_face_that_no_longer_overlaps():
    def relinked(*, back_at_ms=132, back_size=0.04, **options):
        tracker = IoUKalmanTracker(min_hits=1, observation_recovery=True, max_lost_ms=5000, **options)
        original = tracker.update([_face((0.52, 0.20, 0.56, 0.24))], ts_ms=0)[0].attrs["track_id"]
        for t in (33, 66, 99):                           # 3 missed frames at 30 fps
            tracker.update([], ts_ms=t)
        # Back 0.75 face sizes to the left: IoU with its last box is only ~0.14.
        back = tracker.update([_face((0.49, 0.20, 0.49 + back_size, 0.20 + back_size))], ts_ms=back_at_ms)
        return back[0].attrs["track_id"] == original

    nearby = {"recovery_distance": 1.0, "recovery_max_ms": 1000}
    assert relinked(**nearby)
    assert not relinked()                                # off by default
    assert not relinked(back_at_ms=2000, **nearby)       # lost too long ago
    assert not relinked(back_size=0.10, **nearby)        # a much larger face is someone else
