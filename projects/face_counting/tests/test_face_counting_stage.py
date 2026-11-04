"""The full face counting stage on synthetic scenes (fake detector + embedder).

Frames are 100 ms apart (10 fps). Faces are coloured squares; the fake
embedder's identity comes from the colour inside the aligned crop.
"""
import numpy as np
import pytest
from synthetic import FakeModels, face_at, load_object, make_stage, render

from inferops.domain.errors import ConfigError
from inferops.domain.types import StageItem


async def step(stage, models, faces, t_ms, *, stream="cam-1", camera=None):
    models.faces = list(faces)
    payload = {"image_np": render(faces)}
    if camera:
        payload["stream_name"] = camera
    return await stage.process(StageItem(stream_id=stream, ts_ms=t_ms, payload=payload))


def ids(item) -> dict[int, tuple[float, float]]:
    """Track id -> face centre (normalised) for the visible faces."""
    return {t["track_id"]: ((t["box"][0] + t["box"][2]) / 2, (t["box"][1] + t["box"][3]) / 2)
            for t in item.payload["face_tracks"]}


@pytest.mark.asyncio
async def test_one_person_with_a_four_frame_dropout_counts_once():
    models = FakeModels()
    stage = make_stage(models)
    t = 0
    for i in range(6):
        item = await step(stage, models, [face_at(1, 300 + 4 * i, 300)], t)
        t += 100
    (original,) = ids(item)
    for _ in range(4):                                   # head turned / occluded
        item = await step(stage, models, [], t)
        t += 100
    assert item.payload["faces_in_frame"] == 0 and item.payload["unique_faces"] == 1
    for i in range(4):
        item = await step(stage, models, [face_at(1, 330 + 4 * i, 300)], t)
        t += 100
    assert list(ids(item)) == [original]
    assert item.payload["unique_faces"] == 1 and item.payload["faces_in_frame"] == 1


@pytest.mark.asyncio
async def test_a_long_gap_and_a_new_position_relink_by_appearance():
    models = FakeModels()
    stage = make_stage(models, hold_ms=5000)
    t = 0
    for _ in range(5):
        item = await step(stage, models, [face_at(1, 200, 300)], t)
        t += 100
    (original,) = ids(item)
    for _ in range(20):                                  # 2 s without a detection
        await step(stage, models, [], t)
        t += 100
    item = await step(stage, models, [face_at(1, 320, 320)], t)   # 1.5 face widths away: no overlap
    assert list(ids(item)) == [original]
    assert item.payload["face_tracks"][0]["relinked"]
    assert item.payload["unique_faces"] == 1


@pytest.mark.asyncio
async def test_leaving_at_the_border_and_coming_back_counts_again():
    models = FakeModels()
    stage = make_stage(models, exit_ms=500)
    t = 0
    for x in range(300, 600, 30):                        # walks out on the right
        item = await step(stage, models, [face_at(1, min(x, 599), 300)], t)
        t += 100
    assert item.payload["unique_faces"] == 1
    for _ in range(10):                                  # 1 s outside the view (> exit_ms)
        await step(stage, models, [], t)
        t += 100
    for _ in range(4):                                   # back in the middle of the view
        item = await step(stage, models, [face_at(1, 320, 300)], t)
        t += 100
    assert item.payload["unique_faces"] == 2


@pytest.mark.asyncio
async def test_two_people_crossing_keep_their_ids_and_count_two():
    models = FakeModels()
    stage = make_stage(models)
    t = 0
    for i in range(5):                                   # person 1 left, person 2 right
        item = await step(stage, models, [face_at(1, 150 + 10 * i, 300), face_at(2, 490 - 10 * i, 300)], t)
        t += 100
    by_side = sorted(ids(item).items(), key=lambda kv: kv[1][0])
    left_id, right_id = by_side[0][0], by_side[1][0]
    for i in range(6):                                   # person 1 passes in front and hides person 2
        await step(stage, models, [face_at(1, 220 + 40 * i, 300)], t)
        t += 100
    for i in range(4):                                   # both visible again, sides swapped
        item = await step(stage, models, [face_at(1, 470 + 10 * i, 300), face_at(2, 200 - 10 * i, 300)], t)
        t += 100
    by_side = sorted(ids(item).items(), key=lambda kv: kv[1][0])
    assert [by_side[0][0], by_side[1][0]] == [right_id, left_id]   # person 2 is now on the left
    assert item.payload["unique_faces"] == 2 and item.payload["faces_in_frame"] == 2


@pytest.mark.asyncio
async def test_embeddings_are_computed_only_for_new_or_stale_faces():
    models = FakeModels()
    stage = make_stage(models, embed_refresh_ms=1000)
    for t in range(0, 2000, 100):                        # 20 frames, 2 s, one face
        await step(stage, models, [face_at(1, 300, 300)], t)
    assert models.embedded == 2                          # t=0 and the refresh after 1 s
    assert models.calls["face_retinaface"] == 20


@pytest.mark.asyncio
async def test_batched_embedding_falls_back_to_one_face_per_call():
    models = FakeModels()
    models.batch_limit = 1
    stage = make_stage(models, embed_batch_size=4)
    item = await step(stage, models, [face_at(1, 150, 300), face_at(2, 450, 300)], 0)
    assert models.embedded == 2 and len(item.payload["face_tracks"]) == 2


@pytest.mark.asyncio
async def test_a_face_seen_for_fewer_than_min_hits_frames_is_never_counted():
    models = FakeModels()
    stage = make_stage(models, min_hits=3)
    t = 0
    for _ in range(2):
        item = await step(stage, models, [face_at(1, 300, 300)], t)
        t += 100
    assert item.payload["unique_faces"] == 0 and item.payload["face_tracks"][0]["counted"] is False
    for _ in range(70):                                  # gone for 7 s (> hold_ms): track closed
        item = await step(stage, models, [], t)
        t += 100
    assert item.payload["unique_faces"] == 0


@pytest.mark.asyncio
async def test_weak_boxes_keep_a_track_alive_but_never_start_one():
    models = FakeModels()
    stage = make_stage(models, face_confidence=0.6, low_confidence=0.3)
    t = 0
    for _ in range(5):
        item = await step(stage, models, [face_at(2, 450, 300, score=0.4)], t)   # weak only
        t += 100
    assert item.payload["face_tracks"] == [] and item.payload["unique_faces"] == 0
    for _ in range(4):
        item = await step(stage, models, [face_at(1, 200, 300)], t)
        t += 100
    (original,) = ids(item)
    for i in range(4):                                   # blurred: weak boxes, same person
        item = await step(stage, models, [face_at(1, 204 + 4 * i, 300, score=0.4)], t)
        t += 100
    assert list(ids(item)) == [original] and item.payload["unique_faces"] == 1


@pytest.mark.asyncio
async def test_line_and_zone_counts_for_a_configured_camera():
    models = FakeModels()
    counting = {"cameras": {"door": {"line": {"points": [[0.0, 0.5], [1.0, 0.5]]},
                                     "zone": {"points": [[0.0, 0.6], [1.0, 0.6], [1.0, 1.0], [0.0, 1.0]]}}}}
    stage = make_stage(models, counting=counting)
    t = 0
    # Crosses the line on its 2nd frame, while the track is still tentative.
    for y in (300, 340, 380, 420, 460):
        item = await step(stage, models, [face_at(1, 320, y)], t, camera="door")
        t += 100
    assert item.payload["line_in"] == 1 and item.payload["line_out"] == 0
    assert item.payload["zone_entries"] == 1 and item.payload["zone_occupancy"] == 1
    assert item.payload["signals"]["line_in"] == 1
    assert item.payload["counting_geometry"]["line"]["points"] == [[0.0, 0.5], [1.0, 0.5]]
    other = await step(stage, models, [face_at(2, 320, 300)], 0, stream="cam-2", camera="lobby")
    assert "line_in" not in other.payload and "counting_geometry" not in other.payload


@pytest.mark.asyncio
async def test_light_detector_path_and_published_results():
    models = FakeModels(detector="face_detector")
    stage = make_stage(models, min_hits=1, min_track_ms=0)
    item = await step(stage, models, [face_at(1, 320, 320, size=96)], 0)
    (result,) = item.results
    assert result.model == "face_detector" and models.calls["face_detector"] == 1
    det = result.detections[0]
    assert det.label == "face" and det.attrs["counted"] and "embedding" not in det.attrs
    assert det.box == pytest.approx((272 / 640, 272 / 640, 368 / 640, 368 / 640), abs=2e-3)
    assert item.payload["unique_faces"] == 1


@pytest.mark.asyncio
async def test_setup_requires_a_detector_and_params_are_validated():
    stage = load_object(".models.face_tracker.adapter:FaceCountingStage")(model="face_tracker")
    with pytest.raises(ConfigError, match="face detector"):
        await stage.setup(object())
    with pytest.raises(ValueError):
        make_stage(FakeModels(), low_confidence=0.9, face_confidence=0.5)
    with pytest.raises(ValueError):
        make_stage(FakeModels(), counting={"cameras": {"x": {"line": {"points": [[0, 0]]}}}})


@pytest.mark.asyncio
async def test_annotator_draws_ids_line_zone_and_counters():
    models = FakeModels()
    counting = {"default": {"line": {"points": [[0.0, 0.5], [1.0, 0.5]]},
                            "zone": {"points": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9]]}}}
    stage = make_stage(models, counting=counting, min_hits=1)
    item = await step(stage, models, [face_at(1, 320, 200)], 0)
    annotator = load_object(".plugins.frame_annotator:FaceCountAnnotator")()
    frame = np.zeros((640, 640, 3), np.uint8)
    out = annotator(frame.copy(), item)
    assert out.shape == frame.shape and out.any()
    assert annotator(frame.copy(), item.model_copy(update={"payload": {}})).shape == frame.shape


@pytest.mark.asyncio
async def test_the_default_line_applies_to_every_camera_unless_it_has_its_own():
    models = FakeModels()
    own = {"cameras": {"door": {"zone": {"points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]}}}}
    stage = make_stage(models, line="horizontal", line_position=0.5, line_in="down", counting=own)
    t = 0
    for y in (250, 300, 340, 380, 420):                  # walks down across the middle
        item = await step(stage, models, [face_at(1, 320, y)], t, camera="lobby")
        t += 100
    assert item.payload["counting_geometry"]["line"]["points"] == [[0.0, 0.5], [1.0, 0.5]]
    assert item.payload["line_in"] == 1
    door = await step(stage, models, [face_at(2, 320, 300)], 0, stream="cam-2", camera="door")
    assert "line" not in door.payload["counting_geometry"] and "line_in" not in door.payload


@pytest.mark.asyncio
async def test_newly_counted_people_are_numbered_and_marked_new_for_a_while():
    models = FakeModels()
    stage = make_stage(models, new_highlight_ms=1000, min_track_ms=0)
    t = 0
    for _ in range(3):
        item = await step(stage, models, [face_at(1, 200, 300)], t)
        t += 100
    assert item.payload["new_faces"] == 1
    assert [(p["number"], p["track_id"]) for p in item.payload["new_people"]] == [(1, 1)]
    assert item.payload["face_tracks"][0] | {"box": None} == {
        "track_id": 1, "box": None, "counted": True, "number": 1, "new": True, "relinked": False}
    for _ in range(3):                                   # a second person; the first is still NEW
        item = await step(stage, models, [face_at(1, 200, 300), face_at(2, 450, 300)], t)
        t += 100
    assert [p["number"] for p in item.payload["new_people"]] == [2]
    for _ in range(12):                                  # over 1 s later neither is NEW any more
        item = await step(stage, models, [face_at(1, 200, 300), face_at(2, 450, 300)], t)
        t += 100
    assert "new_people" not in item.payload
    assert [(tr["number"], tr["new"]) for tr in item.payload["face_tracks"]] == [(1, False), (2, False)]


@pytest.mark.asyncio
async def test_a_face_must_be_tracked_for_min_track_ms_before_it_counts():
    models = FakeModels()
    stage = make_stage(models, min_hits=3, min_track_ms=300, hold_ms=200, exit_ms=100)
    t = 0
    for _ in range(4):                                   # 30 fps flicker: 4 detections in 0.1 s
        item = await step(stage, models, [face_at(1, 300, 300)], t)
        t += 33
    assert item.payload["unique_faces"] == 0
    for _ in range(15):                                  # gone long enough to close the track
        await step(stage, models, [], t)
        t += 33
    for _ in range(13):                                  # the same face held for 0.4 s
        item = await step(stage, models, [face_at(2, 300, 300)], t)
        t += 33
    assert item.payload["unique_faces"] == 1


@pytest.mark.asyncio
async def test_a_second_box_on_an_already_counted_face_is_not_counted():
    models = FakeModels()
    stage = make_stage(models, min_track_ms=0)
    t = 0
    for _ in range(4):
        item = await step(stage, models, [face_at(1, 300, 300, size=100)], t)
        t += 100
    assert item.payload["unique_faces"] == 1
    for _ in range(6):                                   # the detector adds a smaller box inside it
        item = await step(stage, models, [face_at(1, 300, 300, size=100), face_at(1, 300, 312, size=60)], t)
        t += 100
    assert len(item.payload["face_tracks"]) == 2 and item.payload["unique_faces"] == 1
