"""Line crossing and zone counters."""
import pytest
from synthetic import counting

LineSpec, LineCounter, ZoneCounter = counting["LineSpec"], counting["LineCounter"], counting["ZoneCounter"]
parse_counting = counting["parse_counting"]
HORIZONTAL = {"points": [[0.0, 0.5], [1.0, 0.5]], "margin": 0.02}


def walk(counter, track, ys, x=0.5):
    return [counter.update(track, (x, y)) for y in ys]


def test_moving_down_a_left_to_right_line_is_in_and_up_is_out():
    counter = LineCounter(LineSpec.from_config(HORIZONTAL))
    assert walk(counter, 1, [0.3, 0.4, 0.6, 0.7]) == [None, None, "in", None]
    assert walk(counter, 2, [0.8, 0.6, 0.4]) == [None, None, "out"]
    inverted = LineCounter(LineSpec.from_config({**HORIZONTAL, "invert": True}))
    assert walk(inverted, 1, [0.3, 0.7]) == [None, "out"]


def test_jitter_inside_the_margin_never_counts_and_sides_alternate():
    counter = LineCounter(LineSpec.from_config(HORIZONTAL))
    assert walk(counter, 1, [0.4, 0.51, 0.49, 0.515, 0.485, 0.45]) == [None] * 6
    events = walk(counter, 1, [0.6, 0.62, 0.4, 0.6])
    assert events == ["in", None, "out", "in"]


def test_a_track_first_seen_beyond_the_line_is_not_counted_until_it_crosses():
    counter = LineCounter(LineSpec.from_config(HORIZONTAL))
    assert walk(counter, 1, [0.7, 0.8, 0.9]) == [None, None, None]


def test_crossings_outside_the_segment_do_not_count():
    counter = LineCounter(LineSpec.from_config({"points": [[0.4, 0.5], [0.6, 0.5]], "margin": 0.01}))
    assert walk(counter, 1, [0.3, 0.7], x=0.9) == [None, None]
    assert walk(counter, 2, [0.3, 0.7], x=0.5) == [None, "in"]


def test_zone_counts_each_track_once_and_knows_who_is_inside():
    zone = ZoneCounter(((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)))
    assert zone.contains((0.5, 0.5)) and not zone.contains((0.9, 0.5))
    assert [zone.update(1, p) for p in [(0.1, 0.5), (0.5, 0.5), (0.9, 0.5), (0.5, 0.5)]] == [False, True, False, False]
    assert zone.update(2, (0.3, 0.3))       # first seen inside: an entry
    zone.forget(1)
    assert zone.update(1, (0.5, 0.5))       # a new stay after the track closed


def test_parse_counting_per_camera_and_validation():
    default, cameras = parse_counting({"cameras": {"cam-1": {"line": HORIZONTAL}}})
    assert default.line is None and cameras["cam-1"].line.a == (0.0, 0.5)
    assert cameras["cam-1"].describe()["line"]["points"] == [[0.0, 0.5], [1.0, 0.5]]
    for bad in ({"cameras": {"c": {"line": {"points": [[0, 0]]}}}},
                {"cameras": {"c": {"zone": {"points": [[0, 0], [1, 1]]}}}},
                {"cameras": {"c": {"line": {"points": [[0, 0], [0, 0]]}}}},
                {"cameras": {"c": {"lines": {}}}},
                {"cameras": {"c": {"line": {"points": [[0, 0], [100, 400]]}}}},
                {"camera": {}}):
        with pytest.raises(ValueError):
            parse_counting(bad)


def test_simple_line_settings_give_a_full_width_or_height_line():
    simple_line = counting["simple_line"]
    assert simple_line("off", 0.5, "down") is None
    down = LineCounter(LineSpec.from_config(simple_line("horizontal", 0.4, "down")))
    assert walk(down, 1, [0.3, 0.5]) == [None, "in"]
    up = LineCounter(LineSpec.from_config(simple_line("horizontal", 0.4, "up")))
    assert walk(up, 1, [0.3, 0.5]) == [None, "out"]
    left = LineCounter(LineSpec.from_config(simple_line("vertical", 0.5, "left")))
    assert [left.update(1, (x, 0.5)) for x in (0.7, 0.3)] == [None, "in"]
    right = LineCounter(LineSpec.from_config(simple_line("vertical", 0.5, "right")))
    assert [right.update(1, (x, 0.5)) for x in (0.3, 0.7)] == [None, "in"]
    for bad in (("horizontal", 0.5, "left"), ("vertical", 0.5, "down"), ("diagonal", 0.5, "down"),
                ("horizontal", 1.5, "down")):
        with pytest.raises(ValueError):
            simple_line(*bad)
