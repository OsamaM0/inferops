"""Rolling-time window reducers."""
from inferops.fusion.spec import WindowSpec
from inferops.fusion.window import Window


def test_rolling_time_prune_and_reducers():
    w = Window(WindowSpec(kind="rolling_time", duration_ms=1000))
    w.add(1000, 0.1)
    w.add(1500, 0.5)
    w.add(1900, 0.8)
    assert w.count() == 3
    assert w.latest() == 0.8
    assert abs(w.mean() - (0.1 + 0.5 + 0.8) / 3) < 1e-9
    assert w.fraction_above(0.4) == 2 / 3
    # add t=2600 -> cutoff 1600 -> drops t=1000 and t=1500
    w.add(2600, 0.9)
    assert w.count() == 2
    assert w.latest() == 0.9


def test_ewma():
    w = Window(WindowSpec(kind="ewma", alpha=0.5))
    w.add(1, 1.0)
    w.add(2, 0.0)
    assert w.latest() == 0.5
