"""Driver overlay warnings preserve zero-valued signals and unknown states."""

import pytest

from inferops.core.types import StageItem
from projects.driver_behaviour.plugins.frame_annotator import _danger_classes


@pytest.mark.parametrize(
    ("payload", "warning"),
    [
        ({"hand_on_wheel": 0.0}, True),
        ({"hand_on_wheel": 0.25}, True),
        ({"hand_on_wheel": 0.5}, False),
        ({"hand_on_wheel": 1.0}, False),
        ({"hand_on_wheel": None}, False),
        ({}, False),
    ],
)
def test_wheel_warning_from_driver_signal(payload, warning):
    item = StageItem(stream_id="driver", ts_ms=0, payload=payload)
    assert ("wheel" in _danger_classes(item)) is warning
