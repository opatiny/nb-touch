import numpy as np
import pytest

from nb_touch.contact.filters import (
    Debounce,
    Hysteresis,
    ThresholdFilter,
    make_filter,
)


def test_threshold_filter():
    np.testing.assert_array_equal(
        ThresholdFilter(0.2).update([0.1, 0.3]), [False, True]
    )


def test_hysteresis_holds_between_thresholds():
    hysteresis = Hysteresis(on=0.2, off=0.1)
    assert list(hysteresis.update([0.15])) == [False]
    assert list(hysteresis.update([0.25])) == [True]
    assert list(hysteresis.update([0.15])) == [True]
    assert list(hysteresis.update([0.05])) == [False]


def test_hysteresis_rejects_off_above_on():
    with pytest.raises(ValueError, match="above"):
        Hysteresis(on=0.1, off=0.2)


def test_debounce_needs_frames_in_a_row():
    debounce = Debounce(threshold=0.2, frames=3)
    assert [debounce.update([0.3])[0] for _ in range(3)] == [False, False, True]
    # An interruption resets the count.
    assert debounce.update([0.1])[0]
    assert debounce.update([0.3])[0]
    assert [debounce.update([0.1])[0] for _ in range(3)] == [True, True, False]


def test_debounce_with_one_frame_follows_threshold():
    debounce = Debounce(threshold=0.2, frames=1)
    assert debounce.update([0.3])[0]
    assert not debounce.update([0.1])[0]


def test_debounce_rejects_zero_frames():
    with pytest.raises(ValueError, match="at least 1"):
        Debounce(0.2, 0)


@pytest.mark.parametrize(
    ("name", "kind"),
    [("none", ThresholdFilter), ("hysteresis", Hysteresis), ("debounce", Debounce)],
)
def test_make_filter(name, kind):
    assert isinstance(make_filter(name, 0.2), kind)


def test_make_filter_unknown_raises():
    with pytest.raises(ValueError, match="unknown filter"):
        make_filter("median", 0.2)
