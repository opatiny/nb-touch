import pytest

from nb_touch.utils.units import length_to_metres_factor


@pytest.mark.parametrize(("units", "factor"), [("mm", 1e-3), ("m", 1.0)])
def test_supported_units(units, factor):
    assert length_to_metres_factor(units) == factor


@pytest.mark.parametrize("units", ["cm", "MM", "", None])
def test_unsupported_units_raise(units):
    with pytest.raises(ValueError, match="unsupported"):
        length_to_metres_factor(units)
