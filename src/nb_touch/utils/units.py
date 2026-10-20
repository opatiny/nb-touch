"""Unit conversions."""

LENGTH_UNITS_TO_METRES = {"mm": 1e-3, "m": 1.0}
"""Factor converting each supported length unit to metres."""


def length_to_metres_factor(units: str) -> float:
    """Return the factor converting a length in `units` to metres.

    Args:
        units: One of the keys of `LENGTH_UNITS_TO_METRES`.

    Raises:
        ValueError: If the units are not supported.
    """
    if units not in LENGTH_UNITS_TO_METRES:
        supported = ", ".join(LENGTH_UNITS_TO_METRES)
        raise ValueError(f"unsupported length units {units!r}, expected {supported}")
    return LENGTH_UNITS_TO_METRES[units]
