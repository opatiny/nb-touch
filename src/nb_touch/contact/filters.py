"""Filters deciding, frame after frame, which taxels are in contact, so a force near
the threshold does not make a contact flicker.

Every filter keeps one state per taxel and has an `update(norms)` method, called once
per frame, returning the taxels in contact:

- `ThresholdFilter`: no filtering, a taxel is in contact above the threshold;
- `Hysteresis`: a taxel turns on above the threshold and off below a lower one;
- `Debounce`: a taxel changes state once its raw state has held for some frames.

Taken from `dextHand`, `src/dexthand/touch/contact_filters.py`.
"""

import numpy as np

FILTERS = ("none", "hysteresis", "debounce")
"""Names of the filters, for `make_filter`."""

OFF_THRESHOLD = 0.1
"""Taxel force norm below which a taxel in contact turns off, with hysteresis, in N."""

DEBOUNCE_FRAMES = 5
"""Frames a taxel's raw state must hold before it changes, with debounce."""


class ThresholdFilter:
    """No filtering: a taxel is in contact while its force norm is above `threshold`
    (in N)."""

    def __init__(self, threshold: float):
        self.threshold = threshold

    def update(self, norms) -> np.ndarray:
        """Return the taxels `(n,)` in contact, from force norms `(n,)` in N."""
        return np.asarray(norms, dtype=float) > self.threshold


class Hysteresis:
    """A taxel turns on above `on` and off only below `off` (in N).

    Raises:
        ValueError: If `off` is above `on`.
    """

    def __init__(self, on: float, off: float):
        if off > on:
            raise ValueError(f"off threshold {off} is above on threshold {on}")
        self.on, self.off = on, off
        self.state: np.ndarray | None = None

    def update(self, norms) -> np.ndarray:
        """Return the taxels `(n,)` in contact after this frame, from norms in N."""
        norms = np.asarray(norms, dtype=float)
        if self.state is None:
            self.state = np.zeros(len(norms), dtype=bool)
        self.state = np.where(self.state, norms >= self.off, norms > self.on)
        return self.state.copy()


class Debounce:
    """A taxel changes state once its raw state (above or below `threshold`, in N)
    has differed from its current one for `frames` frames in a row. Taxels start out
    of contact; with `frames` 1, it follows the threshold.

    Raises:
        ValueError: If `frames` is below 1.
    """

    def __init__(self, threshold: float, frames: int):
        if frames < 1:
            raise ValueError(f"debounce needs at least 1 frame, got {frames}")
        self.threshold, self.frames = threshold, frames
        self.state: np.ndarray | None = None
        self.count: np.ndarray | None = None

    def update(self, norms) -> np.ndarray:
        """Return the taxels `(n,)` in contact after this frame, from norms in N."""
        raw = np.asarray(norms, dtype=float) > self.threshold
        if self.state is None:
            self.state = np.zeros(len(raw), dtype=bool)
            self.count = np.zeros(len(raw), dtype=int)
        # Frames in a row the raw state has differed; back to 0 when it agrees.
        self.count = np.where(raw != self.state, self.count + 1, 0)
        flip = self.count >= self.frames
        self.state[flip] = raw[flip]
        self.count[flip] = 0
        return self.state.copy()


def make_filter(
    name: str,
    threshold: float,
    off_threshold: float = OFF_THRESHOLD,
    frames: int = DEBOUNCE_FRAMES,
):
    """Build a filter, with an empty state, from its name.

    Args:
        name: One of `FILTERS`.
        threshold: Force norm above which a taxel is in contact, in N.
        off_threshold: With hysteresis, force norm below which a taxel turns off.
        frames: With debounce, frames in a row the raw state must hold.

    Raises:
        ValueError: If `name` is not a filter.
    """
    if name == "none":
        return ThresholdFilter(threshold)
    if name == "hysteresis":
        return Hysteresis(threshold, off_threshold)
    if name == "debounce":
        return Debounce(threshold, frames)
    raise ValueError(f"unknown filter {name!r}, expected one of {FILTERS}")
