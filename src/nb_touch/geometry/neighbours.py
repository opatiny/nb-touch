"""Which taxels of a sensor are neighbours, from its geometry only.

Taken from `dextHand`, `src/dexthand/touch/contact.py`.
"""

import numpy as np

from nb_touch.geometry.normals import pairwise_distances

LINK_FACTOR = 1.5
"""Two taxels are neighbours when closer than this many times the median taxel
spacing, which links diagonal neighbours on a square grid too."""


def taxel_spacing(positions) -> float:
    """Return the median distance from each taxel to its nearest neighbour.

    Args:
        positions: Taxel positions `(n, 3)`, at least 2.

    Raises:
        ValueError: If there are fewer than 2 taxels.
    """
    if len(positions) < 2:
        raise ValueError("at least 2 taxels are needed for a spacing")
    distances = pairwise_distances(positions)
    np.fill_diagonal(distances, np.inf)
    return float(np.median(distances.min(axis=1)))


def taxel_neighbours(positions, factor: float = LINK_FACTOR) -> list[np.ndarray]:
    """List the neighbours of every taxel. Depends only on the geometry: compute it
    once per sensor.

    Args:
        positions: Taxel positions `(n, 3)`.
        factor: Neighbours are closer than `factor` times the taxel spacing.

    Returns:
        Per taxel, the indices of its neighbours, itself excluded. A single taxel has
        none.
    """
    if len(positions) < 2:
        return [np.zeros(0, dtype=int) for _ in range(len(positions))]
    distances = pairwise_distances(positions)
    linked = distances < factor * taxel_spacing(positions)
    np.fill_diagonal(linked, False)
    return [np.flatnonzero(row) for row in linked]
