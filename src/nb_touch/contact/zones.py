"""Grouping the active taxels into zones of neighbours, one per contact.

Taken from `dextHand`, `src/dexthand/touch/contact.py`.
"""

from collections import deque

import numpy as np

MIN_TAXELS = 3
"""Fewest taxels a zone needs to count as a contact, so a single noisy taxel is not
reported. Value used on the ORCA hand in `dextHand`."""


def connected_groups(active, neighbours: list[np.ndarray]) -> list[np.ndarray]:
    """Group the active taxels linked through active neighbours (breadth-first search).

    Args:
        active: Mask `(n,)` of the taxels in contact.
        neighbours: Per taxel, its neighbours, from `taxel_neighbours`.

    Returns:
        The groups, each a sorted array of taxel indices, ordered by their first taxel.
    """
    active = np.asarray(active, dtype=bool)
    seen = np.zeros(len(active), dtype=bool)
    groups = []
    for start in np.flatnonzero(active):
        if seen[start]:
            continue
        seen[start] = True
        group, queue = [], deque([start])
        while queue:
            taxel = queue.popleft()
            group.append(taxel)
            for neighbour in neighbours[taxel]:
                if active[neighbour] and not seen[neighbour]:
                    seen[neighbour] = True
                    queue.append(neighbour)
        groups.append(np.sort(np.array(group, dtype=int)))
    return groups


def contact_zones(
    active, neighbours: list[np.ndarray], min_taxels: int = MIN_TAXELS
) -> list[np.ndarray]:
    """Return the groups of active neighbours with at least `min_taxels` taxels.

    Args:
        active: Mask `(n,)` of the taxels in contact.
        neighbours: Per taxel, its neighbours, from `taxel_neighbours`.
        min_taxels: Fewest taxels a zone needs.
    """
    groups = connected_groups(active, neighbours)
    return [group for group in groups if len(group) >= min_taxels]
