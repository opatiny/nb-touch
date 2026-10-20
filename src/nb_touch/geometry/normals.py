"""Outward surface normals of a curved sensor, estimated from its taxel positions.

Taken from `dextHand`, `src/dexthand/touch/taxels.py`.
"""

import heapq

import numpy as np

NEIGHBOURS = 7
"""Taxels (the taxel itself included) fitted with a plane to estimate a normal."""


def pairwise_distances(positions) -> np.ndarray:
    """Return the distance `(n, n)` between every pair of points `(n, 3)`."""
    positions = np.asarray(positions, dtype=float)
    return np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)


def nearest_neighbours(positions, count: int) -> np.ndarray:
    """Return the indices of the nearest points to every point, itself first.

    Args:
        positions: Points `(n, 3)`.
        count: Neighbours per point, the point itself included. Clamped to `n`.

    Returns:
        Indices `(n, count)`, nearest first.
    """
    distances = pairwise_distances(positions)
    return np.argsort(distances, axis=1, kind="stable")[:, : min(count, len(distances))]


def plane_normals(positions, neighbours: int = NEIGHBOURS) -> np.ndarray:
    """Fit a plane to every point and its nearest neighbours, and return its normal.

    The normal is the direction in which the neighbourhood spreads least: the right
    singular vector of the centred points with the smallest singular value.

    Args:
        positions: Points `(n, 3)`.
        neighbours: Points per fit, the point itself included.

    Returns:
        Unit normals `(n, 3)`, each with an arbitrary sign.
    """
    positions = np.asarray(positions, dtype=float)
    normals = np.empty_like(positions)
    for i, indices in enumerate(nearest_neighbours(positions, neighbours)):
        nearest = positions[indices]
        _, _, axes = np.linalg.svd(nearest - nearest.mean(axis=0))
        normals[i] = axes[2]
    return normals


def neighbour_graph(positions, neighbours: int = NEIGHBOURS) -> dict[int, set]:
    """Return the symmetric nearest-neighbour graph: per point, its neighbours."""
    graph = nearest_neighbours(positions, neighbours)
    edges = {i: set(row[1:].tolist()) for i, row in enumerate(graph)}
    for i, row in enumerate(graph):
        for j in row[1:]:
            edges[int(j)].add(i)
    return edges


def orient_normals(positions, normals, neighbours: int = NEIGHBOURS) -> np.ndarray:
    """Give a set of normals consistent signs, outward from the sensor.

    The highest point (largest z) faces away from the mounting plane, so its normal is
    turned to +z. That sign then spreads through the neighbour graph, most parallel
    neighbours first (a minimum spanning tree on `1 - |n_i . n_j|`), each normal
    turned to agree with the one it is reached from. Orientation step of Hoppe et al.,
    "Surface reconstruction from unorganized points", SIGGRAPH 1992. Points not
    connected to the highest one keep their sign.

    Args:
        positions: Points `(n, 3)`, with z pointing out of the mounting plane.
        normals: Unit normals `(n, 3)` with arbitrary signs.
        neighbours: Points per neighbourhood, the point itself included.

    Returns:
        The normals with consistent signs.
    """
    positions = np.asarray(positions, dtype=float)
    normals = np.array(normals, dtype=float)
    edges = neighbour_graph(positions, neighbours)
    seed = int(np.argmax(positions[:, 2]))
    if normals[seed, 2] < 0:
        normals[seed] = -normals[seed]

    def cost(i, j):
        return 1.0 - abs(float(normals[i] @ normals[j]))

    visited = {seed}
    frontier = [(cost(seed, j), seed, j) for j in edges[seed]]
    heapq.heapify(frontier)
    while frontier:
        _, parent, child = heapq.heappop(frontier)
        if child in visited:
            continue
        if normals[parent] @ normals[child] < 0:
            normals[child] = -normals[child]
        visited.add(child)
        for j in edges[child] - visited:
            heapq.heappush(frontier, (cost(child, j), child, j))
    return normals


def taxel_normals(positions, neighbours: int = NEIGHBOURS) -> np.ndarray:
    """Estimate the outward surface normal at every taxel.

    See `plane_normals` for the fit and `orient_normals` for the sign.

    Args:
        positions: Taxel positions `(n, 3)`, in the sensor frame.
        neighbours: Points per plane fit, the taxel included.

    Returns:
        Unit normals `(n, 3)`, in the sensor frame.

    Raises:
        ValueError: If there are fewer than 3 taxels, which do not define a plane.
    """
    positions = np.asarray(positions, dtype=float)
    if len(positions) < 3:
        raise ValueError("at least 3 taxels are needed to estimate normals")
    return orient_normals(positions, plane_normals(positions, neighbours), neighbours)
