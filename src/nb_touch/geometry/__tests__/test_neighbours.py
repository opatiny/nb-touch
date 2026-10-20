import numpy as np
import pytest

from nb_touch.geometry.neighbours import taxel_neighbours, taxel_spacing
from nb_touch.geometry.synthetic import plane_sensor


def test_spacing_of_grid():
    assert taxel_spacing(plane_sensor(spacing=0.004).positions) == pytest.approx(0.004)


def test_spacing_needs_two_taxels():
    with pytest.raises(ValueError, match="at least 2"):
        taxel_spacing(np.zeros((1, 3)))


def test_grid_neighbours_include_diagonals():
    neighbours = taxel_neighbours(plane_sensor(rows=3, cols=3).positions)
    # Centre taxel touches all 8 others, a corner touches 3.
    assert len(neighbours[4]) == 8
    assert len(neighbours[0]) == 3
    assert 0 not in neighbours[0]


def test_neighbours_are_symmetric():
    neighbours = taxel_neighbours(plane_sensor().positions)
    for i, row in enumerate(neighbours):
        for j in row:
            assert i in neighbours[j]


def test_single_and_no_taxel():
    assert [len(n) for n in taxel_neighbours(np.zeros((1, 3)))] == [0]
    assert taxel_neighbours(np.zeros((0, 3))) == []
