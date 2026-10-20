import numpy as np

from nb_touch.contact.zones import connected_groups, contact_zones
from nb_touch.geometry.neighbours import taxel_neighbours
from nb_touch.geometry.synthetic import plane_sensor

# 5 x 5 grid, taxel index = row * 5 + col.
NEIGHBOURS = taxel_neighbours(plane_sensor(rows=5, cols=5).positions)


def mask(indices):
    active = np.zeros(25, dtype=bool)
    active[indices] = True
    return active


def test_two_separate_groups():
    groups = connected_groups(mask([0, 1, 5, 23, 24]), NEIGHBOURS)
    assert [g.tolist() for g in groups] == [[0, 1, 5], [23, 24]]


def test_diagonal_taxels_are_one_group():
    assert len(connected_groups(mask([0, 6, 12]), NEIGHBOURS)) == 1


def test_nothing_active():
    assert connected_groups(mask([]), NEIGHBOURS) == []


def test_small_zones_dropped():
    zones = contact_zones(mask([0, 1, 5, 23, 24]), NEIGHBOURS, min_taxels=3)
    assert [z.tolist() for z in zones] == [[0, 1, 5]]


def test_min_one_keeps_single_taxels():
    assert len(contact_zones(mask([0, 24]), NEIGHBOURS, min_taxels=1)) == 2
