import numpy as np

from nb_touch.contact.active import active_taxels, taxel_norms


def test_norms():
    np.testing.assert_allclose(taxel_norms([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]), [5, 0])


def test_active_strictly_above_threshold():
    forces = [[0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.0, 0.3]]
    np.testing.assert_array_equal(active_taxels(forces, 0.2), [False, False, True])


def test_no_taxel():
    assert active_taxels(np.zeros((0, 3))).shape == (0,)
