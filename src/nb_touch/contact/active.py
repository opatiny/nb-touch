"""Which taxels are in contact in one frame.

Taken from `dextHand`, `src/dexthand/touch/contact.py`.
"""

import numpy as np

THRESHOLD = 0.2
"""Taxel force norm above which a taxel is in contact, in N. Value used on the ORCA
hand in `dextHand`."""


def taxel_norms(forces) -> np.ndarray:
    """Return the force norm `(n,)` of every taxel, from forces `(n, 3)`."""
    return np.linalg.norm(np.asarray(forces, dtype=float).reshape(-1, 3), axis=1)


def active_taxels(forces, threshold: float = THRESHOLD) -> np.ndarray:
    """Return a mask `(n,)` of the taxels whose force norm is above `threshold`.

    Args:
        forces: Taxel forces `(n, 3)`, in N.
        threshold: Force norm, in N.
    """
    return taxel_norms(forces) > threshold
