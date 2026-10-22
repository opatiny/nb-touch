"""From taxel forces to contacts, each with its CoP, CoP force and stats. See
`docs/theory/cop.md`.

Stats and normal/shear split adapted from `dextHand`,
`src/dexthand/touch/contact_zone.py`.
"""

from dataclasses import dataclass, replace

import numpy as np

from nb_touch.contact.active import THRESHOLD, active_taxels, taxel_norms
from nb_touch.contact.cop_model import (
    REGULARISATION,
    SIGMA_SPACING_FACTOR,
    cop_normal,
    cop_position,
    spacing_sigma,
    taxel_force_matrices,
    taxel_to_cop_force,
)
from nb_touch.contact.zones import MIN_TAXELS, contact_zones


@dataclass(frozen=True)
class ContactSettings:
    """Settings of the contact detection.

    Attributes:
        threshold: Taxel force norm above which a taxel is in contact, in N.
        min_taxels: Fewest taxels a contact needs.
        sigma: Spread of the CoP model's Gaussian weights, in m. None returns
            `sigma_factor` times the median taxel spacing of the sensor, see
            `from_geometry`.
        sigma_factor: σ over the median taxel spacing when `sigma` is None.
        regularisation: Ridge regularisation λ of the CoP force solve.
    """

    threshold: float = THRESHOLD
    min_taxels: int = MIN_TAXELS
    sigma: float | None = None
    sigma_factor: float = SIGMA_SPACING_FACTOR
    regularisation: float = REGULARISATION

    def from_geometry(self, positions) -> "ContactSettings":
        """Return these settings with `sigma` set for a sensor.

        If sigma is None, a default value is used: `sigma_factor` times the median
        taxel spacing of the sensor, or `SIGMA_M` for a sensor of a single taxel
        (`spacing_sigma`). A set `sigma` is kept.

        Args:
            positions: Taxel positions `(n, 3)` of the whole sensor, in m.
        """
        if self.sigma is not None:
            return self
        return replace(self, sigma=spacing_sigma(positions, self.sigma_factor))


@dataclass(frozen=True)
class Contact:
    """One contact on a sensor. Vectors in the sensor frame.

    Attributes:
        taxels: Indices of the contact's taxels.
        cop: CoP position `(3,)`, in m.
        normal: Unit outward surface normal at the CoP `(3,)`.
        cop_force: CoP force `(3,)` from the model's ridge solve, in N, acting on
            the sensor: against `normal` for a push.
        total_force: Sum of the taxel forces `(3,)`, in N.
        normal_force: Push of `cop_force`, `-cop_force · normal`, in N: positive
            for a push, negative for a pull.
        shear_force: Norm of the part of `cop_force` across `normal`, in N.
        slip_ratio: `shear_force / normal_force`, NaN when the normal force is not
            positive. The contact slips once it reaches the friction coefficient.
        max_force: Largest taxel force norm, in N.
        mean_force: Mean taxel force norm, in N.
    """

    taxels: np.ndarray
    cop: np.ndarray
    normal: np.ndarray
    cop_force: np.ndarray
    total_force: np.ndarray
    normal_force: float
    shear_force: float
    slip_ratio: float
    max_force: float
    mean_force: float

    @property
    def count(self) -> int:
        """Number of taxels in the contact."""
        return len(self.taxels)


def split_force(force, normal) -> tuple[float, float, float]:
    """Split a force into its part along a normal and its part across it.

    Args:
        force: Force `(3,)` acting on the sensor, in N: a push points against the
            outward normal.
        normal: Unit outward normal `(3,)`.

    Returns:
        `(normal_force, shear_force, slip_ratio)`, the normal force `-force · normal`
        positive for a push, the ratio NaN when the normal force is not positive.
        All NaN when `normal` is.
    """
    force = np.asarray(force, dtype=float)
    normal = np.asarray(normal, dtype=float)
    normal_force = float(-force @ normal)
    shear_force = float(np.linalg.norm(force + normal_force * normal))
    ratio = shear_force / normal_force if normal_force > 0.0 else float("nan")
    return normal_force, shear_force, ratio


def contact_from_taxels(
    taxels, positions, normals, forces, settings: ContactSettings = ContactSettings()
) -> Contact:
    """Build the contact of a zone, using only the zone's taxels.

    Args:
        taxels: Indices of the zone's taxels, at least one with a non-zero force.
        positions: Taxel positions `(n, 3)` of the whole sensor, in m.
        normals: Unit outward taxel normals `(n, 3)` of the whole sensor.
        forces: Taxel forces `(n, 3)` of the whole sensor, in the sensor frame, in N.
        settings: `sigma` and `regularisation` are used.
    """
    settings = settings.from_geometry(positions)
    taxels = np.asarray(taxels, dtype=int)
    points = np.asarray(positions, dtype=float)[taxels]
    zone_normals = np.asarray(normals, dtype=float)[taxels]
    zone_forces = np.asarray(forces, dtype=float)[taxels]
    cop = cop_position(points, zone_forces)
    normal = cop_normal(points, zone_normals, cop)
    matrices = taxel_force_matrices(points, zone_normals, cop, normal, settings.sigma)
    cop_force = taxel_to_cop_force(zone_forces, matrices, settings.regularisation)
    normal_force, shear_force, slip_ratio = split_force(cop_force, normal)
    norms = taxel_norms(zone_forces)
    return Contact(
        taxels=taxels,
        cop=cop,
        normal=normal,
        cop_force=cop_force,
        total_force=zone_forces.sum(axis=0),
        normal_force=normal_force,
        shear_force=shear_force,
        slip_ratio=slip_ratio,
        max_force=float(norms.max()),
        mean_force=float(norms.mean()),
    )


def contacts_from_active(
    active,
    positions,
    normals,
    forces,
    neighbours: list[np.ndarray],
    settings: ContactSettings = ContactSettings(),
) -> list[Contact]:
    """Group the taxels already marked active into contacts.

    Args:
        active: Mask `(n,)` of the taxels in contact, for example from a filter.
        positions: Taxel positions `(n, 3)`, in m.
        normals: Unit outward taxel normals `(n, 3)`.
        forces: Taxel forces `(n, 3)`, in the sensor frame, in N.
        neighbours: Per taxel, its neighbours, from `taxel_neighbours`.
        settings: `min_taxels`, `sigma` and `regularisation` are used.

    Returns:
        The contacts, largest CoP force first. Empty when nothing touches.
    """
    settings = settings.from_geometry(positions)
    zones = contact_zones(active, neighbours, settings.min_taxels)
    zones = [zone for zone in zones if taxel_norms(np.asarray(forces)[zone]).any()]
    contacts = [
        contact_from_taxels(zone, positions, normals, forces, settings)
        for zone in zones
    ]
    return sorted(contacts, key=lambda contact: -np.linalg.norm(contact.cop_force))


def find_contacts(
    positions,
    normals,
    forces,
    neighbours: list[np.ndarray],
    settings: ContactSettings = ContactSettings(),
) -> list[Contact]:
    """Find the contacts of one sensor in one frame, without filtering over time.

    Args:
        positions: Taxel positions `(n, 3)`, in m.
        normals: Unit outward taxel normals `(n, 3)`.
        forces: Taxel forces `(n, 3)`, in the sensor frame, in N.
        neighbours: Per taxel, its neighbours, from `taxel_neighbours`.
        settings: Detection settings.

    Returns:
        The contacts, largest CoP force first. Empty when nothing touches.
    """
    active = active_taxels(forces, settings.threshold)
    return contacts_from_active(
        active, positions, normals, forces, neighbours, settings
    )
