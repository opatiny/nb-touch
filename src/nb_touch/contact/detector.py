"""Contact detection for one sensor, frame after frame.

Adapted from `ContactDetector` in `dextHand`, `src/dexthand/touch/contact_detector.py`.
"""

import numpy as np

from nb_touch.contact.active import taxel_norms
from nb_touch.contact.contacts import Contact, ContactSettings, contacts_from_active
from nb_touch.contact.filters import ThresholdFilter
from nb_touch.geometry.frames import normals_from_rotations, with_estimated_rotations
from nb_touch.geometry.neighbours import taxel_neighbours
from nb_touch.geometry.transforms import rotate_vectors
from nb_touch.io.sensor_geometry import SensorGeometry


class ContactDetector:
    """Finds the contacts of one sensor in every frame.

    The geometry-only work (normals, neighbours) is done once here. Missing taxel
    rotations are estimated from the positions.

    Args:
        geometry: Sensor geometry.
        settings: Detection settings.
        active_filter: Filter deciding which taxels are in contact, with an
            `update(norms)` method (`nb_touch.contact.filters`). Default: plain
            threshold at `settings.threshold`.
    """

    def __init__(
        self,
        geometry: SensorGeometry,
        settings: ContactSettings = ContactSettings(),
        active_filter=None,
    ):
        geometry = with_estimated_rotations(geometry)
        self.positions = geometry.positions
        self.rotations = geometry.rotations
        self.normals = normals_from_rotations(geometry.rotations)
        self.neighbours = taxel_neighbours(geometry.positions)
        self.settings = settings
        self.filter = active_filter or ThresholdFilter(settings.threshold)

    def update(self, forces, *, in_taxel_frames: bool) -> list[Contact]:
        """Find the contacts in one frame.

        Args:
            forces: Taxel forces `(n, 3)`, in N.
            in_taxel_frames: True if each force is in its own taxel's frame, as in
                the recording format: it is rotated to the sensor frame first. False
                if the forces are already in the sensor frame, as `orca_core` streams
                them. Required, so the caller states which frame the forces are in.

        Returns:
            The contacts, largest CoP force first.
        """
        forces = np.asarray(forces, dtype=float)
        if in_taxel_frames:
            forces = rotate_vectors(self.rotations, forces)
        active = self.filter.update(taxel_norms(forces))
        return contacts_from_active(
            active, self.positions, self.normals, forces, self.neighbours, self.settings
        )
