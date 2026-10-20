"""Taxel forces to contacts, each with its CoP, CoP force and stats."""

from nb_touch.contact.contacts import Contact, ContactSettings, find_contacts
from nb_touch.contact.cop_model import (
    cop_normal,
    cop_position,
    cop_to_taxel_forces,
    taxel_force_matrices,
    taxel_to_cop_force,
)
from nb_touch.contact.detector import ContactDetector
from nb_touch.contact.filters import make_filter

__all__ = [
    "Contact",
    "ContactDetector",
    "ContactSettings",
    "cop_normal",
    "cop_position",
    "cop_to_taxel_forces",
    "find_contacts",
    "make_filter",
    "taxel_force_matrices",
    "taxel_to_cop_force",
]
