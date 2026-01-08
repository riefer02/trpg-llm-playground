"""Pilot license types for Lancer TTRPG.

Licenses represent a pilot's access to manufacturer equipment.
Each license has 3 ranks, unlocking progressively more powerful gear.

Note: This module contains only mechanical definitions (allowed under
the Lancer Third Party License). No copyrighted flavor text.
"""

from pydantic import Field
from core.shared.models import FrozenModel

from core.shared.enums import ManufacturerType as Manufacturer
from core.shared.id_helpers import LicenseIdField, FrameIdField

# Manufacturer full names (these are proper nouns, allowed)
MANUFACTURER_NAMES: dict[Manufacturer, str] = {
    "GMS": "General Massive Systems",
    "IPS-N": "Interplanetary Shipping-Northstar",
    "SSC": "Smith-Shimano Corpro",
    "HORUS": "HORUS",
    "HA": "Harrison Armory",
}


class LicenseDefinition(FrozenModel):
    """
    A license definition - the template for a manufacturer license.

    Each license corresponds to a frame and unlocks:
    - Rank 1: Basic frame systems and weapons
    - Rank 2: Advanced systems
    - Rank 3: The frame itself plus signature gear

    Note: The 'description' field has been intentionally removed
    to avoid including copyrighted flavor text.
    """

    id: LicenseIdField = Field(
        ..., description="Unique identifier (e.g., 'blackbeard', 'nelson')"
    )
    name: str = Field(..., description="Display name (usually the frame name)")
    manufacturer: Manufacturer
    frame_id: FrameIdField = Field(
        ..., description="ID of the frame this license provides"
    )


class License(FrozenModel):
    """
    A license that a pilot has ranks in.

    Pilots gain 1 license level per level up (starting at LL1).
    They can put this into a new license or increase an existing one.
    """

    license_id: LicenseIdField = Field(..., description="ID of the license definition")
    rank: int = Field(default=1, ge=1, le=3, description="Current rank (1-3)")

    def is_maxed(self) -> bool:
        """Check if this license is at maximum rank."""
        return self.rank >= 3


# Example license definitions (subset of IPS-N frames)
# Note: Only IDs, names, manufacturer - no flavor descriptions
EXAMPLE_LICENSES: list[LicenseDefinition] = [
    LicenseDefinition(
        id="blackbeard",
        name="BLACKBEARD",
        manufacturer="IPS-N",
        frame_id="blackbeard",
    ),
    LicenseDefinition(
        id="drake",
        name="DRAKE",
        manufacturer="IPS-N",
        frame_id="drake",
    ),
    LicenseDefinition(
        id="lancaster",
        name="LANCASTER",
        manufacturer="IPS-N",
        frame_id="lancaster",
    ),
    LicenseDefinition(
        id="nelson",
        name="NELSON",
        manufacturer="IPS-N",
        frame_id="nelson",
    ),
    LicenseDefinition(
        id="raleigh",
        name="RALEIGH",
        manufacturer="IPS-N",
        frame_id="raleigh",
    ),
    LicenseDefinition(
        id="tortuga",
        name="TORTUGA",
        manufacturer="IPS-N",
        frame_id="tortuga",
    ),
    LicenseDefinition(
        id="vlad",
        name="VLAD",
        manufacturer="IPS-N",
        frame_id="vlad",
    ),
]


def get_license_definition(license_id: LicenseIdField) -> LicenseDefinition | None:
    """Look up a license definition by ID."""
    for lic in EXAMPLE_LICENSES:
        if lic.id == license_id:
            return lic
    return None


def get_licenses_by_manufacturer(manufacturer: Manufacturer) -> list[LicenseDefinition]:
    """Get all licenses from a specific manufacturer."""
    return [lic for lic in EXAMPLE_LICENSES if lic.manufacturer == manufacturer]
