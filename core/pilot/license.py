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


# All license definitions organized by manufacturer
# Note: Only IDs, names, manufacturer - no flavor descriptions
ALL_LICENSES: list[LicenseDefinition] = [
    # IPS-N Licenses (7)
    LicenseDefinition(
        id="blackbeard",
        name="BLACKBEARD",
        manufacturer="IPS-N",
        frame_id="ipsn_blackbeard",
    ),
    LicenseDefinition(
        id="drake",
        name="DRAKE",
        manufacturer="IPS-N",
        frame_id="ipsn_drake",
    ),
    LicenseDefinition(
        id="lancaster",
        name="LANCASTER",
        manufacturer="IPS-N",
        frame_id="ipsn_lancaster",
    ),
    LicenseDefinition(
        id="nelson",
        name="NELSON",
        manufacturer="IPS-N",
        frame_id="ipsn_nelson",
    ),
    LicenseDefinition(
        id="raleigh",
        name="RALEIGH",
        manufacturer="IPS-N",
        frame_id="ipsn_raleigh",
    ),
    LicenseDefinition(
        id="tortuga",
        name="TORTUGA",
        manufacturer="IPS-N",
        frame_id="ipsn_tortuga",
    ),
    LicenseDefinition(
        id="vlad",
        name="VLAD",
        manufacturer="IPS-N",
        frame_id="ipsn_vlad",
    ),
    # SSC Licenses (7)
    LicenseDefinition(
        id="black_witch",
        name="BLACK WITCH",
        manufacturer="SSC",
        frame_id="ssc_black_witch",
    ),
    LicenseDefinition(
        id="deaths_head",
        name="DEATH'S HEAD",
        manufacturer="SSC",
        frame_id="ssc_deaths_head",
    ),
    LicenseDefinition(
        id="dusk_wing",
        name="DUSK WING",
        manufacturer="SSC",
        frame_id="ssc_dusk_wing",
    ),
    LicenseDefinition(
        id="metalmark",
        name="METALMARK",
        manufacturer="SSC",
        frame_id="ssc_metalmark",
    ),
    LicenseDefinition(
        id="monarch",
        name="MONARCH",
        manufacturer="SSC",
        frame_id="ssc_monarch",
    ),
    LicenseDefinition(
        id="mourning_cloak",
        name="MOURNING CLOAK",
        manufacturer="SSC",
        frame_id="ssc_mourning_cloak",
    ),
    LicenseDefinition(
        id="swallowtail",
        name="SWALLOWTAIL",
        manufacturer="SSC",
        frame_id="ssc_swallowtail",
    ),
    # HORUS Licenses (7)
    LicenseDefinition(
        id="balor",
        name="BALOR",
        manufacturer="HORUS",
        frame_id="horus_balor",
    ),
    LicenseDefinition(
        id="goblin",
        name="GOBLIN",
        manufacturer="HORUS",
        frame_id="horus_goblin",
    ),
    LicenseDefinition(
        id="gorgon",
        name="GORGON",
        manufacturer="HORUS",
        frame_id="horus_gorgon",
    ),
    LicenseDefinition(
        id="hydra",
        name="HYDRA",
        manufacturer="HORUS",
        frame_id="horus_hydra",
    ),
    LicenseDefinition(
        id="manticore",
        name="MANTICORE",
        manufacturer="HORUS",
        frame_id="horus_manticore",
    ),
    LicenseDefinition(
        id="minotaur",
        name="MINOTAUR",
        manufacturer="HORUS",
        frame_id="horus_minotaur",
    ),
    LicenseDefinition(
        id="pegasus",
        name="PEGASUS",
        manufacturer="HORUS",
        frame_id="horus_pegasus",
    ),
    # HA Licenses (7)
    LicenseDefinition(
        id="sherman",
        name="SHERMAN",
        manufacturer="HA",
        frame_id="ha_sherman",
    ),
    LicenseDefinition(
        id="saladin",
        name="SALADIN",
        manufacturer="HA",
        frame_id="ha_saladin",
    ),
    LicenseDefinition(
        id="napoleon",
        name="NAPOLEON",
        manufacturer="HA",
        frame_id="ha_napoleon",
    ),
    LicenseDefinition(
        id="iskander",
        name="ISKANDER",
        manufacturer="HA",
        frame_id="ha_iskander",
    ),
    LicenseDefinition(
        id="tokugawa",
        name="TOKUGAWA",
        manufacturer="HA",
        frame_id="ha_tokugawa",
    ),
    LicenseDefinition(
        id="genghis",
        name="GENGHIS",
        manufacturer="HA",
        frame_id="ha_genghis",
    ),
    LicenseDefinition(
        id="barbarossa",
        name="BARBAROSSA",
        manufacturer="HA",
        frame_id="ha_barbarossa",
    ),
]


def get_license_definition(license_id: LicenseIdField) -> LicenseDefinition | None:
    """Look up a license definition by ID."""
    for lic in ALL_LICENSES:
        if lic.id == license_id:
            return lic
    return None


def get_licenses_by_manufacturer(manufacturer: Manufacturer) -> list[LicenseDefinition]:
    """Get all licenses from a specific manufacturer."""
    return [lic for lic in ALL_LICENSES if lic.manufacturer == manufacturer]
