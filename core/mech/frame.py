"""Mech frame definitions and core systems for Lancer TTRPG."""

from pydantic import Field
from core.shared.models import FrozenModel

from core.shared.enums import ManufacturerType, SizeClass
from core.shared.effects import MechanicalEffect
from core.mech.mounts import MountSlot
from core.mech.weapon import MountlessWeaponDefinition
from core.shared.id_helpers import SystemIdField, FrameIdField, LicenseIdField


class CoreSystemDefinition(FrozenModel):
    """Core system ability unique to a frame."""

    id: SystemIdField = Field(..., description="Unique core system identifier")
    name: str = Field(..., description="Display name")
    effects: MechanicalEffect = Field(default_factory=MechanicalEffect)
    mountless_weapons: list[MountlessWeaponDefinition] = Field(default_factory=list)
    uses_per_mission: int = Field(default=1, ge=1)


class FrameTrait(FrozenModel):
    """Passive trait provided by a mech frame."""

    name: str = Field(..., description="Display name")
    effects: MechanicalEffect = Field(default_factory=MechanicalEffect)


class MechFrameBaseStats(FrozenModel):
    """Base statistics provided by a frame before pilot bonuses."""

    size: SizeClass
    armor: int = Field(default=0, ge=0, le=4)
    hp: int = Field(default=0, ge=0)
    evasion: int = Field(default=0, ge=0)
    e_defense: int = Field(default=0, ge=0)
    speed: int = Field(default=0, ge=0)
    sensor_range: int = Field(default=0, ge=0)
    tech_attack: int = Field(default=0)
    heat_cap: int = Field(default=0, ge=0)
    repair_cap: int = Field(default=0, ge=0)
    save_target: int = Field(default=10, ge=0)
    structure: int = Field(default=4, ge=0)


class MechFrameDefinition(FrozenModel):
    """Definition for a mech frame."""

    id: FrameIdField = Field(..., description="Unique frame identifier")
    name: str = Field(..., description="Display name")
    manufacturer: ManufacturerType
    license_id: LicenseIdField | None = Field(
        default=None,
        description="License ID required to use this frame (None for GMS/general)",
    )
    license_rank: int | None = Field(
        default=2,
        ge=1,
        le=3,
        description="Required license rank for this frame (typically rank II)",
    )
    base_stats: MechFrameBaseStats
    mounts: list[MountSlot] = Field(default_factory=list)
    system_points: int = Field(default=0, ge=0)
    core_system: CoreSystemDefinition | None = None
    traits: list[FrameTrait] = Field(default_factory=list)


def collect_frame_trait_effects(frame: MechFrameDefinition) -> list[MechanicalEffect]:
    """Collect all passive mechanical effects from frame traits.

    Args:
        frame: The frame definition to collect effects from.

    Returns:
        A list of MechanicalEffect objects from all frame traits.
    """
    return [trait.effects for trait in frame.traits]


def get_core_power_effects(frame: MechFrameDefinition) -> MechanicalEffect | None:
    """Get the mechanical effects from a frame's core system.

    Args:
        frame: The frame definition to get core power effects from.

    Returns:
        The MechanicalEffect from the core system, or None if no core system.
    """
    if frame.core_system is None:
        return None
    return frame.core_system.effects
