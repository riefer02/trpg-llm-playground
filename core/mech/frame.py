"""Mech frame definitions and core systems for Lancer TTRPG."""

from pydantic import BaseModel, Field

from core.shared.enums import ManufacturerType, SizeClass
from core.shared.effects import MechanicalEffect
from core.mech.mounts import MountSlot


class CoreSystemDefinition(BaseModel):
    """Core system ability unique to a frame."""

    id: str = Field(..., description="Unique core system identifier")
    name: str = Field(..., description="Display name")
    effects: MechanicalEffect = Field(default_factory=MechanicalEffect)
    uses_per_mission: int = Field(default=1, ge=1)

    model_config = {"frozen": True}


class FrameTrait(BaseModel):
    """Passive trait provided by a mech frame."""

    name: str = Field(..., description="Display name")
    effects: MechanicalEffect = Field(default_factory=MechanicalEffect)

    model_config = {"frozen": True}


class MechFrameBaseStats(BaseModel):
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

    model_config = {"frozen": True}


class MechFrameDefinition(BaseModel):
    """Definition for a mech frame."""

    id: str = Field(..., description="Unique frame identifier")
    name: str = Field(..., description="Display name")
    manufacturer: ManufacturerType
    base_stats: MechFrameBaseStats
    mounts: list[MountSlot] = Field(default_factory=list)
    system_points: int = Field(default=0, ge=0)
    core_system: CoreSystemDefinition | None = None
    traits: list[FrameTrait] = Field(default_factory=list)

    model_config = {"frozen": True}
