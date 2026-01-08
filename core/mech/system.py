"""Mech system definitions for Lancer TTRPG."""

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel

from core.shared.effects import MechanicalEffect
from core.shared.enums import (
    ActionType,
    CoverType,
    RangeType,
    SaveType,
    SizeClass,
    SystemType,
)
from core.shared.payloads import (
    AreaEffect,
    DamageSpec,
    GrenadePayload,
    MineDetonation,
    MinePayload,
)
from core.shared.id_helpers import SystemIdField, LicenseIdField


SystemTagType = Literal[
    "ai",
    "shield",
    "deployable",
    "drone",
    "protocol",
    "reaction",
    "mod",
    "quick_action",
    "full_action",
    "grenade",
    "mine",
]


class SystemTag(FrozenModel):
    """Structured tag for a mech system."""

    tag: SystemTagType
    value: int | None = None


EnvironmentType = Literal["low_g", "zero_g", "submarine"]


class FlightEffect(FrozenModel):
    """Flight behavior granted by a system."""

    mode: Literal["always", "move", "boost", "move_or_boost", "environmental"]
    environment: list[EnvironmentType] | None = None
    must_end_on_surface: bool = False
    heat_on_turn_end: int | Literal["size_plus_1"] | None = None
    ignores_slowed_in_environment: bool = False


class DeployableObject(FrozenModel):
    """Single deployable object definition."""

    size: int = Field(..., ge=0)
    cover: CoverType | None = None
    evasion: int = Field(default=5, ge=0)
    hp: int = Field(default=10, ge=0)


class DeployableEffect(FrozenModel):
    """Deployable system payload."""

    count: int = Field(default=1, ge=1)
    obj: DeployableObject
    pickup_action: ActionType | None = None


DroneReactionTrigger = Literal["ally_hit_target_within_range"]


class DroneReaction(FrozenModel):
    """Reaction granted by a drone."""

    name: str
    trigger: DroneReactionTrigger
    range: int = Field(..., ge=0)
    damage: DamageSpec
    uses_per_round: int = Field(default=1, ge=1)


class DronePayload(FrozenModel):
    """Drone system payload."""

    name: str
    size: SizeClass
    hp: int = Field(default=10, ge=0)
    evasion: int = Field(default=10, ge=0)
    e_defense: int = Field(default=10, ge=0)
    reactions: list[DroneReaction] = Field(default_factory=list)
    deploy_range_type: RangeType = "sensors"
    deploy_requires_line_of_sight: bool = False
    invisible: bool = False
    attach_to_surface: bool = True
    redeploy_action: ActionType | None = None
    redeploy_requires_line_of_sight: bool = False
    recall_action: ActionType | None = None
    recall_requires_line_of_sight: bool = False


class MechSystemDefinition(FrozenModel):
    """Definition for a mech system or chassis mod."""

    id: SystemIdField = Field(..., description="Unique system identifier")
    name: str = Field(..., description="Display name")
    system_type: SystemType = "system"
    sp_cost: int = Field(default=0, ge=0)
    license_id: LicenseIdField | None = Field(
        default=None,
        description="License ID required to use this system (None for GMS/general)",
    )
    license_rank: int | None = Field(
        default=None,
        ge=1,
        le=3,
        description="Required license rank if gated by a specific license",
    )
    unique: bool = False
    limited_uses: int | None = Field(default=None, ge=0)
    tags: list[SystemTag] = Field(default_factory=list)
    grenades: list[GrenadePayload] = Field(default_factory=list)
    mines: list[MinePayload] = Field(default_factory=list)
    flight: FlightEffect | None = None
    deployable: DeployableEffect | None = None
    drone: DronePayload | None = None
    effects: MechanicalEffect = Field(default_factory=MechanicalEffect)
