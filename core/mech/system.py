"""Mech system definitions for Lancer TTRPG."""

from typing import Literal
from pydantic import BaseModel, Field

from core.shared.effects import MechanicalEffect
from core.shared.enums import ActionType, CoverType, DamageType, RangeType, SaveType, SizeClass, SystemType
from core.shared.dice import DiceExpression


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


class SystemTag(BaseModel):
    """Structured tag for a mech system."""

    tag: SystemTagType
    value: int | None = None

    model_config = {"frozen": True}


class DamageSpec(BaseModel):
    """Damage specification for area effects."""

    damage_type: DamageType | Literal["heat", "burn"]
    dice: DiceExpression | None = None
    flat: int = 0
    ap: bool = False

    model_config = {"frozen": True}


class AreaEffect(BaseModel):
    """Area effect payload (grenade, mine, etc)."""

    pattern: RangeType
    size: int = Field(..., ge=0)
    range: int | None = Field(default=None, ge=0)
    duration: Literal["instant", "end_of_turn", "end_of_next_turn", "scene"] = "instant"
    cover: CoverType | None = None
    damage: DamageSpec | None = None
    save: SaveType | None = None
    half_on_success: bool = True
    attack_vs: Literal["evasion", "e_defense"] | None = None
    object_damage: DamageSpec | None = None
    objects_auto_hit: bool = False

    model_config = {"frozen": True}


class GrenadePayload(BaseModel):
    """Grenade option for a system."""

    name: str
    range: int = Field(..., ge=0)
    area: AreaEffect

    model_config = {"frozen": True}


MineDetonation = Literal[
    "adjacent_movement",
    "ally_adjacent_movement",
    "manual",
]


class MinePayload(BaseModel):
    """Mine option for a system."""

    name: str
    area: AreaEffect
    detonation: MineDetonation = "adjacent_movement"
    can_attach_to_terrain: bool = False
    detonation_action: ActionType | None = None

    model_config = {"frozen": True}


EnvironmentType = Literal["low_g", "zero_g", "submarine"]


class FlightEffect(BaseModel):
    """Flight behavior granted by a system."""

    mode: Literal["always", "move", "boost", "move_or_boost", "environmental"]
    environment: list[EnvironmentType] | None = None
    must_end_on_surface: bool = False
    heat_on_turn_end: int | Literal["size_plus_1"] | None = None
    ignores_slowed_in_environment: bool = False

    model_config = {"frozen": True}


class DeployableObject(BaseModel):
    """Single deployable object definition."""

    size: int = Field(..., ge=0)
    cover: CoverType | None = None
    evasion: int = Field(default=5, ge=0)
    hp: int = Field(default=10, ge=0)

    model_config = {"frozen": True}


class DeployableEffect(BaseModel):
    """Deployable system payload."""

    count: int = Field(default=1, ge=1)
    obj: DeployableObject
    pickup_action: ActionType | None = None

    model_config = {"frozen": True}


DroneReactionTrigger = Literal["ally_hit_target_within_range"]


class DroneReaction(BaseModel):
    """Reaction granted by a drone."""

    name: str
    trigger: DroneReactionTrigger
    range: int = Field(..., ge=0)
    damage: DamageSpec
    uses_per_round: int = Field(default=1, ge=1)

    model_config = {"frozen": True}


class DronePayload(BaseModel):
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

    model_config = {"frozen": True}


class MechSystemDefinition(BaseModel):
    """Definition for a mech system or chassis mod."""

    id: str = Field(..., description="Unique system identifier")
    name: str = Field(..., description="Display name")
    system_type: SystemType = "system"
    sp_cost: int = Field(default=0, ge=0)
    license_id: str | None = Field(
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

    model_config = {"frozen": True}
