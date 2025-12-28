"""Mech weapon types for Lancer TTRPG."""

from typing import Literal
from pydantic import BaseModel, Field

from core.shared.enums import DamageType
from core.shared.dice import DiceExpression
from core.shared.enums import RangeType
from core.shared.effects import MechanicalEffect


WeaponSize = Literal["aux", "main", "heavy", "superheavy"]
WeaponType = Literal["cqb", "rifle", "launcher", "cannon", "melee", "nexus"]
WeaponDamageType = DamageType | Literal["heat"]
WeaponTagType = Literal[
    "accurate",
    "inaccurate",
    "loading",
    "ordnance",
    "arcing",
    "seeking",
    "smart",
    "ap",
    "overkill",
    "reliable",
    "heat_self",
    "heat_target",
    "burn",
    "knockback",
    "limited",
    "threat",
    "thrown",
    "line",
    "cone",
    "blast",
    "burst",
]


class WeaponRange(BaseModel):
    """Range profile for a weapon (including threat, blast, etc)."""

    range_type: RangeType
    value: int = Field(..., ge=0)

    model_config = {"frozen": True}


class WeaponDamage(BaseModel):
    """Damage component for a weapon."""

    damage_type: WeaponDamageType
    dice: DiceExpression | None = None
    flat: int = 0

    model_config = {"frozen": True}


class WeaponTag(BaseModel):
    """A weapon tag with an optional numeric value."""

    tag: WeaponTagType
    value: int | None = None

    model_config = {"frozen": True}


class MechWeaponDefinition(BaseModel):
    """Definition for a mech weapon."""

    id: str = Field(..., description="Unique weapon identifier")
    name: str = Field(..., description="Display name")
    size: WeaponSize
    weapon_type: WeaponType
    damage_type: WeaponDamageType | None = Field(
        default=None,
        description="Base damage type (None for non-damaging weapons)",
    )
    license_id: str | None = Field(
        default=None,
        description="License ID required to use this weapon (None for GMS/general)",
    )
    license_rank: int | None = Field(
        default=None,
        ge=1,
        le=3,
        description="Required license rank if gated by a specific license",
    )
    integrated_only: bool = False
    integrated_frame_id: str | None = Field(
        default=None,
        description="Frame ID that provides this weapon as an integrated mount",
    )
    unique: bool = False
    ranges: list[WeaponRange] = Field(default_factory=list)
    damage: list[WeaponDamage] = Field(default_factory=list)
    tags: list[WeaponTag] = Field(default_factory=list)
    limited_uses: int | None = Field(default=None, ge=0)
    effects: MechanicalEffect = Field(default_factory=MechanicalEffect)

    model_config = {"frozen": True}
