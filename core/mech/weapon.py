"""Mech weapon types for Lancer TTRPG."""

from typing import Literal
from pydantic import BaseModel, Field

from core.shared.enums import DamageType


WeaponSize = Literal["aux", "main", "heavy", "superheavy"]
WeaponType = Literal["cqb", "rifle", "launcher", "cannon", "melee", "nexus"]
WeaponDamageType = DamageType | Literal["heat"]


class MechWeaponDefinition(BaseModel):
    """Definition for a mech weapon."""

    id: str = Field(..., description="Unique weapon identifier")
    name: str = Field(..., description="Display name")
    size: WeaponSize
    weapon_type: WeaponType
    damage_type: WeaponDamageType
    limited_uses: int | None = Field(default=None, ge=0)

    model_config = {"frozen": True}
