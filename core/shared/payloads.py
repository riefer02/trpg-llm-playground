"""Shared payload types for area effects and damage."""

from typing import Literal
from pydantic import Field

from core.shared.dice import DiceExpression
from core.shared.enums import ActionType, CoverType, DamageType, RangeType, SaveType
from core.shared.models import FrozenModel


AttackVs = Literal["evasion", "e_defense"]


class DamageSpecBase(FrozenModel):
    """Common damage fields for gear/system payloads."""

    flat: int = 0
    ap: bool = False


class DamageSpec(DamageSpecBase):
    """Damage specification for mech systems."""

    damage_type: DamageType | Literal["heat", "burn"]
    dice: DiceExpression | None = None


class PilotDamageSpec(DamageSpecBase):
    """Damage specification for pilot gear."""

    damage_type: DamageType
    damage_type_options: list[DamageType] | None = None


class AreaEffectBase(FrozenModel):
    """Shared fields for area effects."""

    pattern: RangeType
    size: int = Field(..., ge=0)
    attack_vs: AttackVs | None = None


class AreaEffect(AreaEffectBase):
    """Area effect payload (grenade, mine, etc)."""

    range: int | None = Field(default=None, ge=0)
    duration: Literal["instant", "end_of_turn", "end_of_next_turn", "scene"] = "instant"
    cover: CoverType | None = None
    damage: DamageSpec | None = None
    save: SaveType | None = None
    half_on_success: bool = True
    object_damage: DamageSpec | None = None
    objects_auto_hit: bool = False


class PilotAreaEffect(AreaEffectBase):
    """Area effect payload for pilot gear."""

    damage: PilotDamageSpec | None = None
    object_damage: PilotDamageSpec | None = None
    objects_auto_hit: bool = False


PilotWeaponRangeType = Literal["range", "threat"]


class PilotWeaponProfile(FrozenModel):
    """Weapon profile for pilot-scale weapons."""

    range_type: PilotWeaponRangeType
    range: int = Field(..., ge=0)
    damage: PilotDamageSpec
    loaded: bool = True


class GrenadePayloadBase(FrozenModel):
    """Shared grenade fields."""

    name: str
    range: int = Field(..., ge=0)


class GrenadePayload(GrenadePayloadBase):
    """Grenade option for a system."""

    area: AreaEffect


class PilotGrenadePayload(GrenadePayloadBase):
    """Grenade option for pilot gear."""

    area: PilotAreaEffect


MineDetonation = Literal[
    "adjacent_movement",
    "ally_adjacent_movement",
    "manual",
]


class MinePayload(FrozenModel):
    """Mine option for a system."""

    name: str
    area: AreaEffect
    detonation: MineDetonation = "adjacent_movement"
    can_attach_to_terrain: bool = False
    detonation_action: ActionType | None = None
