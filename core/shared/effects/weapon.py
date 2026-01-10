"""Weapon modification and granting effects.

Effects for weapon modifications, grants, and AI control.

Effects:
    - WeaponTagGrant: Adds tags to weapons
    - WeaponRangeSpec: Specifies weapon range entries
    - WeaponSizeBonus: Size-based bonuses for weapons
    - WeaponGrantEffect: Grants a weapon profile
    - WeaponModEffect: Modifies weapon properties
    - WeaponSpinUpEffect: Spin-up mode for weapons
    - WeaponAIControlEffect: AI control over weapons
    - AISystemLimitEffect: AI system installation limits
    - AIControlTransferEffect: AI control transfer

See Also:
    - PR2 4444-4490: Weapon systems
"""

from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from pydantic import Field

from core.shared.models import FrozenModel
from core.shared.enums import ActionType, RangeType
from core.shared.id_helpers import WeaponIdField
from core.shared.payloads import DamageSpec
from core.shared.effects.types import (
    EffectTargetNoAll,
    UsesPer,
    WeaponSizeType,
    WeaponTypeType,
)
from core.shared.effects.conditions import EffectCondition

if TYPE_CHECKING:
    from core.shared.effects.core import MechanicalEffect

__all__ = [
    "WeaponTagGrant",
    "WeaponRangeSpec",
    "WeaponSizeBonus",
    "WeaponGrantEffect",
    "WeaponModEffect",
    "WeaponSpinUpEffect",
    "WeaponAIControlEffect",
    "AISystemLimitEffect",
    "AIControlTransferEffect",
]


class WeaponTagGrant(FrozenModel):
    """
    Tag granted to a weapon.

    Examples:
        WeaponTagGrant(tag="ordnance")
    """

    tag: str
    value: int | None = None


class WeaponRangeSpec(FrozenModel):
    """
    Range entry for a granted weapon profile.

    Examples:
        WeaponRangeSpec(range_type="range", value=8)
    """

    range_type: RangeType
    value: int = Field(..., ge=0)


class WeaponSizeBonus(FrozenModel):
    """
    Size-based bonus for a weapon mod.

    Examples:
        WeaponSizeBonus(size="main", burn=2)
    """

    size: WeaponSizeType
    burn: int = Field(..., ge=0)


class WeaponGrantEffect(FrozenModel):
    """
    Grants a specific weapon profile as part of an effect.

    Examples:
        WeaponGrantEffect(
            weapon_id="nuclear_cavalier_fuel_rod_gun",
            name="Fuel Rod Gun",
            size="main",
            weapon_type="cqb",
            ranges=[WeaponRangeSpec(range_type="range", value=8)],
            damage=[DamageSpec(damage_type="energy", dice=DiceExpression.parse("1d3+2"))],
            limited_uses=3,
            integrated_mount=True,
        )
    """

    weapon_id: WeaponIdField | None = None
    name: str
    size: WeaponSizeType
    weapon_type: WeaponTypeType
    ranges: list[WeaponRangeSpec] = Field(default_factory=list)
    damage: list[DamageSpec] = Field(default_factory=list)
    tags: list[WeaponTagGrant] = Field(default_factory=list)
    limited_uses: int | None = Field(default=None, ge=0)
    unique: bool = False
    integrated_mount: bool = False
    target: EffectTargetNoAll = "self"
    condition: EffectCondition | None = None


class WeaponModEffect(FrozenModel):
    """
    Modifies a selected weapon with tags or bonuses.

    Examples:
        WeaponModEffect(
            allowed_weapon_types=["launcher", "cannon"],
            range_bonus=5,
            add_tags=[WeaponTagGrant(tag="ordnance")],
        )
    """

    allowed_weapon_types: list[WeaponTypeType] = Field(default_factory=list)
    allowed_weapon_sizes: list[WeaponSizeType] = Field(default_factory=list)
    range_bonus: int = 0
    add_tags: list[WeaponTagGrant] = Field(default_factory=list)
    burn_by_size: list[WeaponSizeBonus] = Field(default_factory=list)
    increase_existing_burn: bool = False
    damage_unreducible: bool = False
    condition: EffectCondition | None = None


class WeaponSpinUpEffect(FrozenModel):
    """
    Spin-up mode for a weapon with alternate behavior while active.

    Examples:
        WeaponSpinUpEffect(
            spin_up_action_type="quick",
            spin_down_action_type="free",
            spin_down_timing="start_of_turn",
        )
    """

    spin_up_action_type: ActionType
    spin_down_action_type: ActionType | None = None
    spin_down_timing: Literal["start_of_turn", "end_of_turn", "anytime"] | None = None
    allow_skirmish_with_base_profile: bool = False
    requires_barrage_while_spun_up: bool = False
    effects_while_spun_up: "MechanicalEffect | None" = None
    condition: EffectCondition | None = None


class WeaponAIControlEffect(FrozenModel):
    """
    AI-controlled weapon handling rules.

    Examples:
        WeaponAIControlEffect(
            allowed_weapon_sizes=["aux", "main", "heavy"],
            free_attack_difficulty=2,
            cannot_fire_if_used_this_turn=True,
        )
    """

    allowed_weapon_sizes: list[WeaponSizeType] = Field(default_factory=list)
    free_attack_action_type: ActionType = "free"
    free_attack_uses_per: UsesPer = "round"
    free_attack_difficulty: int = 0
    cannot_fire_if_used_this_turn: bool = False
    weapon_locked_until_next_turn_after_ai_attack: bool = False
    ai_selects_target_if_unshackled: bool = False
    ai_control_scope: Literal["weapon_only", "mech"] = "weapon_only"
    condition: EffectCondition | None = None


class AISystemLimitEffect(FrozenModel):
    """
    Adjusts the maximum number of AI-tagged systems that can be installed.

    Examples:
        AISystemLimitEffect(bonus_systems=1, max_ai_systems=2)
    """

    bonus_systems: int = 0
    max_ai_systems: int | None = Field(default=None, ge=0)
    condition: EffectCondition | None = None


class AIControlTransferEffect(FrozenModel):
    """
    Handles AI control transfer when another AI becomes unshackled.

    Examples:
        AIControlTransferEffect(transfer_on_unshackle=True)
    """

    transfer_on_unshackle: bool = True
    source_tag: str = "ai"
    target_tag: str = "ai"
    condition: EffectCondition | None = None
