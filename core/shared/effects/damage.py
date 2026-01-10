"""Direct damage, reduction, and sharing effects.

Effects for dealing, modifying, and absorbing damage.

Effects:
    - DamageModifier: Bonus or penalty to damage
    - DamageMultiplierEffect: Multiplies damage, heat, or burn
    - RangeModifier: Modifies range or threat
    - DirectDamage: Direct damage not tied to weapon attacks
    - DamageReduction: Reduces incoming damage by a flat value
    - DamageReductionRollEffect: Reduces damage based on a roll
    - DamageShareEffect: Shares damage with another target
    - DamageNegationEffect: Negates damage under conditions
    - DamageAbsorption: Absorbs damage as shields/barriers

See Also:
    - PR2 3960-3964: Damage resolution
    - PR2 4618-4650: Armor and damage reduction
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from core.shared.models import FrozenModel
from core.shared.dice import DiceExpression
from core.shared.enums import DamageType
from core.shared.effects.types import (
    DamageTypeScope,
    DirectDamageType,
    EffectTarget,
    EffectTargetNoAll,
    EffectTargetWithObject,
)
from core.shared.effects.conditions import EffectCondition

__all__ = [
    "DamageModifier",
    "DamageMultiplierEffect",
    "RangeModifier",
    "DirectDamage",
    "DamageReduction",
    "DamageReductionRollEffect",
    "DamageShareEffect",
    "DamageNegationEffect",
    "DamageAbsorption",
]


class DamageModifier(FrozenModel):
    """
    Bonus damage under specific conditions.

    Examples:
        DamageModifier(flat=1, condition="melee_attack")  # +1 melee damage
        DamageModifier(dice=DiceExpression.parse("1d6"), damage_type="kinetic")  # +1d6 kinetic
    """

    dice: DiceExpression | None = Field(
        default=None, description="Bonus dice (e.g., DiceExpression.parse('1d6'))"
    )
    flat: int = Field(default=0, description="Flat bonus damage")
    damage_type: DamageType | None = Field(default=None)
    condition: EffectCondition | None = Field(default=None)


class DamageMultiplierEffect(FrozenModel):
    """
    Multiplies damage, heat, or burn dealt or taken.

    Examples:
        DamageMultiplierEffect(multiplier=0.5, applies_to="outgoing", condition="target_range_gt_5")
    """

    multiplier: float = Field(..., ge=0)
    damage_types: list[DamageTypeScope] = Field(default_factory=list)
    applies_to: Literal["outgoing", "incoming"] = "outgoing"
    target: EffectTarget = "enemy"
    condition: EffectCondition | None = None


class RangeModifier(FrozenModel):
    """
    Modifier to range or threat.

    Examples:
        RangeModifier(range_type="threat", value=1)  # +1 Threat
        RangeModifier(range_type="range", value=5)  # +5 Range
    """

    range_type: Literal["range", "threat", "sensors"]
    value: int
    condition: EffectCondition | None = None


class DirectDamage(FrozenModel):
    """Direct damage not tied to a standard weapon attack.

    Per PR2 3960-3964: Damage is calculated from dice and flat modifiers,
    then reduced by armor and resistance. AP ignores armor.

    Examples:
        DirectDamage(damage_type="explosive", dice=DiceExpression.parse("1d3"), ap=True)
    """

    damage_type: DirectDamageType
    dice: DiceExpression | None = None
    flat: int = 0
    multiplier: float = Field(default=1.0, ge=0)
    ap: bool = False
    target: EffectTargetWithObject = "enemy"
    condition: EffectCondition | None = None


class DamageReduction(FrozenModel):
    """
    Flat damage reduction that applies before resistance.

    Examples:
        DamageReduction(amount=2, damage_type="all")
    """

    amount: int = Field(..., ge=0)
    damage_type: DamageType | Literal["all", "heat", "burn"] = "all"
    target: EffectTargetWithObject = "self"
    minimum_damage: int | None = Field(default=None, ge=0)
    condition: EffectCondition | None = None


class DamageReductionRollEffect(FrozenModel):
    """
    Damage reduction based on a roll.

    Examples:
        DamageReductionRollEffect(roll=DiceExpression.parse("1d6"))
    """

    roll: DiceExpression
    damage_type: DamageType | Literal["all", "heat", "burn"] = "all"
    target: EffectTargetWithObject = "self"
    minimum_damage: int | None = Field(default=None, ge=0)
    condition: EffectCondition | None = None


class DamageShareEffect(FrozenModel):
    """
    Shares a fraction of damage from a target to a source.

    Examples:
        DamageShareEffect(share_fraction=0.5, source="self", target="ally", requires_adjacent=True)
    """

    share_fraction: float = Field(..., ge=0, le=1)
    source: EffectTargetNoAll = "self"
    target: EffectTargetNoAll = "ally"
    timing: Literal["before_armor_and_reduction", "after_armor_and_reduction"] = (
        "before_armor_and_reduction"
    )
    requires_adjacent: bool = False
    breaks_on_separation: bool = False
    condition: EffectCondition | None = None


class DamageNegationEffect(FrozenModel):
    """
    Negates an incoming damage instance.

    Examples:
        DamageNegationEffect(target="self")
    """

    target: EffectTargetNoAll = "self"
    negate_damage: bool = True
    negate_heat: bool = False
    negate_burn: bool = False
    condition: EffectCondition | None = None


class DamageAbsorption(FrozenModel):
    """
    Absorbs damage before it affects the target.

    Examples:
        DamageAbsorption(target="ally", base_hp=4, bonus_hp_per_grit=1)
    """

    target: EffectTargetNoAll
    base_hp: int = Field(..., ge=0)
    bonus_hp_per_grit: int = Field(default=0, ge=0)
    max_instances_per_target: int = Field(default=1, ge=1)
    spillover: bool = True
    ends_on_zero: bool = True
    duration: Literal["scene", "until_destroyed"] = "until_destroyed"
