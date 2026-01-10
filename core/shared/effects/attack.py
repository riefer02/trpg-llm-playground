"""Attack and damage resolution effects.

Effects for modifying attack rolls, targeting, outcomes, and damage resolution.

Effects:
    - AttackRollOverrideEffect: Overrides attack roll mechanics
    - AttackTargetingEffect: Multi-target attack selection
    - AreaAttackPattern: Blast, burst, line, cone attack patterns
    - LineAttackEffect: Custom line attack behavior
    - AttackSequenceModifierEffect: Accuracy/difficulty across attack sequences
    - AttackRerollEffect: Allows rerolling attack rolls
    - AttackOutcomeEffect: Modifies attack outcomes (hit/miss/crit)
    - CriticalDamageOverrideEffect: Overrides critical damage rules
    - DamageRollOverrideEffect: Overrides damage calculation
    - AccuracyTradeEffect: Trades accuracy for difficulty or vice versa
    - DelayedImpactEffect: Delays damage application

See Also:
    - PR2 3965-3984: Attack and damage resolution
"""

from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from pydantic import Field

from core.shared.models import FrozenModel
from core.shared.dice import DiceExpression
from core.shared.enums import AttackType, DamageType, RangeType
from core.shared.effects.types import (
    DelayedImpactTiming,
    DirectDamageType,
    EffectDuration,
    EffectTarget,
    EffectTargetNoAll,
    IntelAudience,
    TriggerType,
    UsesPer,
    WeaponTypeType,
)
from core.shared.effects.conditions import EffectCondition
from core.shared.effects.damage import DirectDamage

if TYPE_CHECKING:
    pass

__all__ = [
    "AttackRollOverrideEffect",
    "AttackTargetingEffect",
    "AreaAttackPattern",
    "LineAttackEffect",
    "AttackSequenceModifierEffect",
    "AttackRerollEffect",
    "AttackOutcomeEffect",
    "CriticalDamageOverrideEffect",
    "DamageRollOverrideEffect",
    "AccuracyTradeEffect",
    "DelayedImpactEffect",
]


class AttackRollOverrideEffect(FrozenModel):
    """Overrides attack roll resolution details for a weapon or effect.

    Per PR2 3965-3969: Attack rolls use d20 + relevant skill + any bonuses.
    This effect allows overriding the target defense, attack type restrictions,
    and cover/line-of-sight handling.

    Examples:
        AttackRollOverrideEffect(attack_vs="evasion", fixed_target_defense=8, target="ally",
                                 allowed_attack_types=["ranged"])
    """

    attack_vs: Literal["evasion", "e_defense"] = "e_defense"
    fixed_target_defense: int | None = Field(default=None, ge=0)
    target: EffectTarget = "enemy"
    allowed_attack_types: list[AttackType] = Field(default_factory=list)
    respects_cover: bool = True
    respects_line_of_sight: bool = True
    condition: EffectCondition | None = None


class AttackTargetingEffect(FrozenModel):
    """
    Describes multi-target attack selection.

    Examples:
        AttackTargetingEffect(target_count_options=[1, 2], separate_attack_rolls=True)
    """

    target_count_options: list[int] = Field(default_factory=list)
    separate_attack_rolls: bool = True
    require_distinct_targets: bool = True
    condition: EffectCondition | None = None


class AreaAttackPattern(FrozenModel):
    """Defines multi-area attack patterns for weapons.

    Per PR2 3970-3974: Area attacks use geometric patterns (blast, burst, line, cone)
    to affect multiple targets simultaneously.

    Examples:
        AreaAttackPattern(area_shape="blast", area_size=1, area_count_options=[1, 2], non_overlapping=True)
    """

    area_shape: Literal["blast", "burst", "line", "cone"]
    area_size: int = Field(..., ge=0)
    area_count_options: list[int] = Field(default_factory=list)
    non_overlapping: bool = False


class LineAttackEffect(FrozenModel):
    """Line-based attack behavior that augments a weapon or action.

    Per PR2 3970-3974: Line attacks affect all targets in a straight line.
    This effect allows customizing line length, origin point, and obstacle handling.

    Examples:
        LineAttackEffect(between_self_and_target=True, heat_per_additional_target=1)
        LineAttackEffect(
            length=3,
            attack_origin_at_line_end=True,
            measure_range_from_line_end=True,
            measure_line_of_sight_from_line_end=True,
            line_passes_through_obstacles=True,
            line_damage=DirectDamage(damage_type="kinetic", dice=DiceExpression.parse("1d3"), ap=True),
            line_hits_objects=True,
        )
    """

    length: int | None = Field(
        default=None,
        ge=0,
        description="Line length in spaces; None = between self and target",
    )
    between_self_and_target: bool = False
    attack_origin_at_line_end: bool = False
    measure_range_from_line_end: bool = False
    measure_line_of_sight_from_line_end: bool = False
    measure_cover_from_line_end: bool = False
    line_passes_through_obstacles: bool = False
    fixed_direction_after_firing: bool = False
    line_hits_characters: bool = True
    line_hits_objects: bool = False
    line_damage: DirectDamage | None = None
    heat_per_additional_target: int | None = Field(default=None, ge=0)
    condition: EffectCondition | None = None


class AttackSequenceModifierEffect(FrozenModel):
    """
    Modifies accuracy/difficulty across a sequence of attacks in a turn.

    Examples:
        AttackSequenceModifierEffect(
            first_attack_accuracy=1,
            first_attack_optional=True,
            subsequent_attack_difficulty=1,
            duration="end_of_turn",
        )
    """

    trigger: TriggerType = "on_turn_start"
    first_attack_accuracy: int = 0
    first_attack_optional: bool = False
    subsequent_attack_difficulty: int = 0
    duration: EffectDuration = "end_of_turn"
    condition: EffectCondition | None = None


class AttackRerollEffect(FrozenModel):
    """
    Allows rerolling and retargeting attacks under constraints.

    Examples:
        AttackRerollEffect(trigger="on_miss", allowed_attack_types=["melee", "ranged"],
                           require_new_target=True, max_rerolls_per_attack=1)
    """

    trigger: TriggerType = "on_miss"
    allowed_attack_types: list[AttackType] = Field(default_factory=list)
    require_new_target: bool = False
    allow_same_area_for_aoe: bool = False
    aoe_types: list[RangeType] = Field(default_factory=list)
    max_rerolls_per_attack: int = Field(default=1, ge=1)
    disallow_already_hit_targets: bool = False
    keep_second: bool = False
    uses_reaction: bool = False
    duration: EffectDuration = "end_of_turn"
    uses_per: UsesPer = "unlimited"
    condition: EffectCondition | None = None


class AttackOutcomeEffect(FrozenModel):
    """
    Modifies the outcome of an attack (e.g., upgrade to critical).

    Examples:
        AttackOutcomeEffect(trigger="on_ally_hit", upgrade_to_crit=True, uses_reaction=True)
    """

    trigger: TriggerType
    upgrade_to_crit: bool = False
    force_miss: bool = False
    target: EffectTarget = "enemy"
    attacker: EffectTarget = "ally"
    uses_reaction: bool = False
    duration: EffectDuration = "end_of_turn"
    uses_per: UsesPer = "unlimited"
    condition: EffectCondition | None = None


class CriticalDamageOverrideEffect(FrozenModel):
    """
    Overrides critical damage resolution.

    Examples:
        CriticalDamageOverrideEffect(mode="max", requires_natural_20=True)
    """

    trigger: TriggerType = "on_crit"
    applies_to: list[AttackType] = Field(default_factory=list)
    mode: Literal["max"] = "max"
    requires_natural_20: bool = False
    target: EffectTarget = "enemy"
    attacker: EffectTarget = "self"
    condition: EffectCondition | None = None


class DamageRollOverrideEffect(FrozenModel):
    """
    Overrides how damage rolls are resolved.

    Examples:
        DamageRollOverrideEffect(mode="average", optional=True)
    """

    mode: Literal["average"] = "average"
    optional: bool = True
    target: EffectTargetNoAll = "self"
    condition: EffectCondition | None = None


class AccuracyTradeEffect(FrozenModel):
    """
    Trade accuracy dice for bonus effects.

    Examples:
        AccuracyTradeEffect(accuracy_cost=1, bonus_damage=DiceExpression.parse("1d6"), requires_crit=True)
    """

    accuracy_cost: int = Field(..., ge=1)
    bonus_damage: DiceExpression
    bonus_damage_type: DamageType | DirectDamageType | None = None
    requires_crit: bool = False
    trigger: TriggerType = "on_attack_roll"
    applies_to_attack_types: list[AttackType] = Field(default_factory=list)
    allowed_weapon_types: list[WeaponTypeType] = Field(default_factory=list)
    condition: EffectCondition | None = None


class DelayedImpactEffect(FrozenModel):
    """
    Optional delayed impact behavior for weapon attacks.

    Examples:
        DelayedImpactEffect(
            delay_timing="end_of_next_round",
            delayed_damage=DiceExpression.parse("3d6"),
            delayed_damage_type="explosive",
            self_slow_duration="end_of_next_turn",
            reveal_area=True,
        )
    """

    delay_optional: bool = True
    delay_timing: DelayedImpactTiming = "end_of_next_round"
    delayed_damage: DiceExpression | None = None
    delayed_damage_type: DamageType | None = None
    self_slow_duration: EffectDuration | None = None
    reveal_area: bool = False
    reveal_audience: IntelAudience = "all"
