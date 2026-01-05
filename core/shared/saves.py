"""Save resolution helpers for Lancer combat.

This module provides type-safe helpers for resolving saves (HULL, AGI, SYS, ENG)
including:
- Save roll resolution with modifiers
- Condition-based difficulty modifiers
- Integration with check modifiers from effects

Save Rules (per PR2 1360-1364):
- A save is a defensive roll made to avoid some effect
- Roll 1d20 + relevant bonuses vs target number
- If total >= target, save succeeds
- If total < target, save fails
- Critical: natural 20 always succeeds, natural 1 always fails

Save Types:
- HULL: Physical resilience, knockdown, push effects
- AGILITY: Dodging, area effects, movement-based threats
- SYSTEMS: Electronic warfare, hacking, tech effects
- ENGINEERING: Heat, reactor, systems stability effects
"""

from __future__ import annotations

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.enums import SaveType, StatusType
from core.shared.rolls import SaveRoll, RollModifiers, AccuracyDifficulty
from core.shared.dice import roll_dice
from core.shared.effects import CheckModifierEffect


class SaveRequest(FrozenModel):
    """Input for save resolution.

    Args:
        save_type: Type of save (hull, agility, systems, engineering)
        save_target: The save target that must be matched or beaten
        save_bonus: Flat bonus to the save (from mech skill, grit, etc.)
        target_conditions: Current conditions on the save target
        check_modifiers: CheckModifierEffects from systems/talents
        force_roll: Optional forced roll value for deterministic testing
    """

    save_type: SaveType
    save_target: int = Field(..., ge=0, description="Save target to beat")
    save_bonus: int = Field(default=0, description="Flat bonus to save roll")
    target_conditions: list[StatusType] = Field(
        default_factory=list, description="Conditions on save target"
    )
    check_modifiers: list[CheckModifierEffect] = Field(
        default_factory=list, description="Check modifiers from effects"
    )
    force_roll: int | None = Field(
        default=None, ge=1, le=20, description="Forced d20 roll for testing"
    )


class SaveDifficultyModifier(FrozenModel):
    """A save difficulty modifier from a specific source.

    Args:
        source: Description of the modifier source
        value: Positive = harder (difficulty), negative = easier (accuracy)
        applies_to: Save types this applies to (None = all)
    """

    source: str
    value: int = Field(..., description="Positive = difficulty, negative = accuracy")
    applies_to: list[SaveType] | None = Field(
        default=None, description="Save types this applies to"
    )


class SaveResult(FrozenModel):
    """Result of a save resolution.

    Attributes:
        save_type: Type of save that was made
        roll: The d20 roll (None if auto-succeed/fail)
        roll_with_bonus: Roll plus flat bonus
        total: Final total including all modifiers
        target: Save target that was needed
        success: Whether the save succeeded (total >= target)
        degree: Success/failure classification including criticals
        difficulty_modifier: Total difficulty from conditions
        accuracy_modifier: Total accuracy from effects
        flat_bonus: Flat bonus from skill/grit
        modifier_breakdown: Human-readable breakdown of modifiers
        reason: Explanation of the result
    """

    save_type: SaveType
    roll: int | None
    roll_with_bonus: int | None
    total: int
    target: int
    success: bool
    degree: Literal["critical_success", "success", "failure", "critical_failure"]
    difficulty_modifier: int
    accuracy_modifier: int
    flat_bonus: int
    modifier_breakdown: dict[str, int] = Field(default_factory=dict)
    reason: str = ""


def resolve_save(request: SaveRequest) -> SaveResult:
    """Resolve a save roll against a target value.

    This function handles:
    - Rolling or using forced d20 value
    - Applying flat bonuses (skill, grit)
    - Calculating condition-based difficulty modifiers
    - Applying check modifiers from effects
    - Determining success/failure and degree

    Args:
        request: SaveRequest with all save parameters

    Returns:
        SaveResult with resolution details
    """
    from core.shared.conditions import get_condition_difficulty_modifier

    modifiers_from_conditions = 0
    modifiers_from_effects = 0
    modifier_breakdown: dict[str, int] = {}

    for condition in request.target_conditions:
        cond_mod = get_condition_difficulty_modifier(condition, "save")
        if cond_mod != 0:
            modifiers_from_conditions += cond_mod
            modifier_breakdown[f"condition:{condition}"] = (
                modifier_breakdown.get(f"condition:{condition}", 0) + cond_mod
            )

    for check_mod in request.check_modifiers:
        if not check_mod.check_types or request.save_type in check_mod.check_types:
            if not check_mod.check_kinds or "save" in check_mod.check_kinds:
                modifiers_from_effects += check_mod.value
                modifier_breakdown[f"effect:{check_mod}"] = (
                    modifier_breakdown.get(f"effect:{check_mod}", 0) + check_mod.value
                )

    total_difficulty = modifiers_from_conditions + modifiers_from_effects

    if request.force_roll is not None:
        roll = request.force_roll
    else:
        roll = roll_dice("1d20")

    roll_with_bonus = roll + request.save_bonus
    total = roll_with_bonus - total_difficulty

    critical_failure = roll == 1
    critical_success = roll == 20

    if critical_success:
        degree: Literal[
            "critical_success", "success", "failure", "critical_failure"
        ] = "critical_success"
        success = True
    elif critical_failure:
        degree = "critical_failure"
        success = False
    elif total >= request.save_target:
        degree = "success"
        success = True
    else:
        degree = "failure"
        success = False

    net_direction = "accuracy" if total_difficulty < 0 else "difficulty"
    net_value = abs(total_difficulty)

    if roll == 20:
        reason = f"Natural 20: automatic success"
    elif roll == 1:
        reason = f"Natural 1: automatic failure"
    elif success:
        reason = f"{request.save_type.upper()} save succeeds: {total} vs {request.save_target}"
    else:
        reason = (
            f"{request.save_type.upper()} save fails: {total} < {request.save_target}"
        )

    return SaveResult(
        save_type=request.save_type,
        roll=roll,
        roll_with_bonus=roll_with_bonus,
        total=total,
        target=request.save_target,
        success=success,
        degree=degree,
        difficulty_modifier=modifiers_from_conditions,
        accuracy_modifier=0,
        flat_bonus=request.save_bonus,
        modifier_breakdown=modifier_breakdown,
        reason=reason,
    )


def compute_save_target(
    base_save_target: int,
    grit_bonus: int = 0,
    condition_modifiers: int = 0,
    check_modifiers: int = 0,
) -> int:
    """Compute final save target with all modifiers applied.

    Save target = base + grit + conditions + check modifiers
    Difficulty modifiers increase the target (harder to save).

    Args:
        base_save_target: Base save target from mech stat block
        grit_bonus: Grit bonus (1/2 pilot level)
        condition_modifiers: Total from conditions (Impaired +1, etc.)
        check_modifiers: Total from CheckModifierEffects

    Returns:
        Final save target
    """
    return base_save_target + grit_bonus + condition_modifiers + check_modifiers


def get_save_skill_bonus(save_type: SaveType, skill_bonus: int) -> int:
    """Get the skill bonus for a save type.

    Args:
        save_type: Type of save being made
        skill_bonus: The pilot's relevant mech skill level

    Returns:
        Bonus to add to save roll
    """
    return skill_bonus


def resolve_save_against_damage(
    save_type: SaveType,
    damage_amount: int,
    save_target: int,
    save_bonus: int = 0,
    target_conditions: list[StatusType] = [],
    half_on_save: bool = True,
    force_roll: int | None = None,
) -> tuple[bool, int]:
    """Resolve a save against damage with standard half-damage on success.

    Common pattern: target makes save vs effect, takes full damage on fail,
    half damage on success.

    Args:
        save_type: Type of save (hull, agility, etc.)
        damage_amount: Base damage if save fails
        save_target: Target number for the save
        save_bonus: Flat bonus to save roll
        target_conditions: Conditions on the save target
        half_on_save: Whether damage is halved on successful save
        force_roll: Optional forced roll for testing

    Returns:
        Tuple of (save_succeeded, damage_taken)
    """
    request = SaveRequest(
        save_type=save_type,
        save_target=save_target,
        save_bonus=save_bonus,
        target_conditions=target_conditions,
        force_roll=force_roll,
    )

    result = resolve_save(request)

    if result.success:
        if half_on_save:
            damage_taken = damage_amount // 2
        else:
            damage_taken = damage_amount
    else:
        damage_taken = damage_amount

    return result.success, damage_taken
