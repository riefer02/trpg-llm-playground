"""Condition resolution helpers for Lancer combat.

This module provides type-safe helpers for applying, removing, and resolving
mechanical effects of conditions in mech combat.

Conditions (per PR2):
- IMPAIRED: +1 difficulty on attacks, saves, and checks
- SHREDDED: Cannot benefit from armor/resistance
- JAMMED: Only improvised attacks/grapples allowed
- LOCK_ON: Consume on attack for +1 accuracy
- EXPOSED: Take double damage until stabilized

Conditions are temporary and can be cleared by:
- Stabilize action
- Duration expiry (end_of_turn, etc)
- Rest
- Full repair
"""

from __future__ import annotations

from typing import Literal, Sequence
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.enums import StatusType
from core.mech.statuses import (
    StatusDefinition,
    STATUS_DEFINITIONS_BY_ID,
    get_status_definition,
)


CONDITIONS: frozenset[StatusType] = frozenset(
    [
        "impaired",
        "shredded",
        "jammed",
        "lock_on",
        "exposed",
        "immobilized",
        "slowed",
        "stunned",
    ]
)


class ConditionApplicationResult(FrozenModel):
    """Result of applying a condition to a target."""

    condition: StatusType
    applied: bool
    reason: str = ""
    duration_applied: str | None = None
    stacks: int = 1


class ConditionRemovalResult(FrozenModel):
    """Result of removing a condition from a target."""

    condition: StatusType
    removed: bool
    reason: str = ""
    remaining_stacks: int = 0


class ConditionEffectResult(FrozenModel):
    """Resolved mechanical effects of a condition on a target."""

    condition: StatusType
    attack_difficulty_modifier: int = 0
    save_difficulty_modifier: int = 0
    check_difficulty_modifier: int = 0
    damage_multiplier: float = 1.0
    ignore_armor: bool = False
    ignore_resistance: bool = False
    movement_speed_cap: int | None = None
    only_regular_move: bool = False
    action_restrictions: dict[str, bool] = Field(default_factory=dict)
    targeting_restrictions: dict[str, bool] = Field(default_factory=dict)
    consumable_accuracy_bonus: int | None = None
    auto_fail_checks: list[str] = Field(default_factory=list)


def resolve_condition_effects(condition: StatusType) -> ConditionEffectResult:
    """Resolve the mechanical effects of a condition.

    Args:
        condition: The condition to resolve effects for

    Returns:
        ConditionEffectResult with all mechanical modifiers
    """
    definition = get_status_definition(condition)

    if definition is None:
        return ConditionEffectResult(condition=condition)

    effects = definition.effects

    action_restrictions = {}
    if effects.action_restrictions:
        ar = effects.action_restrictions
        action_restrictions = {
            "disallow_actions": ar.disallow_actions,
            "disallow_full_actions": ar.disallow_full_actions,
            "disallow_reactions": ar.disallow_reactions,
            "disallow_free_actions": ar.disallow_free_actions,
            "disallow_move": ar.disallow_move,
            "disallow_overcharge": ar.disallow_overcharge,
            "disallow_tech_actions": ar.disallow_tech_actions,
            "disallow_comms": ar.disallow_comms,
        }

    targeting_restrictions = {}
    if effects.targeting_restrictions:
        tr = effects.targeting_restrictions
        targeting_restrictions = {
            "cannot_be_targeted": tr.cannot_be_targeted,
            "area_attacks_can_target": tr.area_attacks_can_target,
        }

    auto_fail_checks = []
    if effects.auto_fail_hull_checks:
        auto_fail_checks.append("hull")
    if effects.auto_fail_agility_checks:
        auto_fail_checks.append("agility")

    movement_speed_cap = None
    only_regular_move = False
    if effects.movement_restrictions:
        mr = effects.movement_restrictions
        movement_speed_cap = mr.max_voluntary_speed
        only_regular_move = mr.only_regular_move

    return ConditionEffectResult(
        condition=condition,
        attack_difficulty_modifier=effects.all_attack_difficulty,
        save_difficulty_modifier=effects.save_difficulty,
        check_difficulty_modifier=effects.skill_check_difficulty,
        damage_multiplier=effects.damage_multiplier or 1.0,
        ignore_armor=effects.ignore_armor,
        ignore_resistance=effects.ignore_resistance,
        movement_speed_cap=movement_speed_cap,
        only_regular_move=only_regular_move,
        action_restrictions=action_restrictions,
        targeting_restrictions=targeting_restrictions,
        consumable_accuracy_bonus=effects.consumable_accuracy_bonus,
        auto_fail_checks=auto_fail_checks,
    )


def can_apply_condition(
    target_conditions: Sequence[StatusType],
    condition: StatusType,
    max_stacks: int | None = None,
) -> tuple[bool, str]:
    """Check if a condition can be applied to a target.

    Args:
        target_conditions: Current conditions on the target
        condition: The condition to check
        max_stacks: Maximum stacks allowed (None = unlimited)

    Returns:
        Tuple of (can_apply, reason)
    """
    if condition not in CONDITIONS:
        return False, f"'{condition}' is not a valid condition"

    current_stacks = target_conditions.count(condition)

    if max_stacks is not None and current_stacks >= max_stacks:
        return False, f"Condition '{condition}' already at max stacks ({max_stacks})"

    return True, ""


def apply_condition(
    target_conditions: list[StatusType],
    condition: StatusType,
    max_stacks: int | None = None,
) -> ConditionApplicationResult:
    """Apply a condition to a target's condition list.

    Args:
        target_conditions: Current conditions list to modify
        condition: The condition to apply
        max_stacks: Maximum stacks allowed (None = unlimited)

    Returns:
        ConditionApplicationResult with application details
    """
    can_apply, reason = can_apply_condition(target_conditions, condition, max_stacks)

    if not can_apply:
        return ConditionApplicationResult(
            condition=condition,
            applied=False,
            reason=reason,
        )

    target_conditions.append(condition)

    stacks = target_conditions.count(condition)
    return ConditionApplicationResult(
        condition=condition,
        applied=True,
        reason="",
        stacks=stacks,
    )


def remove_condition(
    target_conditions: list[StatusType],
    condition: StatusType,
    count: int = 1,
) -> ConditionRemovalResult:
    """Remove a condition from a target's condition list.

    Args:
        target_conditions: Current conditions list to modify
        condition: The condition to remove
        count: Number of stacks to remove (default 1)

    Returns:
        ConditionRemovalResult with removal details
    """
    current_stacks = target_conditions.count(condition)

    if current_stacks == 0:
        return ConditionRemovalResult(
            condition=condition,
            removed=False,
            reason=f"Condition '{condition}' not present",
            remaining_stacks=0,
        )

    stacks_to_remove = min(count, current_stacks)

    for _ in range(stacks_to_remove):
        target_conditions.remove(condition)

    remaining = current_stacks - stacks_to_remove
    return ConditionRemovalResult(
        condition=condition,
        removed=True,
        reason="",
        remaining_stacks=remaining,
    )


def clear_all_conditions(
    target_conditions: list[StatusType],
) -> list[ConditionRemovalResult]:
    """Clear all conditions from a target.

    Args:
        target_conditions: Current conditions list to clear

    Returns:
        List of ConditionRemovalResult for each condition type cleared
    """
    results = []
    conditions_to_clear = {c for c in target_conditions if c in CONDITIONS}

    for condition in conditions_to_clear:
        result = remove_condition(target_conditions, condition, count=999)
        results.append(result)

    return results


def get_condition_difficulty_modifier(
    condition: StatusType,
    check_type: Literal["attack", "save", "skill"],
) -> int:
    """Get the difficulty modifier for a specific check type.

    Args:
        condition: The condition to check
        check_type: Type of check ("attack", "save", "skill")

    Returns:
        Difficulty modifier (positive = harder)
    """
    if condition not in CONDITIONS:
        return 0

    effects = resolve_condition_effects(condition)

    if check_type == "attack":
        return effects.attack_difficulty_modifier
    elif check_type == "save":
        return effects.save_difficulty_modifier
    else:
        return effects.check_difficulty_modifier


def is_condition_active(
    condition: StatusType, target_conditions: Sequence[StatusType]
) -> bool:
    """Check if a condition is currently active on a target.

    Args:
        condition: The condition to check
        target_conditions: Current conditions on the target

    Returns:
        True if the condition is present
    """
    return condition in target_conditions


def get_active_conditions(target_conditions: Sequence[StatusType]) -> list[StatusType]:
    """Get all active conditions from a target's condition list.

    Args:
        target_conditions: Current conditions on the target

    Returns:
        List of unique active conditions
    """
    return [c for c in target_conditions if c in CONDITIONS]


def get_condition_stacks(
    target_conditions: Sequence[StatusType],
) -> dict[StatusType, int]:
    """Count stacks for each condition type.

    Args:
        target_conditions: Current conditions on the target

    Returns:
        Dictionary of condition to stack count
    """
    stacks: dict[StatusType, int] = {}
    for condition in target_conditions:
        if condition in CONDITIONS:
            stacks[condition] = stacks.get(condition, 0) + 1
    return stacks


def conditions_prevent_attacks(target_conditions: Sequence[StatusType]) -> bool:
    """Check if conditions prevent making attacks.

    Args:
        target_conditions: Current conditions on the target

    Returns:
        True if attack is prevented
    """
    if "stunned" not in target_conditions:
        return False

    definition = get_status_definition("stunned")
    if definition and definition.effects.action_restrictions:
        return definition.effects.action_restrictions.disallow_actions

    return False


def conditions_prevent_movement(target_conditions: Sequence[StatusType]) -> bool:
    """Check if conditions prevent movement.

    Args:
        target_conditions: Current conditions on the target

    Returns:
        True if movement is prevented
    """
    return "immobilized" in target_conditions or "slowed" in target_conditions


def conditions_affect_damage_multiplier(
    target_conditions: Sequence[StatusType],
) -> float:
    """Get the total damage multiplier from active conditions.

    Args:
        target_conditions: Current conditions on the target

    Returns:
        Damage multiplier (1.0 = normal, 2.0 = doubled)
    """
    if "exposed" not in target_conditions:
        return 1.0

    definition = get_status_definition("exposed")
    if definition and definition.effects.damage_multiplier:
        return definition.effects.damage_multiplier

    return 1.0


def is_condition(condition: StatusType) -> bool:
    """Check if a status type is a condition (not a status).

    Args:
        status: The status type to check

    Returns:
        True if this is a condition
    """
    return condition in CONDITIONS
