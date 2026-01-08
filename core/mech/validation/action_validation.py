"""Action validation helpers for mech combat.

Covers action timing, cooldowns, per-target effects, and overcharge escalation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from core.mech.combat_rules import DEFAULT_MECH_COMBAT_RULES
from core.mech.timing import (
    ActionTimingValidationSettings,
    validate_action_while_prepared,
    validate_per_round_reaction,
    validate_protocol_timing,
)
from core.shared.effects import CooldownState, PerTargetCounter

if TYPE_CHECKING:
    from core.mech.combat_actions import ActionRule
    from core.mech.combat_state import ActionUse, CombatantState, OverchargeState
    from core.mech.combat_rules import OverchargeRules
    from core.mech.timing import TurnPhase


class CombatValidationIssue:
    """A combat validation issue."""

    code: str
    message: str
    severity: Literal["error", "warning"] = "error"


__all__ = [
    "CombatValidationIssue",
]


def _index_per_target_counters(
    combatant: CombatantState,
) -> tuple[dict[tuple[str, str], PerTargetCounter], dict[str, PerTargetCounter]]:
    counters_by_target: dict[tuple[str, str], PerTargetCounter] = {}
    templates_by_effect: dict[str, PerTargetCounter] = {}
    for counter in combatant.per_target_counters.values():
        if counter.target_id:
            counters_by_target[(counter.effect_id, counter.target_id)] = counter
        else:
            templates_by_effect[counter.effect_id] = counter
    return counters_by_target, templates_by_effect


def _index_cooldown_states(
    combatant: CombatantState,
) -> tuple[dict[str, CooldownState], dict[str, CooldownState]]:
    global_cooldowns: dict[str, CooldownState] = {}
    per_target_cooldowns: dict[str, CooldownState] = {}
    for cooldown in combatant.cooldown_states.values():
        if cooldown.per_target and cooldown.target_id:
            key = f"{cooldown.effect_id}:{cooldown.target_id}"
            per_target_cooldowns[key] = cooldown
        else:
            global_cooldowns[cooldown.effect_id] = cooldown
    return global_cooldowns, per_target_cooldowns


def _check_action_on_cooldown(
    action: ActionUse,
    actor_cooldowns: tuple[dict[str, CooldownState], dict[str, CooldownState]],
    issues: list[CombatValidationIssue],
) -> bool:
    global_cooldowns, per_target_cooldowns = actor_cooldowns
    is_blocked = False
    if action.applied_per_target_effects:
        for applied in action.applied_per_target_effects:
            key = (
                f"{applied.effect_id}:{applied.target_id}"
                if applied.target_id
                else applied.effect_id
            )
            cooldown = per_target_cooldowns.get(key) or global_cooldowns.get(
                applied.effect_id
            )
            if cooldown and cooldown.turns_remaining > 0:
                issues.append(
                    CombatValidationIssue(
                        code="action_on_cooldown",
                        message=(
                            f"Action {action.action_id} uses {applied.effect_id} "
                            f"which is on cooldown ({cooldown.turns_remaining} turns remaining)."
                        ),
                    )
                )
                is_blocked = True
    return is_blocked


def _validate_overcharge_escalation(
    action: ActionUse,
    actor_overcharge_state: OverchargeState | None,
    rules: OverchargeRules = DEFAULT_MECH_COMBAT_RULES.overcharge_rules,
    strict_mode: bool = True,
) -> list[CombatValidationIssue]:
    """Validate that overcharge heat cost matches expected escalation level.

    Args:
        action: The action being validated
        actor_overcharge_state: Current overcharge state for the actor
        rules: Overcharge rules to use
        strict_mode: If True, produce errors; otherwise produce warnings

    Returns:
        List of validation issues (empty if valid)
    """
    from core.mech.combat_resolution import compute_overcharge_escalation
    from core.shared.dice import DiceExpression

    issues: list[CombatValidationIssue] = []

    if action.action_id != "overcharge":
        return issues

    escalation = compute_overcharge_escalation(actor_overcharge_state)
    expected_cost = rules.costs[escalation.current_level]
    declared_cost = action.heat_generated

    if declared_cost is not None:
        expected_value: int
        if isinstance(expected_cost, DiceExpression):
            expected_value = expected_cost.min_value()
        else:
            expected_value = expected_cost

        if declared_cost != expected_value and not isinstance(
            expected_cost, DiceExpression
        ):
            severity = "error" if strict_mode else "warning"
            issues.append(
                CombatValidationIssue(
                    code="overcharge_cost_mismatch",
                    message=(
                        f"Overcharge at level {escalation.current_level} expects "
                        f"heat cost {expected_cost}, but action declares {declared_cost}."
                    ),
                    severity=severity,
                )
            )

    return issues


def _validate_action_timing(
    action: ActionUse,
    actor: CombatantState | None,
    current_phase: TurnPhase,
    round_reaction_counts: dict[str, int],
    issues: list[CombatValidationIssue],
    timing_settings: ActionTimingValidationSettings | None = None,
) -> bool:
    """Validate timing constraints for an action.

    Validates:
    - Protocol timing (must be at start of turn)
    - Prepared action lockout (cannot act/react/move while prepared)
    - Per-round reaction limits (brace/overwatch once per round)

    Args:
        action: The action being validated
        actor: The combatant taking the action
        current_phase: Current turn phase
        round_reaction_counts: Per-round reaction counts for this actor
        issues: List to append validation issues to
        timing_settings: Validation settings (uses defaults if None)

    Returns:
        True if all timing constraints are valid
    """
    if timing_settings is None:
        timing_settings = ActionTimingValidationSettings(strict_mode=True)

    is_valid = True

    if actor:
        protocol_timing_result = validate_protocol_timing(
            action.action_id,
            action.action_type == "protocol",
            current_phase,
            timing_settings,
        )
        if not protocol_timing_result.valid:
            is_valid = False
            for error in protocol_timing_result.errors:
                issues.append(
                    CombatValidationIssue(code="protocol_timing", message=error)
                )
        for warning in protocol_timing_result.warnings:
            issues.append(
                CombatValidationIssue(
                    code="protocol_timing", message=warning, severity="warning"
                )
            )

        prepared_result = validate_action_while_prepared(
            action.action_id,
            action.action_type,
            actor.prepared_action,
            timing_settings,
        )
        if not prepared_result.valid:
            is_valid = False
            for error in prepared_result.errors:
                issues.append(
                    CombatValidationIssue(code="prepared_action_lockout", message=error)
                )
        for warning in prepared_result.warnings:
            issues.append(
                CombatValidationIssue(
                    code="prepared_action_lockout", message=warning, severity="warning"
                )
            )

    if action.action_type == "reaction" or action.used_as_reaction:
        max_per_round = DEFAULT_MECH_COMBAT_RULES.reaction_rules.max_reactions_per_turn
        reaction_result = validate_per_round_reaction(
            action.action_id,
            0,
            actor.id if actor else "",
            {actor.id if actor else "": round_reaction_counts} if actor else {},
            max_per_round,
        )
        if not reaction_result.valid:
            is_valid = False
            for error in reaction_result.errors:
                issues.append(
                    CombatValidationIssue(
                        code="per_round_reaction_limit", message=error
                    )
                )

    return is_valid


def _apply_per_target_effects(
    action: ActionUse,
    actor_id: str,
    combatants_by_id: dict[str, CombatantState],
    counters_by_target: dict[tuple[str, str], PerTargetCounter],
    templates_by_effect: dict[str, PerTargetCounter],
    issues: list[CombatValidationIssue],
) -> None:
    for applied in action.applied_per_target_effects:
        if applied.target_id not in combatants_by_id:
            issues.append(
                CombatValidationIssue(
                    code="per_target_unknown_target",
                    message=(
                        f"Action {action.action_id} applies {applied.effect_id} "
                        f"to unknown target {applied.target_id}."
                    ),
                )
            )
            continue

        key = (applied.effect_id, applied.target_id)
        counter = counters_by_target.get(key)
        if counter is None:
            template = templates_by_effect.get(applied.effect_id)
            if template:
                counter = template.model_copy(
                    update={
                        "target_id": applied.target_id,
                        "current_count": template.current_count,
                    }
                )
            else:
                if applied.max_count is None:
                    issues.append(
                        CombatValidationIssue(
                            code="per_target_limit_unknown",
                            message=(
                                f"Action {action.action_id} applies {applied.effect_id} "
                                f"to {applied.target_id} without a max_count."
                            ),
                            severity="warning",
                        )
                    )
                    continue
                counter = PerTargetCounter(
                    effect_id=applied.effect_id,
                    max_count=applied.max_count,
                    reset_on=applied.reset_on or "scene_end",
                    target_id=applied.target_id,
                )

        new_count = counter.current_count + applied.count
        if new_count > counter.max_count:
            issues.append(
                CombatValidationIssue(
                    code="per_target_limit_exceeded",
                    message=(
                        f"{actor_id} applies {applied.effect_id} to {applied.target_id} "
                        f"{new_count} times (max {counter.max_count})."
                    ),
                )
            )

        counters_by_target[key] = counter.model_copy(
            update={"current_count": new_count}
        )


def validate_action_cooldowns(
    action: ActionUse,
    actor_cooldowns: tuple[dict[str, CooldownState], dict[str, CooldownState]],
) -> list[CombatValidationIssue]:
    """Validate that action effects are not on cooldown.

    Args:
        action: The action being validated
        actor_cooldowns: Tuple of (global_cooldowns, per_target_cooldowns)

    Returns:
        List of validation issues (empty if valid)
    """
    issues: list[CombatValidationIssue] = []
    _check_action_on_cooldown(action, actor_cooldowns, issues)
    return issues


def validate_overcharge_escalation(
    action: ActionUse,
    actor_overcharge_state: OverchargeState | None,
    strict_mode: bool = True,
) -> list[CombatValidationIssue]:
    """Validate that overcharge heat cost matches expected escalation level.

    Args:
        action: The action being validated (should have action_id='overcharge')
        actor_overcharge_state: Current overcharge state for the actor
        strict_mode: If True, produce errors; otherwise produce warnings

    Returns:
        List of validation issues (empty if valid)
    """
    return _validate_overcharge_escalation(
        action,
        actor_overcharge_state,
        DEFAULT_MECH_COMBAT_RULES.overcharge_rules,
        strict_mode,
    )
