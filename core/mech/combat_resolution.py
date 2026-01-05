"""Structure and overheat resolution helpers."""

from __future__ import annotations

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel

from core.shared.dice import DiceExpression
from core.mech.combat_rules import (
    DEFAULT_STRUCTURE_DAMAGE_RULES,
    DEFAULT_OVERHEAT_RULES,
    StructureDamageRules,
    OverheatRules,
    StructureOutcomeType,
    OverheatOutcomeType,
)
from core.mech.combat_state import (
    MechInventory,
    WeaponState,
    WeaponMountState,
    MechSystemState,
)
from core.shared.effects import (
    PerTargetCounter,
    CooldownState,
    CooldownEffect,
    CooldownResetTrigger,
    TriggerType,
)


class DiceRollResult(FrozenModel):
    """Raw dice roll outcome."""

    rolls: list[int] = Field(default_factory=list)
    chosen: list[int] = Field(default_factory=list)


class StructureResolution(FrozenModel):
    """Resolution result for structure damage."""

    outcome: StructureOutcomeType
    dice: DiceRollResult
    direct_hit_outcome: StructureOutcomeType | None = None
    system_trauma: SystemTraumaSelection | None = None
    updated_inventory: MechInventory | None = None
    structure_damage: int = 1
    spillover_damage: int = 0


class OverheatResolution(FrozenModel):
    """Resolution result for overheat checks."""

    outcome: OverheatOutcomeType
    dice: DiceRollResult
    meltdown_outcome: OverheatOutcomeType | None = None
    stress_damage: int = 1


class ResolutionSettings(FrozenModel):
    """Settings for deterministic or forced resolution."""

    forced_rolls: list[int] | None = None
    forced_system_trauma_roll: int | None = Field(default=None, ge=1, le=6)


class SystemTraumaSelection(FrozenModel):
    """Resolved selection for system trauma results."""

    roll: int
    initial_target: Literal["mount", "system"]
    resolved_target: Literal["mount", "system", "direct_hit"]
    eligible_mounts: list[int] = Field(default_factory=list)
    eligible_systems: list[str] = Field(default_factory=list)
    destroyed_mount_index: int | None = None
    destroyed_system_id: str | None = None
    fallback_reason: Literal["none", "no_mounts", "no_systems", "none_available"] = (
        "none"
    )


class PerTargetCounterResolution(FrozenModel):
    """Resolution result for per-target counter application."""

    effect_id: str
    target_id: str
    previous_count: int
    new_count: int
    was_applied: bool
    limit_exceeded: bool


def resolve_per_target_counter(
    *,
    counter: PerTargetCounter,
    applied_count: int = 1,
) -> PerTargetCounterResolution:
    """Apply a per-target counter increment, respecting max_count limits.

    Used for effects like Basilisk stun or H0r_OS invasions where each target
    can only be affected a limited number of times per combat/scene.

    Args:
        counter: The per-target counter to apply
        applied_count: Number of times to increment (default 1)

    Returns:
        PerTargetCounterResolution with the result of applying the counter
    """
    previous_count = counter.current_count
    new_count = previous_count + applied_count
    limit_exceeded = new_count > counter.max_count

    if limit_exceeded:
        new_count = previous_count
        was_applied = False
    else:
        was_applied = True

    return PerTargetCounterResolution(
        effect_id=counter.effect_id,
        target_id=counter.target_id or "",
        previous_count=previous_count,
        new_count=new_count,
        was_applied=was_applied,
        limit_exceeded=limit_exceeded,
    )


class CooldownCheckResult(FrozenModel):
    """Result of checking if an action is on cooldown."""

    is_on_cooldown: bool
    effect_id: str
    turns_remaining: int | None = None
    target_id: str | None = None


class CooldownApplicationResult(FrozenModel):
    """Result of applying a cooldown to a combatant."""

    applied: bool
    effect_id: str
    duration: int
    turns_remaining: int
    target_id: str | None = None
    previous_turns_remaining: int | None = None


class CooldownDecrementResult(FrozenModel):
    """Result of decrementing cooldowns at turn/round boundary."""

    effect_id: str
    was_decremented: bool
    turns_remaining_before: int
    turns_remaining_after: int
    target_id: str | None = None
    was_expired: bool = False


def check_action_on_cooldown(
    *,
    actor_cooldown_states: dict[str, CooldownState],
    effect_id: str,
    target_id: str | None = None,
) -> CooldownCheckResult:
    """Check if an action/effect is currently on cooldown for an actor.

    Args:
        actor_cooldown_states: Dict of cooldown states for the actor
        effect_id: The effect/action to check
        target_id: Optional target ID for per-target cooldowns

    Returns:
        CooldownCheckResult indicating if the action is on cooldown
    """
    if target_id:
        key = f"{effect_id}:{target_id}"
    else:
        key = effect_id

    cooldown = actor_cooldown_states.get(key)
    if cooldown is None:
        return CooldownCheckResult(
            is_on_cooldown=False,
            effect_id=effect_id,
        )

    if cooldown.turns_remaining > 0:
        return CooldownCheckResult(
            is_on_cooldown=True,
            effect_id=effect_id,
            turns_remaining=cooldown.turns_remaining,
            target_id=cooldown.target_id,
        )

    return CooldownCheckResult(
        is_on_cooldown=False,
        effect_id=effect_id,
        turns_remaining=0,
        target_id=cooldown.target_id,
    )


def apply_cooldown(
    *,
    actor_cooldown_states: dict[str, CooldownState],
    effect_id: str,
    duration: int,
    target_id: str | None = None,
    trigger_on: TriggerType | None = None,
    reset_on: CooldownResetTrigger = "scene_end",
) -> CooldownApplicationResult:
    """Apply a cooldown to an actor when an action is used.

    Args:
        actor_cooldown_states: Dict of cooldown states for the actor
        effect_id: The effect/action being used
        duration: How many turns the cooldown lasts
        target_id: Optional target ID for per-target cooldowns
        trigger_on: Optional trigger condition for the cooldown
        reset_on: When the cooldown resets (scene_end, rest, full_repair, turn_start, turn_end, round_end, never)

    Returns:
        CooldownApplicationResult with the result of applying the cooldown
    """
    if target_id:
        key = f"{effect_id}:{target_id}"
    else:
        key = effect_id

    existing = actor_cooldown_states.get(key)
    previous_turns = existing.turns_remaining if existing else None

    cooldown = CooldownState(
        effect_id=effect_id,
        turns_remaining=duration,
        duration=duration,
        trigger_on=trigger_on,
        reset_on=reset_on,
        per_target=target_id is not None,
        target_id=target_id,
    )

    actor_cooldown_states[key] = cooldown

    return CooldownApplicationResult(
        applied=True,
        effect_id=effect_id,
        duration=duration,
        turns_remaining=duration,
        target_id=target_id,
        previous_turns_remaining=previous_turns,
    )


def decrement_cooldowns_on_turn_start(
    *,
    actor_cooldown_states: dict[str, CooldownState],
) -> list[CooldownDecrementResult]:
    """Decrement cooldowns at the start of a turn.

    Only decrements cooldowns with reset_on == "turn_start".
    Cooldowns that reach 0 are removed.

    Args:
        actor_cooldown_states: Dict of cooldown states for the actor

    Returns:
        List of CooldownDecrementResult for each affected cooldown
    """
    results: list[CooldownDecrementResult] = []
    to_remove: list[str] = []

    for key, cooldown in actor_cooldown_states.items():
        if cooldown.reset_on != "turn_start":
            continue

        before = cooldown.turns_remaining
        if before > 1:
            cooldown = cooldown.model_copy(update={"turns_remaining": before - 1})
            actor_cooldown_states[key] = cooldown
            results.append(
                CooldownDecrementResult(
                    effect_id=cooldown.effect_id,
                    was_decremented=True,
                    turns_remaining_before=before,
                    turns_remaining_after=cooldown.turns_remaining,
                    target_id=cooldown.target_id,
                )
            )
        elif before == 1:
            results.append(
                CooldownDecrementResult(
                    effect_id=cooldown.effect_id,
                    was_decremented=True,
                    turns_remaining_before=1,
                    turns_remaining_after=0,
                    target_id=cooldown.target_id,
                    was_expired=True,
                )
            )
            to_remove.append(key)
        else:
            results.append(
                CooldownDecrementResult(
                    effect_id=cooldown.effect_id,
                    was_decremented=False,
                    turns_remaining_before=0,
                    turns_remaining_after=0,
                    target_id=cooldown.target_id,
                    was_expired=True,
                )
            )
            to_remove.append(key)

    for key in to_remove:
        del actor_cooldown_states[key]

    return results


def decrement_cooldowns_on_turn_end(
    *,
    actor_cooldown_states: dict[str, CooldownState],
) -> list[CooldownDecrementResult]:
    """Decrement cooldowns at the end of a turn.

    Only decrements cooldowns with reset_on == "turn_end".

    Args:
        actor_cooldown_states: Dict of cooldown states for the actor

    Returns:
        List of CooldownDecrementResult for each affected cooldown
    """
    results: list[CooldownDecrementResult] = []
    to_remove: list[str] = []

    for key, cooldown in actor_cooldown_states.items():
        if cooldown.reset_on != "turn_end":
            continue

        before = cooldown.turns_remaining
        if before > 0:
            cooldown = cooldown.model_copy(update={"turns_remaining": before - 1})
            actor_cooldown_states[key] = cooldown
            results.append(
                CooldownDecrementResult(
                    effect_id=cooldown.effect_id,
                    was_decremented=True,
                    turns_remaining_before=before,
                    turns_remaining_after=cooldown.turns_remaining,
                    target_id=cooldown.target_id,
                )
            )
        else:
            was_expired = cooldown.turns_remaining == 0 and cooldown.duration > 0
            results.append(
                CooldownDecrementResult(
                    effect_id=cooldown.effect_id,
                    was_decremented=False,
                    turns_remaining_before=0,
                    turns_remaining_after=0,
                    target_id=cooldown.target_id,
                    was_expired=was_expired,
                )
            )
            to_remove.append(key)

    for key in to_remove:
        del actor_cooldown_states[key]

    return results


def decrement_cooldowns_on_round_end(
    *,
    actor_cooldown_states: dict[str, CooldownState],
) -> list[CooldownDecrementResult]:
    """Decrement cooldowns at the end of a round.

    Only decrements cooldowns with reset_on == "round_end".

    Args:
        actor_cooldown_states: Dict of cooldown states for the actor

    Returns:
        List of CooldownDecrementResult for each affected cooldown
    """
    results: list[CooldownDecrementResult] = []
    to_remove: list[str] = []

    for key, cooldown in actor_cooldown_states.items():
        if cooldown.reset_on != "round_end":
            continue

        before = cooldown.turns_remaining
        if before > 0:
            cooldown = cooldown.model_copy(update={"turns_remaining": before - 1})
            actor_cooldown_states[key] = cooldown
            results.append(
                CooldownDecrementResult(
                    effect_id=cooldown.effect_id,
                    was_decremented=True,
                    turns_remaining_before=before,
                    turns_remaining_after=cooldown.turns_remaining,
                    target_id=cooldown.target_id,
                )
            )
        else:
            was_expired = cooldown.turns_remaining == 0 and cooldown.duration > 0
            results.append(
                CooldownDecrementResult(
                    effect_id=cooldown.effect_id,
                    was_decremented=False,
                    turns_remaining_before=0,
                    turns_remaining_after=0,
                    target_id=cooldown.target_id,
                    was_expired=was_expired,
                )
            )
            to_remove.append(key)

    for key in to_remove:
        del actor_cooldown_states[key]

    return results


def reset_cooldowns_on_scene_end(
    *,
    actor_cooldown_states: dict[str, CooldownState],
) -> list[str]:
    """Reset all cooldowns at scene end.

    Clears all cooldowns with reset_on == "scene_end" or "never" are retained.

    Args:
        actor_cooldown_states: Dict of cooldown states for the actor

    Returns:
        List of effect_ids that were cleared
    """
    cleared: list[str] = []
    to_remove: list[str] = []

    for key, cooldown in actor_cooldown_states.items():
        if cooldown.reset_on in ("scene_end", "rest", "full_repair"):
            to_remove.append(key)
            cleared.append(cooldown.effect_id)

    for key in to_remove:
        del actor_cooldown_states[key]

    return cleared


def get_cooldown_state(
    *,
    actor_cooldown_states: dict[str, CooldownState],
    effect_id: str,
    target_id: str | None = None,
) -> CooldownState | None:
    """Get the current cooldown state for an effect.

    Args:
        actor_cooldown_states: Dict of cooldown states for the actor
        effect_id: The effect to look up
        target_id: Optional target ID for per-target cooldowns

    Returns:
        CooldownState if found, None otherwise
    """
    if target_id:
        key = f"{effect_id}:{target_id}"
    else:
        key = effect_id

    return actor_cooldown_states.get(key)


def resolve_structure_damage(
    *,
    remaining_structure: int,
    incoming_damage: int,
    hp_before: int,
    structure_damage_marked: int = 1,
    inventory: MechInventory | None = None,
    rules: StructureDamageRules = DEFAULT_STRUCTURE_DAMAGE_RULES,
    settings: ResolutionSettings | None = None,
) -> StructureResolution:
    """
    Resolve a structure check, including spillover and direct hit outcomes.

    Args:
        structure_damage_marked: Total structure damage marked (including the one just taken).
        inventory: Inventory state used to resolve system trauma selections and fallbacks.
    """
    structure_damage = 1
    hp_after = max(hp_before - incoming_damage, 0)
    spillover = (
        max(incoming_damage - hp_before, 0) if rules.spillover_damage_applies else 0
    )

    dice_count = max(structure_damage_marked * rules.dice_per_structure_marked, 1)
    rolls = _roll_dice(dice_count, settings)
    chosen = _choose_lowest(rolls, 1)

    direct_hit_outcome = None
    system_trauma = None
    updated_inventory = None
    if rules.multiple_ones_crushing and rolls.count(1) >= 2:
        outcome = rules.crushing_hit_outcome
    else:
        outcome = _lookup_structure_outcome(chosen[0], rules)
        if outcome.name == "direct_hit":
            direct_hit_outcome = _lookup_direct_hit_outcome(remaining_structure, rules)
        elif outcome.name == "system_trauma" and inventory:
            system_trauma = _resolve_system_trauma(inventory, rules, settings)
            if system_trauma.resolved_target == "direct_hit":
                outcome = _direct_hit_outcome_base(rules)
                direct_hit_outcome = _lookup_direct_hit_outcome(
                    remaining_structure, rules
                )
            else:
                updated_inventory = _apply_system_trauma(inventory, system_trauma)

    return StructureResolution(
        outcome=outcome,
        dice=DiceRollResult(rolls=rolls, chosen=chosen),
        direct_hit_outcome=direct_hit_outcome,
        system_trauma=system_trauma,
        updated_inventory=updated_inventory,
        structure_damage=structure_damage,
        spillover_damage=spillover,
    )


def resolve_overheat(
    *,
    stress_marked: int,
    remaining_stress: int,
    rules: OverheatRules = DEFAULT_OVERHEAT_RULES,
    settings: ResolutionSettings | None = None,
) -> OverheatResolution:
    """
    Resolve an overheat check, including meltdown subtable outcomes.

    Args:
        stress_marked: Total stress boxes marked (including the one just taken).
        remaining_stress: Stress boxes remaining after marking.
    """
    dice_count = max(stress_marked, 1) if rules.roll_dice_per_stress else 1
    rolls = _roll_dice(dice_count, settings)
    chosen = _choose_lowest(rolls, 1)

    if rules.irreversible_meltdown_on_multiple_ones and rolls.count(1) >= 2:
        outcome = rules.irreversible_meltdown_outcome
    else:
        outcome = _lookup_overheat_outcome(chosen[0], rules)
    meltdown_outcome = None
    if outcome.name == "meltdown":
        meltdown_outcome = _lookup_meltdown_outcome(remaining_stress, rules)

    return OverheatResolution(
        outcome=outcome,
        dice=DiceRollResult(rolls=rolls, chosen=chosen),
        meltdown_outcome=meltdown_outcome,
        stress_damage=rules.stress_per_overheat,
    )


def _roll_dice(count: int, settings: ResolutionSettings | None) -> list[int]:
    if settings and settings.forced_rolls:
        if len(settings.forced_rolls) < count:
            raise ValueError("Forced rolls provided fewer results than dice count")
        return list(settings.forced_rolls[:count])
    return DiceExpression.parse(f"{count}d6").roll()


def _choose_lowest(rolls: list[int], count: int) -> list[int]:
    return sorted(rolls)[:count]


def _direct_hit_outcome_base(rules: StructureDamageRules) -> StructureOutcomeType:
    for entry in rules.table:
        if entry.outcome.name == "direct_hit":
            return entry.outcome
    return StructureOutcomeType(name="direct_hit")


def _resolve_system_trauma(
    inventory: MechInventory,
    rules: StructureDamageRules,
    settings: ResolutionSettings | None,
) -> SystemTraumaSelection:
    trauma_rules = rules.system_trauma_rules
    roll = (
        settings.forced_system_trauma_roll
        if settings and settings.forced_system_trauma_roll
        else None
    )
    if roll is None:
        roll_result = trauma_rules.roll.roll()
        roll = roll_result[0] if roll_result else 1

    if trauma_rules.mount_on.roll_min <= roll <= trauma_rules.mount_on.roll_max:
        initial_target: Literal["mount", "system"] = "mount"
    elif trauma_rules.system_on.roll_min <= roll <= trauma_rules.system_on.roll_max:
        initial_target = "system"
    else:
        raise ValueError("System trauma roll outside defined ranges")

    eligible_mounts = _eligible_mounts(
        inventory, trauma_rules.exclude_limited_no_charges
    )
    eligible_systems = _eligible_systems(
        inventory, trauma_rules.exclude_limited_no_charges
    )

    resolved_target: Literal["mount", "system", "direct_hit"] = initial_target
    destroyed_mount_index = None
    destroyed_system_id = None
    fallback_reason: Literal["none", "no_mounts", "no_systems", "none_available"] = (
        "none"
    )

    if initial_target == "mount":
        if eligible_mounts:
            destroyed_mount_index = eligible_mounts[0]
        elif trauma_rules.fallback_to_other_if_none and eligible_systems:
            resolved_target = "system"
            destroyed_system_id = eligible_systems[0]
            fallback_reason = "no_mounts"
        elif trauma_rules.fallback_to_direct_hit_if_none:
            resolved_target = "direct_hit"
            fallback_reason = "none_available"
        else:
            fallback_reason = "no_mounts"
    else:
        if eligible_systems:
            destroyed_system_id = eligible_systems[0]
        elif trauma_rules.fallback_to_other_if_none and eligible_mounts:
            resolved_target = "mount"
            destroyed_mount_index = eligible_mounts[0]
            fallback_reason = "no_systems"
        elif trauma_rules.fallback_to_direct_hit_if_none:
            resolved_target = "direct_hit"
            fallback_reason = "none_available"
        else:
            fallback_reason = "no_systems"

    return SystemTraumaSelection(
        roll=roll,
        initial_target=initial_target,
        resolved_target=resolved_target,
        eligible_mounts=eligible_mounts,
        eligible_systems=eligible_systems,
        destroyed_mount_index=destroyed_mount_index,
        destroyed_system_id=destroyed_system_id,
        fallback_reason=fallback_reason,
    )


def _eligible_mounts(
    inventory: MechInventory,
    exclude_limited_no_charges: bool,
) -> list[int]:
    mounts: list[int] = []
    for mount in inventory.mounts:
        if mount.destroyed:
            continue
        has_valid_weapon = any(
            _weapon_valid(weapon, exclude_limited_no_charges)
            for weapon in mount.weapons
        )
        if has_valid_weapon:
            mounts.append(mount.mount_index)
    return sorted(mounts)


def _eligible_systems(
    inventory: MechInventory,
    exclude_limited_no_charges: bool,
) -> list[str]:
    systems: list[str] = []
    for system in inventory.systems:
        if system.destroyed:
            continue
        if exclude_limited_no_charges and system.limited_charges_remaining == 0:
            continue
        systems.append(system.system_id)
    return sorted(systems)


def _weapon_valid(weapon: WeaponState, exclude_limited_no_charges: bool) -> bool:
    if weapon.destroyed:
        return False
    if exclude_limited_no_charges and weapon.limited_charges_remaining == 0:
        return False
    return True


def _apply_system_trauma(
    inventory: MechInventory,
    selection: SystemTraumaSelection,
) -> MechInventory:
    if (
        selection.resolved_target == "mount"
        and selection.destroyed_mount_index is not None
    ):
        mounts: list[WeaponMountState] = []
        for mount in inventory.mounts:
            if mount.mount_index == selection.destroyed_mount_index:
                destroyed_weapons = [
                    weapon.model_copy(update={"destroyed": True})
                    for weapon in mount.weapons
                ]
                mounts.append(mount.model_copy(update={"weapons": destroyed_weapons}))
            else:
                mounts.append(mount)
        return inventory.model_copy(update={"mounts": mounts})

    if (
        selection.resolved_target == "system"
        and selection.destroyed_system_id is not None
    ):
        systems: list[MechSystemState] = []
        for system in inventory.systems:
            if system.system_id == selection.destroyed_system_id:
                systems.append(system.model_copy(update={"destroyed": True}))
            else:
                systems.append(system)
        return inventory.model_copy(update={"systems": systems})

    return inventory


def _lookup_structure_outcome(
    roll: int, rules: StructureDamageRules
) -> StructureOutcomeType:
    for entry in rules.table:
        if entry.roll_min <= roll <= entry.roll_max:
            return entry.outcome
    raise ValueError("No structure outcome for roll")


def _lookup_direct_hit_outcome(
    remaining_structure: int,
    rules: StructureDamageRules,
) -> StructureOutcomeType:
    for entry in rules.direct_hit_outcomes:
        if remaining_structure < entry.remaining_structure_min:
            continue
        if (
            entry.remaining_structure_max is None
            or remaining_structure <= entry.remaining_structure_max
        ):
            return entry.outcome
    raise ValueError("No direct hit outcome for remaining structure")


def _lookup_overheat_outcome(roll: int, rules: OverheatRules) -> OverheatOutcomeType:
    for entry in rules.table:
        if entry.roll_min <= roll <= entry.roll_max:
            return entry.outcome
    raise ValueError("No overheat outcome for roll")


def _lookup_meltdown_outcome(
    remaining_stress: int,
    rules: OverheatRules,
) -> OverheatOutcomeType:
    for entry in rules.meltdown_outcomes:
        if remaining_stress < entry.remaining_stress_min:
            continue
        if (
            entry.remaining_stress_max is None
            or remaining_stress <= entry.remaining_stress_max
        ):
            return entry.outcome
    raise ValueError("No meltdown outcome for remaining stress")


class PreparedActionResult(FrozenModel):
    """Result of creating a prepared action."""

    success: bool
    prepared_action: "PreparedActionState | None" = None
    message: str = ""


class PreparedActionTriggerResult(FrozenModel):
    """Result of triggering a prepared action."""

    success: bool
    executed_action_id: str | None = None
    prepared_action_cleared: bool = False
    message: str = ""


class PerRoundReactionResult(FrozenModel):
    """Result of consuming a per-round reaction."""

    success: bool
    reaction_consumed: bool = False
    uses_remaining: int = 0
    message: str = ""


from core.mech.timing import PreparedActionState
from core.mech.combat_state import CombatantState, CombatRound


def prepare_action(
    combatant: CombatantState,
    held_action_id: str,
    held_action_type: Literal["quick", "full"],
    trigger_condition: str,
    current_round: int,
    expires_on_turn: int,
) -> PreparedActionResult:
    """Create a prepared action for a combatant.

    After using the Prepare action, the combatant holds a prepared action
    and cannot take other actions/reactions/movement until the trigger occurs
    or the prepared action expires at the start of their next turn.

    Args:
        combatant: The combatant preparing an action
        held_action_id: ID of the action to execute when triggered
        held_action_type: Type of the prepared action (quick or full)
        trigger_condition: Description of the trigger condition
        current_round: Current round number
        expires_on_turn: Turn number when prepared action expires

    Returns:
        PreparedActionResult indicating success and the prepared action state
    """
    prepared = PreparedActionState(
        held_action_id=held_action_id,
        held_action_type=held_action_type,
        trigger_condition=trigger_condition,
        created_on_turn=current_round,
        expires_on_turn=expires_on_turn,
        blocks_actions=True,
        blocks_reactions=True,
        blocks_movement=True,
    )
    return PreparedActionResult(
        success=True, prepared_action=prepared, message="Prepared action created"
    )


def trigger_prepared_action(
    combatant: CombatantState,
) -> PreparedActionTriggerResult:
    """Trigger a prepared action, executing it and clearing the prepared state.

    Args:
        combatant: The combatant with a prepared action

    Returns:
        PreparedActionTriggerResult indicating success and what happened
    """
    if not combatant.prepared_action:
        return PreparedActionTriggerResult(
            success=False,
            message="No prepared action to trigger",
        )

    action_id = combatant.prepared_action.held_action_id
    return PreparedActionTriggerResult(
        success=True,
        executed_action_id=action_id,
        prepared_action_cleared=True,
        message=f"Triggered prepared action: {action_id}",
    )


def expire_prepared_action(
    combatant: CombatantState,
) -> PreparedActionTriggerResult:
    """Expire a prepared action without triggering it.

    Called at the start of the combatant's turn when the prepared action expires.

    Args:
        combatant: The combatant with a prepared action

    Returns:
        PreparedActionTriggerResult indicating the prepared action was cleared
    """
    if not combatant.prepared_action:
        return PreparedActionTriggerResult(
            success=False,
            message="No prepared action to expire",
        )

    return PreparedActionTriggerResult(
        success=True,
        prepared_action_cleared=True,
        message="Prepared action expired at start of turn",
    )


def consume_per_round_reaction(
    combatant: CombatantState,
    action_id: str,
    max_per_round: int,
) -> PerRoundReactionResult:
    """Consume a per-round reaction (brace, overwatch).

    Args:
        combatant: The combatant using the reaction
        action_id: The reaction action ID
        max_per_round: Maximum uses per round for this reaction

    Returns:
        PerRoundReactionResult indicating if the reaction was consumed
    """
    current_count = combatant.per_round_reactions.get(action_id, 0)

    if current_count >= max_per_round:
        return PerRoundReactionResult(
            success=False,
            uses_remaining=0,
            message=f"Reaction {action_id} already used {current_count} time(s) this round",
        )

    new_count = current_count + 1
    return PerRoundReactionResult(
        success=True,
        reaction_consumed=True,
        uses_remaining=max_per_round - new_count,
        message=f"Consumed {action_id} (uses remaining: {max_per_round - new_count})",
    )


def reset_per_round_reactions(
    combatant: CombatantState,
) -> None:
    """Reset per-round reaction counts for a combatant.

    Called at the start of a new round.

    Args:
        combatant: The combatant to reset
    """
    pass


def reset_round_reaction_counts(
    round_: CombatRound,
) -> CombatRound:
    """Reset all per-round reaction counts for a new round.

    Args:
        round_: The combat round to reset reaction counts for

    Returns:
        Updated CombatRound with cleared reaction counts
    """
    return round_.model_copy(update={"reaction_counts_by_actor": {}})
