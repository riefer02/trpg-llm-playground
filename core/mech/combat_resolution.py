"""Structure and overheat resolution helpers."""

from __future__ import annotations

from typing import Literal, Generic, TypeVar
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
    DeployableState,
    MechCombatScenario,
    HexPosition,
)
from core.mech.grid import HexCoord, is_adjacent_by_size
from core.shared.effects import (
    PerTargetCounter,
    CooldownState,
    CooldownEffect,
    CooldownResetTrigger,
    TriggerType,
)
from core.shared.id_helpers import (
    SystemIdField,
    EffectIdField,
    CombatantIdField,
    ActionIdField,
    DeployableIdField,
)

T = TypeVar("T")


class DiceRollResult(FrozenModel):
    """Raw dice roll outcome."""

    rolls: list[int] = Field(default_factory=list)
    chosen: list[int] = Field(default_factory=list)


class ResolutionResult(FrozenModel, Generic[T]):
    """Generic base for resolution results containing outcome and dice information."""

    outcome: T
    dice: DiceRollResult


class StructureResolution(ResolutionResult[StructureOutcomeType]):
    """Resolution result for structure damage.

    Per PR2 4618-4636: Structure checks determine damage to systems and mounts
    when a mech takes damage that exceeds its remaining HP.
    """

    direct_hit_outcome: StructureOutcomeType | None = None
    system_trauma: SystemTraumaSelection | None = None
    updated_inventory: MechInventory | None = None
    structure_damage: int = 1
    spillover_damage: int = 0


class OverheatResolution(ResolutionResult[OverheatOutcomeType]):
    """Resolution result for overheat checks.

    Per PR2 4654-4706: Overheat checks determine stress damage and potential
    meltdown when a mech's heat exceeds its threshold.
    """

    meltdown_outcome: OverheatOutcomeType | None = None
    stress_damage: int = 1


class ResolutionSettings(FrozenModel):
    """Settings for deterministic or forced resolution."""

    forced_rolls: list[int] | None = None
    forced_system_trauma_roll: int | None = Field(default=None, ge=1, le=6)
    # New fields for attack resolution
    forced_roll: int | None = Field(
        default=None, ge=1, le=20, description="Forced d20 roll for attack resolution"
    )
    forced_accuracy_rolls: list[int] | None = Field(
        default=None, description="Forced accuracy d6 rolls"
    )
    forced_difficulty_rolls: list[int] | None = Field(
        default=None, description="Forced difficulty d6 rolls"
    )


class SystemTraumaSelection(FrozenModel):
    """Resolved selection for system trauma results."""

    roll: int
    initial_target: Literal["mount", "system"]
    resolved_target: Literal["mount", "system", "direct_hit"]
    eligible_mounts: list[int] = Field(default_factory=list)
    eligible_systems: list[str] = Field(default_factory=list)
    destroyed_mount_index: int | None = None
    destroyed_system_id: SystemIdField | None = None
    fallback_reason: Literal["none", "no_mounts", "no_systems", "none_available"] = (
        "none"
    )


class PerTargetCounterResolution(FrozenModel):
    """Resolution result for per-target counter application."""

    effect_id: EffectIdField
    target_id: CombatantIdField
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
    effect_id: EffectIdField
    turns_remaining: int | None = None
    target_id: CombatantIdField | None = None


class CooldownApplicationResult(FrozenModel):
    """Result of applying a cooldown to a combatant."""

    applied: bool
    effect_id: EffectIdField
    duration: int
    turns_remaining: int
    target_id: CombatantIdField | None = None
    previous_turns_remaining: int | None = None


class CooldownDecrementResult(FrozenModel):
    """Result of decrementing cooldowns at turn/round boundary."""

    effect_id: EffectIdField
    was_decremented: bool
    turns_remaining_before: int
    turns_remaining_after: int
    target_id: CombatantIdField | None = None
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
    executed_action_id: ActionIdField | None = None
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
    held_action_id: ActionIdField,
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
    action_id: ActionIdField,
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
) -> CombatantState:
    """Reset per-round reaction counts for a combatant.

    Called at the start of a new round per PR2 reaction rules.

    Args:
        combatant: The combatant to reset

    Returns:
        CombatantState with cleared per_round_reactions dict
    """
    return combatant.model_copy(update={"per_round_reactions": {}})


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


class BalorScouringSwarmResult(FrozenModel):
    """Result of Scouring Swarm zone damage application."""

    damage_dealt: bool = Field(..., description="Whether damage was dealt")
    damage_per_target: int = Field(..., description="Damage applied per target")
    affected_targets: list[str] = Field(
        default_factory=list, description="List of affected target IDs"
    )
    is_core_power_active: bool = Field(
        ..., description="Whether core power is active (affects damage)"
    )


class BalorRegenerationResult(FrozenModel):
    """Result of Regeneration healing application."""

    healing_applied: bool = Field(..., description="Whether healing was applied")
    healing_amount: int = Field(..., description="Amount of HP healed")
    was_paused: bool = Field(..., description="Whether regeneration was paused")
    pause_reason: str | None = Field(
        default=None, description="Reason for pause if applicable"
    )


class BalorSelfPerpetuatingResult(FrozenModel):
    """Result of Self-Perpetuating full restore on rest."""

    hp_restored: bool = Field(..., description="Whether HP was restored")
    previous_hp: int = Field(..., description="HP before restoration")
    new_hp: int = Field(..., description="HP after restoration")


class HellswarmProtocolResult(FrozenModel):
    """Result of HELLSWARM protocol activation."""

    cover_granted: bool = Field(..., description="Whether cover was granted")
    soft_cover_targets: list[str] = Field(
        default_factory=list, description="Targets that received soft cover"
    )
    shredded_applied: bool = Field(..., description="Whether shredded was applied")
    structure_avoidance_triggered: bool = Field(
        ..., description="Whether structure damage was avoided"
    )
    structure_roll: list[int] | None = Field(
        default=None, description="Dice roll for structure avoidance"
    )
    avoided_structure_damage: bool = Field(
        ..., description="Whether structure damage was avoided on this check"
    )
    healing_applied: bool = Field(..., description="Whether HP healing was applied")
    healing_amount: int | None = Field(
        default=None, description="Amount of HP healed (half max)"
    )


class HiveDroneResult(FrozenModel):
    """Result of Hive Drone activation."""

    drone_deployed: bool = Field(..., description="Whether drone was deployed")
    drone_position: HexCoord | None = Field(
        default=None, description="Position of deployed drone (hex coordinates)"
    )
    damage_dealt: bool = Field(..., description="Whether drone damage was applied")
    affected_targets: list[str] = Field(
        default_factory=list, description="Targets affected by drone"
    )


class SwarmBodyResult(FrozenModel):
    """Result of Swarm Body zone effect."""

    zone_active: bool = Field(..., description="Whether zone is active")
    condition_met: bool = Field(..., description="Whether condition for zone is met")
    save_triggered: bool = Field(..., description="Whether save was triggered")
    affected_targets: list[str] = Field(
        default_factory=list, description="Targets affected by zone"
    )
    damage_per_target: int = Field(
        ..., description="Damage applied per failed save target"
    )
    damage_applied_to: list[str] = Field(
        default_factory=list, description="Targets that took damage"
    )
    condition_ended: bool = Field(
        default=False, description="Whether condition ended (e.g., movement)"
    )


def resolve_scouring_swarm(
    *,
    combatant_id: CombatantIdField,
    is_core_power_active: bool,
    affected_target_ids: list[str],
    zone_shape: str = "burst",
    zone_size: int = 1,
) -> BalorScouringSwarmResult:
    """Resolve Scouring Swarm zone damage at start of turn.

    The Balor's Scouring Swarm trait deals kinetic damage to all targets in
    the zone at the start of the Balor's turn. Damage increases from 2 to 4
    when the Hellswarm core power is active.

    Args:
        combatant_id: The Balor combatant ID (source of the swarm)
        is_core_power_active: Whether Hellswarm core power is currently active
        affected_target_ids: List of target IDs in the swarm zone
        zone_shape: Shape of the zone (default: burst)
        zone_size: Size of the zone (default: 1)

    Returns:
        BalorScouringSwarmResult with damage application details
    """
    damage_per_target = 4 if is_core_power_active else 2
    damage_dealt = len(affected_target_ids) > 0

    return BalorScouringSwarmResult(
        damage_dealt=damage_dealt,
        damage_per_target=damage_per_target,
        affected_targets=affected_target_ids,
        is_core_power_active=is_core_power_active,
    )


def resolve_balor_regeneration(
    *,
    combatant_id: str,
    max_hp: int,
    is_overheated: bool,
    has_structure_damage: bool,
    is_core_power_active: bool,
    current_hp: int,
) -> BalorRegenerationResult:
    """Resolve Balor Regeneration trait at end of turn.

    The Balor's Regeneration trait heals 1/4 max HP at end of turn,
    but pauses if the Balor is overheated, has structure damage,
    or has the core power active.

    Args:
        combatant_id: The Balor combatant ID
        max_hp: Maximum HP for calculating healing amount
        is_overheated: Whether the Balor is currently overheated
        has_structure_damage: Whether the Balor has taken structure damage
        is_core_power_active: Whether Hellswarm core power is active
        current_hp: Current HP before regeneration

    Returns:
        BalorRegenerationResult with healing application details
    """
    pause_reasons: list[str] = []
    if is_overheated:
        pause_reasons.append("overheated")
    if has_structure_damage:
        pause_reasons.append("structure_damage")
    if is_core_power_active:
        pause_reasons.append("core_power_active")

    was_paused = len(pause_reasons) > 0
    pause_reason = ", ".join(pause_reasons) if was_paused else None

    if was_paused:
        return BalorRegenerationResult(
            healing_applied=False,
            healing_amount=0,
            was_paused=True,
            pause_reason=pause_reason,
        )

    healing_amount = max(1, max_hp // 4)
    new_hp = min(current_hp + healing_amount, max_hp)
    actual_healing = new_hp - current_hp

    return BalorRegenerationResult(
        healing_applied=actual_healing > 0,
        healing_amount=actual_healing,
        was_paused=False,
        pause_reason=None,
    )


def resolve_self_perpetuating(
    *,
    combatant_id: str,
    current_hp: int,
    max_hp: int,
    is_during_rest: bool,
) -> BalorSelfPerpetuatingResult:
    """Resolve Balor Self-Perpetuating trait on rest activation.

    The Balor's Self-Perpetuating trait fully restores HP when
    the Balor takes a rest action.

    Args:
        combatant_id: The Balor combatant ID
        current_hp: Current HP before restoration
        max_hp: Maximum HP
        is_during_rest: Whether a rest action is being taken

    Returns:
        BalorSelfPerpetuatingResult with restoration details
    """
    if not is_during_rest:
        return BalorSelfPerpetuatingResult(
            hp_restored=False,
            previous_hp=current_hp,
            new_hp=current_hp,
        )

    return BalorSelfPerpetuatingResult(
        hp_restored=True,
        previous_hp=current_hp,
        new_hp=max_hp,
    )


def activate_hellswarm_protocol(
    *,
    combatant_id: str,
    adjacent_ally_ids: list[str],
    structure_damage_marked: int,
    current_hp: int,
    max_hp: int,
    settings: ResolutionSettings | None = None,
) -> HellswarmProtocolResult:
    """Activate HELLSWARM protocol (Hellswarm core power).

    The Hellswarm protocol provides the following effects:
    - Grants soft cover to self and adjacent allies
    - Applies shredded status to self
    - At end of turn (if not overheated/structure damage), heals half max HP
    - Structure damage avoidance: on structure loss, roll 1d6; if 6+, heal to 1 HP

    Args:
        combatant_id: The Balor combatant ID
        adjacent_ally_ids: IDs of allies adjacent to the Balor
        structure_damage_marked: Amount of structure damage marked (triggers avoidance)
        current_hp: Current HP before any healing
        max_hp: Maximum HP for calculating healing amount
        settings: Resolution settings for deterministic rolls

    Returns:
        HellswarmProtocolResult with protocol activation details
    """
    cover_granted = True
    soft_cover_targets = [combatant_id] + adjacent_ally_ids
    shredded_applied = True

    structure_avoidance_triggered = structure_damage_marked > 0
    avoided_structure_damage = False
    structure_roll: list[int] | None = None

    if structure_avoidance_triggered:
        roll_result = _roll_dice(1, settings)
        structure_roll = roll_result
        if roll_result[0] >= 6:
            avoided_structure_damage = True

    healing_applied = False
    healing_amount: int | None = None

    if not structure_avoidance_triggered:
        healing_amount = max(1, max_hp // 2)
        new_hp = min(current_hp + healing_amount, max_hp)
        actual_healing = new_hp - current_hp
        healing_applied = actual_healing > 0
        healing_amount = actual_healing if healing_applied else None

    return HellswarmProtocolResult(
        cover_granted=cover_granted,
        soft_cover_targets=soft_cover_targets,
        shredded_applied=shredded_applied,
        structure_avoidance_triggered=structure_avoidance_triggered,
        structure_roll=structure_roll,
        avoided_structure_damage=avoided_structure_damage,
        healing_applied=healing_applied,
        healing_amount=healing_amount,
    )


def deploy_hive_drone(
    *,
    combatant_id: str,
    deploy_position: HexCoord,
    enemy_target_ids: list[str],
) -> HiveDroneResult:
    """Deploy a Hive Drone to a specific position.

    The Hive Drone creates a burst 2 zone that applies AP kinetic damage
    to enemies at start of turn and on zone entry.

    Args:
        combatant_id: The Balor combatant deploying the drone
        deploy_position: Position to deploy the drone (hex coordinates)
        enemy_target_ids: IDs of potential enemy targets in range

    Returns:
        HiveDroneResult with deployment details
    """
    return HiveDroneResult(
        drone_deployed=True,
        drone_position=deploy_position,
        damage_dealt=False,
        affected_targets=[],
    )


def resolve_hive_drone_turn_start(
    *,
    drone_position: HexCoord,
    enemy_target_ids: list[str],
    zone_size: int = 2,
) -> HiveDroneResult:
    """Resolve Hive Drone damage at start of turn.

    The Hive Drone deals 1 AP kinetic damage to all enemies in its
    burst 2 zone at the start of the drone's turn.

    Args:
        drone_position: Position of the Hive Drone
        enemy_target_ids: IDs of enemies potentially in the zone
        zone_size: Size of the drone's burst zone (default: 2)

    Returns:
        HiveDroneResult with damage application details
    """
    damage_dealt = len(enemy_target_ids) > 0

    return HiveDroneResult(
        drone_deployed=True,
        drone_position=drone_position,
        damage_dealt=damage_dealt,
        affected_targets=enemy_target_ids,
    )


def resolve_hive_drone_zone_entry(
    *,
    entering_combatant_id: str,
    drone_position: HexCoord,
    zone_size: int = 2,
) -> HiveDroneResult:
    """Resolve Hive Drone damage on zone entry.

    The Hive Drone deals 1 AP kinetic damage to any combatant
    that enters its burst 2 zone.

    Args:
        entering_combatant_id: ID of the combatant entering the zone
        drone_position: Position of the Hive Drone
        zone_size: Size of the drone's burst zone (default: 2)

    Returns:
        HiveDroneResult with damage application details
    """
    return HiveDroneResult(
        drone_deployed=True,
        drone_position=drone_position,
        damage_dealt=True,
        affected_targets=[entering_combatant_id],
    )


def activate_swarm_body(
    *,
    combatant_id: str,
    current_hp: int,
    has_moved_this_turn: bool,
    enemy_target_ids: list[str],
) -> SwarmBodyResult:
    """Activate Swarm Body zone effect.

    The Swarm Body creates a burst 1 zone around the Balor while the
    Balor has not moved this turn. Enemies in the zone must pass a
    systems save or take 3 kinetic damage on turn start and zone entry.
    The zone ends when the Balor moves.

    Args:
        combatant_id: The Balor combatant ID
        current_hp: Current HP (not directly used, for context)
        has_moved_this_turn: Whether the Balor has moved this turn
        enemy_target_ids: IDs of enemies potentially in the zone

    Returns:
        SwarmBodyResult with zone effect details
    """
    condition_met = not has_moved_this_turn
    zone_active = condition_met

    return SwarmBodyResult(
        zone_active=zone_active,
        condition_met=condition_met,
        save_triggered=False,
        affected_targets=[],
        damage_per_target=3,
        damage_applied_to=[],
        condition_ended=False,
    )


def resolve_swarm_body_turn_start(
    *,
    combatant_id: str,
    has_moved_this_turn: bool,
    enemy_target_ids: list[str],
) -> SwarmBodyResult:
    """Resolve Swarm Body save check at start of turn.

    Enemies in the Swarm Body zone must make a systems save at the
    start of their turn or take 3 kinetic damage.

    Args:
        combatant_id: The Balor combatant ID (zone source)
        has_moved_this_turn: Whether the Balor has moved this turn
        enemy_target_ids: IDs of enemies in the zone

    Returns:
        SwarmBodyResult with save and damage details
    """
    condition_met = not has_moved_this_turn

    if not condition_met:
        return SwarmBodyResult(
            zone_active=False,
            condition_met=False,
            save_triggered=False,
            affected_targets=[],
            damage_per_target=3,
            damage_applied_to=[],
            condition_ended=True,
        )

    return SwarmBodyResult(
        zone_active=True,
        condition_met=True,
        save_triggered=True,
        affected_targets=enemy_target_ids,
        damage_per_target=3,
        damage_applied_to=enemy_target_ids,
        condition_ended=False,
    )


def resolve_swarm_body_zone_entry(
    *,
    entering_combatant_id: str,
    balor_has_moved: bool,
) -> SwarmBodyResult:
    """Resolve Swarm Body save check on zone entry.

    A combatant entering the Swarm Body zone must make a systems save
    or take 3 kinetic damage.

    Args:
        entering_combatant_id: ID of the combatant entering the zone
        balor_has_moved: Whether the Balor has moved this turn (zone inactive if true)

    Returns:
        SwarmBodyResult with save and damage details
    """
    zone_active = not balor_has_moved

    if not zone_active:
        return SwarmBodyResult(
            zone_active=False,
            condition_met=False,
            save_triggered=False,
            affected_targets=[],
            damage_per_target=3,
            damage_applied_to=[],
            condition_ended=False,
        )

    return SwarmBodyResult(
        zone_active=True,
        condition_met=True,
        save_triggered=True,
        affected_targets=[entering_combatant_id],
        damage_per_target=3,
        damage_applied_to=[entering_combatant_id],
        condition_ended=False,
    )


def end_swarm_body_condition(
    *,
    combatant_id: str,
) -> SwarmBodyResult:
    """End Swarm Body zone effect due to movement.

    The Swarm Body zone ends when the Balor moves.

    Args:
        combatant_id: The Balor combatant ID

    Returns:
        SwarmBodyResult indicating the condition has ended
    """
    return SwarmBodyResult(
        zone_active=False,
        condition_met=False,
        save_triggered=False,
        affected_targets=[],
        damage_per_target=3,
        damage_applied_to=[],
        condition_ended=True,
    )


class OverchargeEscalationResult(FrozenModel):
    """Result of computing overcharge escalation state."""

    current_level: int = Field(..., description="Current escalation level (0-3)")
    next_cost: "int | DiceExpression" = Field(
        ..., description="Heat cost for next overcharge"
    )
    can_overcharge: bool = Field(
        ..., description="Whether overcharge can be used this turn"
    )
    uses_this_turn: int = Field(..., description="Number of overcharge uses this turn")


class OverchargeUsageResult(FrozenModel):
    """Result of using overcharge."""

    level_before: int = Field(..., description="Escalation level before use")
    level_after: int = Field(..., description="Escalation level after use")
    cost: "int | DiceExpression" = Field(..., description="Cost of this overcharge")
    rolled_cost: int | None = Field(
        default=None, description="Evaluated cost if dice were rolled"
    )
    uses_this_turn_before: int = Field(..., description="Uses before this action")
    uses_this_turn_after: int = Field(..., description="Uses after this action")


class OverchargeResetResult(FrozenModel):
    """Result of resetting overcharge (on full repair)."""

    level_before: int = Field(..., description="Level before reset")
    level_after: int = Field(..., description="Level after reset (always 0)")
    uses_cleared: bool = Field(..., description="Whether uses_this_turn was cleared")


from core.mech.combat_state import OverchargeState


def compute_overcharge_escalation(
    overcharge_state: OverchargeState | None,
) -> OverchargeEscalationResult:
    """Compute the current overcharge escalation state for a combatant.

    Args:
        overcharge_state: Current overcharge state (None for initial)

    Returns:
        OverchargeEscalationResult with current escalation info
    """
    if overcharge_state is None:
        overcharge_state = OverchargeState()

    return OverchargeEscalationResult(
        current_level=overcharge_state.current_level,
        next_cost=overcharge_state.next_cost,
        can_overcharge=overcharge_state.can_overcharge,
        uses_this_turn=overcharge_state.uses_this_turn,
    )


def use_overcharge(
    overcharge_state: OverchargeState | None,
    force_roll: int | None = None,
) -> tuple[OverchargeState, OverchargeUsageResult]:
    """Use overcharge, escalating the cost level.

    Args:
        overcharge_state: Current overcharge state (None for initial)
        force_roll: Optional forced roll value for deterministic testing

    Returns:
        Tuple of (updated OverchargeState, OverchargeUsageResult)
    """
    from core.mech.rules import DEFAULT_OVERCHARGE_RULES

    rules = DEFAULT_OVERCHARGE_RULES

    if overcharge_state is None:
        overcharge_state = OverchargeState()

    level_before = overcharge_state.current_level
    uses_before = overcharge_state.uses_this_turn

    cost = rules.costs[level_before]
    rolled_cost: int | None = None

    if isinstance(cost, DiceExpression):
        if force_roll is not None:
            rolled_cost = force_roll
        else:
            roll_result = cost.roll()
            rolled_cost = sum(roll_result) if roll_result else 0
    else:
        rolled_cost = cost

    new_level = min(level_before + 1, 3)
    new_uses = uses_before + 1

    new_state = OverchargeState(
        current_level=new_level,
        uses_this_turn=new_uses,
    )

    result = OverchargeUsageResult(
        level_before=level_before,
        level_after=new_level,
        cost=cost,
        rolled_cost=rolled_cost,
        uses_this_turn_before=uses_before,
        uses_this_turn_after=new_uses,
    )

    return new_state, result


def reset_overcharge(
    overcharge_state: OverchargeState | None,
) -> tuple[OverchargeState, OverchargeResetResult]:
    """Reset overcharge escalation (called on full repair).

    Args:
        overcharge_state: Current overcharge state (None for initial)

    Returns:
        Tuple of (reset OverchargeState, OverchargeResetResult)
    """
    if overcharge_state is None:
        return OverchargeState(), OverchargeResetResult(
            level_before=0, level_after=0, uses_cleared=True
        )

    result = OverchargeResetResult(
        level_before=overcharge_state.current_level,
        level_after=0,
        uses_cleared=overcharge_state.uses_this_turn > 0,
    )

    return OverchargeState(), result


def increment_overcharge_on_turn_start(
    overcharge_state: OverchargeState | None,
) -> OverchargeState:
    """Reset overcharge uses at the start of a new turn.

    Args:
        overcharge_state: Current overcharge state (None for initial)

    Returns:
        Updated OverchargeState with uses_this_turn reset to 0
    """
    if overcharge_state is None:
        return OverchargeState()

    if overcharge_state.uses_this_turn == 0:
        return overcharge_state

    return OverchargeState(
        current_level=overcharge_state.current_level,
        uses_this_turn=0,
    )


class DeploymentResult(FrozenModel):
    """Result of deploying an object."""

    deployable_id: DeployableIdField
    deployable_name: str
    kind: Literal["drone", "mine", "deployable", "other"]
    owner_id: CombatantIdField | None
    position_q: int
    position_r: int
    size: int
    hp: int
    max_hp: int
    evasion: int
    is_armed: bool = False
    arming_turn: int | None = None


class DeployableDamageResult(FrozenModel):
    """Result of damaging a deployable."""

    deployable_id: str
    damage_dealt: int
    hp_before: int
    hp_after: int
    is_destroyed: bool
    destroyed: bool = False


class DroneActionResult(FrozenModel):
    """Result of a drone taking action on owner's turn."""

    deployable_id: str
    drone_name: str
    action_taken: str
    can_act: bool
    can_move: bool


class MineTriggerResult(FrozenModel):
    """Result of a mine being triggered."""

    mine_id: DeployableIdField
    mine_name: str
    triggered_by_id: CombatantIdField
    triggered_by_name: str
    was_armed: bool
    detonated: bool
    position_q: int
    position_r: int


class DangerZoneStatus(FrozenModel):
    """Danger zone status for a combatant."""

    combatant_id: str
    combatant_name: str
    heat_current: int
    heat_cap: int
    danger_zone_threshold: int
    in_danger_zone: bool


def deploy_object(
    *,
    scenario: MechCombatScenario,
    deployable_id: str,
    name: str,
    kind: Literal["drone", "mine", "deployable", "other"],
    owner_id: str | None,
    position: HexPosition,
    size: int,
    hp: int,
    max_hp: int,
    armor: int = 0,
    evasion: int = 5,
    cover: Literal["soft", "hard"] | None = None,
    can_act: bool = False,
    can_move: bool = False,
    acts_on_owner_turn: bool = True,
    is_armed: bool = False,
    arming_turn: int | None = None,
    trigger_on_adjacent_entry: bool = False,
    detection_dc: int | None = None,
    disarm_dc: int | None = None,
    e_defense: int = 10,
    reactions: list[str] | None = None,
) -> tuple[MechCombatScenario, DeploymentResult]:
    """Deploy an object (drone, mine, or deployable) to a position.

    Per PR2 rules:
    - Deployables: Quick action, free adjacent valid space, 10 HP/size, evasion 5
    - Drones: Size ½, evasion 10, HP 10, armor 0, act on owner's turn
    - Mines: Quick action, free adjacent space, arm at start of next turn

    Args:
        scenario: Current combat scenario
        deployable_id: Unique ID for this deployable
        name: Display name
        kind: Type of deployable
        owner_id: ID of who deployed it
        position: Position to deploy at
        size: Size category
        hp: Current HP
        max_hp: Maximum HP
        armor: Armor value (default 0)
        evasion: Evasion value (default 5 for deployables, 10 for drones)
        cover: Cover type if any
        can_act: Whether drone can take actions
        can_move: Whether drone can move
        acts_on_owner_turn: Whether drone acts on owner's turn
        is_armed: Whether mine is already armed
        arming_turn: Turn number when mine will arm
        trigger_on_adjacent_entry: Whether mine triggers on adjacent entry
        detection_dc: Systems check DC to detect
        disarm_dc: Systems check DC to disarm
        e_defense: E-Defense value
        reactions: Available reactions

    Returns:
        Tuple of (updated MechCombatScenario, DeploymentResult)
    """
    from core.mech.combat_state import DeployableState

    deployable = DeployableState(
        id=deployable_id,
        name=name,
        kind=kind,
        owner_id=owner_id,
        position=position,
        size=size,
        hp=hp,
        max_hp=max_hp,
        armor=armor,
        evasion=evasion,
        cover=cover,
        can_act=can_act,
        can_move=can_move,
        acts_on_owner_turn=acts_on_owner_turn,
        is_armed=is_armed,
        arming_turn=arming_turn,
        trigger_on_adjacent_entry=trigger_on_adjacent_entry,
        detection_dc=detection_dc,
        disarm_dc=disarm_dc,
        e_defense=e_defense,
        reactions=reactions or [],
    )

    new_scenario = MechCombatScenario(
        combatants=scenario.combatants,
        grapples=scenario.grapples,
        rounds=scenario.rounds,
        terrain=scenario.terrain,
        environment=scenario.environment,
        deployables={**scenario.deployables, deployable_id: deployable},
        sitrep_resolution=scenario.sitrep_resolution,
        pending_decisions=list(scenario.pending_decisions),
    )

    result = DeploymentResult(
        deployable_id=deployable_id,
        deployable_name=name,
        kind=kind,
        owner_id=owner_id,
        position_q=position.coord.q,
        position_r=position.coord.r,
        size=size,
        hp=hp,
        max_hp=max_hp,
        evasion=evasion,
        is_armed=is_armed,
        arming_turn=arming_turn,
    )

    return new_scenario, result


def damage_deployable(
    *,
    scenario: MechCombatScenario,
    deployable_id: str,
    damage: int,
    armor_piercing: int = 0,
) -> tuple[MechCombatScenario, DeployableDamageResult]:
    """Apply damage to a deployable.

    Args:
        scenario: Current combat scenario
        deployable_id: ID of deployable to damage
        damage: Amount of damage
        armor_piercing: Armor piercing value

    Returns:
        Tuple of (updated MechCombatScenario, DeployableDamageResult)
    """
    if deployable_id not in scenario.deployables:
        raise ValueError(f"Deployable {deployable_id} not found in scenario")

    deployable = scenario.deployables[deployable_id]
    hp_before = deployable.hp

    effective_armor = max(0, deployable.armor - armor_piercing)
    net_damage = max(0, damage - effective_armor)
    new_hp = max(0, deployable.hp - net_damage)
    is_destroyed = new_hp == 0

    updated_deployable = DeployableState(
        **{
            k: v
            for k, v in deployable.model_dump().items()
            if k not in ("hp", "is_destroyed")
        },
        hp=new_hp,
        is_destroyed=is_destroyed,
    )

    new_scenario = MechCombatScenario(
        combatants=scenario.combatants,
        grapples=scenario.grapples,
        rounds=scenario.rounds,
        terrain=scenario.terrain,
        environment=scenario.environment,
        deployables={**scenario.deployables, deployable_id: updated_deployable},
        sitrep_resolution=scenario.sitrep_resolution,
        pending_decisions=list(scenario.pending_decisions),
    )

    result = DeployableDamageResult(
        deployable_id=deployable_id,
        damage_dealt=net_damage,
        hp_before=hp_before,
        hp_after=new_hp,
        is_destroyed=is_destroyed,
        destroyed=is_destroyed,
    )

    return new_scenario, result


def check_mine_trigger(
    scenario: MechCombatScenario,
    moving_combatant_id: str,
    new_position: HexPosition,
) -> list[MineTriggerResult]:
    """Check if any armed mines are triggered by moving to a position.

    Per PR2: Mines trigger when any character enters an adjacent space.
    Leaving a space does NOT trigger mines.

    Args:
        scenario: Current combat scenario
        moving_combatant_id: ID of combatant moving
        new_position: Position being moved to

    Returns:
        List of MineTriggerResult for each triggered mine
    """
    moving_combatant = None
    for c in scenario.combatants:
        if c.id == moving_combatant_id:
            moving_combatant = c
            break

    if moving_combatant is None:
        return []

    # Get mover's size for adjacency calculation
    mover_size = moving_combatant.stats.size if moving_combatant.stats else "size_1"

    results = []

    for mine_id, mine in scenario.deployables.items():
        if mine.kind != "mine" or not mine.is_armed:
            continue

        mine_coord = HexCoord(q=mine.position.coord.q, r=mine.position.coord.r)
        new_coord = HexCoord(q=new_position.coord.q, r=new_position.coord.r)

        # Use size-aware adjacency - mines are effectively size 1
        if is_adjacent_by_size(mine_coord, new_coord, "size_1", mover_size):
            result = MineTriggerResult(
                mine_id=mine_id,
                mine_name=mine.name,
                triggered_by_id=moving_combatant_id,
                triggered_by_name=moving_combatant.name
                if moving_combatant
                else "Unknown",
                was_armed=mine.is_armed,
                detonated=True,
                position_q=mine.position.coord.q,
                position_r=mine.position.coord.r,
            )
            results.append(result)

    return results


def get_combatants_in_danger_zone(
    scenario: MechCombatScenario,
    danger_zone_fraction: float = 0.5,
    rounding: Literal["up", "down"] = "up",
) -> list[DangerZoneStatus]:
    """Get all combatants currently in danger zone.

    Per PR2: When a mech has 1/2 of its total heat capacity filled (rounded up),
    it's in the danger zone.

    Args:
        scenario: Current combat scenario
        danger_zone_fraction: Fraction of heat cap for danger zone
        rounding: How to round the threshold

    Returns:
        List of DangerZoneStatus for each combatant
    """
    results = []

    for combatant in scenario.combatants:
        if combatant.resources.heat_cap <= 0:
            continue

        if rounding == "up":
            threshold = int(
                (combatant.resources.heat_cap * danger_zone_fraction) + 0.999
            )
        else:
            threshold = int(combatant.resources.heat_cap * danger_zone_fraction)

        in_danger = combatant.resources.heat_current >= threshold

        status = DangerZoneStatus(
            combatant_id=combatant.id,
            combatant_name=combatant.name,
            heat_current=combatant.resources.heat_current,
            heat_cap=combatant.resources.heat_cap,
            danger_zone_threshold=threshold,
            in_danger_zone=in_danger,
        )
        results.append(status)

    return results


import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.shared.structure import StructureInput, StructureResolutionResult
    from core.shared.heat import OverheatInput, OverheatResolutionResult, MeltdownState


def resolve_structure_damage_deprecated(
    *,
    remaining_structure: int,
    incoming_damage: int,
    hp_before: int,
    structure_damage_marked: int = 1,
    inventory: "MechInventory | None" = None,
    rules: "StructureDamageRules" = DEFAULT_STRUCTURE_DAMAGE_RULES,
    settings: "ResolutionSettings | None" = None,
) -> StructureResolution:
    """[DEPRECATED] Use core.shared.structure.resolve_structure_damage instead.

    Resolve a structure check, including spillover and direct hit outcomes.

    Args:
        structure_damage_marked: Total structure damage marked (including the one just taken).
        inventory: Inventory state used to resolve system trauma selections and fallbacks.
    """
    warnings.warn(
        "resolve_structure_damage is deprecated. Use core.shared.structure.resolve_structure_damage instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return resolve_structure_damage(
        remaining_structure=remaining_structure,
        incoming_damage=incoming_damage,
        hp_before=hp_before,
        structure_damage_marked=structure_damage_marked,
        inventory=inventory,
        rules=rules,
        settings=settings,
    )


def resolve_overheat_deprecated(
    *,
    stress_marked: int,
    remaining_stress: int,
    rules: "OverheatRules" = DEFAULT_OVERHEAT_RULES,
    settings: "ResolutionSettings | None" = None,
) -> OverheatResolution:
    """[DEPRECATED] Use core.shared.heat.resolve_overheat instead.

    Resolve an overheat check, including meltdown subtable outcomes.

    Args:
        stress_marked: Total stress boxes marked (including the one just taken).
        remaining_stress: Stress boxes remaining after marking.
    """
    warnings.warn(
        "resolve_overheat is deprecated. Use core.shared.heat.resolve_overheat instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return resolve_overheat(
        stress_marked=stress_marked,
        remaining_stress=remaining_stress,
        rules=rules,
        settings=settings,
    )
