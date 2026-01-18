"""Combat turn execution and action resolution.

Provides pure functions for executing combat turns:
- Turn initialization (reset economy, expire prepared actions, decrement cooldowns)
- Action execution (validate economy, resolve effects, apply mutations)
- Turn finalization (end-of-turn effects, advance sequence)

All functions are pure - they take state and return new state + results.
The API layer handles persistence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.shared.enums import ActionType, StatusType
from core.shared.dice import roll_dice

if TYPE_CHECKING:
    from core.shared.structure import StructureResolutionResult
    from core.shared.heat import OverheatResolutionResult

# Import models from combat_models (re-export for backward compatibility)
from core.mech.combat_models import (
    StabilizePrimary,
    StabilizeSecondary,
    ActionExecutionInput,
    ResourceChange,
    ActionExecutionResult,
    TurnStartResult,
    TurnEndResult,
    ReactionInput,
    ReactionResult,
    AvailableAction,
    AvailableActionsResult,
)

# Import helpers from combat_helpers
from core.mech.combat_helpers import (
    _resolve_weapon_profile,
    _extract_tag_value,
    _extract_area_pattern,
    _roll_weapon_damage,
    _build_full_tech_option,
    _apply_tech_result,
    _record_statuses_applied,
    _apply_statuses_to_target,
    _remove_status_from_target,
    _get_basic_available_actions,
    _get_attacker_status_modifiers,
    _get_target_status_modifiers,
    _check_invisibility_miss,
    _get_cover_modifier,
    _resolve_stabilize,
    _resolve_hide,
    _resolve_ram,
    _resolve_grapple,
    _resolve_search,
    _resolve_movement,
    _resolve_mount,
    _resolve_dismount,
    _resolve_eject,
)

from core.mech.combat_state import (
    MechCombatScenario,
    CombatantState,
    CombatResources,
    CombatTurn,
    CombatRound,
    ActionUse,
    OverchargeState,
)
from core.mech.action_economy import (
    ActionEconomyState,
    validate_action_economy,
    use_full_action,
    use_quick_action,
    use_overcharge as mark_overcharge_used,
    use_reaction,
    reset_economy_for_new_turn,
)
from core.mech.combat_resolution import (
    decrement_cooldowns_on_turn_start,
    decrement_cooldowns_on_turn_end,
    reset_per_round_reactions,
    consume_per_round_reaction,
    increment_overcharge_on_turn_start,
)
from core.mech.grid import HexPosition
from core.mech.grid import (
    HexCoord,
    hex_add,
    hex_cone,
    hex_cone_centered,
    hex_line_from_direction,
    hex_scale,
    hexes_in_radius,
    normalize_hex_direction,
)
from core.mech.compendium import get_weapon_definition
from core.mech.combat_rules import AttackPatternDefinition
from core.mech.weapon import WeaponProfile, WeaponTag, resolve_weapon_profile
from core.mech.tech_actions import (
    ScanResult,
    BolsterResult,
    LockOnResult,
    InvadeResult,
    resolve_scan,
    resolve_bolster,
    resolve_lock_on,
    resolve_invade,
)
from core.shared.full_tech import (
    FullTechOptionSelection,
    FullTechInput,
    FullTechFirstOption,
    FullTechSecondOption,
    ScanTechParams,
    BolsterTechParams,
    LockOnTechParams,
    InvadeTechParams,
    resolve_full_tech,
)


# =============================================================================
# Turn Management Functions
# =============================================================================


def get_current_actor(
    scenario: MechCombatScenario,
    current_round: int,
    current_turn_index: int,
) -> CombatantState | None:
    """Get the combatant whose turn it currently is.

    Args:
        scenario: Current combat scenario
        current_round: Current round number (1-indexed)
        current_turn_index: Current turn index within the round (0-indexed)

    Returns:
        CombatantState for current actor, or None if not found
    """
    if not scenario.rounds:
        return None

    round_idx = current_round - 1
    if round_idx < 0 or round_idx >= len(scenario.rounds):
        return None

    current_round_data = scenario.rounds[round_idx]
    if current_turn_index >= len(current_round_data.turns):
        return None

    actor_id = current_round_data.turns[current_turn_index].actor_id

    for combatant in scenario.combatants:
        if combatant.id == actor_id:
            return combatant

    return None


def start_turn(
    scenario: MechCombatScenario,
    actor_id: str,
) -> tuple[MechCombatScenario, TurnStartResult]:
    """Initialize a combatant's turn.

    This function:
    - Resets the action economy for the new turn
    - Expires any prepared actions
    - Decrements cooldowns that trigger on turn start
    - Resets overcharge uses for the turn

    Args:
        scenario: Current combat scenario
        actor_id: ID of the combatant starting their turn

    Returns:
        Tuple of (updated scenario, TurnStartResult)
    """
    # Find the actor
    actor: CombatantState | None = None
    actor_index: int = -1
    for i, c in enumerate(scenario.combatants):
        if c.id == actor_id:
            actor = c
            actor_index = i
            break

    if actor is None:
        # Return unchanged scenario with error result
        return scenario, TurnStartResult(
            actor_id=actor_id,
            actor_name="Unknown",
            economy=ActionEconomyState(),
            available_actions=[],
            prepared_action_expired=False,
        )

    # Initialize fresh action economy
    economy = ActionEconomyState()

    # Check for prepared action expiration
    prepared_action_expired = False
    updated_actor = actor
    if actor.prepared_action is not None:
        prepared_action_expired = True
        updated_actor = actor.model_copy(update={"prepared_action": None})

    # Decrement cooldowns
    cooldowns_decremented: list[str] = []
    if updated_actor.cooldown_states:
        mutable_cooldowns = dict(updated_actor.cooldown_states)
        results = decrement_cooldowns_on_turn_start(actor_cooldown_states=mutable_cooldowns)
        cooldowns_decremented = [r.effect_id for r in results if r.was_decremented]
        updated_actor = updated_actor.model_copy(update={"cooldown_states": mutable_cooldowns})

    # Reset overcharge uses for the turn
    if updated_actor.overcharge_state is not None:
        new_overcharge_state = increment_overcharge_on_turn_start(updated_actor.overcharge_state)
        updated_actor = updated_actor.model_copy(update={"overcharge_state": new_overcharge_state})

    # Build updated combatants list
    updated_combatants = list(scenario.combatants)
    updated_combatants[actor_index] = updated_actor

    # Build updated scenario
    updated_scenario = MechCombatScenario(
        combatants=updated_combatants,
        grapples=list(scenario.grapples),
        rounds=list(scenario.rounds),
        terrain=scenario.terrain,
        environment=scenario.environment,
        deployables=dict(scenario.deployables),
    )

    # Determine available actions
    available_actions = _get_basic_available_actions(updated_actor)

    return updated_scenario, TurnStartResult(
        actor_id=actor_id,
        actor_name=actor.name,
        economy=economy,
        available_actions=available_actions,
        prepared_action_expired=prepared_action_expired,
        cooldowns_decremented=cooldowns_decremented,
    )


def end_turn(
    scenario: MechCombatScenario,
    current_round: int,
    current_turn_index: int,
    current_turn: CombatTurn,
) -> tuple[MechCombatScenario, TurnEndResult, int, int]:
    """Finalize a combatant's turn and determine the next actor.

    This function:
    - Applies end-of-turn effects
    - Decrements cooldowns that trigger on turn end
    - Determines the next actor or advances the round

    Args:
        scenario: Current combat scenario
        current_round: Current round number (1-indexed)
        current_turn_index: Current turn index (0-indexed)
        current_turn: The turn being ended

    Returns:
        Tuple of (updated scenario, TurnEndResult, new_round, new_turn_index)
    """
    actor_id = current_turn.actor_id

    # Find the actor
    actor: CombatantState | None = None
    actor_index: int = -1
    for i, c in enumerate(scenario.combatants):
        if c.id == actor_id:
            actor = c
            actor_index = i
            break

    end_of_turn_effects: list[dict] = []
    cooldowns_decremented: list[str] = []
    updated_actor = actor

    if actor is not None:
        # Decrement cooldowns on turn end
        if actor.cooldown_states:
            mutable_cooldowns = dict(actor.cooldown_states)
            results = decrement_cooldowns_on_turn_end(actor_cooldown_states=mutable_cooldowns)
            cooldowns_decremented = [r.effect_id for r in results if r.was_decremented]
            updated_actor = actor.model_copy(update={"cooldown_states": mutable_cooldowns})

    # Update combatants if actor was modified
    updated_combatants = list(scenario.combatants)
    if updated_actor is not None and actor_index >= 0:
        updated_combatants[actor_index] = updated_actor

    # Get current round data
    round_idx = current_round - 1
    current_round_data = scenario.rounds[round_idx] if round_idx < len(scenario.rounds) else None

    # Determine next actor
    next_turn_index = current_turn_index + 1
    next_round = current_round
    round_advanced = False
    next_actor_id: str | None = None
    next_actor_name: str | None = None

    if current_round_data and next_turn_index < len(current_round_data.turns):
        # More turns in this round
        next_actor_id = current_round_data.turns[next_turn_index].actor_id
        for c in updated_combatants:
            if c.id == next_actor_id:
                next_actor_name = c.name
                break
    else:
        # Round ends, advance to next round
        round_advanced = True
        next_round = current_round + 1
        next_turn_index = 0

        # Reset per-round reactions for all combatants
        reset_combatants: list[CombatantState] = []
        for c in updated_combatants:
            reset_combatants.append(reset_per_round_reactions(c))
        updated_combatants = reset_combatants

        # Check if there's a next round
        if next_round - 1 < len(scenario.rounds):
            next_round_data = scenario.rounds[next_round - 1]
            if next_round_data.turns:
                next_actor_id = next_round_data.turns[0].actor_id
                for c in updated_combatants:
                    if c.id == next_actor_id:
                        next_actor_name = c.name
                        break

    # Build updated scenario
    updated_scenario = MechCombatScenario(
        combatants=updated_combatants,
        grapples=list(scenario.grapples),
        rounds=list(scenario.rounds),
        terrain=scenario.terrain,
        environment=scenario.environment,
        deployables=dict(scenario.deployables),
    )

    result = TurnEndResult(
        actor_id=actor_id,
        next_actor_id=next_actor_id,
        next_actor_name=next_actor_name,
        round_advanced=round_advanced,
        new_round_number=next_round if round_advanced else None,
        end_of_turn_effects=end_of_turn_effects,
        cooldowns_decremented=cooldowns_decremented,
    )

    return updated_scenario, result, next_round, next_turn_index


# =============================================================================
# Action Execution Functions
# =============================================================================


def lookup_weapon_damage_and_ap(weapon_id: str | None) -> tuple[int, int]:
    """Look up weapon damage and AP value from compendium.

    Rolls the weapon's damage dice and extracts the AP tag value.

    Args:
        weapon_id: Weapon ID to look up, or None for default damage

    Returns:
        Tuple of (damage_rolled, armor_piercing)
        Falls back to (6, 0) if weapon not found.
    """
    if weapon_id is None:
        return 6, 0

    weapon_def = get_weapon_definition(weapon_id)
    if weapon_def is None:
        return 6, 0  # Graceful fallback for unknown weapons

    profile = resolve_weapon_profile(weapon_def)

    # Roll damage from all damage components
    total_damage = 0
    for damage_component in profile.damage:
        if damage_component.dice is not None:
            total_damage += roll_dice(damage_component.dice)
        total_damage += damage_component.flat

    if total_damage == 0:
        total_damage = 6  # Fallback for weapons with no damage

    # Extract AP tag value
    armor_piercing = 0
    for tag in profile.tags:
        if tag.tag == "ap":
            armor_piercing = tag.value if tag.value is not None else 1
            break

    return total_damage, armor_piercing


def execute_action(
    scenario: MechCombatScenario,
    current_turn: CombatTurn,
    economy: ActionEconomyState,
    action_input: ActionExecutionInput,
) -> tuple[MechCombatScenario, CombatTurn, ActionEconomyState, ActionExecutionResult]:
    """Execute a combat action.

    This function:
    - Validates the action against the current economy
    - Resolves the action effects
    - Updates the scenario with results
    - Records the action in the turn log

    Args:
        scenario: Current combat scenario
        current_turn: Current turn being played
        economy: Current action economy state
        action_input: The action to execute

    Returns:
        Tuple of (updated scenario, updated turn, updated economy, ActionExecutionResult)
    """
    # Validate action economy
    validation = validate_action_economy(
        economy,
        action_input.action_type,
        is_overcharge=action_input.is_overcharge,
    )

    if not validation.can_take_action:
        return scenario, current_turn, economy, ActionExecutionResult(
            success=False,
            error=validation.errors[0] if validation.errors else "Action not available",
        )

    # Find the actor
    actor: CombatantState | None = None
    for c in scenario.combatants:
        if c.id == action_input.actor_id:
            actor = c
            break

    if actor is None:
        return scenario, current_turn, economy, ActionExecutionResult(
            success=False,
            error=f"Actor {action_input.actor_id} not found",
        )

    full_tech_result = None
    full_tech_targets: list[str] = []
    if action_input.action_id == "full_tech":
        if not action_input.full_tech_first or not action_input.full_tech_second:
            return scenario, current_turn, economy, ActionExecutionResult(
                success=False,
                error="Full Tech requires two tech options",
            )

        first_target = next(
            (c for c in scenario.combatants if c.id == action_input.full_tech_first.target_id),
            None,
        )
        if first_target is None:
            return scenario, current_turn, economy, ActionExecutionResult(
                success=False,
                error=f"Full Tech target {action_input.full_tech_first.target_id} not found",
            )

        second_target = next(
            (c for c in scenario.combatants if c.id == action_input.full_tech_second.target_id),
            None,
        )
        if second_target is None:
            return scenario, current_turn, economy, ActionExecutionResult(
                success=False,
                error=f"Full Tech target {action_input.full_tech_second.target_id} not found",
            )

        full_tech_targets = [first_target.id, second_target.id]
        full_tech_input = FullTechInput(
            actor_id=actor.id,
            first_option=_build_full_tech_option(
                action_input.full_tech_first, actor, first_target, FullTechFirstOption
            ),
            second_option=_build_full_tech_option(
                action_input.full_tech_second, actor, second_target, FullTechSecondOption
            ),
        )
        full_tech_result = resolve_full_tech(full_tech_input)
        if not full_tech_result.is_valid:
            return scenario, current_turn, economy, ActionExecutionResult(
                success=False,
                error="Full Tech options invalid: " + "; ".join(full_tech_result.validation_errors),
            )

    # Update economy based on action type
    updated_economy = economy
    if action_input.action_type == "full":
        updated_economy = use_full_action(economy)
    elif action_input.action_type == "quick":
        updated_economy = use_quick_action(economy)
    elif action_input.is_overcharge:
        updated_economy = mark_overcharge_used(economy)

    # Resolve attack metadata and area targets (if applicable)
    is_attack = action_input.action_id in ("skirmish", "barrage", "fight") or \
                action_input.weapon_id is not None
    attack_target_ids = list(action_input.target_ids)
    area_pattern: AttackPatternDefinition | None = None
    area_origin: HexPosition | None = None
    area_direction: HexCoord | None = None
    area_affected: list[HexCoord] = []
    weapon_profile = _resolve_weapon_profile(action_input.weapon_id)
    weapon_tags: list[WeaponTag] = list(weapon_profile.tags) if weapon_profile else []
    accuracy_bonus = sum(1 for tag in weapon_tags if tag.tag == "accurate")
    difficulty_bonus = sum(1 for tag in weapon_tags if tag.tag == "inaccurate")
    armor_piercing = _extract_tag_value(weapon_tags, "ap") or 0
    reliable_value = _extract_tag_value(weapon_tags, "reliable")
    heat_self = _extract_tag_value(weapon_tags, "heat_self") or 0
    heat_target = _extract_tag_value(weapon_tags, "heat_target") or 0
    burn_value = _extract_tag_value(weapon_tags, "burn")
    has_overkill = any(tag.tag == "overkill" for tag in weapon_tags)
    smart_attack = any(tag.tag == "smart" for tag in weapon_tags)

    if is_attack and weapon_profile is not None:
        area_pattern = _extract_area_pattern(weapon_profile)
        if area_pattern is not None:
            if actor.position is None:
                return scenario, current_turn, economy, ActionExecutionResult(
                    success=False,
                    error="Area attack requires actor position",
                )

            if area_pattern.pattern == "burst":
                area_origin = actor.position
            elif area_pattern.pattern == "blast":
                area_origin = action_input.target_position
                if area_origin is None and action_input.target_ids:
                    target = next(
                        (c for c in scenario.combatants if c.id == action_input.target_ids[0]),
                        None,
                    )
                    if target is not None:
                        area_origin = target.position
            else:
                area_origin = actor.position

            if area_origin is None:
                return scenario, current_turn, economy, ActionExecutionResult(
                    success=False,
                    error="Area attack requires a target position",
                )

            if area_pattern.pattern in ("line", "cone"):
                direction_source = action_input.target_position
                if direction_source is None and action_input.target_ids:
                    target = next(
                        (c for c in scenario.combatants if c.id == action_input.target_ids[0]),
                        None,
                    )
                    if target is not None:
                        direction_source = target.position
                if direction_source is None:
                    return scenario, current_turn, economy, ActionExecutionResult(
                        success=False,
                        error="Line/cone attacks require a target position",
                    )
                area_direction = HexCoord(
                    q=direction_source.coord.q - area_origin.coord.q,
                    r=direction_source.coord.r - area_origin.coord.r,
                )
                if normalize_hex_direction(area_direction) is None:
                    return scenario, current_turn, economy, ActionExecutionResult(
                        success=False,
                        error="Line/cone attacks require a straight-line direction",
                    )

            if area_pattern.pattern == "line" and area_direction is not None:
                area_affected = hex_line_from_direction(
                    area_origin.coord,
                    area_direction,
                    area_pattern.size,
                )
            elif area_pattern.pattern == "cone" and area_direction is not None:
                if area_pattern.cone_mode == "axis":
                    area_affected = hex_cone_centered(
                        area_origin.coord,
                        area_direction,
                        area_pattern.size,
                    )
                else:
                    area_affected = hex_cone(
                        area_origin.coord,
                        area_direction,
                        area_pattern.size,
                    )
            elif area_pattern.pattern in ("blast", "burst"):
                area_affected = hexes_in_radius(
                    area_origin.coord,
                    area_pattern.size,
                )

            area_coords = {(coord.q, coord.r) for coord in area_affected}
            attack_target_ids = [
                combatant.id
                for combatant in scenario.combatants
                if combatant.position
                and (combatant.position.coord.q, combatant.position.coord.r) in area_coords
            ]

    # Create action record
    action_target_ids = list(attack_target_ids)
    if action_input.action_id == "full_tech" and full_tech_targets:
        action_target_ids = full_tech_targets

    action_use = ActionUse(
        action_id=action_input.action_id,
        action_type=action_input.action_type,
        target_ids=action_target_ids,
        target_position=action_input.target_position,
        weapon_tags=[tag.tag for tag in weapon_tags],
        area_pattern=area_pattern,
        area_origin=area_origin,
        area_direction=area_direction,
        area_affected=area_affected,
        granted_by_overcharge=action_input.granted_by_overcharge,
    )

    # Record action in turn
    updated_actions = list(current_turn.actions) + [action_use]
    updated_turn = CombatTurn(
        actor_id=current_turn.actor_id,
        move_used=current_turn.move_used or (action_input.action_id == "move"),
        movement_mode=current_turn.movement_mode,
        movement_path=current_turn.movement_path,
        actions=updated_actions,
    )

    # Initialize effect tracking
    resource_changes: list[ResourceChange] = []
    effects_applied: list[dict] = []
    damage_dealt = 0
    heat_generated = 0
    statuses_applied: dict[str, list[StatusType]] = {}
    structure_checks: list[dict] = []
    overheat_checks: list[dict] = []

    # Check if this is an attack action with targets
    if is_attack and attack_target_ids:
        from core.shared.rolls import resolve_attack

        # Get attack bonus from actor's grit
        attack_bonus = actor.stats.grit if actor.stats else 0
        self_overkill_heat = 0

        # Determine if attack is ranged (check weapon ranges)
        is_ranged_attack = True  # Default to ranged
        if weapon_profile is not None:
            for range_entry in weapon_profile.ranges:
                if range_entry.range_type == "threat":
                    is_ranged_attack = False
                    break

        # Get attacker status modifiers
        attacker_acc_mod, attacker_diff_mod = _get_attacker_status_modifiers(actor)

        # Track lock-on targets for consumption after resolution
        targets_with_lock_on: list[str] = []

        attack_results: list[tuple[str, CombatantState, "AttackResolutionResult", bool]] = []
        for target_id in attack_target_ids:
            target = next((c for c in scenario.combatants if c.id == target_id), None)
            if target is None:
                continue

            target_defense = target.stats.e_defense if smart_attack else target.stats.evasion
            if target.stats is None:
                target_defense = 8 if smart_attack else 10

            # Get target status modifiers
            target_acc_mod, target_diff_mod, has_lock_on = _get_target_status_modifiers(
                target, is_ranged_attack
            )

            # Get cover modifier for ranged attacks
            cover_difficulty = 0
            cover_info = None
            if is_ranged_attack:
                cover_difficulty, cover_info = _get_cover_modifier(
                    scenario, actor, target
                )
                if cover_info is not None:
                    effects_applied.append(cover_info)

            # Combine all accuracy/difficulty modifiers
            final_accuracy_bonus = accuracy_bonus + attacker_acc_mod + target_acc_mod
            final_difficulty_bonus = (
                difficulty_bonus + attacker_diff_mod + target_diff_mod + cover_difficulty
            )

            attack_result = resolve_attack(
                attack_bonus=attack_bonus,
                target_defense=target_defense,
                accuracy_bonus=final_accuracy_bonus,
                difficulty_bonus=final_difficulty_bonus,
            )

            # Check for invisibility miss (50% miss chance)
            invisibility_miss = False
            if attack_result.hit and _check_invisibility_miss(target):
                invisibility_miss = True
                # Create a modified attack result with hit=False
                attack_result = attack_result.model_copy(update={"hit": False})
                effects_applied.append({
                    "type": "invisibility_miss",
                    "target_id": target_id,
                    "reason": "50% miss chance from invisible status",
                })

            attack_results.append((target_id, target, attack_result, has_lock_on))

            effects_applied.append({
                "type": "attack",
                "target_id": target_id,
                "roll": attack_result.roll,
                "total": attack_result.total_accuracy,
                "hit": attack_result.hit,
                "critical": attack_result.is_critical,
                "accuracy_bonus": final_accuracy_bonus,
                "difficulty_bonus": final_difficulty_bonus,
                "status_modifiers": {
                    "attacker_acc": attacker_acc_mod,
                    "attacker_diff": attacker_diff_mod,
                    "target_acc": target_acc_mod,
                    "target_diff": target_diff_mod,
                    "cover_diff": cover_difficulty,
                },
            })

            # Track lock-on for consumption if hit
            if attack_result.hit and has_lock_on:
                targets_with_lock_on.append(target_id)

        single_damage_roll = area_pattern.single_damage_roll if area_pattern else False
        shared_damage = 0
        if single_damage_roll and any(result.hit for _, _, result, _ in attack_results):
            shared_damage, shared_overkill_heat = _roll_weapon_damage(
                weapon_profile,
                apply_overkill=has_overkill,
            )
            self_overkill_heat += shared_overkill_heat

        for target_id, target, attack_result, _ in attack_results:
            if attack_result.hit:
                if single_damage_roll:
                    base_damage = shared_damage
                else:
                    base_damage, overkill_heat = _roll_weapon_damage(
                        weapon_profile,
                        apply_overkill=has_overkill,
                    )
                    self_overkill_heat += overkill_heat

                final_damage = base_damage * 2 if attack_result.is_critical else base_damage

                # Apply exposed damage multiplier (stacks with critical: crit + exposed = 4x)
                if "exposed" in target.statuses:
                    final_damage = final_damage * 2
                    effects_applied.append({
                        "type": "exposed_multiplier",
                        "target_id": target_id,
                        "multiplier": 2,
                    })

                if reliable_value is not None and final_damage < reliable_value:
                    final_damage = reliable_value

                # Shredded ignores armor
                effective_ap = armor_piercing
                if "shredded" in target.statuses:
                    effective_ap = target.stats.armor if target.stats else 0
                    effects_applied.append({
                        "type": "shredded_armor_bypass",
                        "target_id": target_id,
                        "armor_bypassed": effective_ap,
                    })

                scenario, change, structure_result = apply_damage(
                    scenario, target_id, final_damage, effective_ap
                )
                resource_changes.append(change)
                damage_dealt += abs(change.hp_change or 0)

                if structure_result:
                    structure_checks.append({
                        "type": "structure_check",
                        "target_id": target_id,
                        "outcome": structure_result.outcome,
                        "mech_destroyed": structure_result.mech_destroyed,
                        "statuses": [str(s) for s in structure_result.statuses_to_apply],
                        "dice_rolls": structure_result.dice_rolls,
                        "lowest_roll": structure_result.lowest_roll,
                    })

                if burn_value is not None and burn_value > 0:
                    scenario, burn_change, burn_structure = apply_damage(
                        scenario,
                        target_id,
                        burn_value,
                        armor_piercing=target.stats.armor if target.stats else 0,
                    )
                    resource_changes.append(burn_change)
                    damage_dealt += abs(burn_change.hp_change or 0)

                    if burn_structure:
                        structure_checks.append({
                            "type": "structure_check",
                            "target_id": target_id,
                            "outcome": burn_structure.outcome,
                            "mech_destroyed": burn_structure.mech_destroyed,
                            "statuses": [str(s) for s in burn_structure.statuses_to_apply],
                            "dice_rolls": burn_structure.dice_rolls,
                            "lowest_roll": burn_structure.lowest_roll,
                        })

                    scenario, added_statuses = _apply_statuses_to_target(
                        scenario, target_id, ["burn"]
                    )
                    _record_statuses_applied(statuses_applied, target_id, added_statuses)
                    effects_applied.append({
                        "type": "burn",
                        "target_id": target_id,
                        "amount": burn_value,
                    })

                if heat_target > 0:
                    scenario, change, overheat_result = apply_heat(
                        scenario, target_id, heat_target
                    )
                    resource_changes.append(change)
                    heat_generated += heat_target

                    if overheat_result:
                        overheat_checks.append({
                            "type": "overheat_check",
                            "target_id": target_id,
                            "outcome": overheat_result.outcome,
                            "statuses": [str(s) for s in overheat_result.statuses_to_apply],
                            "dice_rolls": overheat_result.dice_rolls,
                            "lowest_roll": overheat_result.lowest_roll,
                            "meltdown_state": overheat_result.meltdown_state is not None,
                        })
            elif reliable_value is not None:
                scenario, change, structure_result = apply_damage(
                    scenario, target_id, reliable_value, armor_piercing
                )
                resource_changes.append(change)
                damage_dealt += abs(change.hp_change or 0)
                effects_applied.append({
                    "type": "reliable_damage",
                    "target_id": target_id,
                    "amount": reliable_value,
                })

                if structure_result:
                    structure_checks.append({
                        "type": "structure_check",
                        "target_id": target_id,
                        "outcome": structure_result.outcome,
                        "mech_destroyed": structure_result.mech_destroyed,
                        "statuses": [str(s) for s in structure_result.statuses_to_apply],
                        "dice_rolls": structure_result.dice_rolls,
                        "lowest_roll": structure_result.lowest_roll,
                    })

        if heat_self > 0 or self_overkill_heat > 0:
            total_self_heat = heat_self + self_overkill_heat
            scenario, change, overheat_result = apply_heat(
                scenario, actor.id, total_self_heat
            )
            resource_changes.append(change)
            heat_generated += total_self_heat
            effects_applied.append({
                "type": "heat_self",
                "target_id": actor.id,
                "amount": total_self_heat,
                "overkill": self_overkill_heat,
            })

            if overheat_result:
                overheat_checks.append({
                    "type": "overheat_check",
                    "target_id": actor.id,
                    "outcome": overheat_result.outcome,
                    "statuses": [str(s) for s in overheat_result.statuses_to_apply],
                    "dice_rolls": overheat_result.dice_rolls,
                    "lowest_roll": overheat_result.lowest_roll,
                    "meltdown_state": overheat_result.meltdown_state is not None,
                })

        # Consume lock-on status from targets that were hit
        for lock_on_target_id in targets_with_lock_on:
            scenario = _remove_status_from_target(scenario, lock_on_target_id, "lock_on")
            effects_applied.append({
                "type": "lock_on_consumed",
                "target_id": lock_on_target_id,
                "reason": "Consumed after successful hit",
            })

    # Handle tech actions (scan, bolster, lock on, invade)
    if action_input.action_id in ("scan", "bolster", "lock_on", "invade") and action_input.target_ids:
        target_id = action_input.target_ids[0]
        target = next((c for c in scenario.combatants if c.id == target_id), None)

        if target is not None:
            tech_result: ScanResult | BolsterResult | LockOnResult | InvadeResult | None = None
            if action_input.action_id == "scan":
                tech_result = resolve_scan(
                    actor_id=actor.id,
                    target_id=target_id,
                    scan_options=["stats", "hidden_info", "public_info"],
                )
            elif action_input.action_id == "bolster":
                tech_result = resolve_bolster(
                    actor_id=actor.id,
                    target_id=target_id,
                    attacker_systems=actor.stats.tech_attack if actor.stats else 0,
                    settings=None,
                )
            elif action_input.action_id == "lock_on":
                tech_result = resolve_lock_on(
                    actor_id=actor.id,
                    target_id=target_id,
                    accuracy_bonus=1,
                )
            elif action_input.action_id == "invade":
                tech_result = resolve_invade(
                    actor_id=actor.id,
                    target_id=target_id,
                    attacker_systems=actor.stats.tech_attack if actor.stats else 0,
                    target_e_defense=target.stats.e_defense if target.stats else 10,
                    settings=None,
                )

            if tech_result is not None:
                scenario, added_heat = _apply_tech_result(
                    scenario,
                    tech_result,
                    effects_applied,
                    resource_changes,
                    statuses_applied,
                    overheat_checks,
                    apply_heat,
                )
                heat_generated += added_heat

    # Handle Full Tech (two tech options in sequence)
    if action_input.action_id == "full_tech" and full_tech_result is not None:
        if full_tech_result.first_result is not None:
            scenario, added_heat = _apply_tech_result(
                scenario,
                full_tech_result.first_result,
                effects_applied,
                resource_changes,
                statuses_applied,
                overheat_checks,
                apply_heat,
            )
            heat_generated += added_heat
        if full_tech_result.second_result is not None:
            scenario, added_heat = _apply_tech_result(
                scenario,
                full_tech_result.second_result,
                effects_applied,
                resource_changes,
                statuses_applied,
                overheat_checks,
                apply_heat,
            )
            heat_generated += added_heat

    # Handle Stabilize action (PR2 4275-4286)
    if action_input.action_id == "stabilize":
        scenario, stab_effects, stab_changes = _resolve_stabilize(
            scenario,
            actor,
            action_input.stabilize_primary,
            action_input.stabilize_secondary,
            action_input.target_ids[0] if action_input.target_ids else None,
        )
        effects_applied.extend(stab_effects)
        resource_changes.extend(stab_changes)

    # Handle Disengage action (PR2 4288-4291)
    if action_input.action_id == "disengage":
        effects_applied.append({
            "type": "disengage",
            "effect": "ignore_engagement_and_reactions",
            "duration": "until_end_of_turn",
        })

    # Handle Hide action (PR2 4221-4237)
    if action_input.action_id == "hide":
        scenario, hide_success, hide_reason = _resolve_hide(scenario, actor)
        if hide_success:
            scenario, added_statuses = _apply_statuses_to_target(
                scenario, actor.id, ["hidden"]
            )
            _record_statuses_applied(statuses_applied, actor.id, added_statuses)
        effects_applied.append({
            "type": "hide",
            "success": hide_success,
            "reason": hide_reason,
        })

    # Handle Ram action (PR2 4152-4155)
    if action_input.action_id == "ram" and action_input.target_ids:
        target_id = action_input.target_ids[0]
        target = next((c for c in scenario.combatants if c.id == target_id), None)
        if target is not None:
            scenario, ram_effects = _resolve_ram(
                scenario,
                actor,
                target,
                apply_knockback=action_input.apply_knockback,
            )
            effects_applied.extend(ram_effects)
            # Apply prone status if ram hit
            if any(e.get("target_becomes_prone") for e in ram_effects):
                scenario, added_statuses = _apply_statuses_to_target(
                    scenario, target_id, ["prone"]
                )
                _record_statuses_applied(statuses_applied, target_id, added_statuses)

    # Handle Grapple action (PR2 4157-4177)
    if action_input.action_id == "grapple" and action_input.target_ids:
        target_id = action_input.target_ids[0]
        target = next((c for c in scenario.combatants if c.id == target_id), None)
        if target is not None:
            scenario, grapple_effects = _resolve_grapple(scenario, actor, target)
            effects_applied.extend(grapple_effects)

    # Handle Search action (PR2 4241-4249)
    if action_input.action_id == "search" and action_input.target_ids:
        target_id = action_input.target_ids[0]
        target = next((c for c in scenario.combatants if c.id == target_id), None)
        if target is not None:
            scenario, search_effects = _resolve_search(scenario, actor, target)
            effects_applied.extend(search_effects)
            # Remove hidden status if search succeeded
            if any(e.get("search_success") for e in search_effects):
                scenario = _remove_status_from_target(scenario, target_id, "hidden")

    # Handle overcharge heat
    if action_input.is_overcharge and actor.overcharge_state is not None:
        from core.mech.combat_resolution import use_overcharge as apply_overcharge
        new_overcharge_state, overcharge_result = apply_overcharge(actor.overcharge_state)
        heat_generated = overcharge_result.rolled_cost or 0

        # Update actor's overcharge state and heat
        actor_idx = next(i for i, c in enumerate(scenario.combatants) if c.id == actor.id)
        new_heat = actor.resources.heat_current + heat_generated
        new_resources = actor.resources.model_copy(update={"heat_current": new_heat})
        updated_actor = actor.model_copy(
            update={
                "overcharge_state": new_overcharge_state,
                "resources": new_resources,
            }
        )

        # Check for overheat cascade if heat meets or exceeds cap
        overheat_result = None
        if new_heat >= updated_actor.resources.heat_cap:
            updated_actor, overheat_result = check_overheat_cascade(updated_actor)
            if overheat_result:
                overheat_checks.append({
                    "type": "overheat_check",
                    "target_id": actor.id,
                    "outcome": overheat_result.outcome,
                    "statuses": [str(s) for s in overheat_result.statuses_to_apply],
                    "dice_rolls": overheat_result.dice_rolls,
                    "lowest_roll": overheat_result.lowest_roll,
                    "meltdown_state": overheat_result.meltdown_state is not None,
                })

        updated_combatants = list(scenario.combatants)
        updated_combatants[actor_idx] = updated_actor

        scenario = MechCombatScenario(
            combatants=updated_combatants,
            grapples=list(scenario.grapples),
            rounds=list(scenario.rounds),
            terrain=scenario.terrain,
            environment=scenario.environment,
            deployables=dict(scenario.deployables),
        )

        # Calculate heat change (may be negative if heat was cleared by overheat)
        final_heat_change = heat_generated
        stress_change = 0
        if overheat_result:
            # Heat was cleared by overheat
            final_heat_change = updated_actor.resources.heat_current - actor.resources.heat_current
            stress_change = -overheat_result.stress_damage

        resource_changes.append(ResourceChange(
            combatant_id=actor.id,
            heat_change=final_heat_change,
            stress_change=stress_change,
        ))
        effects_applied.append({
            "type": "overcharge",
            "heat": heat_generated,
            "new_level": new_overcharge_state.current_level,
        })

    # Handle movement actions (move, boost)
    if action_input.action_id in ("move", "boost") and action_input.movement_path:
        # Re-fetch actor from scenario as it may have been updated
        actor = next((c for c in scenario.combatants if c.id == action_input.actor_id), actor)
        scenario, move_effects = _resolve_movement(
            scenario,
            actor,
            action_input.movement_path,
            is_boost=(action_input.action_id == "boost"),
            apply_damage_func=apply_damage,
        )
        effects_applied.extend(move_effects)

    # Handle mount/dismount/eject actions
    if action_input.action_id == "mount" and action_input.target_ids:
        actor = next((c for c in scenario.combatants if c.id == action_input.actor_id), actor)
        target_mech_id = action_input.target_ids[0]
        scenario, mount_effects = _resolve_mount(scenario, actor, target_mech_id)
        effects_applied.extend(mount_effects)

    if action_input.action_id == "dismount":
        actor = next((c for c in scenario.combatants if c.id == action_input.actor_id), actor)
        scenario, dismount_effects = _resolve_dismount(scenario, actor)
        effects_applied.extend(dismount_effects)

    if action_input.action_id == "eject":
        actor = next((c for c in scenario.combatants if c.id == action_input.actor_id), actor)
        scenario, eject_effects = _resolve_eject(scenario, actor, action_input.eject_direction)
        effects_applied.extend(eject_effects)

    result = ActionExecutionResult(
        success=True,
        action_use=action_use,
        effects_applied=effects_applied,
        damage_dealt=damage_dealt,
        heat_generated=heat_generated,
        resource_changes=resource_changes,
        statuses_applied=statuses_applied,
        structure_checks=structure_checks,
        overheat_checks=overheat_checks,
    )

    return scenario, updated_turn, updated_economy, result


def execute_reaction(
    scenario: MechCombatScenario,
    economy: ActionEconomyState,
    reaction_input: ReactionInput,
) -> tuple[MechCombatScenario, ActionEconomyState, ReactionResult]:
    """Execute a reaction.

    Args:
        scenario: Current combat scenario
        economy: Current action economy (for the reacting combatant)
        reaction_input: The reaction to execute

    Returns:
        Tuple of (updated scenario, updated economy, ReactionResult)
    """
    # Find the reactor
    reactor: CombatantState | None = None
    reactor_idx: int = -1
    for i, c in enumerate(scenario.combatants):
        if c.id == reaction_input.reactor_id:
            reactor = c
            reactor_idx = i
            break

    if reactor is None:
        return scenario, economy, ReactionResult(
            success=False,
            error=f"Reactor {reaction_input.reactor_id} not found",
        )

    # Check if reaction is available this round
    max_per_round = 1  # Standard reactions are 1/round
    reaction_result = consume_per_round_reaction(
        reactor,
        reaction_input.reaction_type,
        max_per_round,
    )

    if not reaction_result.success:
        return scenario, economy, ReactionResult(
            success=False,
            error=reaction_result.message,
        )

    # Use reaction in economy
    updated_economy = use_reaction(economy)

    # Update reactor's per-round reaction tracking
    new_per_round = dict(reactor.per_round_reactions)
    current_count = new_per_round.get(reaction_input.reaction_type, 0)
    new_per_round[reaction_input.reaction_type] = current_count + 1
    updated_reactor = reactor.model_copy(update={"per_round_reactions": new_per_round})

    # Apply reaction effects
    effects_applied: list[dict] = []
    damage_dealt = 0

    if reaction_input.reaction_type == "brace":
        # Brace grants resistance to triggering attack
        effects_applied.append({
            "type": "brace",
            "effect": "resistance_to_triggering_attack",
            "duration": "until_start_of_next_turn",
        })
    elif reaction_input.reaction_type == "overwatch":
        # Overwatch allows a skirmish attack
        effects_applied.append({
            "type": "overwatch",
            "effect": "skirmish_attack",
            "targets": reaction_input.target_ids,
        })
        # Actual damage would be resolved by calling combat_helpers

    # Update scenario
    updated_combatants = list(scenario.combatants)
    updated_combatants[reactor_idx] = updated_reactor

    updated_scenario = MechCombatScenario(
        combatants=updated_combatants,
        grapples=list(scenario.grapples),
        rounds=list(scenario.rounds),
        terrain=scenario.terrain,
        environment=scenario.environment,
        deployables=dict(scenario.deployables),
    )

    return updated_scenario, updated_economy, ReactionResult(
        success=True,
        reaction_used=reaction_input.reaction_type,
        effects_applied=effects_applied,
        damage_dealt=damage_dealt,
    )


# =============================================================================
# Available Actions Query
# =============================================================================


def get_available_actions(
    scenario: MechCombatScenario,
    actor_id: str,
    economy: ActionEconomyState,
) -> AvailableActionsResult:
    """Get available actions for an actor given current economy.

    Args:
        scenario: Current combat scenario
        actor_id: ID of the actor to check
        economy: Current action economy state

    Returns:
        AvailableActionsResult with categorized available actions
    """
    # Find the actor
    actor: CombatantState | None = None
    for c in scenario.combatants:
        if c.id == actor_id:
            actor = c
            break

    if actor is None:
        return AvailableActionsResult(
            actor_id=actor_id,
            economy=economy,
            can_overcharge=False,
        )

    # Check economy for each action type
    can_full = economy.full_actions_remaining > 0
    can_quick = economy.quick_actions_remaining > 0
    can_react = economy.reactions_remaining_this_turn > 0
    can_overcharge = economy.can_overcharge

    # Standard full actions
    full_actions: list[AvailableAction] = []
    if can_full:
        full_actions.extend([
            AvailableAction(
                action_id="barrage",
                action_name="Barrage",
                action_type="full",
                is_available=True,
                requires_target=True,
                requires_weapon=True,
                max_targets=2,  # Barrage can attack up to 2 targets
            ),
            AvailableAction(
                action_id="full_tech",
                action_name="Full Tech",
                action_type="full",
                is_available=True,
            ),
            AvailableAction(
                action_id="improvised_attack",
                action_name="Improvised Attack",
                action_type="full",
                is_available=True,
                requires_target=True,
            ),
            AvailableAction(
                action_id="stabilize",
                action_name="Stabilize",
                action_type="full",
                is_available=True,
            ),
            AvailableAction(
                action_id="disengage",
                action_name="Disengage",
                action_type="full",
                is_available=True,
            ),
        ])
    else:
        full_actions.extend([
            AvailableAction(
                action_id="barrage",
                action_name="Barrage",
                action_type="full",
                is_available=False,
                unavailable_reason="Full action already used",
                requires_target=True,
                requires_weapon=True,
                max_targets=2,  # Barrage can attack up to 2 targets
            ),
        ])

    # Standard quick actions
    quick_actions: list[AvailableAction] = []
    if can_quick:
        quick_actions.extend([
            AvailableAction(
                action_id="skirmish",
                action_name="Skirmish",
                action_type="quick",
                is_available=True,
                requires_target=True,
                requires_weapon=True,
            ),
            AvailableAction(
                action_id="boost",
                action_name="Boost",
                action_type="quick",
                is_available=True,
                requires_path=True,
            ),
            AvailableAction(
                action_id="ram",
                action_name="Ram",
                action_type="quick",
                is_available=True,
                requires_target=True,
            ),
            AvailableAction(
                action_id="grapple",
                action_name="Grapple",
                action_type="quick",
                is_available=True,
                requires_target=True,
            ),
            AvailableAction(
                action_id="scan",
                action_name="Scan",
                action_type="quick",
                is_available=True,
                requires_target=True,
            ),
            AvailableAction(
                action_id="bolster",
                action_name="Bolster",
                action_type="quick",
                is_available=True,
                requires_target=True,
            ),
            AvailableAction(
                action_id="lock_on",
                action_name="Lock On",
                action_type="quick",
                is_available=True,
                requires_target=True,
            ),
            AvailableAction(
                action_id="invade",
                action_name="Invade",
                action_type="quick",
                is_available=True,
                requires_target=True,
            ),
            AvailableAction(
                action_id="hide",
                action_name="Hide",
                action_type="quick",
                is_available=True,
            ),
            AvailableAction(
                action_id="search",
                action_name="Search",
                action_type="quick",
                is_available=True,
            ),
            AvailableAction(
                action_id="activate",
                action_name="Activate",
                action_type="quick",
                is_available=True,
                requires_system=True,
            ),
        ])
    else:
        quick_actions.extend([
            AvailableAction(
                action_id="skirmish",
                action_name="Skirmish",
                action_type="quick",
                is_available=False,
                unavailable_reason="No quick actions remaining",
                requires_target=True,
                requires_weapon=True,
            ),
            AvailableAction(
                action_id="activate",
                action_name="Activate",
                action_type="quick",
                is_available=False,
                unavailable_reason="No quick actions remaining",
                requires_system=True,
            ),
        ])

    # Free actions (always available)
    free_actions: list[AvailableAction] = [
        AvailableAction(
            action_id="move",
            action_name="Move",
            action_type="free",
            is_available=True,
            requires_path=True,
        ),
        AvailableAction(
            action_id="overcharge",
            action_name="Overcharge",
            action_type="free",
            is_available=can_overcharge,
            unavailable_reason=None if can_overcharge else "Already overcharged this turn",
        ),
        AvailableAction(
            action_id="mount_dismount",
            action_name="Mount/Dismount",
            action_type="free",
            is_available=True,
        ),
    ]

    # Reactions
    reactions: list[AvailableAction] = [
        AvailableAction(
            action_id="brace",
            action_name="Brace",
            action_type="reaction",
            is_available=can_react and actor.per_round_reactions.get("brace", 0) < 1,
            unavailable_reason=None if can_react else "No reactions remaining this turn",
        ),
        AvailableAction(
            action_id="overwatch",
            action_name="Overwatch",
            action_type="reaction",
            is_available=can_react and actor.per_round_reactions.get("overwatch", 0) < 1,
            unavailable_reason=None if can_react else "No reactions remaining this turn",
            requires_target=True,
            requires_weapon=True,
        ),
    ]

    # Protocols (at start of turn only - simplified)
    protocols: list[AvailableAction] = []

    return AvailableActionsResult(
        actor_id=actor_id,
        economy=economy,
        full_actions=full_actions,
        quick_actions=quick_actions,
        free_actions=free_actions,
        reactions=reactions,
        protocols=protocols,
        can_overcharge=can_overcharge,
    )


# =============================================================================
# Cascade Resolution Helpers
# =============================================================================


def check_structure_cascade(
    combatant: CombatantState,
    excess_damage: int,
) -> tuple[CombatantState, "StructureResolutionResult | None"]:
    """Check for structure damage when HP reaches 0.

    Per PR2 4592-4637: When HP reaches 0, roll structure check.
    Results can be glancing blow, system trauma, direct hit, or crushing hit.

    Args:
        combatant: The combatant that took damage
        excess_damage: Damage that exceeded remaining HP

    Returns:
        Tuple of (updated combatant, resolution result or None if no check triggered)
    """
    from core.shared.structure import resolve_structure_damage, StructureInput
    from core.mech.combat_rules import DEFAULT_STRUCTURE_DAMAGE_RULES

    # Only trigger if HP is at 0
    if combatant.resources.hp_current > 0:
        return combatant, None

    # Don't trigger if already at 0 structure
    if combatant.resources.structure_current <= 0:
        return combatant, None

    # Resolve structure check
    result = resolve_structure_damage(StructureInput(
        damage_dealt=excess_damage,
        remaining_structure=combatant.resources.structure_current,
        inventory=combatant.inventory,
        rules=DEFAULT_STRUCTURE_DAMAGE_RULES,
    ))

    # Apply result to combatant
    new_structure = combatant.resources.structure_current - 1
    # If not destroyed, reset HP to max; if destroyed, HP stays at 0
    new_hp = combatant.stats.hp_max if new_structure > 0 and not result.mech_destroyed else 0

    new_resources = combatant.resources.model_copy(update={
        "structure_current": max(0, new_structure),
        "hp_current": new_hp,
    })

    # Apply statuses from outcome
    new_statuses = list(combatant.statuses)
    for status in result.statuses_to_apply:
        if status not in new_statuses:
            new_statuses.append(status)

    updated_combatant = combatant.model_copy(update={
        "resources": new_resources,
        "statuses": new_statuses,
        "inventory": result.inventory_update or combatant.inventory,
    })

    return updated_combatant, result


def check_overheat_cascade(
    combatant: CombatantState,
) -> tuple[CombatantState, "OverheatResolutionResult | None"]:
    """Check for stress when heat exceeds capacity.

    Per PR2 4660-4706: When heat exceeds cap, mark stress and roll overheat check.
    Results can be emergency shunt, power plant destabilize, meltdown, or irreversible meltdown.

    Args:
        combatant: The combatant that gained heat

    Returns:
        Tuple of (updated combatant, resolution result or None if no check triggered)
    """
    from core.shared.heat import resolve_overheat, OverheatInput
    from core.mech.combat_rules import DEFAULT_OVERHEAT_RULES

    # Only trigger if heat meets or exceeds cap
    if combatant.resources.heat_current < combatant.resources.heat_cap:
        return combatant, None

    # Don't trigger if already at 0 stress
    if combatant.resources.stress_current <= 0:
        return combatant, None

    # Resolve overheat check
    result = resolve_overheat(OverheatInput(
        stress_marked=combatant.resources.stress_current,
        remaining_stress=combatant.resources.stress_current,
        rules=DEFAULT_OVERHEAT_RULES,
    ))

    # Apply result to combatant
    new_stress = max(0, combatant.resources.stress_current - result.stress_damage)
    # Per PR2: heat always clears on overheat
    new_heat = 0

    new_resources = combatant.resources.model_copy(update={
        "stress_current": new_stress,
        "heat_current": new_heat,
    })

    # Apply statuses from outcome
    new_statuses = list(combatant.statuses)
    for status in result.statuses_to_apply:
        if status not in new_statuses:
            new_statuses.append(status)

    updated_combatant = combatant.model_copy(update={
        "resources": new_resources,
        "statuses": new_statuses,
        "meltdown_state": result.meltdown_state,
    })

    return updated_combatant, result


# =============================================================================
# Resource Mutation Helpers
# =============================================================================


def apply_damage(
    scenario: MechCombatScenario,
    target_id: str,
    damage: int,
    armor_piercing: int = 0,
) -> tuple[MechCombatScenario, ResourceChange, "StructureResolutionResult | None"]:
    """Apply damage to a combatant, triggering structure check if HP reaches 0.

    Args:
        scenario: Current combat scenario
        target_id: ID of the target taking damage
        damage: Amount of damage
        armor_piercing: AP value to bypass armor

    Returns:
        Tuple of (updated scenario, ResourceChange record, structure result or None)
    """
    # Find target
    target: CombatantState | None = None
    target_idx: int = -1
    for i, c in enumerate(scenario.combatants):
        if c.id == target_id:
            target = c
            target_idx = i
            break

    if target is None:
        return scenario, ResourceChange(combatant_id=target_id), None

    # Calculate effective damage
    effective_armor = max(0, target.stats.armor - armor_piercing)
    net_damage = max(0, damage - effective_armor)

    # Calculate excess damage for structure check (damage beyond current HP)
    excess_damage = max(0, net_damage - target.resources.hp_current)

    # Apply to HP
    new_hp = max(0, target.resources.hp_current - net_damage)
    hp_change = target.resources.hp_current - new_hp

    new_resources = target.resources.model_copy(update={"hp_current": new_hp})
    updated_target = target.model_copy(update={"resources": new_resources})

    # Check for structure cascade if HP reached 0
    structure_result: "StructureResolutionResult | None" = None
    if new_hp == 0:
        updated_target, structure_result = check_structure_cascade(
            updated_target, excess_damage
        )

    updated_combatants = list(scenario.combatants)
    updated_combatants[target_idx] = updated_target

    updated_scenario = MechCombatScenario(
        combatants=updated_combatants,
        grapples=list(scenario.grapples),
        rounds=list(scenario.rounds),
        terrain=scenario.terrain,
        environment=scenario.environment,
        deployables=dict(scenario.deployables),
    )

    # Calculate structure change for resource change record
    structure_change = 0
    if structure_result:
        structure_change = -1  # Structure always decrements by 1 on structure check

    return updated_scenario, ResourceChange(
        combatant_id=target_id,
        hp_change=-hp_change,
        structure_change=structure_change,
    ), structure_result


def apply_heat(
    scenario: MechCombatScenario,
    target_id: str,
    heat: int,
) -> tuple[MechCombatScenario, ResourceChange, "OverheatResolutionResult | None"]:
    """Apply heat to a combatant, triggering stress check if heat exceeds cap.

    Args:
        scenario: Current combat scenario
        target_id: ID of the target gaining heat
        heat: Amount of heat

    Returns:
        Tuple of (updated scenario, ResourceChange record, overheat result or None)
    """
    # Find target
    target: CombatantState | None = None
    target_idx: int = -1
    for i, c in enumerate(scenario.combatants):
        if c.id == target_id:
            target = c
            target_idx = i
            break

    if target is None:
        return scenario, ResourceChange(combatant_id=target_id), None

    # Apply heat (can exceed cap - triggers overheat check)
    new_heat = target.resources.heat_current + heat

    new_resources = target.resources.model_copy(update={"heat_current": new_heat})
    updated_target = target.model_copy(update={"resources": new_resources})

    # Check for overheat cascade if heat meets or exceeds cap
    overheat_result: "OverheatResolutionResult | None" = None
    if new_heat >= target.resources.heat_cap:
        updated_target, overheat_result = check_overheat_cascade(updated_target)

    updated_combatants = list(scenario.combatants)
    updated_combatants[target_idx] = updated_target

    updated_scenario = MechCombatScenario(
        combatants=updated_combatants,
        grapples=list(scenario.grapples),
        rounds=list(scenario.rounds),
        terrain=scenario.terrain,
        environment=scenario.environment,
        deployables=dict(scenario.deployables),
    )

    # Calculate stress change for resource change record
    stress_change = 0
    final_heat_change = heat
    if overheat_result:
        stress_change = -overheat_result.stress_damage
        # Heat was cleared by overheat, so actual heat change is negative
        final_heat_change = -target.resources.heat_current  # Cleared all original heat

    return updated_scenario, ResourceChange(
        combatant_id=target_id,
        heat_change=final_heat_change if not overheat_result else -target.resources.heat_current,
        stress_change=stress_change,
    ), overheat_result


def clear_heat(
    scenario: MechCombatScenario,
    target_id: str,
    amount: int,
) -> tuple[MechCombatScenario, ResourceChange]:
    """Clear heat from a combatant.

    Args:
        scenario: Current combat scenario
        target_id: ID of the target clearing heat
        amount: Amount of heat to clear

    Returns:
        Tuple of (updated scenario, ResourceChange record)
    """
    # Find target
    target: CombatantState | None = None
    target_idx: int = -1
    for i, c in enumerate(scenario.combatants):
        if c.id == target_id:
            target = c
            target_idx = i
            break

    if target is None:
        return scenario, ResourceChange(combatant_id=target_id)

    # Clear heat (minimum 0)
    heat_cleared = min(amount, target.resources.heat_current)
    new_heat = target.resources.heat_current - heat_cleared

    new_resources = target.resources.model_copy(update={"heat_current": new_heat})
    updated_target = target.model_copy(update={"resources": new_resources})

    updated_combatants = list(scenario.combatants)
    updated_combatants[target_idx] = updated_target

    updated_scenario = MechCombatScenario(
        combatants=updated_combatants,
        grapples=list(scenario.grapples),
        rounds=list(scenario.rounds),
        terrain=scenario.terrain,
        environment=scenario.environment,
        deployables=dict(scenario.deployables),
    )

    return updated_scenario, ResourceChange(
        combatant_id=target_id,
        heat_change=-heat_cleared,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Models (re-exported from combat_models for backward compatibility)
    "StabilizePrimary",
    "StabilizeSecondary",
    "ActionExecutionInput",
    "ResourceChange",
    "ActionExecutionResult",
    "TurnStartResult",
    "TurnEndResult",
    "ReactionInput",
    "ReactionResult",
    "AvailableAction",
    "AvailableActionsResult",
    # Turn Management
    "get_current_actor",
    "start_turn",
    "end_turn",
    # Action Execution
    "lookup_weapon_damage_and_ap",
    "execute_action",
    "execute_reaction",
    # Available Actions
    "get_available_actions",
    # Cascade Resolution
    "check_structure_cascade",
    "check_overheat_cascade",
    # Resource Mutation
    "apply_damage",
    "apply_heat",
    "clear_heat",
]
