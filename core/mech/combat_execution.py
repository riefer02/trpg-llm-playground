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

from core.shared.enums import ActionType, StatusType, DamageType
from core.shared.dice import roll_dice, round_up

if TYPE_CHECKING:
    from core.shared.structure import StructureResolutionResult
    from core.shared.heat import OverheatResolutionResult

# Import models from combat_models (re-export for backward compatibility)
from core.mech.combat_models import (
    StabilizePrimary,
    StabilizeSecondary,
    ActionExecutionInput,
    ResourceChange,
    OverwatchOpportunityInfo,
    ActionExecutionResult,
    TurnStartResult,
    TurnEndResult,
    BurnTickResult,
    ReactionInput,
    ReactionResult,
    AvailableAction,
    AvailableActionsResult,
)

# Import overwatch trigger detection
from core.shared.overwatch import check_overwatch_triggers_for_movement
from core.shared.damage import DamageInput, DamageResolutionContext, DamageBreakdown, resolve_damage_on_target

# Import helpers from combat_helpers
from core.mech.combat_helpers import (
    _resolve_weapon_profile,
    _extract_tag_value,
    _extract_area_pattern,
    _roll_weapon_damage_components,
    _roll_weapon_damage_components_critical,
    _get_primary_damage_type,
    _build_damage_context,
    _collect_damage_resistances,
    _collect_heat_resistance_multiplier,
    _build_full_tech_option,
    _apply_tech_result,
    _record_statuses_applied,
    _apply_statuses_to_target,
    _remove_status_from_target,
    _get_basic_available_actions,
    _get_attacker_status_modifiers,
    _get_target_status_modifiers,
    _get_talent_accuracy_modifiers,
    _check_invisibility_miss,
    _get_cover_modifier,
    _validate_attack_range_and_los,
    _validate_blast_origin,
    _is_melee_weapon,
    _get_thrown_range,
    _has_weapon_tag,
    _get_weapon_state,
    _validate_weapon_usable,
    _update_weapon_after_attack,
    _get_system_state,
    _validate_system_usable,
    _resolve_stabilize,
    _resolve_hide,
    _resolve_ram,
    _apply_knockback_on_hit,
    _resolve_grapple,
    _resolve_search,
    _resolve_burn_tick,
    _resolve_falling,
    _resolve_movement,
    _resolve_mount,
    _resolve_dismount,
    _resolve_eject,
    _resolve_deploy,
    _check_mine_triggers,
    _clear_statuses_by_trigger,
    _expire_turn_duration_statuses,
    resolve_single_attack,
)

from core.mech.combat_state import (
    MechCombatScenario,
    CombatantState,
    CombatResources,
    CombatTurn,
    CombatRound,
    ActionLogEffect,
    ActionUse,
    OverchargeState,
    GrappleLink,
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
from core.shared.heat import decrement_meltdown_countdown, trigger_meltdown
from core.shared.self_destruct import (
    SelfDestructExplosionInput,
    resolve_self_destruct_explosion,
)
from core.shared.deployables import arm_mines_at_turn_start
from core.shared.terrain import terrain_index, resolve_dangerous_terrain
from core.mech.combat_rules import DEFAULT_MECH_COMBAT_RULES
from core.shared.drone_turn import (
    DroneTurnStartInput,
    DroneTurnEndInput,
    resolve_drone_turn_start,
    resolve_drone_turn_end,
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
    - Processes meltdown countdown (triggers explosion at 0)

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

    # Track meltdown result fields
    meltdown_countdown_active = False
    meltdown_countdown_remaining: int | None = None
    meltdown_triggered = False
    meltdown_explosion_damage = 0
    meltdown_affected_targets: list[str] = []

    # Process meltdown countdown (PR2 4692-4706)
    # Countdown decrements at the start of the mech's turn
    if updated_actor.meltdown_state is not None:
        meltdown_countdown_active = True
        updated_actor, meltdown_triggered = decrement_meltdown_countdown(updated_actor)

        if meltdown_triggered:
            # Meltdown explosion: burst 2, 4d6 explosive damage (same as self-destruct)
            # Only resolve if actor has a position
            if updated_actor.position is not None:
                explosion_input = SelfDestructExplosionInput(
                    mech_id=updated_actor.id,
                    mech_position=updated_actor.position,
                )
                # Build combatants list for explosion (with updated actor state)
                temp_combatants = list(scenario.combatants)
                temp_combatants[actor_index] = updated_actor
                explosion_result = resolve_self_destruct_explosion(
                    explosion_input,
                    all_combatants=temp_combatants,
                )
                meltdown_explosion_damage = explosion_result.total_damage

                # Apply destruction to actor via trigger_meltdown
                destroyed_actor, _wreckage = trigger_meltdown(updated_actor)
                updated_actor = destroyed_actor
                meltdown_countdown_remaining = None

                # Build scenario with destroyed actor for damage application
                updated_combatants = list(scenario.combatants)
                updated_combatants[actor_index] = updated_actor
                temp_scenario = MechCombatScenario(
                    combatants=updated_combatants,
                    grapples=list(scenario.grapples),
                    rounds=list(scenario.rounds),
                    terrain=scenario.terrain,
                    environment=scenario.environment,
                    deployables=dict(scenario.deployables),
                    sitrep_resolution=scenario.sitrep_resolution,
                    pending_decisions=list(scenario.pending_decisions),
                )

                # Apply damage to affected combatants
                for target_result in explosion_result.target_results:
                    if target_result.damage_dealt > 0:
                        meltdown_affected_targets.append(target_result.target_id)
                        temp_scenario, _, _ = apply_damage(
                            temp_scenario,
                            target_result.target_id,
                            target_result.damage_dealt,
                            armor_piercing=0,
                        )

                # Update actor from temp_scenario (in case damage cascaded to them somehow)
                for c in temp_scenario.combatants:
                    if c.id == actor_id:
                        updated_actor = c
                        break

                scenario = temp_scenario
        else:
            # Countdown decremented but not triggered
            if updated_actor.meltdown_state is not None:
                meltdown_countdown_remaining = updated_actor.meltdown_state.turns_remaining

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

    # Check if flying actor should fall due to status (PR2 flight rules - Phase 52)
    # Flying units fall if immobilized, stunned, or shutdown
    started_falling = False
    falling_from_altitude_value: int | None = None

    if (
        updated_actor.flying_status is not None
        and updated_actor.flying_status.is_flying
        and updated_actor.flying_status.altitude_level > 0
    ):
        from core.shared.flying import should_fall_from_flying

        should_fall, _reason = should_fall_from_flying(
            is_flying=True,
            is_immobilized="immobilized" in updated_actor.statuses,
            is_stunned="stunned" in updated_actor.statuses,
            is_shutdown="shutdown" in updated_actor.statuses,
        )

        if should_fall:
            started_falling = True
            falling_from_altitude_value = updated_actor.flying_status.altitude_level

            # Mark actor as falling - will resolve at end of turn
            new_statuses = list(updated_actor.statuses)
            if "falling" not in new_statuses:
                new_statuses.append("falling")

            updated_actor = updated_actor.model_copy(update={
                "statuses": new_statuses,
                "falling_from_altitude": falling_from_altitude_value,
            })

    # Check for dangerous terrain at start of turn (PR2 3859-3860)
    # "A character only needs to make one check a round for dangerous terrain"
    dangerous_terrain_check_required = False
    dangerous_terrain_decision_created = False
    dangerous_terrain_auto_resolved = False
    dangerous_terrain_check_passed: bool | None = None
    dangerous_terrain_damage = 0

    current_round_num = len(scenario.rounds) if scenario.rounds else 1
    terrain_rules = DEFAULT_MECH_COMBAT_RULES.terrain

    if updated_actor.position is not None and scenario.terrain is not None:
        terrain_idx = terrain_index(scenario.terrain)
        terrain_hex = terrain_idx.get(updated_actor.position.coord)

        if terrain_hex and terrain_hex.dangerous:
            # Check if already made a check this round
            check_once_per_round = terrain_rules.dangerous_terrain_check_once_per_round
            already_checked = (
                check_once_per_round
                and updated_actor.dangerous_terrain_last_check_round == current_round_num
            )

            if not already_checked:
                dangerous_terrain_check_required = True
                skill_bonus = updated_actor.stats.engineering_skill if updated_actor.stats else 0

                # Player combatants get a decision prompt (unless AI-controlled)
                if updated_actor.side == "players" and not updated_actor.ai_controlled:
                    from core.shared.decisions import (
                        add_decision_to_scenario,
                        check_dangerous_terrain_decision,
                    )

                    decision = check_dangerous_terrain_decision(
                        combatant=updated_actor,
                        terrain_name="dangerous",
                        check_target=10,
                        current_round=current_round_num,
                    )
                    # We'll add the decision to the scenario after building it
                    dangerous_terrain_decision_created = True
                    # Update last check round to prevent duplicate checks during movement
                    updated_actor = updated_actor.model_copy(
                        update={"dangerous_terrain_last_check_round": current_round_num}
                    )
                else:
                    # Auto-resolve for non-player combatants
                    danger_result = resolve_dangerous_terrain(
                        terrain=scenario.terrain,
                        coord=updated_actor.position.coord,
                        skill_bonus=skill_bonus,
                        damage=terrain_rules.dangerous_terrain_damage,
                        damage_type=terrain_rules.dangerous_terrain_damage_type,
                        check_once_per_round=check_once_per_round,
                        round_checked=current_round_num,
                    )
                    dangerous_terrain_auto_resolved = True
                    dangerous_terrain_check_passed = danger_result.check_passed
                    dangerous_terrain_damage = danger_result.damage_dealt

                    # Update last check round
                    updated_actor = updated_actor.model_copy(
                        update={"dangerous_terrain_last_check_round": current_round_num}
                    )

                    # Apply damage if check failed (will be applied after scenario is built)

    # Build updated combatants list
    updated_combatants = list(scenario.combatants)
    updated_combatants[actor_index] = updated_actor

    # Build intermediate scenario for deployable processing
    intermediate_scenario = MechCombatScenario(
        combatants=updated_combatants,
        grapples=list(scenario.grapples),
        rounds=list(scenario.rounds),
        terrain=scenario.terrain,
        environment=scenario.environment,
        deployables=dict(scenario.deployables),
        sitrep_resolution=scenario.sitrep_resolution,
        pending_decisions=list(scenario.pending_decisions),
    )

    # Arm mines at turn start (PR2 5083-5084)
    # Mines arm at the start of the deployer's next turn
    current_round_num = len(scenario.rounds) if scenario.rounds else 1
    intermediate_scenario, mines_armed = arm_mines_at_turn_start(
        intermediate_scenario, current_round_num
    )

    # Process drone turn start for owner's drones (PR2 5070-5074)
    # Filter to drones owned by this actor
    owner_drones = {
        drone_id: drone
        for drone_id, drone in intermediate_scenario.deployables.items()
        if drone.kind == "drone" and drone.owner_id == actor_id
    }

    drone_heat_to_owner = 0
    drones_ready_to_act: list[str] = []

    if owner_drones:
        drone_start_input = DroneTurnStartInput(
            owner_id=actor_id,
            deployed_drones=owner_drones,
            current_turn=current_round_num,
            latch_drone_active=False,  # Would need to check for active latch mode
            latch_drone_target_id=None,
            tier=1,
        )
        drone_start_result = resolve_drone_turn_start(drone_start_input)
        drone_heat_to_owner = drone_start_result.heat_to_owner
        drones_ready_to_act = drone_start_result.drones_ready_to_act

        # Apply latch drone heat to owner if applicable
        if drone_heat_to_owner > 0:
            new_heat = updated_actor.resources.heat_current + drone_heat_to_owner
            new_resources = updated_actor.resources.model_copy(update={"heat_current": new_heat})
            updated_actor = updated_actor.model_copy(update={"resources": new_resources})

            # Update combatants with new actor state
            updated_combatants = list(intermediate_scenario.combatants)
            updated_combatants[actor_index] = updated_actor
            intermediate_scenario = MechCombatScenario(
                combatants=updated_combatants,
                grapples=list(intermediate_scenario.grapples),
                rounds=list(intermediate_scenario.rounds),
                terrain=intermediate_scenario.terrain,
                environment=intermediate_scenario.environment,
                deployables=dict(intermediate_scenario.deployables),
                sitrep_resolution=intermediate_scenario.sitrep_resolution,
                pending_decisions=list(intermediate_scenario.pending_decisions),
            )

    updated_scenario = intermediate_scenario

    # Add pending decision for dangerous terrain if needed
    if dangerous_terrain_decision_created:
        from core.shared.decisions import (
            add_decision_to_scenario,
            check_dangerous_terrain_decision,
        )

        decision = check_dangerous_terrain_decision(
            combatant=updated_actor,
            terrain_name="dangerous",
            check_target=10,
            current_round=current_round_num,
        )
        updated_scenario = add_decision_to_scenario(updated_scenario, decision)

    # Apply dangerous terrain damage for auto-resolved checks
    if dangerous_terrain_auto_resolved and dangerous_terrain_damage > 0:
        updated_scenario, _change, _structure_result = apply_damage(
            updated_scenario, actor_id, dangerous_terrain_damage, armor_piercing=0
        )
        # Update actor reference from scenario after damage
        for c in updated_scenario.combatants:
            if c.id == actor_id:
                updated_actor = c
                break

    # Determine available actions
    available_actions = _get_basic_available_actions(updated_actor)

    return updated_scenario, TurnStartResult(
        actor_id=actor_id,
        actor_name=actor.name,
        economy=economy,
        available_actions=available_actions,
        prepared_action_expired=prepared_action_expired,
        cooldowns_decremented=cooldowns_decremented,
        meltdown_countdown_active=meltdown_countdown_active,
        meltdown_countdown_remaining=meltdown_countdown_remaining,
        meltdown_triggered=meltdown_triggered,
        meltdown_explosion_damage=meltdown_explosion_damage,
        meltdown_affected_targets=meltdown_affected_targets,
        mines_armed=mines_armed,
        drone_heat_to_owner=drone_heat_to_owner,
        drones_ready_to_act=drones_ready_to_act,
        dangerous_terrain_check_required=dangerous_terrain_check_required,
        dangerous_terrain_decision_created=dangerous_terrain_decision_created,
        dangerous_terrain_auto_resolved=dangerous_terrain_auto_resolved,
        dangerous_terrain_check_passed=dangerous_terrain_check_passed,
        dangerous_terrain_damage=dangerous_terrain_damage,
        started_falling=started_falling,
        falling_from_altitude=falling_from_altitude_value,
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
    burn_tick_result: BurnTickResult | None = None
    fall_resolved = False
    fall_damage = 0
    fell_from_altitude: int | None = None
    updated_actor: CombatantState | None = actor

    if actor is not None:
        updated_actor = actor

        # Resolve burn tick at end of turn (PR2 5017-5021)
        if "burn" in actor.statuses and actor.resources.burn_marked > 0:
            scenario, burn_tick_result = _resolve_burn_tick(scenario, actor)

            # Re-fetch updated actor from scenario
            for i, c in enumerate(scenario.combatants):
                if c.id == actor_id:
                    updated_actor = c
                    actor_index = i
                    break

            if burn_tick_result:
                end_of_turn_effects.append({
                    "type": "burn_tick",
                    "target_id": actor_id,
                    "roll": burn_tick_result.engineering_roll,
                    "bonus": burn_tick_result.engineering_bonus,
                    "total": burn_tick_result.total,
                    "success": burn_tick_result.success,
                    "damage_taken": burn_tick_result.damage_taken,
                    "burn_cleared": burn_tick_result.burn_cleared,
                })

        # Resolve falling at end of turn (PR2 flight rules - Phase 52)
        if "falling" in updated_actor.statuses and updated_actor.falling_from_altitude is not None:
            fell_from_altitude = updated_actor.falling_from_altitude
            scenario, falling_effects = _resolve_falling(scenario, updated_actor)
            end_of_turn_effects.extend(falling_effects)
            fall_resolved = True

            # Extract fall damage from effects
            for effect in falling_effects:
                if effect.get("type") == "falling_damage":
                    fall_damage = effect.get("damage", 0)
                    break

            # Re-fetch updated actor from scenario
            for i, c in enumerate(scenario.combatants):
                if c.id == actor_id:
                    updated_actor = c
                    actor_index = i
                    break

        # Decrement cooldowns on turn end
        if updated_actor.cooldown_states:
            mutable_cooldowns = dict(updated_actor.cooldown_states)
            results = decrement_cooldowns_on_turn_end(actor_cooldown_states=mutable_cooldowns)
            cooldowns_decremented = [r.effect_id for r in results if r.was_decremented]
            updated_actor = updated_actor.model_copy(update={"cooldown_states": mutable_cooldowns})

    # Update combatants if actor was modified
    updated_combatants = list(scenario.combatants)
    if updated_actor is not None and actor_index >= 0:
        updated_combatants[actor_index] = updated_actor

    # Create intermediate scenario for status expiration
    intermediate_scenario = MechCombatScenario(
        combatants=updated_combatants,
        grapples=list(scenario.grapples),
        rounds=list(scenario.rounds),
        terrain=scenario.terrain,
        environment=scenario.environment,
        deployables=dict(scenario.deployables),
        sitrep_resolution=scenario.sitrep_resolution,
        pending_decisions=list(scenario.pending_decisions),
    )

    # Expire duration-based statuses (braced, stunned, etc.)
    intermediate_scenario, expired_effects = _expire_turn_duration_statuses(
        intermediate_scenario, actor_id, current_round
    )
    end_of_turn_effects.extend(expired_effects)

    # Process drone turn end for owner's drones (PR2 5070-5074)
    # Drones prime at end of owner's turn
    owner_drones = {
        drone_id: drone
        for drone_id, drone in intermediate_scenario.deployables.items()
        if drone.kind == "drone" and drone.owner_id == actor_id
    }

    drones_primed: list[str] = []
    if owner_drones:
        drone_end_input = DroneTurnEndInput(
            owner_id=actor_id,
            deployed_drones=owner_drones,
            current_turn=current_round,
            owner_is_stunned="stunned" in (actor.statuses if actor else []),
            latch_drone_active=False,
            latch_drone_target_id=None,
            tier=1,
        )
        drone_end_result = resolve_drone_turn_end(drone_end_input)
        drones_primed = drone_end_result.drones_to_prime

        # Prime drones (set is_armed to True for restock-type drones)
        if drones_primed:
            updated_deployables = dict(intermediate_scenario.deployables)
            for drone_id in drones_primed:
                if drone_id in updated_deployables:
                    drone = updated_deployables[drone_id]
                    updated_deployables[drone_id] = drone.model_copy(update={"is_armed": True})
            intermediate_scenario = MechCombatScenario(
                combatants=list(intermediate_scenario.combatants),
                grapples=list(intermediate_scenario.grapples),
                rounds=list(intermediate_scenario.rounds),
                terrain=intermediate_scenario.terrain,
                environment=intermediate_scenario.environment,
                deployables=updated_deployables,
                sitrep_resolution=intermediate_scenario.sitrep_resolution,
                pending_decisions=list(intermediate_scenario.pending_decisions),
            )

    # Re-fetch combatants after expiration
    updated_combatants = list(intermediate_scenario.combatants)

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
        sitrep_resolution=scenario.sitrep_resolution,
        pending_decisions=list(scenario.pending_decisions),
    )

    result = TurnEndResult(
        actor_id=actor_id,
        next_actor_id=next_actor_id,
        next_actor_name=next_actor_name,
        round_advanced=round_advanced,
        new_round_number=next_round if round_advanced else None,
        end_of_turn_effects=end_of_turn_effects,
        cooldowns_decremented=cooldowns_decremented,
        burn_tick_result=burn_tick_result,
        drones_primed=drones_primed,
        fall_resolved=fall_resolved,
        fall_damage=fall_damage,
        fell_from_altitude=fell_from_altitude,
    )

    return updated_scenario, result, next_round, next_turn_index


# =============================================================================
# Action Execution Functions
# =============================================================================


def lookup_weapon_damage_and_ap(
    weapon_id: str | None,
    profile_id: str | None = None,
) -> tuple[int, int]:
    """Look up weapon damage and AP value from compendium.

    Rolls the weapon's damage dice and extracts the AP tag value.

    Args:
        weapon_id: Weapon ID to look up, or None for default damage
        profile_id: Optional profile ID for weapons with multiple profiles

    Returns:
        Tuple of (damage_rolled, armor_piercing)
        Falls back to (6, 0) if weapon not found.
    """
    if weapon_id is None:
        return 6, 0

    weapon_def = get_weapon_definition(weapon_id)
    if weapon_def is None:
        return 6, 0  # Graceful fallback for unknown weapons

    profile = resolve_weapon_profile(weapon_def, profile_id)

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
    weapon_profile = _resolve_weapon_profile(
        action_input.weapon_id,
        action_input.weapon_profile_id,
    )
    weapon_tags: list[WeaponTag] = list(weapon_profile.tags) if weapon_profile else []
    accuracy_bonus = sum(1 for tag in weapon_tags if tag.tag == "accurate")
    difficulty_bonus = sum(1 for tag in weapon_tags if tag.tag == "inaccurate")
    armor_piercing = _extract_tag_value(weapon_tags, "ap") or 0
    reliable_value = _extract_tag_value(weapon_tags, "reliable")
    heat_self = _extract_tag_value(weapon_tags, "heat_self") or 0
    heat_target = _extract_tag_value(weapon_tags, "heat_target") or 0
    burn_value = _extract_tag_value(weapon_tags, "burn")
    knockback_value = _extract_tag_value(weapon_tags, "knockback") or 0
    has_overkill = any(tag.tag == "overkill" for tag in weapon_tags)
    smart_attack = any(tag.tag == "smart" for tag in weapon_tags)
    primary_damage_type = _get_primary_damage_type(weapon_profile)
    thrown_range = _get_thrown_range(
        action_input.weapon_id,
        action_input.weapon_profile_id,
    ) if action_input.use_thrown else None
    thrown_coord: HexCoord | None = None

    # Validate weapon usability (loading, limited, ordnance restrictions)
    if is_attack and action_input.weapon_id:
        weapon_state = _get_weapon_state(actor, action_input.weapon_id)
        valid, error_msg = _validate_weapon_usable(
            weapon_state=weapon_state,
            weapon_id=action_input.weapon_id,
            actor=actor,
            has_moved_or_acted=current_turn.has_moved_or_acted,
        )
        if not valid:
            return scenario, current_turn, economy, ActionExecutionResult(
                success=False,
                error=f"Cannot attack with {action_input.weapon_id}: {error_msg}",
            )

    # Validate system usability (for activate actions)
    if action_input.system_id:
        system_state = _get_system_state(actor, action_input.system_id)
        valid, error_msg = _validate_system_usable(system_state, action_input.system_id)
        if not valid:
            return scenario, current_turn, economy, ActionExecutionResult(
                success=False,
                error=f"Cannot activate system: {error_msg}",
            )

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
                # Validate blast origin is in range and LOS (PR2 3993-3994)
                if area_origin is not None:
                    valid, error = _validate_blast_origin(
                        scenario=scenario,
                        attacker=actor,
                        blast_origin=area_origin,
                        weapon_id=action_input.weapon_id,
                        profile_id=action_input.weapon_profile_id,
                    )
                    if not valid:
                        return scenario, current_turn, economy, ActionExecutionResult(
                            success=False,
                            error=f"Blast attack failed: {error}",
                        )
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
    # Determine if this action blocks ordnance (sets has_moved_or_acted)
    # Per PR2: Ordnance must fire before any action/movement except protocols
    is_protocol = action_input.action_type == "protocol"
    is_movement = action_input.action_id in ("move", "boost")
    is_action_that_blocks_ordnance = (
        not is_protocol and (
            is_movement or
            action_input.action_type in ("full", "quick")
        )
    )
    new_has_moved_or_acted = (
        current_turn.has_moved_or_acted or is_action_that_blocks_ordnance
    )

    updated_movement_path = current_turn.movement_path
    if action_input.action_id in ("move", "boost") and action_input.movement_path:
        updated_movement_path = action_input.movement_path

    updated_turn = CombatTurn(
        actor_id=current_turn.actor_id,
        move_used=current_turn.move_used or (action_input.action_id == "move"),
        movement_mode=current_turn.movement_mode,
        movement_path=updated_movement_path,
        actions=updated_actions,
        has_moved_or_acted=new_has_moved_or_acted,
    )

    # Initialize effect tracking
    resource_changes: list[ResourceChange] = []
    effects_applied: list[dict] = []
    damage_dealt = 0
    damage_totals = {
        "kinetic": 0,
        "explosive": 0,
        "energy": 0,
        "burn": 0,
        "heat": 0,
    }
    heat_generated = 0
    statuses_applied: dict[str, list[StatusType]] = {}
    structure_checks: list[dict] = []
    overheat_checks: list[dict] = []
    position_updates: dict[str, dict] = {}
    overwatch_opportunities: list[OverwatchOpportunityInfo] = []

    if is_attack and action_input.use_thrown and not attack_target_ids:
        if action_input.weapon_id is None:
            return scenario, current_turn, economy, ActionExecutionResult(
                success=False,
                error="Thrown attack requires a weapon",
            )
        if not _is_melee_weapon(action_input.weapon_id, action_input.weapon_profile_id):
            return scenario, current_turn, economy, ActionExecutionResult(
                success=False,
                error="Only melee weapons can be thrown",
            )
        if thrown_range is None:
            return scenario, current_turn, economy, ActionExecutionResult(
                success=False,
                error="Weapon has no thrown range",
            )
        if actor.position is None or action_input.target_position is None:
            return scenario, current_turn, economy, ActionExecutionResult(
                success=False,
                error="Thrown attack requires a target position",
            )
        distance = actor.position.coord.distance_to(action_input.target_position.coord)
        if distance > thrown_range:
            return scenario, current_turn, economy, ActionExecutionResult(
                success=False,
                error=f"Target out of range ({distance} > {thrown_range} thrown)",
            )
        thrown_coord = action_input.target_position.coord

    # Check if this is an attack action with targets
    if is_attack and attack_target_ids:
        from core.shared.rolls import resolve_attack

        # Validate range and LOS for each target before resolving attacks
        # Skip for area attacks (already validated by area origin)
        if area_pattern is None:
            for target_id in attack_target_ids:
                target = next((c for c in scenario.combatants if c.id == target_id), None)
                if target is None:
                    continue

                valid, error = _validate_attack_range_and_los(
                    scenario=scenario,
                    attacker=actor,
                    target=target,
                    weapon_id=action_input.weapon_id,
                    is_tech_attack=False,
                    use_thrown=action_input.use_thrown,
                    profile_id=action_input.weapon_profile_id,
                )
                if not valid:
                    return scenario, current_turn, economy, ActionExecutionResult(
                        success=False,
                        error=f"Attack on {target_id} failed: {error}",
                    )

        # Get attack bonus from actor's grit
        attack_bonus = actor.stats.grit if actor.stats else 0
        self_overkill_heat = 0

        # Determine if attack is ranged (check weapon ranges)
        is_ranged_attack = True  # Default to ranged
        threat_range = None
        if weapon_profile is not None:
            for range_entry in weapon_profile.ranges:
                if range_entry.range_type == "threat":
                    is_ranged_attack = False
                    threat_range = range_entry.value
                    break
        # Get attacker status modifiers
        attacker_acc_mod, attacker_diff_mod = _get_attacker_status_modifiers(actor)

        # Get talent/frame effect modifiers (Phase 32)
        talent_acc_mod, talent_diff_mod = _get_talent_accuracy_modifiers(
            actor,
            is_melee=not is_ranged_attack,
            is_ranged=is_ranged_attack,
            is_tech=smart_attack,
            context={"is_outgoing": True},
        )

        # Track lock-on targets for consumption after resolution
        targets_with_lock_on: list[str] = []

        attack_results: list[tuple[str, CombatantState, "AttackResolutionResult", bool]] = []
        for target_id in attack_target_ids:
            target = next((c for c in scenario.combatants if c.id == target_id), None)
            if target is None:
                continue

            is_thrown_attack = False
            if (
                action_input.use_thrown
                and thrown_range is not None
                and actor.position is not None
                and target.position is not None
            ):
                is_thrown_attack = True
                if thrown_coord is None:
                    thrown_coord = target.position.coord

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
            if is_ranged_attack or is_thrown_attack:
                cover_difficulty, cover_info = _get_cover_modifier(
                    scenario, actor, target
                )
                if cover_info is not None:
                    effects_applied.append(cover_info)

            # Combine all accuracy/difficulty modifiers (including talents, Phase 32)
            final_accuracy_bonus = accuracy_bonus + attacker_acc_mod + target_acc_mod + talent_acc_mod
            final_difficulty_bonus = (
                difficulty_bonus + attacker_diff_mod + target_diff_mod + cover_difficulty + talent_diff_mod
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
                    "talent_acc": talent_acc_mod,
                    "talent_diff": talent_diff_mod,
                },
            })

            # Track lock-on for consumption if hit
            if attack_result.hit and has_lock_on:
                targets_with_lock_on.append(target_id)

        single_damage_roll = area_pattern.single_damage_roll if area_pattern else False
        shared_damage_components: list[tuple[str, int]] = []
        if single_damage_roll and any(result.hit for _, _, result, _ in attack_results):
            shared_damage_components, shared_overkill_heat = _roll_weapon_damage_components(
                weapon_profile,
                apply_overkill=has_overkill,
            )
            self_overkill_heat += shared_overkill_heat

        for target_id, target, attack_result, _ in attack_results:
            current_target = next(
                (c for c in scenario.combatants if c.id == target_id), target
            )
            damage_context = _build_damage_context(
                attacker=actor,
                target=current_target,
                is_melee=not is_ranged_attack,
                is_ranged=is_ranged_attack,
                is_tech=smart_attack,
            )
            target_resistances = _collect_damage_resistances(
                current_target, damage_context
            )
            heat_multiplier = _collect_heat_resistance_multiplier(
                current_target, damage_context
            )

            if attack_result.hit:
                if single_damage_roll:
                    # For multi-target: use shared roll for non-crits
                    # If this specific target was crit, re-roll with crit mechanics
                    if attack_result.is_critical:
                        base_components, overkill_heat = _roll_weapon_damage_components_critical(
                            weapon_profile,
                            is_critical=True,
                            apply_overkill=has_overkill,
                        )
                        self_overkill_heat += overkill_heat
                    else:
                        base_components = list(shared_damage_components)
                else:
                    # Per PR2 3965-3969: critical = roll dice twice, pick highest
                    base_components, overkill_heat = _roll_weapon_damage_components_critical(
                        weapon_profile,
                        is_critical=attack_result.is_critical,
                        apply_overkill=has_overkill,
                    )
                    self_overkill_heat += overkill_heat

                # No more crit doubling needed - handled by roll function
                scaled_components: list[tuple[str, int]] = list(base_components)

                if "exposed" in current_target.statuses:
                    effects_applied.append({
                        "type": "exposed_multiplier",
                        "target_id": target_id,
                        "multiplier": 2,
                    })

                if reliable_value is not None:
                    total_scaled_damage = sum(
                        amount for dmg_type, amount in scaled_components if dmg_type != "heat"
                    )
                    if total_scaled_damage < reliable_value:
                        scaled_components.append(
                            (primary_damage_type, reliable_value - total_scaled_damage)
                        )

                if burn_value is not None and burn_value > 0:
                    scaled_components.append(("burn", burn_value))

                if heat_target > 0:
                    scaled_components.append(("heat", heat_target))

                if "shredded" in current_target.statuses:
                    effects_applied.append({
                        "type": "shredded_armor_bypass",
                        "target_id": target_id,
                        "armor_bypassed": current_target.stats.armor if current_target.stats else 0,
                    })

                scenario, change, breakdown, structure_result, overheat_result = apply_typed_damage(
                    scenario,
                    target_id,
                    scaled_components,
                    armor_piercing=armor_piercing,
                    attacker_id=actor.id,
                    resistances=target_resistances,
                    heat_resistance_multiplier=heat_multiplier,
                )
                resource_changes.append(change)
                damage_dealt += (
                    breakdown.kinetic
                    + breakdown.explosive
                    + breakdown.energy
                    + breakdown.burn
                )
                heat_generated += breakdown.heat
                damage_totals["kinetic"] += breakdown.kinetic
                damage_totals["explosive"] += breakdown.explosive
                damage_totals["energy"] += breakdown.energy
                damage_totals["burn"] += breakdown.burn
                damage_totals["heat"] += breakdown.heat

                if structure_result:
                    structure_checks.append({
                        "type": "structure_check",
                        "target_id": target_id,
                        "outcome": structure_result.outcome,
                        "direct_hit_outcome": structure_result.direct_hit_outcome,
                        "mech_destroyed": structure_result.mech_destroyed,
                        "statuses": [str(s) for s in structure_result.statuses_to_apply],
                        "dice_rolls": structure_result.dice_rolls,
                        "lowest_roll": structure_result.lowest_roll,
                    })

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

                if breakdown.burn > 0:
                    scenario, added_statuses = _apply_statuses_to_target(
                        scenario, target_id, ["burn"]
                    )
                    _record_statuses_applied(
                        statuses_applied, target_id, added_statuses
                    )

                    burn_target_idx = next(
                        (i for i, c in enumerate(scenario.combatants) if c.id == target_id),
                        -1,
                    )
                    if burn_target_idx >= 0:
                        burn_target = scenario.combatants[burn_target_idx]
                        new_burn = burn_target.resources.burn_marked + breakdown.burn
                        new_burn_resources = burn_target.resources.model_copy(
                            update={"burn_marked": new_burn}
                        )
                        updated_burn_target = burn_target.model_copy(
                            update={"resources": new_burn_resources}
                        )
                        updated_burn_combatants = list(scenario.combatants)
                        updated_burn_combatants[burn_target_idx] = updated_burn_target
                        scenario = MechCombatScenario(
                            combatants=updated_burn_combatants,
                            grapples=list(scenario.grapples),
                            rounds=list(scenario.rounds),
                            terrain=scenario.terrain,
                            environment=scenario.environment,
                            deployables=dict(scenario.deployables),
                            sitrep_resolution=scenario.sitrep_resolution,
                            pending_decisions=list(scenario.pending_decisions),
                        )

                    effects_applied.append({
                        "type": "burn",
                        "target_id": target_id,
                        "amount": breakdown.burn,
                        "total_burn_marked": (
                            scenario.combatants[burn_target_idx].resources.burn_marked
                            if burn_target_idx >= 0 else breakdown.burn
                        ),
                    })

                # Apply knockback if weapon has knockback tag
                if knockback_value > 0:
                    updated_target = next(
                        (c for c in scenario.combatants if c.id == target_id), None
                    )
                    current_actor = next(
                        (c for c in scenario.combatants if c.id == action_input.actor_id), actor
                    )
                    if updated_target and updated_target.resources.structure_current > 0:
                        scenario, knockback_effect = _apply_knockback_on_hit(
                            scenario, current_actor, updated_target, knockback_value
                        )
                        if knockback_effect:
                            effects_applied.append(knockback_effect)
                            position_updates[str(target_id)] = knockback_effect["final_position"]
            elif reliable_value is not None:
                if "shredded" in current_target.statuses:
                    effects_applied.append({
                        "type": "shredded_armor_bypass",
                        "target_id": target_id,
                        "armor_bypassed": current_target.stats.armor if current_target.stats else 0,
                    })

                scenario, change, breakdown, structure_result, overheat_result = apply_typed_damage(
                    scenario,
                    target_id,
                    [(primary_damage_type, reliable_value)],
                    armor_piercing=armor_piercing,
                    attacker_id=actor.id,
                    resistances=target_resistances,
                    heat_resistance_multiplier=heat_multiplier,
                )
                resource_changes.append(change)
                damage_dealt += (
                    breakdown.kinetic
                    + breakdown.explosive
                    + breakdown.energy
                    + breakdown.burn
                )
                damage_totals["kinetic"] += breakdown.kinetic
                damage_totals["explosive"] += breakdown.explosive
                damage_totals["energy"] += breakdown.energy
                damage_totals["burn"] += breakdown.burn
                damage_totals["heat"] += breakdown.heat
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
                        "direct_hit_outcome": structure_result.direct_hit_outcome,
                        "mech_destroyed": structure_result.mech_destroyed,
                        "statuses": [str(s) for s in structure_result.statuses_to_apply],
                        "dice_rolls": structure_result.dice_rolls,
                        "lowest_roll": structure_result.lowest_roll,
                    })

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

        if heat_self > 0 or self_overkill_heat > 0:
            total_self_heat = heat_self + self_overkill_heat
            self_context = _build_damage_context(
                attacker=actor,
                target=actor,
                is_melee=not is_ranged_attack,
                is_ranged=is_ranged_attack,
                is_tech=smart_attack,
            )
            self_heat_multiplier = _collect_heat_resistance_multiplier(
                actor, self_context
            )
            scenario, change, breakdown, _, overheat_result = apply_typed_damage(
                scenario,
                actor.id,
                [("heat", total_self_heat)],
                armor_piercing=0,
                attacker_id=actor.id,
                resistances=[],
                heat_resistance_multiplier=self_heat_multiplier,
            )
            resource_changes.append(change)
            heat_generated += breakdown.heat
            damage_totals["heat"] += breakdown.heat
            effects_applied.append({
                "type": "heat_self",
                "target_id": actor.id,
                "amount": breakdown.heat,
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

        if thrown_coord is not None and action_input.weapon_id:
            effects_applied.append({
                "type": "weapon_thrown",
                "weapon_id": action_input.weapon_id,
                "coord": {"q": thrown_coord.q, "r": thrown_coord.r},
            })

        # Update weapon state after attack (set needs_reload, decrement limited)
        if action_input.weapon_id:
            # Re-fetch actor from scenario (may have been updated)
            actor = next((c for c in scenario.combatants if c.id == action_input.actor_id), actor)
            updated_actor = _update_weapon_after_attack(
                actor,
                action_input.weapon_id,
                thrown_coord=thrown_coord,
            )

            # Apply updated actor to scenario
            actor_idx = next(
                (i for i, c in enumerate(scenario.combatants) if c.id == actor.id), -1
            )
            if actor_idx >= 0:
                updated_combatants = list(scenario.combatants)
                updated_combatants[actor_idx] = updated_actor
                scenario = MechCombatScenario(
                    combatants=updated_combatants,
                    grapples=list(scenario.grapples),
                    rounds=list(scenario.rounds),
                    terrain=scenario.terrain,
                    environment=scenario.environment,
                    deployables=dict(scenario.deployables),
                    sitrep_resolution=scenario.sitrep_resolution,
                    pending_decisions=list(scenario.pending_decisions),
                )

    # Handle tech actions (scan, bolster, lock on, invade)
    if (
        is_attack
        and action_input.use_thrown
        and not attack_target_ids
        and thrown_coord is not None
        and action_input.weapon_id
    ):
        effects_applied.append({
            "type": "weapon_thrown",
            "weapon_id": action_input.weapon_id,
            "coord": {"q": thrown_coord.q, "r": thrown_coord.r},
        })
        actor = next((c for c in scenario.combatants if c.id == action_input.actor_id), actor)
        updated_actor = _update_weapon_after_attack(
            actor,
            action_input.weapon_id,
            thrown_coord=thrown_coord,
        )
        actor_idx = next(
            (i for i, c in enumerate(scenario.combatants) if c.id == actor.id), -1
        )
        if actor_idx >= 0:
            updated_combatants = list(scenario.combatants)
            updated_combatants[actor_idx] = updated_actor
            scenario = MechCombatScenario(
                combatants=updated_combatants,
                grapples=list(scenario.grapples),
                rounds=list(scenario.rounds),
                terrain=scenario.terrain,
                environment=scenario.environment,
                deployables=dict(scenario.deployables),
                sitrep_resolution=scenario.sitrep_resolution,
                pending_decisions=list(scenario.pending_decisions),
            )

    if action_input.action_id in ("scan", "bolster", "lock_on", "invade") and action_input.target_ids:
        target_id = action_input.target_ids[0]
        target = next((c for c in scenario.combatants if c.id == target_id), None)

        if target is not None:
            # Validate range and LOS for tech attack
            valid, error = _validate_attack_range_and_los(
                scenario=scenario,
                attacker=actor,
                target=target,
                weapon_id=None,
                is_tech_attack=True,
            )
            if not valid:
                return scenario, current_turn, economy, ActionExecutionResult(
                    success=False,
                    error=f"Tech attack on {target_id} failed: {error}",
                )

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
                    tech_attack_bonus=actor.stats.tech_attack if actor.stats else 0,
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

    # Handle Activate Core Power action (Phase 33)
    if action_input.action_id == "activate_core_power":
        if not actor.core_power_available:
            return scenario, current_turn, economy, ActionExecutionResult(
                success=False,
                error="Core power already used this mission",
            )
        if actor.core_power_active:
            return scenario, current_turn, economy, ActionExecutionResult(
                success=False,
                error="Core power already active",
            )
        # Activate core power
        actor_idx = next((i for i, c in enumerate(scenario.combatants) if c.id == actor.id), -1)
        if actor_idx >= 0:
            updated_combatants = list(scenario.combatants)
            updated_combatants[actor_idx] = actor.model_copy(update={
                "core_power_available": False,
                "core_power_active": True,
            })
            scenario = scenario.model_copy(update={"combatants": updated_combatants})
            actor = updated_combatants[actor_idx]
        effects_applied.append({
            "type": "activate_core_power",
            "actor_id": actor.id,
            "core_power_effects": actor.core_power_effects.model_dump() if actor.core_power_effects else None,
        })

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

    # Handle Escape Grapple action (PR2 4172-4173: defender can escape via contested HULL)
    if action_input.action_id == "escape_grapple":
        from core.shared.grapple import contest_grapple_check

        # Find the grapple where actor is the defender
        grapple = next(
            (g for g in scenario.grapples if g.target_id == actor.id),
            None,
        )
        if grapple is None:
            return scenario, current_turn, economy, ActionExecutionResult(
                success=False,
                error="Not currently grappled as defender",
            )

        # Find the grappler (opponent)
        grappler = next(
            (c for c in scenario.combatants if c.id == grapple.grappler_id),
            None,
        )
        if grappler is None:
            return scenario, current_turn, economy, ActionExecutionResult(
                success=False,
                error="Grappler not found",
            )

        # Get HULL bonuses (grit represents HULL skill in combat context)
        defender_hull = actor.stats.grit if actor.stats else 0
        attacker_hull = grappler.stats.grit if grappler.stats else 0

        # Contested HULL check
        winner, roll = contest_grapple_check(attacker_hull, defender_hull)

        if winner == "target":  # defender wins
            # Remove grapple from scenario
            updated_grapples = [g for g in scenario.grapples if g.target_id != actor.id]
            scenario = MechCombatScenario(
                combatants=list(scenario.combatants),
                grapples=updated_grapples,
                rounds=list(scenario.rounds),
                terrain=scenario.terrain,
                environment=scenario.environment,
                deployables=dict(scenario.deployables),
                sitrep_resolution=scenario.sitrep_resolution,
                pending_decisions=list(scenario.pending_decisions),
            )
            effects_applied.append({
                "type": "escape_grapple",
                "success": True,
                "roll": roll,
                "winner": "defender",
                "grappler_id": grapple.grappler_id,
            })
        else:
            # Escape failed, grapple remains
            effects_applied.append({
                "type": "escape_grapple",
                "success": False,
                "roll": roll,
                "winner": "attacker" if winner == "attacker" else "tie",
                "grappler_id": grapple.grappler_id,
            })

    # Handle End Grapple action (PR2 4172: attacker can end grapple as free action)
    if action_input.action_id == "end_grapple":
        # Find the grapple where actor is the attacker
        grapple = next(
            (g for g in scenario.grapples if g.grappler_id == actor.id),
            None,
        )
        if grapple is None:
            return scenario, current_turn, economy, ActionExecutionResult(
                success=False,
                error="Not currently grappling as attacker",
            )

        # Remove grapple from scenario
        updated_grapples = [g for g in scenario.grapples if g.grappler_id != actor.id]
        scenario = MechCombatScenario(
            combatants=list(scenario.combatants),
            grapples=updated_grapples,
            rounds=list(scenario.rounds),
            terrain=scenario.terrain,
            environment=scenario.environment,
            deployables=dict(scenario.deployables),
            sitrep_resolution=scenario.sitrep_resolution,
            pending_decisions=list(scenario.pending_decisions),
        )
        effects_applied.append({
            "type": "end_grapple",
            "target_id": grapple.target_id,
        })

    # Handle overcharge heat
    if action_input.is_overcharge and actor.overcharge_state is not None:
        from core.mech.combat_resolution import use_overcharge as apply_overcharge
        new_overcharge_state, overcharge_result = apply_overcharge(actor.overcharge_state)
        raw_heat = overcharge_result.rolled_cost or 0
        heat_context = {
            "is_melee": False,
            "is_ranged": False,
            "is_tech": False,
            "attack_type": None,
            "is_incoming": False,
            "is_outgoing": False,
            "is_engaged": "engaged" in actor.statuses,
            "structure_remaining": actor.resources.structure_current,
            "structure_1_or_less": actor.resources.structure_current <= 1,
        }
        if actor.stats:
            size_map = {
                "size_half": 0.5,
                "size_1": 1,
                "size_2": 2,
                "size_3": 3,
                "size_4": 4,
                "size_5": 5,
            }
            heat_context["actor_size"] = size_map.get(actor.stats.size, 1)

        heat_multiplier = _collect_heat_resistance_multiplier(actor, heat_context)
        if "shredded" in actor.statuses:
            heat_multiplier = 1.0

        heat_generated = round_up(raw_heat * heat_multiplier)
        scenario, change, overheat_result = apply_heat(
            scenario,
            actor.id,
            raw_heat,
            heat_resistance_multiplier=heat_multiplier,
        )

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

        actor_idx = next(i for i, c in enumerate(scenario.combatants) if c.id == actor.id)
        updated_combatants = list(scenario.combatants)
        updated_combatants[actor_idx] = updated_combatants[actor_idx].model_copy(
            update={"overcharge_state": new_overcharge_state}
        )
        scenario = scenario.model_copy(update={"combatants": updated_combatants})

        resource_changes.append(change)
        effects_applied.append({
            "type": "overcharge",
            "heat": heat_generated,
            "new_level": new_overcharge_state.current_level,
        })

    # Handle movement actions (move, boost)
    if action_input.action_id in ("move", "boost") and action_input.movement_path:
        # Re-fetch actor from scenario as it may have been updated
        actor = next((c for c in scenario.combatants if c.id == action_input.actor_id), actor)

        # Check if Disengage was used this turn (affects engagement and reactions)
        disengage_active = any(
            a.action_id == "disengage"
            for a in current_turn.actions
        )

        # Check for overwatch triggers (start/enter/leave threat where permitted)
        if actor.position is not None:
            is_hidden = "hidden" in actor.statuses
            is_invisible = "invisible" in actor.statuses

            overwatch_result = check_overwatch_triggers_for_movement(
                scenario=scenario,
                mover=actor,
                movement_path=action_input.movement_path,
                is_disengaging=disengage_active,
                is_hidden=is_hidden,
                is_invisible=is_invisible,
            )

            # Convert to OverwatchOpportunityInfo for result
            for opp in overwatch_result.opportunities:
                overwatch_opportunities.append(
                    OverwatchOpportunityInfo(
                        reactor_id=str(opp.reactor_id),
                        weapon_id=str(opp.weapon_id),
                        weapon_threat=opp.weapon_threat,
                        can_react=opp.can_react,
                        prevention_reason=opp.prevention_reason,
                    )
                )

        scenario, move_effects = _resolve_movement(
            scenario,
            actor,
            action_input.movement_path,
            is_boost=(action_input.action_id == "boost"),
            apply_damage_func=apply_damage,
            ignore_engagement=disengage_active,
            prompt_dangerous_terrain=action_input.prompt_dangerous_terrain,
        )
        effects_applied.extend(move_effects)

        # Check for mine triggers along movement path (PR2 5085-5086)
        scenario, mine_effects = _check_mine_triggers(
            scenario,
            action_input.actor_id,
            action_input.movement_path,
        )
        effects_applied.extend(mine_effects)

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

    # Handle Deploy action (PR2 5070-5088)
    if action_input.action_id == "deploy" and action_input.target_position and action_input.deploy_kind:
        actor = next((c for c in scenario.combatants if c.id == action_input.actor_id), actor)
        current_round_num = len(scenario.rounds) if scenario.rounds else 1
        scenario, deploy_effects = _resolve_deploy(
            scenario,
            actor,
            action_input.target_position,
            action_input.deploy_kind,
            action_input.deploy_name,
            action_input.system_id,
            action_input.mine_type,
            current_round_num,
        )
        effects_applied.extend(deploy_effects)

    # Clear statuses based on action triggers (PR2 status clear rules)
    statuses_cleared: list[StatusType] = []

    # Attack actions clear hidden status
    if action_input.action_id in ("skirmish", "barrage", "improvised_attack", "fight"):
        scenario, cleared = _clear_statuses_by_trigger(scenario, action_input.actor_id, "attack")
        statuses_cleared.extend(cleared)

    # Boost action clears hidden status
    if action_input.action_id == "boost":
        scenario, cleared = _clear_statuses_by_trigger(scenario, action_input.actor_id, "boost")
        statuses_cleared.extend(cleared)

    # boot_up action clears shutdown status
    if action_input.action_id == "boot_up":
        scenario, cleared = _clear_statuses_by_trigger(scenario, action_input.actor_id, "boot_up")
        statuses_cleared.extend(cleared)

    # stand_up action clears prone status
    if action_input.action_id == "stand_up":
        scenario, cleared = _clear_statuses_by_trigger(scenario, action_input.actor_id, "stand_up")
        statuses_cleared.extend(cleared)

    # Record cleared statuses in effects
    if statuses_cleared:
        effects_applied.append({
            "type": "statuses_cleared",
            "actor_id": action_input.actor_id,
            "statuses": [str(s) for s in statuses_cleared],
            "trigger": action_input.action_id,
        })

    log_effects = _build_action_log_effects(effects_applied, statuses_applied)
    if log_effects:
        action_use = action_use.model_copy(update={"log_effects": log_effects})
        updated_actions = list(updated_turn.actions)
        if updated_actions:
            updated_actions[-1] = action_use
            updated_turn = updated_turn.model_copy(update={"actions": updated_actions})

    result = ActionExecutionResult(
        success=True,
        action_use=action_use,
        effects_applied=effects_applied,
        damage_dealt=damage_dealt,
        damage_breakdown=DamageBreakdown(**damage_totals),
        heat_generated=heat_generated,
        resource_changes=resource_changes,
        statuses_applied=statuses_applied,
        structure_checks=structure_checks,
        overheat_checks=overheat_checks,
        position_updates=position_updates,
        overwatch_opportunities=overwatch_opportunities,
    )

    return scenario, updated_turn, updated_economy, result


def _build_action_log_effects(
    effects_applied: list[dict],
    statuses_applied: dict[str, list[StatusType]],
) -> list[ActionLogEffect]:
    log_effects: list[ActionLogEffect] = []

    for effect in effects_applied:
        effect_type = effect.get("type")
        if effect_type == "weapon_thrown":
            log_effects.append(
                ActionLogEffect(
                    type="weapon_thrown",
                    weapon_id=effect.get("weapon_id"),
                )
            )
        elif effect_type == "retrieve_thrown_weapon":
            log_effects.append(
                ActionLogEffect(
                    type="retrieve_thrown_weapon",
                    weapon_id=effect.get("weapon_id"),
                )
            )

    seen_statuses: set[StatusType] = set()
    for statuses in statuses_applied.values():
        for status in statuses:
            if status in seen_statuses:
                continue
            seen_statuses.add(status)
            log_effects.append(ActionLogEffect(type="status_applied", status=status))

    return log_effects


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

    # Block reactions while grappled (PR2 4170)
    is_reactor_grappled = any(
        g.grappler_id == reaction_input.reactor_id or g.target_id == reaction_input.reactor_id
        for g in scenario.grapples
    )
    if is_reactor_grappled:
        return scenario, economy, ReactionResult(
            success=False,
            error="Cannot take reactions while grappled",
        )

    # Block ordnance weapons from overwatch (PR2 5035-5037)
    if reaction_input.reaction_type == "overwatch" and reaction_input.weapon_id:
        has_ordnance = _has_weapon_tag(
            reaction_input.weapon_id,
            "ordnance",
            reaction_input.weapon_profile_id,
        )
        # Also check weapon state tags
        weapon_state = _get_weapon_state(reactor, reaction_input.weapon_id)
        if weapon_state is not None and "ordnance" in weapon_state.tags:
            has_ordnance = True
        if has_ordnance:
            return scenario, economy, ReactionResult(
                success=False,
                error="Ordnance weapons cannot be used for overwatch",
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
        # Overwatch allows a skirmish attack (PR2 4395-4401)
        # Validate weapon is provided
        weapon_id = reaction_input.weapon_id
        if not weapon_id:
            return scenario, economy, ReactionResult(
                success=False,
                error="Overwatch requires a weapon",
            )

        # Validate target is provided
        if not reaction_input.target_ids:
            return scenario, economy, ReactionResult(
                success=False,
                error="Overwatch requires a target",
            )
        target_id = reaction_input.target_ids[0]  # Overwatch = 1 target per skirmish rules

        # Find target
        target = next((c for c in scenario.combatants if c.id == target_id), None)
        if target is None:
            return scenario, economy, ReactionResult(
                success=False,
                error=f"Target {target_id} not found",
            )

        # Validate weapon usable (loading, limited, destroyed)
        weapon_state = _get_weapon_state(reactor, weapon_id)
        valid, error_msg = _validate_weapon_usable(
            weapon_state=weapon_state,
            weapon_id=weapon_id,
            actor=reactor,
            has_moved_or_acted=False,  # Reactions don't count as having moved/acted
        )
        if not valid:
            return scenario, economy, ReactionResult(
                success=False,
                error=f"Cannot attack with {weapon_id}: {error_msg}",
            )

        # Validate range and LOS (target must be in weapon threat range)
        valid, error_msg = _validate_attack_range_and_los(
            scenario=scenario,
            attacker=reactor,
            target=target,
            weapon_id=weapon_id,
            is_tech_attack=False,
            profile_id=reaction_input.weapon_profile_id,
            force_threat=True,
        )
        if not valid:
            return scenario, economy, ReactionResult(
                success=False,
                error=error_msg,
            )

        # Resolve the attack
        scenario, outcome = resolve_single_attack(
            scenario=scenario,
            attacker=reactor,
            target=target,
            weapon_id=weapon_id,
            profile_id=reaction_input.weapon_profile_id,
        )

        damage_dealt = outcome.damage_dealt

        # Record overwatch attack effect
        effects_applied.append({
            "type": "overwatch_attack",
            "target_id": target_id,
            "hit": outcome.hit,
            "critical": outcome.critical,
            "damage": outcome.damage_dealt,
            "roll": outcome.roll,
            "total": outcome.total,
            "target_defense": outcome.target_defense,
        })
        effects_applied.extend(outcome.effects)

        # Update weapon state after attack (loading, limited charges)
        # Need to re-fetch reactor from scenario after attack resolution
        updated_reactor_after_attack = next(
            (c for c in scenario.combatants if c.id == reaction_input.reactor_id), None
        )
        if updated_reactor_after_attack is not None:
            updated_reactor = _update_weapon_after_attack(
                updated_reactor_after_attack, weapon_id
            )
            # Update per-round reactions on the already-updated reactor
            new_per_round = dict(updated_reactor.per_round_reactions)
            current_count = new_per_round.get(reaction_input.reaction_type, 0)
            new_per_round[reaction_input.reaction_type] = current_count + 1
            updated_reactor = updated_reactor.model_copy(update={"per_round_reactions": new_per_round})

        # Build resource_changes and structure_checks for result
        resource_changes: list[ResourceChange] = []
        structure_checks: list[dict] = []
        if outcome.resource_change is not None:
            resource_changes.append(outcome.resource_change)
        if outcome.structure_check is not None:
            structure_checks.append(outcome.structure_check)

        # Update scenario with updated reactor
        reactor_idx_in_scenario = next(
            (i for i, c in enumerate(scenario.combatants) if c.id == reaction_input.reactor_id), -1
        )
        if reactor_idx_in_scenario >= 0:
            updated_combatants = list(scenario.combatants)
            updated_combatants[reactor_idx_in_scenario] = updated_reactor
            scenario = MechCombatScenario(
                combatants=updated_combatants,
                grapples=list(scenario.grapples),
                rounds=list(scenario.rounds),
                terrain=scenario.terrain,
                environment=scenario.environment,
                deployables=dict(scenario.deployables),
                sitrep_resolution=scenario.sitrep_resolution,
                pending_decisions=list(scenario.pending_decisions),
            )

        # Reactions clear hidden status (PR2 status clear rules)
        scenario, cleared_statuses = _clear_statuses_by_trigger(
            scenario, reaction_input.reactor_id, "reaction"
        )

        # Record cleared statuses in effects
        if cleared_statuses:
            effects_applied.append({
                "type": "statuses_cleared",
                "actor_id": reaction_input.reactor_id,
                "statuses": [str(s) for s in cleared_statuses],
                "trigger": "reaction",
            })

        return scenario, updated_economy, ReactionResult(
            success=True,
            reaction_used=reaction_input.reaction_type,
            effects_applied=effects_applied,
            damage_dealt=damage_dealt,
            damage_breakdown=outcome.damage_breakdown,
            attack_hit=outcome.hit,
            attack_critical=outcome.critical,
            attack_roll=outcome.roll,
            resource_changes=resource_changes,
            structure_checks=structure_checks,
        )

    # Update scenario (for brace reaction)
    updated_combatants = list(scenario.combatants)
    updated_combatants[reactor_idx] = updated_reactor

    updated_scenario = MechCombatScenario(
        combatants=updated_combatants,
        grapples=list(scenario.grapples),
        rounds=list(scenario.rounds),
        terrain=scenario.terrain,
        environment=scenario.environment,
        deployables=dict(scenario.deployables),
        sitrep_resolution=scenario.sitrep_resolution,
        pending_decisions=list(scenario.pending_decisions),
    )

    # Reactions clear hidden status (PR2 status clear rules)
    updated_scenario, cleared_statuses = _clear_statuses_by_trigger(
        updated_scenario, reaction_input.reactor_id, "reaction"
    )

    # Record cleared statuses in effects
    if cleared_statuses:
        effects_applied.append({
            "type": "statuses_cleared",
            "actor_id": reaction_input.reactor_id,
            "statuses": [str(s) for s in cleared_statuses],
            "trigger": "reaction",
        })

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

    # Check grapple status for the actor (PR2 4170: neither party can boost or take reactions)
    is_in_grapple = any(
        g.grappler_id == actor_id or g.target_id == actor_id
        for g in scenario.grapples
    )
    is_grapple_attacker = any(g.grappler_id == actor_id for g in scenario.grapples)
    is_grapple_defender = any(g.target_id == actor_id for g in scenario.grapples)

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
                is_available=not is_in_grapple,
                unavailable_reason="Cannot boost while grappled" if is_in_grapple else None,
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

    # Add escape grapple action for defender (PR2 4172-4173: quick action contested HULL)
    if is_grapple_defender:
        escape_available = can_quick
        escape_reason = None if can_quick else "No quick actions remaining"
        quick_actions.append(AvailableAction(
            action_id="escape_grapple",
            action_name="Escape Grapple",
            action_type="quick",
            is_available=escape_available,
            unavailable_reason=escape_reason,
        ))

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

    # Add end grapple action for attacker (PR2 4172: free action to end grapple)
    if is_grapple_attacker:
        free_actions.append(AvailableAction(
            action_id="end_grapple",
            action_name="End Grapple",
            action_type="free",
            is_available=True,
        ))

    # Reactions (PR2 4170: cannot take reactions while grappled)
    brace_available = can_react and actor.per_round_reactions.get("brace", 0) < 1 and not is_in_grapple
    overwatch_available = can_react and actor.per_round_reactions.get("overwatch", 0) < 1 and not is_in_grapple
    brace_reason = (
        "Cannot react while grappled" if is_in_grapple
        else None if can_react else "No reactions remaining this turn"
    )
    overwatch_reason = (
        "Cannot react while grappled" if is_in_grapple
        else None if can_react else "No reactions remaining this turn"
    )
    reactions: list[AvailableAction] = [
        AvailableAction(
            action_id="brace",
            action_name="Brace",
            action_type="reaction",
            is_available=brace_available,
            unavailable_reason=brace_reason,
        ),
        AvailableAction(
            action_id="overwatch",
            action_name="Overwatch",
            action_type="reaction",
            is_available=overwatch_available,
            unavailable_reason=overwatch_reason,
            requires_target=True,
            requires_weapon=True,
        ),
    ]

    # Protocols (at start of turn only - simplified)
    protocols: list[AvailableAction] = []

    # Add activate_core_power if available (Phase 33)
    if actor.core_power_available and actor.core_power_effects is not None:
        protocols.append(AvailableAction(
            action_id="activate_core_power",
            action_name="Activate Core Power",
            action_type="protocol",
            is_available=True,
        ))

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


def apply_typed_damage(
    scenario: MechCombatScenario,
    target_id: str,
    damage_components: list[tuple[str, int]],
    armor_piercing: int = 0,
    attacker_id: str | None = None,
    resistances: list[DamageType] | None = None,
    heat_resistance_multiplier: float | None = None,
) -> tuple[
    MechCombatScenario,
    ResourceChange,
    DamageBreakdown,
    "StructureResolutionResult | None",
    "OverheatResolutionResult | None",
]:
    """Apply typed damage (including heat) in a single resolution pass."""
    target: CombatantState | None = None
    target_idx: int = -1
    for i, c in enumerate(scenario.combatants):
        if c.id == target_id:
            target = c
            target_idx = i
            break

    if target is None:
        return (
            scenario,
            ResourceChange(combatant_id=target_id),
            DamageBreakdown(),
            None,
            None,
        )

    resistances = resistances or []
    heat_multiplier = heat_resistance_multiplier or 1.0
    if "shredded" in target.statuses:
        heat_multiplier = 1.0

    damage_totals = {
        "kinetic": 0,
        "explosive": 0,
        "energy": 0,
        "burn": 0,
        "heat": 0,
    }

    total_hp_damage = 0
    total_heat = 0
    for damage_type, amount in damage_components:
        if amount <= 0:
            continue
        if damage_type == "heat":
            adjusted_heat = round_up(amount * heat_multiplier)
            total_heat += adjusted_heat
            damage_totals["heat"] += adjusted_heat
            continue

        result = resolve_damage_on_target(
            DamageInput(
                damage=amount,
                damage_type=damage_type,
                armor_piercing=armor_piercing,
            ),
            DamageResolutionContext(
                attacker_id=attacker_id or "attacker",
                target=target,
                resistances=resistances,
            ),
        )
        total_hp_damage += result.damage_to_hp
        damage_totals[result.damage_type] += result.damage_to_hp

    old_hp = target.resources.hp_current
    new_hp = max(0, old_hp - total_hp_damage)
    hp_change = new_hp - old_hp
    excess_damage = max(0, total_hp_damage - old_hp)

    new_resources = target.resources.model_copy(update={"hp_current": new_hp})
    updated_target = target.model_copy(update={"resources": new_resources})

    structure_result: "StructureResolutionResult | None" = None
    if total_hp_damage > 0 and new_hp == 0:
        updated_target, structure_result = check_structure_cascade(
            updated_target, excess_damage
        )

    old_heat = updated_target.resources.heat_current
    old_stress = updated_target.resources.stress_current
    overheat_result: "OverheatResolutionResult | None" = None
    heat_change = 0
    stress_change = 0
    if total_heat > 0:
        new_heat = old_heat + total_heat
        new_resources = updated_target.resources.model_copy(
            update={"heat_current": new_heat}
        )
        updated_target = updated_target.model_copy(update={"resources": new_resources})

        if new_heat >= updated_target.resources.heat_cap:
            updated_target, overheat_result = check_overheat_cascade(updated_target)

        heat_change = updated_target.resources.heat_current - old_heat
        stress_change = updated_target.resources.stress_current - old_stress

    pending_decisions = list(scenario.pending_decisions)
    if structure_result or overheat_result:
        from core.shared.decisions import (
            check_structure_decisions,
            check_overheat_decisions,
        )

        current_round = (
            scenario.rounds[-1].round_index if scenario.rounds else 1
        )
        if structure_result:
            pending_decisions.extend(
                check_structure_decisions(
                    updated_target, structure_result, current_round
                )
            )
        if overheat_result:
            pending_decisions.extend(
                check_overheat_decisions(
                    updated_target, overheat_result, current_round
                )
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
        sitrep_resolution=scenario.sitrep_resolution,
        pending_decisions=pending_decisions,
    )

    structure_change = -1 if structure_result else 0

    return (
        updated_scenario,
        ResourceChange(
            combatant_id=target_id,
            hp_change=hp_change,
            heat_change=heat_change,
            structure_change=structure_change,
            stress_change=stress_change,
        ),
        DamageBreakdown(**damage_totals),
        structure_result,
        overheat_result,
    )


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

    pending_decisions = list(scenario.pending_decisions)
    if structure_result:
        from core.shared.decisions import check_structure_decisions

        current_round = (
            scenario.rounds[-1].round_index if scenario.rounds else 1
        )
        pending_decisions.extend(
            check_structure_decisions(
                updated_target, structure_result, current_round
            )
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
        sitrep_resolution=scenario.sitrep_resolution,
        pending_decisions=pending_decisions,
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
    heat_resistance_multiplier: float | None = None,
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

    heat_multiplier = heat_resistance_multiplier if heat_resistance_multiplier is not None else 1.0
    if "shredded" in target.statuses:
        heat_multiplier = 1.0
    adjusted_heat = round_up(heat * heat_multiplier)

    if adjusted_heat <= 0:
        return scenario, ResourceChange(combatant_id=target_id), None

    # Apply heat (can exceed cap - triggers overheat check)
    new_heat = target.resources.heat_current + adjusted_heat

    new_resources = target.resources.model_copy(update={"heat_current": new_heat})
    updated_target = target.model_copy(update={"resources": new_resources})

    # Check for overheat cascade if heat meets or exceeds cap
    overheat_result: "OverheatResolutionResult | None" = None
    if new_heat >= target.resources.heat_cap:
        updated_target, overheat_result = check_overheat_cascade(updated_target)

    pending_decisions = list(scenario.pending_decisions)
    if overheat_result:
        from core.shared.decisions import check_overheat_decisions

        current_round = (
            scenario.rounds[-1].round_index if scenario.rounds else 1
        )
        pending_decisions.extend(
            check_overheat_decisions(
                updated_target, overheat_result, current_round
            )
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
        sitrep_resolution=scenario.sitrep_resolution,
        pending_decisions=pending_decisions,
    )

    # Calculate stress change for resource change record
    stress_change = 0
    final_heat_change = adjusted_heat
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
        sitrep_resolution=scenario.sitrep_resolution,
        pending_decisions=list(scenario.pending_decisions),
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
