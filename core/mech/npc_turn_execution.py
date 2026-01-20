"""NPC turn execution for automated combat.

This module provides orchestration for NPC turns, using the existing
NPC AI behavior patterns to select actions and the combat execution
system to resolve them.
"""

from __future__ import annotations

from pydantic import Field

from core.shared.models import FrozenModel
from core.shared.enums import ActionType
from core.shared.integration.npc_ai import (
    TargetInfo,
    NPCActionDecision,
    select_npc_action_with_role,
)
from core.npc.models import NPCRole
from core.npc.state import NPCState, NPCCombatStats
from core.mech.combat_state import (
    MechCombatScenario,
    CombatantState,
    CombatTurn,
)
from core.mech.combat_execution import (
    start_turn,
    end_turn,
    execute_action,
    TurnStartResult,
    TurnEndResult,
    ActionExecutionResult,
    ActionExecutionInput,
)


class NPCTurnResult(FrozenModel):
    """Result of executing an automated NPC turn."""

    actor_id: str = Field(..., description="ID of the NPC that took the turn")
    actor_name: str = Field(..., description="Name of the NPC")
    decision: NPCActionDecision | None = Field(
        default=None, description="AI decision (action + target + reasoning)"
    )
    action_result: ActionExecutionResult | None = Field(
        default=None, description="Result of executing the selected action"
    )
    turn_start: TurnStartResult | None = Field(
        default=None, description="Turn start result"
    )
    turn_end: TurnEndResult | None = Field(
        default=None, description="Turn end result"
    )
    actions_taken: int = Field(default=0, description="Number of actions executed")
    skipped: bool = Field(default=False, description="Whether turn was skipped")
    skip_reason: str | None = Field(
        default=None, description="Reason for skipping turn"
    )


def build_target_info_list(
    scenario: MechCombatScenario,
    actor: CombatantState,
) -> list[TargetInfo]:
    """Build a list of potential targets for NPC decision-making.

    Args:
        scenario: Current combat scenario
        actor: The NPC making the decision

    Returns:
        List of TargetInfo for visible enemies within sensor range
    """
    targets: list[TargetInfo] = []
    actor_position = actor.position

    if actor_position is None:
        return targets

    sensor_range = actor.stats.sensor_range

    for combatant in scenario.combatants:
        # Skip self
        if combatant.id == actor.id:
            continue

        # Skip destroyed combatants
        if combatant.resources.hp_current <= 0:
            continue

        # Skip allies (same side)
        if combatant.side == actor.side:
            continue

        # Skip combatants without position
        if combatant.position is None:
            continue

        # Calculate distance
        distance = actor_position.coord.distance_to(combatant.position.coord)

        # Skip if out of sensor range
        if distance > sensor_range:
            continue

        targets.append(
            TargetInfo(
                id=combatant.id,
                distance=distance,
                hp_current=combatant.resources.hp_current,
                hp_max=combatant.stats.hp_max,
                is_objective_holder=False,  # Could be enhanced
                is_ally=False,
            )
        )

    return targets


def get_npc_role(combatant: CombatantState) -> NPCRole:
    """Get the NPC role from a combatant, defaulting to 'striker'.

    Args:
        combatant: The combatant to get the role for

    Returns:
        The NPC role, or 'striker' if not set
    """
    if combatant.npc_role is not None:
        return combatant.npc_role
    return "striker"


def _build_npc_state_for_ai(combatant: CombatantState) -> NPCState:
    """Build an NPCState from a CombatantState for AI decision-making.

    The NPC AI system expects an NPCState, but we have a CombatantState.
    This creates a minimal NPCState with the required fields.

    Args:
        combatant: The combatant to convert

    Returns:
        NPCState suitable for AI decision-making
    """
    return NPCState(
        id=combatant.id,
        name=combatant.name,
        npc_class="grunt",  # Default class, not used for decision-making
        tier="tier_1",  # Default tier, not used for decision-making
        stats=NPCCombatStats(
            size=combatant.stats.size,
            hp_max=combatant.stats.hp_max,
            evasion=combatant.stats.evasion,
            e_defense=combatant.stats.e_defense,
            armor=combatant.stats.armor,
            speed=combatant.stats.speed,
            sensor_range=combatant.stats.sensor_range,
            tech_attack=combatant.stats.tech_attack,
        ),
    )


def _get_first_available_weapon(combatant: CombatantState) -> str | None:
    """Get the first available weapon ID from the combatant's inventory.

    Args:
        combatant: The combatant to check

    Returns:
        Weapon ID if found, None otherwise
    """
    if combatant.inventory is None:
        return None

    for mount in combatant.inventory.mounts:
        if mount.destroyed:
            continue
        for weapon in mount.weapons:
            if weapon.destroyed:
                continue
            if weapon.needs_reload:
                continue
            return weapon.weapon_id

    return None


def execute_npc_turn(
    scenario: MechCombatScenario,
    actor_id: str,
    current_round: int,
    current_turn_index: int,
) -> tuple[MechCombatScenario, NPCTurnResult]:
    """Execute a full NPC turn automatically.

    This function:
    1. Validates the actor is ai_controlled
    2. Calls start_turn()
    3. Builds target list from enemies in sensor range
    4. Calls select_npc_action_with_role() for AI decision
    5. Converts decision to ActionExecutionInput
    6. Calls execute_action() with first available weapon
    7. Calls end_turn()
    8. Returns NPCTurnResult with decision reasoning

    Args:
        scenario: Current combat scenario
        actor_id: ID of the NPC to execute turn for
        current_round: Current round number (1-indexed)
        current_turn_index: Current turn index (0-indexed)

    Returns:
        Tuple of (updated scenario, NPCTurnResult)
    """
    # Find the actor
    actor: CombatantState | None = None
    for combatant in scenario.combatants:
        if combatant.id == actor_id:
            actor = combatant
            break

    if actor is None:
        return scenario, NPCTurnResult(
            actor_id=actor_id,
            actor_name="Unknown",
            skipped=True,
            skip_reason="Actor not found",
        )

    # Validate actor is AI-controlled
    if not actor.ai_controlled:
        return scenario, NPCTurnResult(
            actor_id=actor_id,
            actor_name=actor.name,
            skipped=True,
            skip_reason="Actor is not AI-controlled",
        )

    # Check if actor is destroyed
    if actor.resources.hp_current <= 0:
        return scenario, NPCTurnResult(
            actor_id=actor_id,
            actor_name=actor.name,
            skipped=True,
            skip_reason="Actor is destroyed",
        )

    # Start the turn
    updated_scenario, turn_start_result = start_turn(scenario, actor_id)

    # Refresh actor reference after turn start (state may have changed)
    for combatant in updated_scenario.combatants:
        if combatant.id == actor_id:
            actor = combatant
            break

    # Build target list
    targets = build_target_info_list(updated_scenario, actor)

    # Check if there are valid targets
    if not targets:
        # No targets - skip to end turn
        current_turn = CombatTurn(actor_id=actor_id)
        final_scenario, turn_end_result, _, _ = end_turn(
            updated_scenario,
            current_round,
            current_turn_index,
            current_turn,
        )
        return final_scenario, NPCTurnResult(
            actor_id=actor_id,
            actor_name=actor.name,
            turn_start=turn_start_result,
            turn_end=turn_end_result,
            actions_taken=0,
            skipped=True,
            skip_reason="No valid targets in range",
        )

    # Get NPC role and make decision
    role = get_npc_role(actor)
    npc_state = _build_npc_state_for_ai(actor)

    # Available actions for NPC (simplified)
    available_actions: list[ActionType] = ["full", "quick"]

    decision = select_npc_action_with_role(
        npc=npc_state,
        role=role,
        available_actions=available_actions,
        visible_targets=targets,
    )

    # Build action input
    weapon_id = _get_first_available_weapon(actor)
    target_ids = [decision.target_id] if decision.target_id else []

    # Determine action based on decision
    action_id = "skirmish" if decision.action == "quick" else "barrage"
    action_type = decision.action

    # Create turn for tracking
    current_turn = CombatTurn(actor_id=actor_id)
    economy = turn_start_result.economy

    action_input = ActionExecutionInput(
        actor_id=actor_id,
        action_id=action_id,
        action_type=action_type,
        target_ids=target_ids,
        weapon_id=weapon_id,
    )

    # Execute the action
    action_scenario, updated_turn, updated_economy, action_result = execute_action(
        updated_scenario, current_turn, economy, action_input
    )

    # End the turn
    final_scenario, turn_end_result, new_round, new_turn_idx = end_turn(
        action_scenario,
        current_round,
        current_turn_index,
        updated_turn,
    )

    return final_scenario, NPCTurnResult(
        actor_id=actor_id,
        actor_name=actor.name,
        decision=decision,
        action_result=action_result,
        turn_start=turn_start_result,
        turn_end=turn_end_result,
        actions_taken=1 if action_result.success else 0,
        skipped=False,
    )
