"""Combat statistics integration helpers.

Provides functions to update combat statistics based on action results
and combat state changes.
"""

from typing import Literal
from core.shared.combat.statistics import CombatStatistics, CombatStatisticsTracker
from core.mech.combat_models import ActionExecutionResult
from core.mech.combat_state import MechCombatScenario, CombatantState


def update_statistics_from_action(
    statistics: CombatStatistics,
    scenario: MechCombatScenario,
    actor: CombatantState,
    action_result: ActionExecutionResult,
    action_type: Literal["full", "quick", "free", "reaction", "protocol", "move"],
    is_overcharge: bool = False,
) -> CombatStatistics:
    """Update combat statistics based on an action result.

    Args:
        statistics: Current combat statistics
        scenario: Current combat scenario
        actor: The combatant who took the action
        action_result: Result of the action execution
        action_type: Type of action (full, quick, reaction)
        is_overcharge: Whether this was an overcharge action

    Returns:
        Updated combat statistics
    """
    # Create a tracker from the current statistics
    tracker = CombatStatisticsTracker()
    tracker.rounds_completed = statistics.rounds_completed
    tracker.total_turns = statistics.total_turns
    tracker.total_damage_dealt_by_players = statistics.total_damage_dealt_by_players
    tracker.total_damage_received_by_players = (
        statistics.total_damage_received_by_players
    )
    tracker.total_enemies_destroyed = statistics.total_enemies_destroyed
    tracker.closest_call_hp = (
        statistics.closest_call_hp if statistics.closest_call_hp > 0 else 999999
    )
    tracker.closest_call_combatant = statistics.closest_call_combatant
    tracker.max_overkill = statistics.max_overkill
    tracker.combatant_stats = dict(statistics.combatant_stats)
    tracker.action_totals = statistics.action_totals

    # Ensure actor is initialized in stats
    if actor.id not in tracker.combatant_stats:
        starting_hp = actor.resources.hp_current if actor.resources else 0
        tracker.initialize_combatant(
            combatant_id=actor.id,
            combatant_name=actor.name,
            side=actor.side,
            starting_hp=starting_hp,
        )

    # Track overcharge if applicable
    if is_overcharge:
        tracker.record_action(actor.id, "overcharge")

    # Always track the action type (full/quick/reaction)
    if action_type in ("full", "quick", "reaction"):
        tracker.record_action(actor.id, action_type)  # type: ignore

    # Also track the action category (attack/move/tech) based on result
    if action_result.damage_dealt > 0:
        tracker.record_action(actor.id, "attack")
    elif action_result.position_updates:
        tracker.record_action(actor.id, "move")

    # Track damage dealt
    if action_result.damage_dealt > 0:
        # Get targets and track damage
        from core.mech.combat_models import ResourceChange

        for change in action_result.resource_changes:
            if isinstance(change, ResourceChange):
                target_id = change.combatant_id
                hp_change = change.hp_change

                if target_id == actor.id:
                    continue  # Skip self-damage

                target = next(
                    (c for c in scenario.combatants if c.id == target_id), None
                )
                if target and hp_change < 0:  # Damage (negative HP change)
                    damage = abs(hp_change)
                    # Scenario has BEFORE state, hp_change is negative
                    target_hp_before = (
                        target.resources.hp_current if target.resources else damage
                    )
                    target_hp_after = max(
                        0, target_hp_before + hp_change  # hp_change is negative
                    )
                    target_destroyed = (
                        target_hp_after == 0
                        and target.resources
                        and target.resources.structure_current <= 1
                    )

                    # Ensure target is initialized
                    if target.id not in tracker.combatant_stats:
                        tracker.initialize_combatant(
                            combatant_id=target.id,
                            combatant_name=target.name,
                            side=target.side,
                            starting_hp=target_hp_before,
                        )

                    tracker.record_damage_dealt(
                        dealer_id=actor.id,
                        target_id=target.id,
                        damage=damage,
                        target_hp_before=target_hp_before,
                        target_hp_after=target_hp_after,
                        target_destroyed=target_destroyed,
                    )

    return tracker.to_combat_statistics()


def update_statistics_for_turn_end(
    statistics: CombatStatistics,
    combatant_id: str,
    is_new_round: bool = False,
) -> CombatStatistics:
    """Update combat statistics when a turn ends.

    Args:
        statistics: Current combat statistics
        combatant_id: ID of combatant whose turn ended
        is_new_round: Whether this turn end started a new round

    Returns:
        Updated combat statistics
    """
    tracker = CombatStatisticsTracker()
    tracker.rounds_completed = statistics.rounds_completed
    tracker.total_turns = statistics.total_turns
    tracker.total_damage_dealt_by_players = statistics.total_damage_dealt_by_players
    tracker.total_damage_received_by_players = (
        statistics.total_damage_received_by_players
    )
    tracker.total_enemies_destroyed = statistics.total_enemies_destroyed
    tracker.closest_call_hp = (
        statistics.closest_call_hp if statistics.closest_call_hp > 0 else 999999
    )
    tracker.closest_call_combatant = statistics.closest_call_combatant
    tracker.max_overkill = statistics.max_overkill
    tracker.combatant_stats = dict(statistics.combatant_stats)
    tracker.action_totals = statistics.action_totals

    # Record turn taken
    tracker.record_turn_taken(combatant_id)

    # Record round completion if new round started
    if is_new_round:
        tracker.record_round_completed()

    return tracker.to_combat_statistics()


def initialize_statistics_for_scenario(
    scenario: MechCombatScenario,
) -> CombatStatistics:
    """Initialize combat statistics for a new scenario.

    Args:
        scenario: The combat scenario

    Returns:
        Initialized combat statistics
    """
    tracker = CombatStatisticsTracker()

    # Initialize all combatants
    for combatant in scenario.combatants:
        starting_hp = combatant.resources.hp_current if combatant.resources else 0
        tracker.initialize_combatant(
            combatant_id=combatant.id,
            combatant_name=combatant.name,
            side=combatant.side,
            starting_hp=starting_hp,
        )

    return tracker.to_combat_statistics()
