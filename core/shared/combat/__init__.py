"""Tactical combat primitives for Lancer mech combat."""

from core.shared.combat.statistics import (
    ActionTypeCount,
    CombatStatistics,
    CombatStatisticsTracker,
    CombatantStatistics,
)

# Note: statistics_integration functions are NOT exported here to avoid
# circular imports. Import them directly from:
#   from core.shared.combat.statistics_integration import (
#       initialize_statistics_for_scenario,
#       update_statistics_for_turn_end,
#       update_statistics_from_action,
#   )

from core.shared.combat.tactical_initiative import (
    ActorPriority,
    CombatSide,
    TacticalInitiativeTracker,
    advance_to_next_actor,
    complete_turn,
    get_eligible_nominees,
    get_remaining_actors_on_side,
    get_turn_order_for_display,
    is_valid_nomination,
    nominate_next,
    start_tactical_combat,
)

from core.shared.combat.phased_tracker import (
    PhasedTacticalTracker,
    activate_protocol,
    advance_phase,
    drop_prepared_action,
    end_actor_turn,
    get_eligible_actions,
    get_phase_state,
    get_turn_order_for_display as get_turn_order_for_display_phased,
    nominate_next_phase,
    prepare_action,
    start_actor_turn,
    start_tactical_combat_with_phases,
    use_reaction,
    validate_action_timing,
)

__all__ = [
    "ActionTypeCount",
    "ActorPriority",
    "CombatSide",
    "CombatStatistics",
    "CombatStatisticsTracker",
    "CombatantStatistics",
    # Note: statistics_integration functions not exported to avoid circular imports
    "TacticalInitiativeTracker",
    "PhasedTacticalTracker",
    "advance_to_next_actor",
    "advance_phase",
    "complete_turn",
    "drop_prepared_action",
    "end_actor_turn",
    "get_eligible_actions",
    "get_eligible_nominees",
    "get_phase_state",
    "get_remaining_actors_on_side",
    "get_turn_order_for_display",
    "get_turn_order_for_display_phased",
    "is_valid_nomination",
    "nominate_next",
    "nominate_next_phase",
    "prepare_action",
    "start_actor_turn",
    "start_tactical_combat",
    "start_tactical_combat_with_phases",
    "use_reaction",
    "validate_action_timing",
    "activate_protocol",
]
