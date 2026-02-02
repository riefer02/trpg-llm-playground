"""DTO models for API responses.

DTOs (Data Transfer Objects) provide UI-optimized views of domain models.
They pre-compute common lookups and flatten nested structures to reduce
frontend complexity.
"""

from app.backend.api.dtos.combat_ui import (
    ActionEconomyBrief,
    ActionFeedEntry,
    CombatantBrief,
    CombatUIState,
    CurrentActorState,
    DeployableBrief,
    MovementRangeData,
    ObjectiveBrief,
    PendingDecisionBrief,
)
from app.backend.api.dtos.converters import (
    build_action_feed_entries,
    build_combat_ui_state,
    build_combatant_brief,
    build_current_actor_state,
    build_deployable_brief,
    build_economy_brief,
    build_movement_range,
    build_objective_brief,
    build_pending_decision_brief,
    build_terrain_hash,
    build_turn_order,
)

__all__ = [
    # Models
    "ActionEconomyBrief",
    "ActionFeedEntry",
    "CombatantBrief",
    "CombatUIState",
    "CurrentActorState",
    "DeployableBrief",
    "MovementRangeData",
    "ObjectiveBrief",
    "PendingDecisionBrief",
    # Converters
    "build_action_feed_entries",
    "build_combat_ui_state",
    "build_combatant_brief",
    "build_current_actor_state",
    "build_deployable_brief",
    "build_economy_brief",
    "build_movement_range",
    "build_objective_brief",
    "build_pending_decision_brief",
    "build_terrain_hash",
    "build_turn_order",
]
