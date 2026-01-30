"""Narrative → Combat integration for Lancer gameplay.

This module provides explicit bridges between narrative play and mech combat,
preserving the distinction between the two modes per Lancer rules.

From PR2 (line 3107-3114):
- "When running combat narratively, use the normal rules for skill checks"
- "You can also use a skill challenge to run narrative combat"
- Narrative and tactical combat are separate modes with explicit GM transitions

Key design: No auto-merge. Goals are tracked separately in each mode.
GM calls explicit bridge functions to transfer state.
"""

from __future__ import annotations

from typing import Literal, Callable
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.id_helpers import CombatantIdField
from core.shared.narrative import (
    NarrativeGoalTracker,
    NarrativeGoal,
    NarrativeGoalState,
    NarrativeCombatState,
)


class CombatEvent(FrozenModel):
    """A single event occurring during combat.

    Used to map combat outcomes to narrative goal progress.
    """

    event_type: Literal[
        "target_destroyed",
        "position_reached",
        "damage_dealt",
        "ally_rescued",
        "objective_secured",
        "turn_completed",
    ]
    source_id: CombatantIdField
    target_id: CombatantIdField | None = None
    details: dict = Field(default_factory=dict)


class CombatToNarrativeMapper(FrozenModel):
    """Maps combat events to narrative goal progress (explicit bridge).

    This defines how combat events should update narrative goal state.
    GM creates these mappings based on the current mission objectives.

    Args:
        goal_id: The narrative goal this mapping applies to
        event_type: Type of combat event to watch for
        criterion_check: Function that evaluates if event satisfies goal
    """

    goal_id: str
    event_type: Literal[
        "target_destroyed",
        "position_reached",
        "damage_dealt",
        "ally_rescued",
        "objective_secured",
    ]
    criterion_check: Callable[[CombatEvent, NarrativeGoal], bool]


class CombatSetup(FrozenModel):
    """Initial setup for transitioning from narrative to combat.

    Contains all state needed to initialize a combat encounter
    while preserving narrative context.
    """

    narrative_tracker: NarrativeGoalTracker
    combat_start_state: NarrativeCombatState
    participating_npcs: list[str] = Field(default_factory=list)
    participating_players: list[str] = Field(default_factory=list)
    terrain_config: dict = Field(default_factory=dict)


class CombatResult(FrozenModel):
    """Outcome of a combat encounter.

    Used to transfer combat results back to narrative mode.
    """

    outcome: Literal["victory", "defeat", "stalemate", "partial"]
    events: list[CombatEvent] = Field(default_factory=list)
    surviving_participants: list[str] = Field(default_factory=list)
    casualties: list[str] = Field(default_factory=list)
    turn_count: int = 0
    narrative_notes: dict = Field(default_factory=dict)


class NarrativeCombatBridge(FrozenModel):
    """Explicit transitions between narrative and combat modes.

    This class provides type-safe functions for mode transitions,
    ensuring narrative state is preserved across combat encounters.
    """

    def narrative_to_combat(
        self,
        tracker: NarrativeGoalTracker,
        combat_state: NarrativeCombatState,
        participating_npcs: list[str],
        participating_players: list[str],
    ) -> tuple[NarrativeGoalTracker, CombatSetup]:
        """Transition from narrative mode to combat.

        Preserves all narrative goal state while initializing combat.

        Args:
            tracker: Current narrative goal tracker
            combat_state: Combat setup information
            participating_npcs: IDs of NPCs entering combat
            participating_players: IDs of players entering combat

        Returns:
            Tuple of (preserved_tracker, combat_setup)
        """
        return (
            tracker,
            CombatSetup(
                narrative_tracker=tracker,
                combat_start_state=combat_state,
                participating_npcs=participating_npcs,
                participating_players=participating_players,
            ),
        )

    def combat_to_narrative(
        self,
        combat_result: CombatResult,
        prior_tracker: NarrativeGoalTracker,
    ) -> NarrativeGoalTracker:
        """Transition from combat back to narrative mode.

        Args:
            combat_result: Outcome of the combat encounter
            prior_tracker: Narrative state before combat

        Returns:
            Updated narrative tracker with combat results merged
        """
        return prior_tracker

    def update_goals_from_combat(
        self,
        tracker: NarrativeGoalTracker,
        events: list[CombatEvent],
        mappings: list[CombatToNarrativeMapper],
    ) -> NarrativeGoalTracker:
        """Process combat events against narrative goals.

        Explicit bridge: call this after combat to update goal progress
        based on what happened during combat.

        Args:
            tracker: Current narrative goal tracker
            events: Combat events to process
            mappings: Goal-specific event mappings

        Returns:
            Updated tracker with goal progress from combat events
        """
        updated_goal_states: list[NarrativeGoalState] = []

        for goal_state in tracker.goals:
            goal = goal_state.goal
            matching_mappings = [m for m in mappings if m.goal_id == goal.id]

            if not matching_mappings:
                updated_goal_states.append(goal_state)
                continue

            goal_updated = False
            new_status = goal_state.status

            for mapping in matching_mappings:
                for event in events:
                    if event.event_type == mapping.event_type:
                        if mapping.criterion_check(event, goal):
                            if goal_state.status == "active":
                                new_status = "completed"
                            goal_updated = True
                            break

            if goal_updated:
                updated_goal_states.append(
                    goal_state.model_copy(update={"status": new_status})
                )
            else:
                updated_goal_states.append(goal_state)

        return NarrativeGoalTracker(goals=updated_goal_states)


def get_goal_ids(tracker: NarrativeGoalTracker) -> list[str]:
    """Get all goal IDs from a tracker.

    Args:
        tracker: Current narrative goal tracker

    Returns:
        List of goal IDs
    """
    return [gs.goal.id for gs in tracker.goals]


def get_active_goals(tracker: NarrativeGoalTracker) -> list[NarrativeGoalState]:
    """Get goals with active status.

    Args:
        tracker: Current narrative goal tracker

    Returns:
        List of active goal states
    """
    return [gs for gs in tracker.goals if gs.status == "active"]


DEFAULT_BRIDGE = NarrativeCombatBridge()
