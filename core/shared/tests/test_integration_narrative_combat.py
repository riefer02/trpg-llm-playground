"""Tests for narrative_combat integration module.

Tests explicit bridges between narrative play and mech combat.
"""

import pytest
from core.shared.narrative import (
    NarrativeGoal,
    NarrativeGoalState,
    NarrativeGoalTracker,
    NarrativeGoalCondition,
    NarrativeCombatState,
)
from core.shared.integration.narrative_combat import (
    CombatEvent,
    CombatToNarrativeMapper,
    CombatSetup,
    CombatResult,
    NarrativeCombatBridge,
    get_goal_ids,
    get_active_goals,
    DEFAULT_BRIDGE,
)


@pytest.fixture
def sample_goal() -> NarrativeGoal:
    """Create a sample narrative goal for testing."""
    return NarrativeGoal(
        id="test_goal_1",
        description="Destroy the enemy commander",
        success_conditions=[
            NarrativeGoalCondition(
                id="commander_destroyed",
                condition_type="target_removed",
                description="Enemy commander destroyed",
            )
        ],
    )


@pytest.fixture
def sample_goal_state(sample_goal) -> NarrativeGoalState:
    """Create a sample goal state for testing."""
    return NarrativeGoalState(goal=sample_goal, status="active")


@pytest.fixture
def sample_tracker(sample_goal_state) -> NarrativeGoalTracker:
    """Create a sample goal tracker for testing."""
    return NarrativeGoalTracker(goals=[sample_goal_state])


class TestCombatEvent:
    """Tests for CombatEvent model."""

    def test_create_target_destroyed_event(self):
        """Test creating a target destroyed event."""
        event = CombatEvent(
            event_type="target_destroyed",
            source_id="player_1",
            target_id="npc_commander",
        )
        assert event.event_type == "target_destroyed"
        assert event.source_id == "player_1"
        assert event.target_id == "npc_commander"

    def test_create_damage_dealt_event(self):
        """Test creating a damage dealt event."""
        event = CombatEvent(
            event_type="damage_dealt",
            source_id="npc_1",
            target_id="player_1",
            details={"damage": 10},
        )
        assert event.details["damage"] == 10


class TestNarrativeCombatBridge:
    """Tests for NarrativeCombatBridge class."""

    def test_narrative_to_combat_preserves_tracker(self, sample_tracker):
        """Transition should preserve narrative tracker state."""
        combat_state = NarrativeCombatState()

        preserved, setup = DEFAULT_BRIDGE.narrative_to_combat(
            sample_tracker,
            combat_state,
            participating_npcs=["npc_1"],
            participating_players=["player_1"],
        )

        assert preserved == sample_tracker
        assert setup.narrative_tracker == sample_tracker

    def test_combat_to_narrative_preserves_tracker(self, sample_tracker):
        """Combat result should merge into narrative tracker."""
        result = CombatResult(
            outcome="victory",
            events=[],
            surviving_participants=["player_1"],
            casualties=["npc_1"],
        )

        updated = DEFAULT_BRIDGE.combat_to_narrative(result, sample_tracker)
        assert updated == sample_tracker

    def test_update_goals_from_combat_matching_event(
        self, sample_goal_state, sample_goal
    ):
        """Combat event matching goal should mark goal complete."""
        tracker = NarrativeGoalTracker(goals=[sample_goal_state])

        def check_commander_destroyed(event: CombatEvent, goal: NarrativeGoal) -> bool:
            return (
                event.event_type == "target_destroyed"
                and event.target_id == "npc_commander"
            )

        mapper = CombatToNarrativeMapper(
            goal_id="test_goal_1",
            event_type="target_destroyed",
            criterion_check=check_commander_destroyed,
        )

        combat_event = CombatEvent(
            event_type="target_destroyed",
            source_id="player_1",
            target_id="npc_commander",
        )

        updated = DEFAULT_BRIDGE.update_goals_from_combat(
            tracker, [combat_event], [mapper]
        )

        assert len(updated.goals) == 1
        assert updated.goals[0].status == "completed"

    def test_update_goals_from_combat_no_matching_event(self, sample_goal_state):
        """No matching combat event should not change goal status."""
        tracker = NarrativeGoalTracker(goals=[sample_goal_state])

        def check_commander_destroyed(event: CombatEvent, goal: NarrativeGoal) -> bool:
            return (
                event.event_type == "target_destroyed"
                and event.target_id == "npc_commander"
            )

        mapper = CombatToNarrativeMapper(
            goal_id="test_goal_1",
            event_type="target_destroyed",
            criterion_check=check_commander_destroyed,
        )

        combat_event = CombatEvent(
            event_type="damage_dealt",
            source_id="player_1",
            target_id="npc_grunt",
        )

        updated = DEFAULT_BRIDGE.update_goals_from_combat(
            tracker, [combat_event], [mapper]
        )

        assert updated.goals[0].status == "active"

    def test_update_goals_no_matching_mapping(self, sample_goal_state):
        """Event with no matching mapping should not affect goal."""
        tracker = NarrativeGoalTracker(goals=[sample_goal_state])

        def check_commander_destroyed(event: CombatEvent, goal: NarrativeGoal) -> bool:
            return (
                event.event_type == "target_destroyed"
                and event.target_id == "npc_commander"
            )

        mapper = CombatToNarrativeMapper(
            goal_id="different_goal",
            event_type="target_destroyed",
            criterion_check=check_commander_destroyed,
        )

        combat_event = CombatEvent(
            event_type="target_destroyed",
            source_id="player_1",
            target_id="npc_commander",
        )

        updated = DEFAULT_BRIDGE.update_goals_from_combat(
            tracker, [combat_event], [mapper]
        )

        assert updated.goals[0].status == "active"


class TestGetGoalIds:
    """Tests for get_goal_ids function."""

    def test_get_single_goal_id(self, sample_tracker):
        """Should return ID of single goal."""
        ids = get_goal_ids(sample_tracker)
        assert ids == ["test_goal_1"]

    def test_get_multiple_goal_ids(self, sample_goal):
        """Should return IDs of all goals."""
        goal1 = NarrativeGoalState(goal=sample_goal)
        goal2 = NarrativeGoalState(
            goal=NarrativeGoal(
                id="test_goal_2",
                description="Secure the objective",
                success_conditions=[
                    NarrativeGoalCondition(
                        id="objective_secured",
                        condition_type="other",
                        description="Objective secured",
                    )
                ],
            )
        )
        tracker = NarrativeGoalTracker(goals=[goal1, goal2])

        ids = get_goal_ids(tracker)
        assert ids == ["test_goal_1", "test_goal_2"]


class TestGetActiveGoals:
    """Tests for get_active_goals function."""

    def test_get_active_goals_only(self, sample_goal):
        """Should only return goals with active status."""
        active_goal = NarrativeGoalState(goal=sample_goal, status="active")
        completed_goal = NarrativeGoalState(
            goal=NarrativeGoal(
                id="test_goal_2",
                description="Completed goal",
                success_conditions=[
                    NarrativeGoalCondition(
                        id="done",
                        condition_type="other",
                        description="Task completed",
                    )
                ],
            ),
            status="completed",
        )
        tracker = NarrativeGoalTracker(goals=[active_goal, completed_goal])

        active = get_active_goals(tracker)
        assert len(active) == 1
        assert active[0].status == "active"

    def test_get_active_goals_empty_tracker(self):
        """Empty tracker should return empty list."""
        tracker = NarrativeGoalTracker(goals=[])
        active = get_active_goals(tracker)
        assert active == []


class TestCombatSetup:
    """Tests for CombatSetup model."""

    def test_create_combat_setup(self, sample_tracker):
        """Should create combat setup with all fields."""
        combat_state = NarrativeCombatState()
        setup = CombatSetup(
            narrative_tracker=sample_tracker,
            combat_start_state=combat_state,
            participating_npcs=["npc_1", "npc_2"],
            participating_players=["player_1"],
        )

        assert setup.narrative_tracker == sample_tracker
        assert len(setup.participating_npcs) == 2


class TestCombatResult:
    """Tests for CombatResult model."""

    def test_create_victory_result(self):
        """Should create victory combat result."""
        result = CombatResult(
            outcome="victory",
            events=[],
            surviving_participants=["player_1"],
            casualties=["npc_1"],
            turn_count=5,
        )

        assert result.outcome == "victory"
        assert result.turn_count == 5
