"""Tests for narrative combat loop primitives."""

import pytest
from core.shared.combat_loop import (
    CombatLoopState,
    InitiativeTracker,
    SceneTimer,
    SceneMetrics,
    ScenePhase,
    SceneScenarioType,
    TimePressureLevel,
    VictoryCondition,
    MissionCompletionResult,
    VictoryCheckResult,
    advance_phase,
    transfer_initiative,
    start_scene,
    end_scene,
    check_victory_conditions,
    resolve_mission_completion,
    start_narrative_scenario,
    advance_narrative_turn,
    check_narrative_victory,
)
from core.shared.narrative import (
    NarrativeCombatState,
    NarrativeGoal,
    NarrativeGoalState,
    NarrativeGoalTracker,
)


class TestCombatLoopState:
    """Tests for CombatLoopState model."""

    def test_default_values(self):
        state = CombatLoopState()
        assert state.phase == "opening"
        assert state.round_number == 1
        assert state.has_gm_turn is False
        assert state.active_goals == []
        assert state.active_complications == []

    def test_with_initiative_holder(self):
        state = CombatLoopState(
            scene_id="scene_1",
            initiative_holder="player_1",
            scenario_type="control",
        )
        assert state.scene_id == "scene_1"
        assert state.initiative_holder == "player_1"
        assert state.scenario_type == "control"
        assert "player_1" in state.turn_history

    def test_with_scenario_settings(self):
        state = CombatLoopState(
            scenario_type="escort",
            scenario_settings={"objective_zone": "alpha", "extract_zone": "bravo"},
        )
        assert state.scenario_type == "escort"
        assert state.scenario_settings["objective_zone"] == "alpha"


class TestInitiativeTracker:
    """Tests for InitiativeTracker per PR2 3064-3071."""

    def test_player_has_initiative_by_default(self):
        tracker = InitiativeTracker(current_initiative="player_1")
        assert tracker.current_initiative == "player_1"
        assert not tracker.gm_took_initiative
        assert tracker.player_stall_count == 0

    def test_gm_takes_initiative_on_stall(self):
        tracker = InitiativeTracker(current_initiative="player_1")
        updated = tracker.player_stalls()
        assert updated.gm_took_initiative
        assert updated.player_stall_count == 1
        assert updated.current_initiative is None

    def test_transfer_to_gm_explicit(self):
        tracker = InitiativeTracker(current_initiative="player_1")
        updated = transfer_initiative(tracker, "GM", is_gm=True)
        assert updated.gm_took_initiative
        assert updated.player_stall_count == 1

    def test_player_regains_initiative(self):
        tracker = InitiativeTracker(current_initiative="player_1")
        tracker = tracker.player_stalls()
        assert tracker.gm_took_initiative

        updated = transfer_initiative(tracker, "player_1", roll=15)
        assert not updated.gm_took_initiative
        assert updated.current_initiative == "player_1"
        assert updated.last_player_roll == 15
        assert updated.player_stall_count == 0

    def test_initiative_queue(self):
        tracker = InitiativeTracker(
            current_initiative="player_1",
            initiative_queue=["player_2", "player_3"],
        )
        assert len(tracker.initiative_queue) == 2


class TestSceneTimer:
    """Tests for SceneTimer and time pressure mechanics."""

    def test_default_no_timer(self):
        timer = SceneTimer()
        assert timer.total_turns is None
        assert timer.remaining_turns is None
        assert timer.pressure_level == "none"

    def test_init_from_total(self):
        timer = SceneTimer(total_turns=6)
        assert timer.remaining_turns == 6

    def test_timer_ticks_down(self):
        timer = SceneTimer(total_turns=6, remaining_turns=6)
        updated = timer.tick()
        assert updated.remaining_turns == 5
        assert updated.pressure_level == "none"

    def test_pressure_escalates_at_half(self):
        timer = SceneTimer(total_turns=6, remaining_turns=3)
        updated = timer.tick()
        assert updated.pressure_level == "urgent"

    def test_pressure_critical_at_third(self):
        timer = SceneTimer(total_turns=6, remaining_turns=2)
        updated = timer.tick()
        assert updated.pressure_level == "critical"

    def test_multiple_ticks(self):
        timer = SceneTimer(total_turns=10, remaining_turns=10)
        for _ in range(8):
            timer = timer.tick()
        assert timer.remaining_turns == 2
        assert timer.pressure_level == "critical"


class TestPhaseTransitions:
    """Tests for scene phase transitions."""

    def test_opening_to_action_on_goal(self):
        state = CombatLoopState(phase="opening")
        updated = advance_phase(state, "goal_announced")
        assert updated.phase == "action"

    def test_opening_to_action_on_action(self):
        state = CombatLoopState(phase="opening")
        updated = advance_phase(state, "action_taken")
        assert updated.phase == "action"

    def test_action_to_complication(self):
        state = CombatLoopState(phase="action")
        updated = advance_phase(state, "complication_occurred")
        assert updated.phase == "complication"

    def test_action_to_resolution_on_goal_complete(self):
        state = CombatLoopState(phase="action")
        updated = advance_phase(state, "goal_completed")
        assert updated.phase == "resolution"

    def test_complication_to_action(self):
        state = CombatLoopState(phase="complication")
        updated = advance_phase(state, "action_taken")
        assert updated.phase == "action"

    def test_complication_to_resolution(self):
        state = CombatLoopState(phase="complication")
        updated = advance_phase(state, "goal_completed")
        assert updated.phase == "resolution"

    def test_no_transition_from_resolution(self):
        state = CombatLoopState(phase="resolution")
        updated = advance_phase(state, "action_taken")
        assert updated.phase == "resolution"


class TestSceneLifecycle:
    """Tests for scene start and end."""

    def test_start_scene(self):
        loop_state, init_tracker, timer = start_scene(
            scene_id="scene_1",
            first_actor="player_1",
            scenario_type="control",
            scene_timer=6,
        )
        assert loop_state.scene_id == "scene_1"
        assert loop_state.phase == "opening"
        assert loop_state.initiative_holder == "player_1"
        assert loop_state.scenario_type == "control"
        assert init_tracker.current_initiative == "player_1"
        assert timer is not None
        assert timer.total_turns == 6

    def test_start_scene_no_timer(self):
        loop_state, init_tracker, timer = start_scene(
            scene_id="scene_2",
            first_actor="player_1",
        )
        assert timer is None

    def test_end_scene(self):
        state = CombatLoopState(phase="action")
        updated = end_scene(state, "goals_achieved")
        assert updated.phase == "resolution"


class TestVictoryConditions:
    """Tests for victory condition checking."""

    def test_all_goals_complete(self):
        state = CombatLoopState(
            active_goals=["goal_1", "goal_2"],
            round_number=3,
        )
        conditions = [
            VictoryCondition(
                condition_type="all_goals_complete",
                description="Complete all goals",
            ),
        ]
        result = check_victory_conditions(
            state,
            conditions,
            completed_goal_ids=["goal_1", "goal_2"],
            active_hostile_ids=["enemy_1"],
        )
        assert result.all_met
        assert "all_goals_complete" in result.conditions_met

    def test_goals_incomplete(self):
        state = CombatLoopState(
            active_goals=["goal_1", "goal_2"],
            round_number=3,
        )
        conditions = [
            VictoryCondition(
                condition_type="all_goals_complete",
                description="Complete all goals",
            ),
        ]
        result = check_victory_conditions(
            state,
            conditions,
            completed_goal_ids=["goal_1"],
            active_hostile_ids=["enemy_1"],
        )
        assert not result.all_met
        assert "all_goals_complete" in result.conditions_pending

    def test_time_elapsed(self):
        state = CombatLoopState(round_number=6)
        conditions = [
            VictoryCondition(
                condition_type="time_elapsed",
                description="Survive 6 rounds",
                target_value=6,
            ),
        ]
        result = check_victory_conditions(
            state,
            conditions,
            completed_goal_ids=[],
            active_hostile_ids=["enemy_1"],
        )
        assert result.all_met

    def test_all_enemies_defeated(self):
        state = CombatLoopState(round_number=3)
        conditions = [
            VictoryCondition(
                condition_type="all_enemies_defeated",
                description="Eliminate all hostiles",
            ),
        ]
        result = check_victory_conditions(
            state,
            conditions,
            completed_goal_ids=[],
            active_hostile_ids=[],
        )
        assert result.all_met

    def test_enemies_still_active(self):
        state = CombatLoopState(round_number=3)
        conditions = [
            VictoryCondition(
                condition_type="all_enemies_defeated",
                description="Eliminate all hostiles",
            ),
        ]
        result = check_victory_conditions(
            state,
            conditions,
            completed_goal_ids=[],
            active_hostile_ids=["enemy_1", "enemy_2"],
        )
        assert not result.all_met

    def test_optional_condition_ignored(self):
        state = CombatLoopState(
            active_goals=["goal_1"],
            round_number=3,
        )
        conditions = [
            VictoryCondition(
                condition_type="all_goals_complete",
                description="Complete all goals",
                is_optional=True,
            ),
        ]
        result = check_victory_conditions(
            state,
            conditions,
            completed_goal_ids=[],
            active_hostile_ids=[],
        )
        assert result.all_met


class TestMissionCompletion:
    """Tests for mission completion resolution."""

    def test_full_victory(self):
        state = CombatLoopState(
            active_goals=["goal_1"],
            round_number=3,
        )
        conditions = [
            VictoryCondition(
                condition_type="all_goals_complete",
                description="Complete all goals",
            ),
        ]
        check_result = VictoryCheckResult(
            conditions_met=["all_goals_complete"],
            conditions_pending=[],
            all_met=True,
        )
        result = resolve_mission_completion(
            state,
            conditions,
            check_result,
            completed_goal_ids=["goal_1"],
        )
        assert result.victory
        assert result.goals_achieved == 1
        assert result.goals_total == 1
        assert not result.partial_success

    def test_partial_success(self):
        state = CombatLoopState(
            active_goals=["goal_1", "goal_2"],
            round_number=6,
        )
        conditions = [
            VictoryCondition(
                condition_type="all_goals_complete",
                description="Complete all goals",
            ),
        ]
        check_result = VictoryCheckResult(
            conditions_met=[],
            conditions_pending=["all_goals_complete"],
            all_met=False,
        )
        result = resolve_mission_completion(
            state,
            conditions,
            check_result,
            completed_goal_ids=["goal_1"],
        )
        assert not result.victory
        assert result.partial_success
        assert result.goals_achieved == 1
        assert result.goals_total == 2

    def test_time_limit_victory(self):
        state = CombatLoopState(
            active_goals=[],
            round_number=6,
        )
        conditions = [
            VictoryCondition(
                condition_type="time_elapsed",
                description="Survive 6 rounds",
                target_value=6,
            ),
        ]
        check_result = VictoryCheckResult(
            conditions_met=["time_elapsed"],
            conditions_pending=[],
            all_met=True,
        )
        result = resolve_mission_completion(
            state,
            conditions,
            check_result,
            completed_goal_ids=[],
        )
        assert result.victory
        assert result.completion_type == "time_elapsed"


class TestNarrativeScenarioIntegration:
    """Tests for integration with NarrativeCombatState."""

    def test_start_narrative_scenario(self):
        """Test initializing a complete narrative scenario."""
        from core.shared.combat_loop import (
            start_narrative_scenario,
            InitiativeTracker,
            SceneTimer,
        )
        from core.shared.narrative import NarrativeCombatState, NarrativeGoalTracker

        loop_state, combat_state, init_tracker, timer = start_narrative_scenario(
            scene_id="test_scene",
            first_actor="pilot_1",
            scenario_type="control",
            scene_timer=6,
        )

        assert loop_state.scene_id == "test_scene"
        assert loop_state.phase == "opening"
        assert loop_state.initiative_holder == "pilot_1"
        assert loop_state.scenario_type == "control"
        assert isinstance(combat_state, NarrativeCombatState)
        assert combat_state.scene_id == "test_scene"
        assert init_tracker.current_initiative == "pilot_1"
        assert isinstance(timer, SceneTimer)
        assert timer.total_turns == 6

    def test_advance_narrative_turn(self):
        """Test advancing narrative turn with phase transition."""
        from core.shared.combat_loop import (
            advance_narrative_turn,
            CombatLoopState,
        )

        state = CombatLoopState(
            scene_id="scene_1",
            phase="opening",
            initiative_holder="player_1",
        )

        updated = advance_narrative_turn(state, "player_1", "goal_announced")

        assert updated.phase == "action"
        assert "player_1" in updated.turn_history

    def test_check_narrative_victory_conditions_met(self):
        """Test victory detection when all conditions are met."""
        from core.shared.combat_loop import (
            check_narrative_victory,
            VictoryCondition,
            CombatLoopState,
        )
        from core.shared.narrative import NarrativeCombatState

        loop_state = CombatLoopState(
            scene_id="scene_1",
            active_goals=["goal_1"],
            round_number=3,
        )
        combat_state = NarrativeCombatState(
            scene_id="scene_1",
            goal_tracker=NarrativeGoalTracker(
                goals=[
                    NarrativeGoalState(
                        goal=NarrativeGoal(
                            id="goal_1",
                            description="Complete objective",
                            success_conditions=[],
                        ),
                        status="completed",
                    )
                ]
            ),
        )

        conditions = [
            VictoryCondition(
                condition_type="all_goals_complete",
                description="Complete all goals",
            ),
        ]

        is_victory, result = check_narrative_victory(
            loop_state,
            combat_state,
            active_hostiles=[],
            victory_conditions=conditions,
        )

        assert is_victory is True
        assert result is not None
        assert result.victory is True

    def test_check_narrative_victory_not_met(self):
        """Test victory detection when conditions are not met."""
        from core.shared.combat_loop import (
            check_narrative_victory,
            VictoryCondition,
            CombatLoopState,
        )
        from core.shared.narrative import (
            NarrativeCombatState,
            NarrativeGoalTracker,
            NarrativeGoal,
            NarrativeGoalState,
        )

        loop_state = CombatLoopState(
            scene_id="scene_1",
            active_goals=["goal_1", "goal_2"],
            round_number=3,
        )
        combat_state = NarrativeCombatState(
            scene_id="scene_1",
            goal_tracker=NarrativeGoalTracker(
                goals=[
                    NarrativeGoalState(
                        goal=NarrativeGoal(
                            id="goal_1",
                            description="First objective",
                            success_conditions=[],
                        ),
                        status="completed",
                    )
                ]
            ),
        )

        conditions = [
            VictoryCondition(
                condition_type="all_goals_complete",
                description="Complete all goals",
            ),
        ]

        is_victory, result = check_narrative_victory(
            loop_state,
            combat_state,
            active_hostiles=["enemy_1"],
            victory_conditions=conditions,
        )

        assert is_victory is False
        assert result is None
