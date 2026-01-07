"""Tests for scenario/objective system."""

import pytest
from pydantic import ValidationError
from core.shared.scenario import (
    MissionObjectiveType,
    ObjectiveStatus,
    ObjectiveCriterionType,
    MissionOutcomeType,
    ObjectiveCriterion,
    MissionObjective,
    MissionObjectiveState,
    MissionStakes,
    Mission,
    MissionState,
    MissionBriefing,
    MissionDebrief,
    calculate_mission_progress,
    check_objective_prerequisites,
    update_objective_status,
    mark_objective_criterion_met,
    check_mission_completion,
    resolve_mission_outcome,
    start_mission,
    advance_mission_phase,
    advance_mission_turn,
    SitrepZoneType,
    SitrepDeploymentType,
    SitrepReservePattern,
    VictoryConditionType,
    SitrepZone,
    VictoryCondition,
    SitrepTemplate,
    ESCORT_TEMPLATE,
    CONTROL_TEMPLATE,
    EXTRACT_TEMPLATE,
    HOLDOUT_TEMPLATE,
    GAUNTLET_TEMPLATE,
    RECON_TEMPLATE,
    SITREP_TEMPLATES,
    get_sitrep_template,
    create_mission_from_template,
)


class TestMissionObjectiveType:
    """Tests for MissionObjectiveType enum."""

    def test_all_types_available(self):
        """Verify all 8 mission types are available."""
        expected_types = {
            "escort",
            "defend",
            "extract",
            "destroy",
            "infiltrate",
            "investigate",
            "control",
            "custom",
        }
        available_types = set(MissionObjectiveType.__args__)
        assert expected_types == available_types


class TestObjectiveStatus:
    """Tests for ObjectiveStatus enum."""

    def test_all_statuses_available(self):
        """Verify all 5 statuses are available."""
        expected_statuses = {"pending", "in_progress", "blocked", "completed", "failed"}
        available_statuses = set(ObjectiveStatus.__args__)
        assert expected_statuses == available_statuses


class TestObjectiveCriterion:
    """Tests for ObjectiveCriterion model."""

    def test_create_basic_criterion(self):
        """Create a basic criterion."""
        criterion = ObjectiveCriterion(
            criterion_type="target_destroyed",
            description="Destroy the enemy commander",
        )
        assert criterion.criterion_type == "target_destroyed"
        assert criterion.description == "Destroy the enemy commander"
        assert criterion.target_id is None
        assert criterion.required_amount is None

    def test_create_criterion_with_target(self):
        """Create a criterion with target ID."""
        criterion = ObjectiveCriterion(
            criterion_type="target_destroyed",
            description="Destroy the reactor",
            target_id="reactor_001",
        )
        assert criterion.target_id == "reactor_001"

    def test_create_criterion_with_amount(self):
        """Create a criterion with required amount."""
        criterion = ObjectiveCriterion(
            criterion_type="custom",
            description="Elim enemiesinate 5",
            required_amount=5,
        )
        assert criterion.required_amount == 5

    def test_criterion_default_values(self):
        """Test default values are applied."""
        criterion = ObjectiveCriterion(
            criterion_type="area_secured",
            description="Secure the landing zone",
        )
        assert criterion.target_id is None
        assert criterion.required_amount is None


class TestMissionObjective:
    """Tests for MissionObjective model."""

    def test_create_basic_objective(self):
        """Create a basic mission objective."""
        objective = MissionObjective(
            id="obj_001",
            description="Destroy the enemy reactor",
            objective_type="destroy",
        )
        assert objective.id == "obj_001"
        assert objective.description == "Destroy the enemy reactor"
        assert objective.objective_type == "destroy"
        assert objective.status == "pending"
        assert objective.priority == 1
        assert objective.depends_on == []
        assert objective.is_optional is False

    def test_objective_with_dependencies(self):
        """Create objective with dependencies."""
        objective = MissionObjective(
            id="obj_002",
            description="Secure the command center",
            objective_type="control",
            depends_on=["obj_001"],
        )
        assert objective.depends_on == ["obj_001"]

    def test_objective_with_priority(self):
        """Create objective with custom priority."""
        objective = MissionObjective(
            id="obj_003",
            description="Critical mission objective",
            objective_type="defend",
            priority=3,
        )
        assert objective.priority == 3

    def test_objective_with_criteria(self):
        """Create objective with completion criteria."""
        criteria = [
            ObjectiveCriterion(
                criterion_type="target_destroyed",
                description="Destroy reactor",
                target_id="reactor_01",
            ),
            ObjectiveCriterion(
                criterion_type="area_secured",
                description="Secure the room",
            ),
        ]
        objective = MissionObjective(
            id="obj_004",
            description="Take over the facility",
            objective_type="destroy",
            completion_criteria=criteria,
        )
        assert len(objective.completion_criteria) == 2

    def test_optional_objective(self):
        """Create an optional objective."""
        objective = MissionObjective(
            id="obj_005",
            description="Collect optional intel",
            objective_type="investigate",
            is_optional=True,
        )
        assert objective.is_optional is True

    def test_self_dependency_rejected(self):
        """Objective cannot depend on itself."""
        with pytest.raises(ValueError, match="cannot depend on itself"):
            MissionObjective(
                id="obj_self",
                description="Self-referencing objective",
                objective_type="custom",
                depends_on=["obj_self"],
            )


class TestMissionObjectiveState:
    """Tests for MissionObjectiveState model."""

    def test_state_with_no_criteria(self):
        """State tracks objective without criteria."""
        objective = MissionObjective(
            id="obj_001",
            description="Test objective",
            objective_type="escort",
        )
        state = MissionObjectiveState(objective=objective)
        assert state.objective == objective
        assert state.criteria_met == []
        assert state.attempts == 0
        assert state.is_complete is False

    def test_complete_with_criteria_met(self):
        """State is complete when criteria are met."""
        objective = MissionObjective(
            id="obj_001",
            description="Test objective",
            objective_type="destroy",
            completion_criteria=[
                ObjectiveCriterion(
                    criterion_type="target_destroyed",
                    description="Destroy target",
                ),
            ],
        )
        state = MissionObjectiveState(
            objective=objective,
            criteria_met=["Destroy target"],
        )
        assert state.is_complete is True

    def test_not_complete_without_criteria(self):
        """State is not complete when criteria not met."""
        objective = MissionObjective(
            id="obj_001",
            description="Test objective",
            objective_type="destroy",
            completion_criteria=[
                ObjectiveCriterion(
                    criterion_type="target_destroyed",
                    description="Destroy target",
                ),
            ],
        )
        state = MissionObjectiveState(
            objective=objective,
            criteria_met=[],
        )
        assert state.is_complete is False

    def test_completion_percentage(self):
        """Calculate completion percentage correctly."""
        objective = MissionObjective(
            id="obj_001",
            description="Test objective",
            objective_type="custom",
            completion_criteria=[
                ObjectiveCriterion(criterion_type="custom", description="Crit 1"),
                ObjectiveCriterion(criterion_type="custom", description="Crit 2"),
                ObjectiveCriterion(criterion_type="custom", description="Crit 3"),
                ObjectiveCriterion(criterion_type="custom", description="Crit 4"),
            ],
        )
        state = MissionObjectiveState(
            objective=objective,
            criteria_met=["Crit 1", "Crit 2"],
        )
        assert state.completion_percentage == 0.5

    def test_tracks_attempts(self):
        """State tracks number of attempts."""
        objective = MissionObjective(
            id="obj_001",
            description="Test objective",
            objective_type="defend",
        )
        state = MissionObjectiveState(objective=objective, attempts=3)
        assert state.attempts == 3


class TestMissionStakes:
    """Tests for MissionStakes model."""

    def test_create_stakes(self):
        """Create mission stakes."""
        stakes = MissionStakes(
            stakes_type="immediate",
            description="Stop the attack before it reaches the city",
            consequences_success="City saved, reward granted",
            consequences_failure="City destroyed, reputation ruined",
        )
        assert stakes.stakes_type == "immediate"
        assert stakes.description == "Stop the attack before it reaches the city"

    def test_stakes_without_consequences(self):
        """Stakes can have no consequences defined."""
        stakes = MissionStakes(
            stakes_type="gradual",
            description="Long-term investigation",
        )
        assert stakes.consequences_success is None
        assert stakes.consequences_failure is None
        assert stakes.consequences_partial is None

    def test_all_stake_types(self):
        """All stake types are available."""
        for stake_type in ["personal", "faction", "immediate", "gradual"]:
            stakes = MissionStakes(
                stakes_type=stake_type,  # type: ignore[arg-type]
                description=f"Test {stake_type} stakes",
            )
            assert stakes.stakes_type == stake_type


class TestMission:
    """Tests for Mission model."""

    def test_create_basic_mission(self):
        """Create a basic mission."""
        mission = Mission(
            id="mission_001",
            name="Destroy the Reactor",
            description="Infiltrate the enemy base and destroy their reactor",
        )
        assert mission.id == "mission_001"
        assert mission.name == "Destroy the Reactor"
        assert mission.objectives == []
        assert mission.stakes is None
        assert mission.is_critical is False

    def test_mission_with_objectives(self):
        """Create mission with objectives."""
        objectives = [
            MissionObjective(
                id="obj_001",
                description="Infiltrate the base",
                objective_type="infiltrate",
            ),
            MissionObjective(
                id="obj_002",
                description="Destroy the reactor",
                objective_type="destroy",
                depends_on=["obj_001"],
            ),
        ]
        mission = Mission(
            id="mission_002",
            name="Covert Operation",
            description="Secret mission to disable enemy operations",
            objectives=objectives,
        )
        assert len(mission.objectives) == 2
        assert mission.objectives[1].depends_on == ["obj_001"]

    def test_mission_with_stakes(self):
        """Create mission with stakes."""
        stakes = MissionStakes(
            stakes_type="faction",
            description="Stop the Horus threat",
            consequences_success="Union recognition",
            consequences_failure="Horus expands influence",
        )
        mission = Mission(
            id="mission_003",
            name="Horus Threat",
            description="Investigate Horus activity",
            stakes=stakes,
        )
        assert mission.stakes == stakes

    def test_mission_with_time_limit(self):
        """Create mission with time limit."""
        mission = Mission(
            id="mission_004",
            name="Timed Operation",
            description="Complete before reinforcements arrive",
            time_limit=6,
        )
        assert mission.time_limit == 6

    def test_critical_mission(self):
        """Create a critical mission."""
        mission = Mission(
            id="mission_005",
            name="Last Stand",
            description="Defend the colony at all costs",
            is_critical=True,
        )
        assert mission.is_critical is True

    def test_invalid_dependency_rejected(self):
        """Mission rejects objective with invalid dependency."""
        objectives = [
            MissionObjective(
                id="obj_001",
                description="This objective",
                objective_type="custom",
                depends_on=["nonexistent"],
            ),
        ]
        with pytest.raises(ValueError, match="non-existent objective"):
            Mission(
                id="mission_006",
                name="Invalid Mission",
                description="Has invalid dependency",
                objectives=objectives,
            )


class TestMissionState:
    """Tests for MissionState model."""

    def test_init_from_mission(self):
        """Initialize state from mission definition."""
        mission = Mission(
            id="mission_001",
            name="Test Mission",
            description="A test mission",
            objectives=[
                MissionObjective(
                    id="obj_001",
                    description="Objective 1",
                    objective_type="escort",
                ),
                MissionObjective(
                    id="obj_002",
                    description="Objective 2",
                    objective_type="destroy",
                ),
            ],
        )
        state = MissionState(mission=mission)
        assert state.mission == mission
        assert state.current_phase == "briefing"
        assert state.current_turn == 1
        assert len(state.objective_states) == 2
        assert state.completion_score == 0.0
        assert state.is_victory is None

    def test_get_objective_state(self):
        """Get state for a specific objective."""
        mission = Mission(
            id="mission_001",
            name="Test Mission",
            description="A test mission",
            objectives=[
                MissionObjective(
                    id="obj_001",
                    description="Objective 1",
                    objective_type="escort",
                ),
            ],
        )
        state = MissionState(mission=mission)
        obj_state = state.get_objective_state("obj_001")
        assert obj_state is not None
        assert obj_state.objective.id == "obj_001"

    def test_get_unknown_objective_state(self):
        """Getting unknown objective returns None."""
        mission = Mission(
            id="mission_001",
            name="Test Mission",
            description="A test mission",
            objectives=[],
        )
        state = MissionState(mission=mission)
        obj_state = state.get_objective_state("nonexistent")
        assert obj_state is None

    def test_get_required_objectives(self):
        """Get objectives with met dependencies."""
        mission = Mission(
            id="mission_001",
            name="Test Mission",
            description="A test mission",
            objectives=[
                MissionObjective(
                    id="obj_001",
                    description="First objective",
                    objective_type="escort",
                ),
                MissionObjective(
                    id="obj_002",
                    description="Second objective",
                    objective_type="destroy",
                    depends_on=["obj_001"],
                ),
            ],
        )
        state = MissionState(mission=mission)
        required = state.get_required_objectives()
        assert len(required) == 1
        assert required[0].id == "obj_001"

    def test_get_blocked_objectives(self):
        """Get objectives with unmet dependencies."""
        mission = Mission(
            id="mission_001",
            name="Test Mission",
            description="A test mission",
            objectives=[
                MissionObjective(
                    id="obj_001",
                    description="First objective",
                    objective_type="escort",
                ),
                MissionObjective(
                    id="obj_002",
                    description="Second objective",
                    objective_type="destroy",
                    depends_on=["obj_001"],
                ),
            ],
        )
        state = MissionState(mission=mission)
        blocked = state.get_blocked_objectives()
        assert len(blocked) == 1
        assert blocked[0].id == "obj_002"


class TestMissionBriefing:
    """Tests for MissionBriefing model."""

    def test_create_briefing(self):
        """Create a mission briefing."""
        briefing = MissionBriefing(
            mission_id="mission_001",
            objectives_revealed=["obj_001", "obj_002"],
            reserves_available=["ammo_pack", "repair_drone"],
        )
        assert briefing.mission_id == "mission_001"
        assert len(briefing.objectives_revealed) == 2
        assert briefing.stakes_revealed is True
        assert len(briefing.reserves_available) == 2

    def test_briefing_without_revealed_stakes(self):
        """Briefing can hide stakes."""
        briefing = MissionBriefing(
            mission_id="mission_001",
            stakes_revealed=False,
        )
        assert briefing.stakes_revealed is False


class TestMissionDebrief:
    """Tests for MissionDebrief model."""

    def test_create_debrief(self):
        """Create a mission debrief."""
        debrief = MissionDebrief(
            mission_id="mission_001",
            outcome="success",
            objectives_completed=["obj_001", "obj_002"],
            objectives_failed=[],
            objectives_blocked=[],
            completion_score=1.0,
            experience_gained=1,
            reserves_granted=["bonus_ammo"],
        )
        assert debrief.outcome == "success"
        assert len(debrief.objectives_completed) == 2
        assert debrief.completion_score == 1.0
        assert debrief.experience_gained == 1

    def test_partial_success_debrief(self):
        """Create partial success debrief."""
        debrief = MissionDebrief(
            mission_id="mission_001",
            outcome="partial",
            objectives_completed=["obj_001"],
            objectives_failed=["obj_002"],
            objectives_blocked=["obj_003"],
            completion_score=0.5,
        )
        assert debrief.outcome == "partial"
        assert debrief.completion_score == 0.5

    def test_catastrophic_failure_debrief(self):
        """Create catastrophic failure debrief."""
        debrief = MissionDebrief(
            mission_id="mission_001",
            outcome="catastrophic",
            objectives_completed=[],
            objectives_failed=["obj_001", "obj_002"],
            objectives_blocked=[],
            completion_score=0.0,
            experience_gained=0,
        )
        assert debrief.outcome == "catastrophic"
        assert debrief.completion_score == 0.0
        assert debrief.experience_gained == 0


class TestCalculateMissionProgress:
    """Tests for calculate_mission_progress function."""

    def test_no_objectives(self):
        """Progress is 100% with no objectives."""
        mission = Mission(
            id="mission_001",
            name="Empty Mission",
            description="Has no objectives",
            objectives=[],
        )
        state = MissionState(mission=mission)
        progress = calculate_mission_progress(state)
        assert progress == 1.0

    def test_all_complete(self):
        """Progress is 100% when all complete."""
        mission = Mission(
            id="mission_001",
            name="Complete Mission",
            description="All objectives complete",
            objectives=[
                MissionObjective(
                    id="obj_001",
                    description="Objective 1",
                    objective_type="escort",
                ),
                MissionObjective(
                    id="obj_002",
                    description="Objective 2",
                    objective_type="destroy",
                ),
            ],
        )
        state = MissionState(mission=mission)
        state = update_objective_status(state, "obj_001", "completed")
        state = update_objective_status(state, "obj_002", "completed")
        progress = calculate_mission_progress(state)
        assert progress == 1.0

    def test_half_complete(self):
        """Progress is 50% when half complete."""
        mission = Mission(
            id="mission_001",
            name="Half Mission",
            description="Half objectives complete",
            objectives=[
                MissionObjective(
                    id="obj_001",
                    description="Objective 1",
                    objective_type="escort",
                ),
                MissionObjective(
                    id="obj_002",
                    description="Objective 2",
                    objective_type="destroy",
                ),
            ],
        )
        state = MissionState(mission=mission)
        state = update_objective_status(state, "obj_001", "completed")
        progress = calculate_mission_progress(state)
        assert progress == 0.5

    def test_optional_not_counted(self):
        """Optional objectives don't affect score."""
        mission = Mission(
            id="mission_001",
            name="Optional Mission",
            description="Has optional objectives",
            objectives=[
                MissionObjective(
                    id="obj_001",
                    description="Main objective",
                    objective_type="escort",
                ),
                MissionObjective(
                    id="obj_002",
                    description="Bonus objective",
                    objective_type="investigate",
                    is_optional=True,
                ),
            ],
        )
        state = MissionState(mission=mission)
        progress = calculate_mission_progress(state)
        assert progress == 0.0

    def test_none_complete(self):
        """Progress is 0% when nothing complete."""
        mission = Mission(
            id="mission_001",
            name="Failed Mission",
            description="No objectives complete",
            objectives=[
                MissionObjective(
                    id="obj_001",
                    description="Objective 1",
                    objective_type="escort",
                ),
                MissionObjective(
                    id="obj_002",
                    description="Objective 2",
                    objective_type="destroy",
                ),
            ],
        )
        state = MissionState(mission=mission)
        progress = calculate_mission_progress(state)
        assert progress == 0.0


class TestCheckObjectivePrerequisites:
    """Tests for check_objective_prerequisites function."""

    def test_no_dependencies(self):
        """Objective with no dependencies passes."""
        mission = Mission(
            id="mission_001",
            name="Test Mission",
            description="Test",
            objectives=[
                MissionObjective(
                    id="obj_001",
                    description="Independent objective",
                    objective_type="escort",
                ),
            ],
        )
        state = MissionState(mission=mission)
        all_met, unmet = check_objective_prerequisites(state, "obj_001")
        assert all_met is True
        assert unmet == []

    def test_dependencies_met(self):
        """Objective with met dependencies passes."""
        mission = Mission(
            id="mission_001",
            name="Test Mission",
            description="Test",
            objectives=[
                MissionObjective(
                    id="obj_001",
                    description="First objective",
                    objective_type="escort",
                ),
                MissionObjective(
                    id="obj_002",
                    description="Second objective",
                    objective_type="destroy",
                    depends_on=["obj_001"],
                ),
            ],
        )
        state = MissionState(mission=mission)
        state = update_objective_status(state, "obj_001", "completed")
        all_met, unmet = check_objective_prerequisites(state, "obj_002")
        assert all_met is True
        assert unmet == []

    def test_dependencies_unmet(self):
        """Objective with unmet dependencies fails."""
        mission = Mission(
            id="mission_001",
            name="Test Mission",
            description="Test",
            objectives=[
                MissionObjective(
                    id="obj_001",
                    description="First objective",
                    objective_type="escort",
                ),
                MissionObjective(
                    id="obj_002",
                    description="Second objective",
                    objective_type="destroy",
                    depends_on=["obj_001"],
                ),
            ],
        )
        state = MissionState(mission=mission)
        all_met, unmet = check_objective_prerequisites(state, "obj_002")
        assert all_met is False
        assert unmet == ["obj_001"]

    def test_unknown_objective(self):
        """Unknown objective returns False."""
        mission = Mission(
            id="mission_001",
            name="Test Mission",
            description="Test",
            objectives=[],
        )
        state = MissionState(mission=mission)
        all_met, unmet = check_objective_prerequisites(state, "nonexistent")
        assert all_met is False


class TestUpdateObjectiveStatus:
    """Tests for update_objective_status function."""

    def test_update_to_in_progress(self):
        """Update objective status to in_progress."""
        mission = Mission(
            id="mission_001",
            name="Test Mission",
            description="Test",
            objectives=[
                MissionObjective(
                    id="obj_001",
                    description="Objective 1",
                    objective_type="escort",
                ),
            ],
        )
        state = MissionState(mission=mission)
        new_state = update_objective_status(state, "obj_001", "in_progress")
        assert new_state.objective_states["obj_001"].objective.status == "in_progress"

    def test_update_to_completed(self):
        """Update objective status to completed."""
        mission = Mission(
            id="mission_001",
            name="Test Mission",
            description="Test",
            objectives=[
                MissionObjective(
                    id="obj_001",
                    description="Objective 1",
                    objective_type="escort",
                ),
            ],
        )
        state = MissionState(mission=mission)
        new_state = update_objective_status(state, "obj_001", "completed")
        assert new_state.completion_score == 1.0

    def test_unknown_objective_raises(self):
        """Unknown objective raises ValueError."""
        mission = Mission(
            id="mission_001",
            name="Test Mission",
            description="Test",
            objectives=[],
        )
        state = MissionState(mission=mission)
        with pytest.raises(ValueError, match="Unknown objective"):
            update_objective_status(state, "nonexistent", "completed")


class TestMarkObjectiveCriterionMet:
    """Tests for mark_objective_criterion_met function."""

    def test_mark_criterion(self):
        """Mark a completion criterion as met."""
        mission = Mission(
            id="mission_001",
            name="Test Mission",
            description="Test",
            objectives=[
                MissionObjective(
                    id="obj_001",
                    description="Objective 1",
                    objective_type="destroy",
                    completion_criteria=[
                        ObjectiveCriterion(
                            criterion_type="target_destroyed",
                            description="Destroy reactor",
                        ),
                        ObjectiveCriterion(
                            criterion_type="area_secured",
                            description="Secure the room",
                        ),
                    ],
                ),
            ],
        )
        state = MissionState(mission=mission)
        new_state = mark_objective_criterion_met(state, "obj_001", "Destroy reactor")
        assert "Destroy reactor" in new_state.objective_states["obj_001"].criteria_met
        assert (
            "Secure the room" not in new_state.objective_states["obj_001"].criteria_met
        )

    def test_mark_all_criteria(self):
        """Mark all criteria to complete objective."""
        mission = Mission(
            id="mission_001",
            name="Test Mission",
            description="Test",
            objectives=[
                MissionObjective(
                    id="obj_001",
                    description="Objective 1",
                    objective_type="destroy",
                    completion_criteria=[
                        ObjectiveCriterion(
                            criterion_type="target_destroyed",
                            description="Destroy reactor",
                        ),
                        ObjectiveCriterion(
                            criterion_type="area_secured",
                            description="Secure the room",
                        ),
                    ],
                ),
            ],
        )
        state = MissionState(mission=mission)
        state = mark_objective_criterion_met(state, "obj_001", "Destroy reactor")
        state = mark_objective_criterion_met(state, "obj_001", "Secure the room")
        assert state.completion_score == 1.0


class TestCheckMissionCompletion:
    """Tests for check_mission_completion function."""

    def test_all_complete(self):
        """Mission is complete when all objectives complete."""
        mission = Mission(
            id="mission_001",
            name="Test Mission",
            description="Test",
            objectives=[
                MissionObjective(
                    id="obj_001",
                    description="Objective 1",
                    objective_type="escort",
                ),
            ],
        )
        state = MissionState(mission=mission)
        state = update_objective_status(state, "obj_001", "completed")
        assert check_mission_completion(state) is True

    def test_not_complete(self):
        """Mission is not complete when objectives pending."""
        mission = Mission(
            id="mission_001",
            name="Test Mission",
            description="Test",
            objectives=[
                MissionObjective(
                    id="obj_001",
                    description="Objective 1",
                    objective_type="escort",
                ),
            ],
        )
        state = MissionState(mission=mission)
        assert check_mission_completion(state) is False

    def test_time_limit_reached(self):
        """Mission complete when time limit reached."""
        mission = Mission(
            id="mission_001",
            name="Timed Mission",
            description="Test",
            time_limit=5,
            objectives=[
                MissionObjective(
                    id="obj_001",
                    description="Objective 1",
                    objective_type="escort",
                ),
            ],
        )
        state = MissionState(mission=mission)
        state = state.model_copy(update={"current_turn": 5})
        assert check_mission_completion(state) is True

    def test_time_limit_not_reached(self):
        """Mission not complete when time limit not reached."""
        mission = Mission(
            id="mission_001",
            name="Timed Mission",
            description="Test",
            time_limit=5,
            objectives=[
                MissionObjective(
                    id="obj_001",
                    description="Objective 1",
                    objective_type="escort",
                ),
            ],
        )
        state = MissionState(mission=mission)
        state = state.model_copy(update={"current_turn": 3})
        assert check_mission_completion(state) is False

    def test_no_time_limit(self):
        """Mission without time limit doesn't end early."""
        mission = Mission(
            id="mission_001",
            name="Untimed Mission",
            description="Test",
            objectives=[
                MissionObjective(
                    id="obj_001",
                    description="Objective 1",
                    objective_type="escort",
                ),
            ],
        )
        state = MissionState(mission=mission)
        state = state.model_copy(update={"current_turn": 100})
        assert check_mission_completion(state) is False


class TestResolveMissionOutcome:
    """Tests for resolve_mission_outcome function."""

    def test_full_success(self):
        """Full success outcome when all complete."""
        mission = Mission(
            id="mission_001",
            name="Success Mission",
            description="Test",
            objectives=[
                MissionObjective(
                    id="obj_001",
                    description="Objective 1",
                    objective_type="escort",
                ),
                MissionObjective(
                    id="obj_002",
                    description="Objective 2",
                    objective_type="destroy",
                ),
            ],
        )
        state = MissionState(mission=mission)
        state = update_objective_status(state, "obj_001", "completed")
        state = update_objective_status(state, "obj_002", "completed")
        debrief = resolve_mission_outcome(state)
        assert debrief.outcome == "success"
        assert len(debrief.objectives_completed) == 2
        assert debrief.completion_score == 1.0

    def test_partial_success(self):
        """Partial outcome when some complete."""
        mission = Mission(
            id="mission_001",
            name="Partial Mission",
            description="Test",
            objectives=[
                MissionObjective(
                    id="obj_001",
                    description="Objective 1",
                    objective_type="escort",
                ),
                MissionObjective(
                    id="obj_002",
                    description="Objective 2",
                    objective_type="destroy",
                ),
            ],
        )
        state = MissionState(mission=mission)
        state = update_objective_status(state, "obj_001", "completed")
        debrief = resolve_mission_outcome(state)
        assert debrief.outcome == "partial"
        assert debrief.completion_score == 0.5

    def test_failure(self):
        """Failure outcome when some attempted but failed."""
        mission = Mission(
            id="mission_001",
            name="Failed Mission",
            description="Test",
            objectives=[
                MissionObjective(
                    id="obj_001",
                    description="Objective 1",
                    objective_type="escort",
                ),
                MissionObjective(
                    id="obj_002",
                    description="Objective 2",
                    objective_type="destroy",
                ),
            ],
        )
        state = MissionState(mission=mission)
        state = update_objective_status(state, "obj_001", "failed")
        debrief = resolve_mission_outcome(state)
        assert debrief.outcome == "failure"

    def test_catastrophic(self):
        """Catastrophic outcome when nothing done."""
        mission = Mission(
            id="mission_001",
            name="Catastrophic Mission",
            description="Test",
            objectives=[
                MissionObjective(
                    id="obj_001",
                    description="Objective 1",
                    objective_type="escort",
                ),
                MissionObjective(
                    id="obj_002",
                    description="Objective 2",
                    objective_type="destroy",
                ),
            ],
        )
        state = MissionState(mission=mission)
        debrief = resolve_mission_outcome(state)
        assert debrief.outcome == "catastrophic"
        assert debrief.completion_score == 0.0

    def test_experience_granted(self):
        """Experience is granted per PR2 rules."""
        mission = Mission(
            id="mission_001",
            name="Test Mission",
            description="Test",
            objectives=[
                MissionObjective(
                    id="obj_001",
                    description="Objective 1",
                    objective_type="escort",
                ),
            ],
        )
        state = MissionState(mission=mission)
        state = update_objective_status(state, "obj_001", "completed")
        debrief = resolve_mission_outcome(state)
        assert debrief.experience_gained == 1


class TestStartMission:
    """Tests for start_mission function."""

    def test_start_mission(self):
        """Start a mission creates correct initial state."""
        mission = Mission(
            id="mission_001",
            name="Test Mission",
            description="Test",
            objectives=[
                MissionObjective(
                    id="obj_001",
                    description="Objective 1",
                    objective_type="escort",
                ),
            ],
        )
        state = start_mission(mission)
        assert state.mission == mission
        assert state.current_phase == "briefing"
        assert state.current_turn == 1
        assert state.completion_score == 0.0
        assert state.is_victory is None


class TestAdvanceMissionPhase:
    """Tests for advance_mission_phase function."""

    def test_advance_to_active(self):
        """Advance from briefing to active."""
        mission = Mission(
            id="mission_001",
            name="Test Mission",
            description="Test",
        )
        state = MissionState(mission=mission)
        new_state = advance_mission_phase(state, "active")
        assert new_state.current_phase == "active"

    def test_advance_to_debrief(self):
        """Advance from active to debrief."""
        mission = Mission(
            id="mission_001",
            name="Test Mission",
            description="Test",
        )
        state = MissionState(mission=mission, current_phase="active")
        new_state = advance_mission_phase(state, "debrief")
        assert new_state.current_phase == "debrief"

    def test_cannot_regress(self):
        """Cannot regress to earlier phase."""
        mission = Mission(
            id="mission_001",
            name="Test Mission",
            description="Test",
        )
        state = MissionState(mission=mission, current_phase="active")
        with pytest.raises(ValueError, match="Cannot regress"):
            advance_mission_phase(state, "briefing")


class TestAdvanceMissionTurn:
    """Tests for advance_mission_turn function."""

    def test_advance_turn(self):
        """Advance mission turn during active phase."""
        mission = Mission(
            id="mission_001",
            name="Test Mission",
            description="Test",
        )
        state = MissionState(mission=mission, current_phase="active")
        new_state = advance_mission_turn(state)
        assert new_state.current_turn == 2

    def test_cannot_advance_in_briefing(self):
        """Cannot advance turns during briefing."""
        mission = Mission(
            id="mission_001",
            name="Test Mission",
            description="Test",
        )
        state = MissionState(mission=mission, current_phase="briefing")
        with pytest.raises(ValueError, match="active phase"):
            advance_mission_turn(state)


# ============================================================================
# SITREP Mission Type Template Tests (Priority 48)
# ============================================================================


class TestSitrepZoneType:
    """Tests for SitrepZoneType enum."""

    def test_all_zone_types_available(self):
        """Verify all zone types are available."""
        expected_types = {"deployment", "extraction", "objective", "ingress"}
        available_types = set(SitrepZoneType.__args__)
        assert expected_types == available_types


class TestSitrepDeploymentType:
    """Tests for SitrepDeploymentType enum."""

    def test_all_deployment_types_available(self):
        """Verify all deployment types are available."""
        expected_types = {"players_first", "enemies_first", "roll_off"}
        available_types = set(SitrepDeploymentType.__args__)
        assert expected_types == available_types


class TestSitrepReservePattern:
    """Tests for SitrepReservePattern enum."""

    def test_all_reserve_patterns_available(self):
        """Verify all reserve patterns are available."""
        expected_patterns = {"none", "half", "normal", "double", "increasing"}
        available_patterns = set(SitrepReservePattern.__args__)
        assert expected_patterns == available_patterns


class TestVictoryConditionType:
    """Tests for VictoryConditionType enum."""

    def test_all_condition_types_available(self):
        """Verify all victory condition types are available."""
        expected_types = {
            "extract_objective",
            "control_zones",
            "score_above_threshold",
            "outnumber_enemies",
            "control_real_objective",
            "survive_rounds",
        }
        available_types = set(VictoryConditionType.__args__)
        assert expected_types == available_types


class TestSitrepZone:
    """Tests for SitrepZone model."""

    def test_create_basic_zone(self):
        """Create a basic zone."""
        zone = SitrepZone(zone_type="deployment")
        assert zone.zone_type == "deployment"
        assert zone.width is None
        assert zone.height is None
        assert zone.location is None
        assert zone.terrain_notes is None

    def test_create_zone_with_dimensions(self):
        """Create a zone with width and height."""
        zone = SitrepZone(
            zone_type="objective",
            width=4,
            height=4,
            location="quadrant_nw",
        )
        assert zone.width == 4
        assert zone.height == 4
        assert zone.location == "quadrant_nw"

    def test_create_zone_with_terrain_notes(self):
        """Create a zone with terrain notes."""
        zone = SitrepZone(
            zone_type="ingress",
            location="flank",
            terrain_notes="hard cover",
        )
        assert zone.terrain_notes == "hard cover"

    def test_zone_type_required(self):
        """Zone type is required."""
        zone = SitrepZone(zone_type="deployment", width=4)
        assert zone.zone_type == "deployment"
        assert zone.width == 4


class TestVictoryCondition:
    """Tests for VictoryCondition model."""

    def test_create_basic_condition(self):
        """Create a basic victory condition."""
        condition = VictoryCondition(
            condition_type="extract_objective",
            description="Extract the objective",
        )
        assert condition.condition_type == "extract_objective"
        assert condition.description == "Extract the objective"
        assert condition.threshold is None

    def test_create_condition_with_threshold(self):
        """Create a condition with threshold."""
        condition = VictoryCondition(
            condition_type="control_zones",
            threshold=3,
            description="Control 3 zones",
        )
        assert condition.threshold == 3

    def test_condition_type_required(self):
        """Condition type is required."""
        condition = VictoryCondition(
            condition_type="extract_objective",
            description="Extract the objective",
            threshold=5,
        )
        assert condition.condition_type == "extract_objective"
        assert condition.threshold == 5

    def test_description_required(self):
        """Description is required."""
        condition = VictoryCondition(
            condition_type="extract_objective",
            description="Test description",
        )
        assert condition.description == "Test description"


class TestSitrepTemplate:
    """Tests for SitrepTemplate model."""

    def test_create_basic_template(self):
        """Create a basic template."""
        template = SitrepTemplate(
            sitrep_type="escort",
            name="ESCORT",
            description="Test escort mission",
        )
        assert template.sitrep_type == "escort"
        assert template.name == "ESCORT"
        assert template.duration_rounds == 6
        assert template.deployment_type == "players_first"

    def test_template_with_zones(self):
        """Create a template with zones."""
        template = SitrepTemplate(
            sitrep_type="control",
            name="CONTROL",
            description="Test control mission",
            objective_zones=[
                SitrepZone(zone_type="objective", width=4, height=4, location="nw"),
                SitrepZone(zone_type="objective", width=4, height=4, location="ne"),
            ],
        )
        assert len(template.objective_zones) == 2

    def test_template_with_victory_conditions(self):
        """Create a template with victory conditions."""
        template = SitrepTemplate(
            sitrep_type="hold_out",
            name="HOLDOUT",
            description="Test holdout mission",
            victory_conditions=[
                VictoryCondition(
                    condition_type="score_above_threshold",
                    threshold=1,
                    description="Maintain score >= 1",
                )
            ],
        )
        assert len(template.victory_conditions) == 1
        assert template.victory_conditions[0].threshold == 1

    def test_template_with_special_rules(self):
        """Create a template with special rules."""
        template = SitrepTemplate(
            sitrep_type="extract",
            name="EXTRACT",
            description="Test extract mission",
            special_rules=["Round 1: no enemies", "Free action to extract"],
        )
        assert len(template.special_rules) == 2

    def test_template_defaults(self):
        """Test default values are applied."""
        template = SitrepTemplate(
            sitrep_type="gauntlet",
            name="GAUNTLET",
            description="Test",
        )
        assert template.duration_rounds == 6
        assert template.deployment_type == "players_first"
        assert template.reserve_pattern == "normal"
        assert template.deployment_zones == []
        assert template.victory_conditions == []
        assert template.special_rules == []

    def test_sitrep_type_is_literal(self):
        """Sitrep type accepts valid literal values."""
        for sitrep_type in [
            "escort",
            "control",
            "extract",
            "hold_out",
            "gauntlet",
            "recon",
        ]:
            template = SitrepTemplate(
                sitrep_type=sitrep_type,  # type: ignore[arg-type]
                name=sitrep_type.upper(),
                description="Test",
            )
            assert template.sitrep_type == sitrep_type


class TestPreBuiltTemplates:
    """Tests for pre-built SITREP templates."""

    def test_escort_template_exists(self):
        """ESCORT template exists with correct values."""
        assert ESCORT_TEMPLATE.sitrep_type == "escort"
        assert ESCORT_TEMPLATE.name == "ESCORT"
        assert ESCORT_TEMPLATE.duration_rounds == 6
        assert ESCORT_TEMPLATE.deployment_type == "players_first"
        assert len(ESCORT_TEMPLATE.victory_conditions) == 1
        assert (
            ESCORT_TEMPLATE.victory_conditions[0].condition_type == "extract_objective"
        )

    def test_control_template_exists(self):
        """CONTROL template exists with correct values."""
        assert CONTROL_TEMPLATE.sitrep_type == "control"
        assert CONTROL_TEMPLATE.name == "CONTROL"
        assert CONTROL_TEMPLATE.duration_rounds == 6
        assert CONTROL_TEMPLATE.deployment_type == "roll_off"
        assert len(CONTROL_TEMPLATE.objective_zones) == 4

    def test_extract_template_exists(self):
        """EXTRACT template exists with correct values."""
        assert EXTRACT_TEMPLATE.sitrep_type == "extract"
        assert EXTRACT_TEMPLATE.name == "EXTRACT"
        assert EXTRACT_TEMPLATE.duration_rounds == 8
        assert EXTRACT_TEMPLATE.reserve_pattern == "double"
        assert EXTRACT_TEMPLATE.reserve_per_round == 2

    def test_holdout_template_exists(self):
        """HOLDOUT template exists with correct values."""
        assert HOLDOUT_TEMPLATE.sitrep_type == "hold_out"
        assert HOLDOUT_TEMPLATE.name == "HOLDOUT"
        assert HOLDOUT_TEMPLATE.reserve_pattern == "half"
        assert len(HOLDOUT_TEMPLATE.victory_conditions) == 1
        assert (
            HOLDOUT_TEMPLATE.victory_conditions[0].condition_type
            == "score_above_threshold"
        )
        assert HOLDOUT_TEMPLATE.victory_conditions[0].threshold == 1

    def test_gauntlet_template_exists(self):
        """GAUNTLET template exists with correct values."""
        assert GAUNTLET_TEMPLATE.sitrep_type == "gauntlet"
        assert GAUNTLET_TEMPLATE.name == "GAUNTLET"
        assert GAUNTLET_TEMPLATE.deployment_type == "enemies_first"

    def test_recon_template_exists(self):
        """RECON template exists with correct values."""
        assert RECON_TEMPLATE.sitrep_type == "recon"
        assert RECON_TEMPLATE.name == "RECON"
        assert len(RECON_TEMPLATE.objective_zones) == 4
        assert (
            RECON_TEMPLATE.victory_conditions[0].condition_type
            == "control_real_objective"
        )


class TestSitrepTemplateRegistry:
    """Tests for SITREP template registry."""

    def test_all_templates_registered(self):
        """All templates are in the registry."""
        expected = {"escort", "control", "extract", "hold_out", "gauntlet", "recon"}
        assert set(SITREP_TEMPLATES.keys()) == expected

    def test_get_sitrep_template_found(self):
        """get_sitrep_template returns template when found."""
        template = get_sitrep_template("escort")
        assert template is not None
        assert template.sitrep_type == "escort"

    def test_get_sitrep_template_not_found(self):
        """get_sitrep_template returns None when not found."""
        template = get_sitrep_template("invalid_type")
        assert template is None


class TestCreateMissionFromTemplate:
    """Tests for create_mission_from_template function."""

    def test_create_mission_from_escort_template(self):
        """Create a mission from ESCORT template."""
        mission = create_mission_from_template(
            ESCORT_TEMPLATE,
            mission_id="test_escort",
            name="VIP Escort",
            description="Escort the VIP to safety",
        )
        assert mission.id == "test_escort"
        assert mission.name == "VIP Escort"
        assert mission.scenario_type == "escort"
        assert mission.time_limit == 6
        assert len(mission.objectives) == 1

    def test_create_mission_from_control_template(self):
        """Create a mission from CONTROL template with 4 objectives."""
        mission = create_mission_from_template(
            CONTROL_TEMPLATE,
            mission_id="test_control",
            name="Zone Control",
            description="Control the zones",
        )
        assert mission.scenario_type == "control"
        assert len(mission.objectives) == 4

    def test_create_mission_with_custom_duration(self):
        """Create a mission with custom duration override."""
        mission = create_mission_from_template(
            ESCORT_TEMPLATE,
            mission_id="test",
            name="Test",
            description="Test",
            duration_rounds=8,
        )
        assert mission.time_limit == 8

    def test_create_mission_from_template_with_empty_zones(self):
        """Create a mission from template with no objective zones."""
        template = SitrepTemplate(
            sitrep_type="escort",  # type: ignore[arg-type]
            name="CUSTOM",
            description="Custom mission",
            objective_zones=[],
        )
        mission = create_mission_from_template(
            template,
            mission_id="test",
            name="Test",
            description="Test",
        )
        assert len(mission.objectives) == 0

    def test_created_mission_is_valid(self):
        """Created mission passes validation."""
        mission = create_mission_from_template(
            ESCORT_TEMPLATE,
            mission_id="test_mission",
            name="Test Mission",
            description="A test mission",
        )
        assert mission.id == "test_mission"
        assert mission.objectives is not None
        assert isinstance(mission.objectives, list)
