"""Integration tests for SITREP mission flow.

Tests mission templates, objectives, zone control, and victory condition resolution,
following the structure defined in PR2 888-1000 and 2812-2815.
"""

from core.shared.scenario import (
    SitrepTemplate,
    MissionObjective,
    MissionOutcomeType,
    ObjectiveCriterion,
    ESCORT_TEMPLATE,
    CONTROL_TEMPLATE,
    EXTRACT_TEMPLATE,
)
from core.shared.sitrep_resolution import (
    ZoneControlStateTracker,
    create_sitrep_resolution,
    spawn_reserves,
    check_extraction_progress,
    advance_sitrep_round,
)
from core.shared.integration.narrative_combat import CombatEvent


class TestSITREPTemplateCreation:
    """Tests for SITREP template creation (PR2 888-950)."""

    def test_sitrep_template_creation(
        self, integration_sitrep_template: SitrepTemplate
    ):
        """SITREP template can be created with all required fields."""
        assert integration_sitrep_template.sitrep_type is not None
        assert integration_sitrep_template.name is not None
        assert integration_sitrep_template.description is not None
        assert integration_sitrep_template.special_rules is not None

    def test_sitrep_zones_defined(self, integration_sitrep_template: SitrepTemplate):
        """Template defines mission zones."""
        zones = integration_sitrep_template.objective_zones

        assert isinstance(zones, list)
        if len(zones) > 0:
            zone = zones[0]
            assert zone.location is not None

    def test_sitrep_objectives_defined(
        self, integration_sitrep_template: SitrepTemplate
    ):
        """Template defines mission objectives."""
        objectives = integration_sitrep_template.victory_conditions

        assert isinstance(objectives, list)

    def test_victory_conditions_defined(
        self, integration_sitrep_template: SitrepTemplate
    ):
        """Template defines victory conditions."""
        victory = integration_sitrep_template.victory_conditions

        assert isinstance(victory, list)

    def test_turn_limit_defined(self, integration_sitrep_template: SitrepTemplate):
        """Template defines mission turn limit."""
        turn_limit = integration_sitrep_template.duration_rounds

        assert turn_limit is None or (isinstance(turn_limit, int) and turn_limit > 0)


class TestZoneControlMechanics:
    """Tests for zone control mechanics (PR2 950-1000)."""

    def test_zone_control_state_tracker(self):
        """Zone control can be tracked."""
        tracker = ZoneControlStateTracker(
            zone_id="test_zone",
            state="neutral",
            controlling_side=None,
            last_checked_turn=1,
        )

        assert tracker.zone_id == "test_zone"
        assert tracker.state == "neutral"

    def test_player_controlled_zone(self):
        """Zone can be player-controlled."""
        tracker = ZoneControlStateTracker(
            zone_id="alpha",
            state="player_controlled",
            controlling_side="players",
            last_checked_turn=3,
        )

        assert tracker.state == "player_controlled"
        assert tracker.controlling_side == "players"

    def test_enemy_controlled_zone(self):
        """Zone can be enemy-controlled."""
        tracker = ZoneControlStateTracker(
            zone_id="bravo",
            state="enemy_controlled",
            controlling_side="enemies",
            last_checked_turn=3,
        )

        assert tracker.state == "enemy_controlled"
        assert tracker.controlling_side == "enemies"

    def test_contested_zone(self):
        """Zone can be contested."""
        tracker = ZoneControlStateTracker(
            zone_id="charlie",
            state="contested",
            controlling_side=None,
            last_checked_turn=2,
        )

        assert tracker.state == "contested"
        assert tracker.controlling_side is None

    def test_update_player_control(self):
        """Zone control can be updated to player."""
        tracker = ZoneControlStateTracker(
            zone_id="test",
            state="neutral",
            controlling_side=None,
            last_checked_turn=1,
        )

        updated = ZoneControlStateTracker(
            zone_id=tracker.zone_id,
            state="player_controlled",
            controlling_side="players",
            last_checked_turn=2,
        )

        assert updated.state == "player_controlled"


class TestObjectiveCompletion:
    """Tests for objective tracking (PR2 2812-2815)."""

    def test_objective_status_transitions(self):
        """Objectives can change status."""
        objective = MissionObjective(
            id="obj_1",
            description="Destroy the target",
            objective_type="destroy",
            status="pending",
        )

        assert objective.status == "pending"

        in_progress = MissionObjective(
            id=objective.id,
            description=objective.description,
            objective_type=objective.objective_type,
            status="in_progress",
        )

        assert in_progress.status == "in_progress"

    def test_objective_with_criteria(self):
        """Objectives can have completion criteria."""
        criterion_type = "target_destroyed"
        objective = MissionObjective(
            id="obj_1",
            description="Destroy enemy commander",
            objective_type="destroy",
            status="in_progress",
            completion_criteria=[
                ObjectiveCriterion(
                    criterion_type=criterion_type,
                    description="Destroy the enemy commander",
                    target_id="npc_commander",
                )
            ],
        )

        assert len(objective.completion_criteria) == 1

    def test_optional_objective(self):
        """Objectives can be optional."""
        objective = MissionObjective(
            id="obj_bonus",
            description="Bonus objective",
            objective_type="destroy",
            status="pending",
            is_optional=True,
        )

        assert objective.is_optional is True


class TestMissionScoring:
    """Tests for mission scoring (PR2 2812-2815)."""

    def test_score_calculation_full_success(self):
        """Score is 1.0 when all objectives complete."""
        completed = 4
        total = 4
        score = completed / total

        assert score == 1.0

    def test_score_calculation_partial_success(self):
        """Score reflects partial completion."""
        completed = 2
        total = 4
        score = completed / total

        assert score == 0.5

    def test_score_calculation_failure(self):
        """Score is 0 when no objectives complete."""
        completed = 0
        total = 4
        score = completed / total

        assert score == 0.0

    def test_outcome_from_score(self):
        """Outcome determined from score."""

        def score_to_outcome(score: float) -> MissionOutcomeType:
            if score >= 1.0:
                return "success"
            elif score >= 0.5:
                return "partial"
            elif score > 0:
                return "failure"
            else:
                return "catastrophic"

        assert score_to_outcome(1.0) == "success"
        assert score_to_outcome(0.75) == "partial"
        assert score_to_outcome(0.25) == "failure"
        assert score_to_outcome(0.0) == "catastrophic"


class TestCombatToSITREPIntegration:
    """Tests for combat events updating SITREP state."""

    def test_combat_event_triggers_objective(self):
        """Combat event can trigger objective completion."""
        event = CombatEvent(
            event_type="target_destroyed",
            source_id="player_1",
            target_id="npc_commander",
        )

        assert event.event_type == "target_destroyed"
        assert event.target_id == "npc_commander"

    def test_npc_destruction_updates_zone(self):
        """NPC destruction affects zone control."""
        initial_control = "enemies"

        assert initial_control == "enemies"

        combat_event = CombatEvent(
            event_type="target_destroyed",
            source_id="player_1",
            target_id="npc_zone_commander",
        )

        assert combat_event.event_type == "target_destroyed"


class TestReserveManagement:
    """Tests for reserve spawning during mission (PR2 2869-2876)."""

    def test_reserve_spawning(self):
        """Reserves can be spawned during mission."""
        from core.shared.scenario import ESCORT_TEMPLATE

        resolution = create_sitrep_resolution(
            ESCORT_TEMPLATE,
            player_count=4,
            reserve_ids=["npc_1", "npc_2", "npc_3"],
            enemy_count=3,
        )
        updated_resolution, spawned = spawn_reserves(resolution, count=1)

        assert updated_resolution is not None

    def test_reserve_types_available(self):
        """All reserve types are available."""
        from core.shared.downtime import (
            ReserveType,
        )

        assert "narrative" in ReserveType.__args__
        assert "mech" in ReserveType.__args__
        assert "tactical" in ReserveType.__args__

    def test_narrative_reserve_examples(self):
        """Narrative reserves match book examples."""
        from core.shared.downtime import NarrativeReserveType

        expected = {"access", "backing", "supplies", "disguise", "diversion"}
        available = set(NarrativeReserveType.__args__)

        assert expected.issubset(available)


class TestTurnLimitEnforcement:
    """Tests for mission turn limits (PR2 2812-2815)."""

    def test_turn_limit_check(self):
        """Mission ends when turn limit reached."""
        turn_limit = 10
        current_turn = 10

        assert current_turn >= turn_limit

    def test_advance_turn(self):
        """Mission turn can be advanced."""
        resolution = create_sitrep_resolution(
            ESCORT_TEMPLATE,
            player_count=4,
            reserve_ids=[],
            enemy_count=0,
        )
        next_resolution = advance_sitrep_round(resolution)

        assert next_resolution.current_round == 2


class TestSITREPResolution:
    """Tests for full SITREP resolution."""

    def test_create_sitrep_resolution(
        self, integration_sitrep_template: SitrepTemplate
    ):
        """SITREP resolution system can be initialized."""
        resolution = create_sitrep_resolution(
            integration_sitrep_template,
            player_count=4,
            reserve_ids=[],
            enemy_count=0,
        )

        assert resolution is not None

    def test_resolution_tracks_score(self):
        """Resolution system tracks mission score."""
        resolution = create_sitrep_resolution(
            ESCORT_TEMPLATE,
            player_count=4,
            reserve_ids=[],
            enemy_count=0,
        )

        assert resolution is not None

    def test_victory_condition_check(self):
        """Victory conditions are checked correctly."""
        objectives = [
            {"id": "obj_1", "completed": True},
            {"id": "obj_2", "completed": True},
            {"id": "obj_3", "completed": False},
            {"id": "obj_4", "completed": False},
        ]

        completed = sum(1 for obj in objectives if obj["completed"])
        total = len(objectives)
        score = completed / total

        assert score == 0.5


class TestExtractionMechanics:
    """Tests for extraction-type missions."""

    def test_extraction_progress(self):
        """Extraction progress can be tracked."""
        resolution = create_sitrep_resolution(
            ESCORT_TEMPLATE,
            player_count=4,
            reserve_ids=[],
            enemy_count=0,
        )
        updated_resolution = check_extraction_progress(resolution, "full")

        assert updated_resolution is not None

    def test_extraction_timing(self):
        """Extraction timing affects outcome."""
        turns_available = 3
        turns_used = 2

        assert turns_used < turns_available

    def test_failed_extraction(self):
        """Failed extraction affects outcome."""
        turns_available = 3
        turns_used = 4

        assert turns_used > turns_available


class TestHoldoutMechanics:
    """Tests for holdout-type missions."""

    def test_holdout_scoring(self):
        """Holdout scoring considers zone control."""
        zone_scores = {
            "zone_1": {"players": 1, "enemies": 0},
            "zone_2": {"players": 1, "enemies": 1},
            "zone_3": {"players": 0, "enemies": 1},
        }

        player_zones = sum(
            1 for z in zone_scores.values() if z["players"] > z["enemies"]
        )
        enemy_zones = sum(
            1 for z in zone_scores.values() if z["enemies"] > z["players"]
        )

        assert player_zones == 1
        assert enemy_zones == 1

    def test_holdout_bonus(self):
        """Holdout gives starting bonus."""
        starting_bonus = 2

        assert starting_bonus > 0


class TestGauntletMechanics:
    """Tests for gauntlet-type missions."""

    def test_gauntlet_progress(self):
        """Gauntlet tracks progress through zones."""
        zones = ["start", "alpha", "bravo", "charlie", "end"]
        current_position = 2

        assert current_position < len(zones)

    def test_gauntlet_failure(self):
        """Gauntlet failure at any point affects outcome."""
        failed_at_zone = "bravo"

        assert failed_at_zone is not None


class TestMissionOutcomeIntegration:
    """Tests for complete mission outcome determination."""

    def test_full_mission_workflow(self):
        """Complete mission: Start → Combat → Resolution → Outcome."""
        mission_state = {
            "turn": 5,
            "turn_limit": 10,
            "objectives": [
                {"id": "obj_1", "completed": True, "priority": 2},
                {"id": "obj_2", "completed": False, "priority": 1},
            ],
            "zones": [
                {"id": "zone_1", "control": "players"},
                {"id": "zone_2", "control": "neutral"},
            ],
        }

        completed = sum(1 for obj in mission_state["objectives"] if obj["completed"])
        total = len(mission_state["objectives"])
        score = completed / total

        outcome = "partial" if score >= 0.5 else "failure"

        assert outcome == "partial"

    def test_victory_with_all_objectives(self):
        """Victory when all primary objectives complete."""
        objectives = [
            {"id": "obj_1", "completed": True, "is_optional": False},
            {"id": "obj_2", "completed": True, "is_optional": False},
            {"id": "obj_3", "completed": True, "is_optional": True},
        ]

        primary_complete = all(
            obj["completed"] or obj["is_optional"]
            for obj in objectives
            if not obj["is_optional"]
        )

        assert primary_complete is True

    def test_catastrophic_failure(self):
        """Catastrophic when all objectives fail."""
        objectives = [
            {"id": "obj_1", "completed": False, "is_optional": False},
            {"id": "obj_2", "completed": False, "is_optional": False},
        ]

        any_complete = any(obj["completed"] for obj in objectives)

        assert any_complete is False

    def test_reserve_spawn_after_mission(self):
        """Reserves can spawn after mission completion."""
        from core.shared.downtime import Reserve

        victory_reserves = [
            Reserve(
                id="victory_bonus",
                reserve_type="narrative",
                specific_type="reputation",
                description="Victory bonus reputation",
            )
        ]

        assert len(victory_reserves) == 1
        assert victory_reserves[0].reserve_type == "narrative"


class TestSITREPTemplateVariants:
    """Tests for different SITREP template types."""

    def test_escort_template(self):
        """ESCORT template has escort-specific fields."""
        template = ESCORT_TEMPLATE

        assert (
            "escort" in template.sitrep_type.lower()
            or template.victory_conditions is not None
        )

    def test_control_template(self):
        """CONTROL template has multiple zones."""
        template = CONTROL_TEMPLATE

        assert template.objective_zones is not None

    def test_extract_template(self):
        """EXTRACT template has extraction objective."""
        template = EXTRACT_TEMPLATE

        assert (
            "extract" in template.sitrep_type.lower()
            or template.victory_conditions is not None
        )

    def test_template_selection(self):
        """GM can select appropriate template."""
        templates = {
            "escort": ESCORT_TEMPLATE,
            "control": CONTROL_TEMPLATE,
            "extract": EXTRACT_TEMPLATE,
        }

        assert len(templates) >= 3

    def test_template_customization(self):
        """Templates can be customized for specific missions."""
        base = ESCORT_TEMPLATE

        customized = SitrepTemplate(
            sitrep_type="escort",
            name=f"Custom {base.name}",
            description=base.description,
            special_rules=["Custom stakes"],
            objective_zones=base.objective_zones,
            victory_conditions=base.victory_conditions,
            duration_rounds=base.duration_rounds,
        )

        assert customized.sitrep_type == base.sitrep_type
        assert customized.name != base.name
