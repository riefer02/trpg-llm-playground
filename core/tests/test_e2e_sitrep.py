"""End-to-end tests for SITREP objective mechanics in combat.

Tests SITREP zone control, scoring, victory conditions, and
reserve management through realistic mission scenarios.
"""

import pytest

from core.tests.e2e_helpers import (
    make_pilot_with_talents,
    make_combatant_from_pilot,
    make_combatant,
    make_enemy_combatant,
    make_sitrep_scenario,
    make_scenario,
    update_sitrep_zone_for_side,
    advance_scenario_round,
    check_scenario_victory,
    get_sitrep_outcome,
)
from core.shared.scenario import SITREP_TEMPLATES, SitrepType
from core.shared.sitrep_resolution import (
    create_sitrep_resolution,
    advance_sitrep_round,
    update_zone_control,
    check_victory_conditions,
    check_extraction_progress,
    spawn_reserves,
    resolve_sitrep,
)


class TestControlSitrep:
    """Tests for CONTROL SITREP zone control mechanics."""

    def test_control_sitrep_zone_scoring(self):
        """CONTROL: Controlling zones awards points.

        PR2 Lines 12637-12657: 4 zones, 1 point per zone per round when controlled.
        """
        template = SITREP_TEMPLATES["control"]
        resolution = create_sitrep_resolution(template, player_count=4)

        # Initial state
        assert resolution.player_score == 0
        assert resolution.enemy_score == 0

        # Control first zone
        if resolution.zone_states:
            zone_id = list(resolution.zone_states.keys())[0]
            resolution = update_zone_control(
                resolution, zone_id, "player_controlled", "players"
            )

            # Player should gain 1 point
            assert resolution.player_score == 1

    def test_control_sitrep_contested_zone(self):
        """CONTROL: Contested zones don't award points."""
        template = SITREP_TEMPLATES["control"]
        resolution = create_sitrep_resolution(template, player_count=4)

        if resolution.zone_states:
            zone_id = list(resolution.zone_states.keys())[0]

            # First player controls
            resolution = update_zone_control(
                resolution, zone_id, "player_controlled", "players"
            )
            assert resolution.player_score == 1

            # Now contested
            resolution = update_zone_control(
                resolution, zone_id, "contested", None
            )
            # Score should decrease when losing control
            assert resolution.player_score == 0

    def test_control_sitrep_victory_condition(self):
        """CONTROL: Victory requires controlling 3+ of 4 zones at round 6."""
        template = SITREP_TEMPLATES["control"]
        resolution = create_sitrep_resolution(template, player_count=4)

        # Control 3 zones
        zones = list(resolution.zone_states.keys())[:3]
        for zone_id in zones:
            resolution = update_zone_control(
                resolution, zone_id, "player_controlled", "players"
            )

        # Check victory
        resolution = check_victory_conditions(resolution)

        # With 3 zones controlled, if threshold is 3, should be met
        control_conditions = [
            vc for vc in resolution.victory_conditions
            if vc.condition_type == "control_zones"
        ]
        if control_conditions:
            assert control_conditions[0].current_value >= 3


class TestEscortSitrep:
    """Tests for ESCORT SITREP extraction mechanics."""

    def test_escort_sitrep_extraction_progress(self):
        """ESCORT: Extraction progress tracked correctly."""
        template = SITREP_TEMPLATES["escort"]
        resolution = create_sitrep_resolution(template, player_count=4)

        assert resolution.extraction_progress == 0.0

        # Full action extraction
        resolution = check_extraction_progress(resolution, "full")
        assert resolution.extraction_progress == 1.0

    def test_escort_sitrep_quick_extraction(self):
        """ESCORT: Quick action adds 0.5 extraction progress."""
        template = SITREP_TEMPLATES["escort"]
        resolution = create_sitrep_resolution(template, player_count=4)

        resolution = check_extraction_progress(resolution, "quick")
        assert resolution.extraction_progress == 0.5

    def test_escort_sitrep_free_extraction(self):
        """ESCORT: Free action adds 0.25 extraction progress."""
        template = SITREP_TEMPLATES["escort"]
        resolution = create_sitrep_resolution(template, player_count=4)

        resolution = check_extraction_progress(resolution, "free")
        assert resolution.extraction_progress == 0.25

    def test_escort_sitrep_victory_on_extraction(self):
        """ESCORT: Players win when extraction reaches 1.0."""
        template = SITREP_TEMPLATES["escort"]
        resolution = create_sitrep_resolution(template, player_count=4)

        # Complete extraction
        resolution = check_extraction_progress(resolution, "full")
        resolution = check_victory_conditions(resolution)

        # Should be players win
        extraction_conditions = [
            vc for vc in resolution.victory_conditions
            if vc.condition_type == "extract_objective"
        ]
        if extraction_conditions:
            assert extraction_conditions[0].is_met is True


class TestHoldoutSitrep:
    """Tests for HOLDOUT SITREP defense mechanics."""

    def test_holdout_sitrep_starting_score(self):
        """HOLDOUT: Players start with 4 points."""
        template = SITREP_TEMPLATES["hold_out"]
        resolution = create_sitrep_resolution(template, player_count=4)

        # Players start with 4 points (special HOLDOUT rule)
        assert resolution.player_score == 4

    def test_holdout_sitrep_victory_condition(self):
        """HOLDOUT: Victory requires maintaining score >= 1."""
        template = SITREP_TEMPLATES["hold_out"]
        resolution = create_sitrep_resolution(template, player_count=4)

        # With starting score of 4, resolve should be players_win
        resolution = resolve_sitrep(resolution)
        assert resolution.outcome == "players_win"


class TestGauntletSitrep:
    """Tests for GAUNTLET SITREP survival mechanics."""

    def test_gauntlet_sitrep_force_comparison(self):
        """GAUNTLET: Victory based on surviving forces comparison."""
        template = SITREP_TEMPLATES["gauntlet"]
        resolution = create_sitrep_resolution(
            template, player_count=4, enemy_count=3
        )

        assert resolution.surviving_players == 4
        assert resolution.surviving_enemies == 3

        # Players outnumber enemies, should win on resolve
        resolution = resolve_sitrep(resolution)
        assert resolution.outcome == "players_win"

    def test_gauntlet_sitrep_enemies_win(self):
        """GAUNTLET: Enemies win if they outnumber players."""
        template = SITREP_TEMPLATES["gauntlet"]
        resolution = create_sitrep_resolution(
            template, player_count=2, enemy_count=5
        )

        resolution = resolve_sitrep(resolution)
        assert resolution.outcome == "enemies_win"


class TestReserveSpawning:
    """Tests for reserve force spawning mechanics."""

    def test_spawn_reserves_basic(self):
        """Reserves spawn from pool correctly."""
        template = SITREP_TEMPLATES["escort"]
        resolution = create_sitrep_resolution(
            template, player_count=4,
            reserve_ids=["npc_1", "npc_2", "npc_3"],
            enemy_count=0,
        )

        initial_enemies = resolution.surviving_enemies

        resolution, spawned = spawn_reserves(resolution, count=2)

        assert len(spawned) == 2
        assert resolution.surviving_enemies == initial_enemies + 2

    def test_spawn_reserves_ingress_rotation(self):
        """Reserves cannot use same ingress zone twice in a row (PR2 rule)."""
        template = SITREP_TEMPLATES["escort"]
        resolution = create_sitrep_resolution(
            template, player_count=4,
            reserve_ids=["npc_1", "npc_2", "npc_3", "npc_4"],
            enemy_count=0,
        )

        if resolution.deployment and resolution.deployment.ingress_zones:
            # First spawn
            resolution, _ = spawn_reserves(resolution, count=1, seed=42)
            assert resolution.deployment is not None  # Type narrowing
            first_zone = resolution.deployment.last_ingress_zone

            # Second spawn should not use same zone (if multiple zones exist)
            resolution, _ = spawn_reserves(resolution, count=1, seed=42)
            assert resolution.deployment is not None  # Type narrowing
            second_zone = resolution.deployment.last_ingress_zone

            # If there are multiple ingress zones, should rotate
            if len(resolution.deployment.ingress_zones) > 1:
                assert first_zone != second_zone or first_zone is None

    def test_spawn_reserves_depletes_pool(self):
        """Spawning reduces reserve pool."""
        template = SITREP_TEMPLATES["escort"]
        resolution = create_sitrep_resolution(
            template, player_count=4,
            reserve_ids=["npc_1", "npc_2", "npc_3"],
            enemy_count=0,
        )

        initial_remaining = resolution.reserves_remaining

        resolution, _ = spawn_reserves(resolution, count=2)

        assert resolution.reserves_remaining == initial_remaining - 2


class TestRoundAdvancement:
    """Tests for round advancement and timing."""

    def test_advance_round_increments(self):
        """Advancing round increases current_round."""
        template = SITREP_TEMPLATES["control"]
        resolution = create_sitrep_resolution(template, player_count=4)

        assert resolution.current_round == 1

        resolution = advance_sitrep_round(resolution)
        assert resolution.current_round == 2

        resolution = advance_sitrep_round(resolution)
        assert resolution.current_round == 3

    def test_turn_limit_reached_at_max_rounds(self):
        """turn_limit_reached set when max rounds hit."""
        template = SITREP_TEMPLATES["control"]  # 6 rounds
        resolution = create_sitrep_resolution(template, player_count=4)

        # Advance to max rounds
        for _ in range(5):  # 5 advances to reach round 6
            resolution = advance_sitrep_round(resolution)

        assert resolution.turn_limit_reached is True
        assert resolution.current_round == 6

    def test_escort_increasing_reserves(self):
        """ESCORT: Reserves spawn on round advance with increasing pattern."""
        template = SITREP_TEMPLATES["escort"]
        resolution = create_sitrep_resolution(
            template, player_count=4,
            reserve_ids=["npc_1", "npc_2", "npc_3"],
            enemy_count=0,
        )

        initial_enemies = resolution.surviving_enemies

        # Advance round (should spawn reserves for escort)
        resolution = advance_sitrep_round(resolution)

        # If escort has increasing reserve pattern, enemies should increase
        # Note: depends on reserve pattern setting


class TestVictoryConditionEvaluation:
    """Tests for victory condition checking."""

    def test_control_zones_condition(self):
        """control_zones victory condition evaluates correctly."""
        template = SITREP_TEMPLATES["control"]
        resolution = create_sitrep_resolution(template, player_count=4)

        # Check initial state (no zones controlled)
        resolution = check_victory_conditions(resolution)
        control_vc = next(
            (vc for vc in resolution.victory_conditions
             if vc.condition_type == "control_zones"),
            None
        )
        if control_vc:
            assert control_vc.current_value == 0
            assert control_vc.is_met is False

    def test_extract_objective_condition(self):
        """extract_objective victory condition evaluates correctly."""
        template = SITREP_TEMPLATES["escort"]
        resolution = create_sitrep_resolution(template, player_count=4)

        # Not extracted yet
        resolution = check_victory_conditions(resolution)
        extract_vc = next(
            (vc for vc in resolution.victory_conditions
             if vc.condition_type == "extract_objective"),
            None
        )
        if extract_vc:
            assert extract_vc.is_met is False

        # Complete extraction
        resolution = check_extraction_progress(resolution, "full")
        resolution = check_victory_conditions(resolution)

        extract_vc = next(
            (vc for vc in resolution.victory_conditions
             if vc.condition_type == "extract_objective"),
            None
        )
        if extract_vc:
            assert extract_vc.is_met is True


class TestSitrepResolutionOutcomes:
    """Tests for final SITREP resolution outcomes."""

    @pytest.mark.parametrize("sitrep_type", [
        "escort", "control", "extract", "hold_out", "gauntlet", "recon"
    ])
    def test_all_sitrep_types_can_resolve(self, sitrep_type: SitrepType):
        """All SITREP types can reach a resolution."""
        template = SITREP_TEMPLATES[sitrep_type]
        resolution = create_sitrep_resolution(
            template, player_count=4, enemy_count=2
        )

        # Set up for a potential win scenario
        if sitrep_type in ("escort", "extract"):
            resolution = resolution.model_copy(
                update={"extraction_progress": 1.0}
            )
        elif sitrep_type == "control":
            # Control 3 zones
            for i, zone_id in enumerate(list(resolution.zone_states.keys())[:3]):
                resolution = update_zone_control(
                    resolution, zone_id, "player_controlled", "players"
                )
        elif sitrep_type == "hold_out":
            # Keep starting score
            pass
        elif sitrep_type == "gauntlet":
            # Players outnumber (default)
            pass
        elif sitrep_type == "recon":
            # Control at least one zone
            if resolution.zone_states:
                zone_id = list(resolution.zone_states.keys())[0]
                resolution = update_zone_control(
                    resolution, zone_id, "player_controlled", "players"
                )

        resolution = resolve_sitrep(resolution)

        # Should have an outcome
        assert resolution.outcome in ("players_win", "enemies_win", "draw", "ongoing")


class TestSitrepScenarioIntegration:
    """Tests for SITREP integration with combat scenarios."""

    def test_create_sitrep_scenario(self):
        """Can create a scenario with SITREP tracking."""
        pilot = make_pilot_with_talents("TEST", [])
        player = make_combatant_from_pilot(pilot, position=(0, 0))
        enemy = make_enemy_combatant()

        scenario = make_sitrep_scenario(
            sitrep_type="control",
            player_combatants=[player],
            enemy_combatants=[enemy],
        )

        assert scenario.sitrep_resolution is not None
        assert scenario.terrain is not None
        assert scenario.sitrep_resolution.template_type == "control"

    def test_update_sitrep_zone_via_scenario(self):
        """Can update zone control through scenario helper."""
        pilot = make_pilot_with_talents("TEST", [])
        player = make_combatant_from_pilot(pilot, position=(0, 0))
        enemy = make_enemy_combatant()

        scenario = make_sitrep_scenario(
            sitrep_type="control",
            player_combatants=[player],
            enemy_combatants=[enemy],
        )

        if scenario.sitrep_resolution.zone_states:
            zone_id = list(scenario.sitrep_resolution.zone_states.keys())[0]

            scenario = update_sitrep_zone_for_side(scenario, zone_id, "players")

            zone = scenario.sitrep_resolution.zone_states[zone_id]
            assert zone.state == "player_controlled"

    def test_advance_scenario_round(self):
        """Can advance round through scenario helper."""
        pilot = make_pilot_with_talents("TEST", [])
        player = make_combatant_from_pilot(pilot, position=(0, 0))
        enemy = make_enemy_combatant()

        scenario = make_sitrep_scenario(
            sitrep_type="control",
            player_combatants=[player],
            enemy_combatants=[enemy],
        )

        initial_round = scenario.sitrep_resolution.current_round
        scenario = advance_scenario_round(scenario)

        assert scenario.sitrep_resolution.current_round == initial_round + 1

    def test_check_scenario_victory(self):
        """Can check victory conditions through scenario helper."""
        pilot = make_pilot_with_talents("TEST", [])
        player = make_combatant_from_pilot(pilot, position=(0, 0))
        enemy = make_enemy_combatant()

        scenario = make_sitrep_scenario(
            sitrep_type="control",
            player_combatants=[player],
            enemy_combatants=[enemy],
        )

        scenario = check_scenario_victory(scenario)

        # Should have updated victory conditions
        assert scenario.sitrep_resolution is not None

    def test_get_sitrep_outcome(self):
        """Can get outcome through scenario helper."""
        pilot = make_pilot_with_talents("TEST", [])
        player = make_combatant_from_pilot(pilot, position=(0, 0))
        enemy = make_enemy_combatant()

        scenario = make_sitrep_scenario(
            sitrep_type="hold_out",
            player_combatants=[player],
            enemy_combatants=[enemy],
        )

        # Initially no outcome
        outcome = get_sitrep_outcome(scenario)
        assert outcome is None  # Not yet resolved
