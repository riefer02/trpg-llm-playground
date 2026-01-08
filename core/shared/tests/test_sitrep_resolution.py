"""Tests for enhanced SITREP resolution mechanics."""

import pytest
from core.shared.sitrep_resolution import (
    ZoneControlStateTracker,
    SitrepVictoryCondition,
    SitrepDeployment,
    SitrepResolution,
    create_sitrep_resolution,
    spawn_reserves,
    update_zone_control,
    check_extraction_progress,
    check_victory_conditions,
    advance_sitrep_round,
    resolve_sitrep,
)
from core.shared.scenario import (
    SitrepZone,
    VictoryCondition,
    SitrepTemplate,
    ESCORT_TEMPLATE,
    CONTROL_TEMPLATE,
    EXTRACT_TEMPLATE,
    HOLDOUT_TEMPLATE,
    GAUNTLET_TEMPLATE,
    RECON_TEMPLATE,
)


class TestZoneControlStateTracker:
    """Tests for ZoneControlStateTracker model."""

    def test_create_neutral_zone(self):
        """Create a neutral zone tracker."""
        tracker = ZoneControlStateTracker(
            zone_id="zone_1",
            state="neutral",
            controlling_side=None,
            last_checked_turn=1,
        )
        assert tracker.zone_id == "zone_1"
        assert tracker.state == "neutral"
        assert tracker.controlling_side is None
        assert tracker.last_checked_turn == 1

    def test_create_player_controlled_zone(self):
        """Create a player-controlled zone."""
        tracker = ZoneControlStateTracker(
            zone_id="quadrant_nw",
            state="player_controlled",
            controlling_side="players",
            last_checked_turn=3,
        )
        assert tracker.state == "player_controlled"
        assert tracker.controlling_side == "players"
        assert tracker.last_checked_turn == 3

    def test_create_enemy_controlled_zone(self):
        """Create an enemy-controlled zone."""
        tracker = ZoneControlStateTracker(
            zone_id="center",
            state="enemy_controlled",
            controlling_side="enemies",
            last_checked_turn=5,
        )
        assert tracker.state == "enemy_controlled"
        assert tracker.controlling_side == "enemies"

    def test_create_contested_zone(self):
        """Create a contested zone."""
        tracker = ZoneControlStateTracker(
            zone_id="objective_a",
            state="contested",
            controlling_side=None,
            last_checked_turn=2,
        )
        assert tracker.state == "contested"
        assert tracker.controlling_side is None


class TestSitrepVictoryCondition:
    """Tests for SitrepVictoryCondition model."""

    def test_create_control_zones_condition(self):
        """Create a control zones victory condition."""
        vc = SitrepVictoryCondition(
            condition_type="control_zones",
            target_value=3,
            current_value=0,
            is_met=False,
            description="Control 3 zones at end of round 6",
        )
        assert vc.condition_type == "control_zones"
        assert vc.target_value == 3
        assert vc.current_value == 0
        assert vc.is_met is False

    def test_create_extract_objective_condition(self):
        """Create an extract objective victory condition."""
        vc = SitrepVictoryCondition(
            condition_type="extract_objective",
            target_value=1,
            current_value=50,
            is_met=False,
            description="Extract the objective to the extraction zone",
        )
        assert vc.condition_type == "extract_objective"
        assert vc.current_value == 50

    def test_create_survive_rounds_condition(self):
        """Create a survive rounds victory condition."""
        vc = SitrepVictoryCondition(
            condition_type="survive_rounds",
            target_value=6,
            current_value=6,
            is_met=True,
            description="Survive 6 rounds",
        )
        assert vc.is_met is True


class TestSitrepDeployment:
    """Tests for SitrepDeployment model."""

    def test_create_empty_deployment(self):
        """Create an empty deployment configuration."""
        deployment = SitrepDeployment()
        assert deployment.player_zones == []
        assert deployment.enemy_zones == []
        assert deployment.ingress_zones == []
        assert deployment.reserve_pool == []

    def test_create_deployment_with_zones(self):
        """Create a deployment with zone configurations."""
        player_zone = SitrepZone(zone_type="deployment", location="map_edge")
        ingress_zone = SitrepZone(zone_type="ingress", location="left_flank")
        deployment = SitrepDeployment(
            player_zones=[player_zone],
            ingress_zones=[ingress_zone],
            reserves_spawned_per_round=1,
        )
        assert len(deployment.player_zones) == 1
        assert len(deployment.ingress_zones) == 1
        assert deployment.reserves_spawned_per_round == 1


class TestSitrepResolution:
    """Tests for SitrepResolution model."""

    def test_create_escort_resolution(self):
        """Create an escort SITREP resolution."""
        resolution = SitrepResolution(
            template_type="escort",
            current_round=1,
            max_rounds=6,
            surviving_players=4,
            surviving_enemies=6,
        )
        assert resolution.template_type == "escort"
        assert resolution.current_round == 1
        assert resolution.max_rounds == 6
        assert resolution.surviving_players == 4
        assert resolution.surviving_enemies == 6
        assert resolution.outcome is None

    def test_create_control_resolution(self):
        """Create a control SITREP resolution with zone tracking."""
        zone_states = {
            "quadrant_nw": ZoneControlStateTracker(
                zone_id="quadrant_nw",
                state="player_controlled",
                controlling_side="players",
            )
        }
        resolution = SitrepResolution(
            template_type="control",
            current_round=3,
            max_rounds=6,
            player_score=2,
            enemy_score=1,
            zone_states=zone_states,
            surviving_players=4,
            surviving_enemies=4,
        )
        assert resolution.template_type == "control"
        assert resolution.player_score == 2
        assert len(resolution.zone_states) == 1


class TestCreateSitrepResolution:
    """Tests for create_sitrep_resolution function."""

    def test_create_escort_resolution(self):
        """Create an escort resolution from template."""
        resolution = create_sitrep_resolution(
            template=ESCORT_TEMPLATE,
            player_count=4,
            reserve_ids=["grunt_1", "grunt_2", "grunt_3"],
            enemy_count=6,
        )
        assert resolution.template_type == "escort"
        assert resolution.current_round == 1
        assert resolution.max_rounds == 6
        assert resolution.surviving_players == 4
        assert resolution.surviving_enemies == 6
        assert resolution.reserves_remaining == 3
        assert resolution.deployment is not None

    def test_create_control_resolution_with_zones(self):
        """Create a control resolution with zone states."""
        resolution = create_sitrep_resolution(
            template=CONTROL_TEMPLATE,
            player_count=4,
            enemy_count=8,
        )
        assert resolution.template_type == "control"
        assert len(resolution.zone_states) == 4  # 4 objective zones in CONTROL
        assert len(resolution.victory_conditions) == 1

    def test_create_extract_resolution(self):
        """Create an extract resolution."""
        resolution = create_sitrep_resolution(
            template=EXTRACT_TEMPLATE,
            player_count=4,
            reserve_ids=["elite_1", "elite_2"],
            enemy_count=4,
        )
        assert resolution.template_type == "extract"
        assert resolution.max_rounds == 8  # EXTRACT has 8 rounds
        assert resolution.reserves_remaining == 2

    def test_create_holdout_resolution(self):
        """Create a holdout resolution."""
        resolution = create_sitrep_resolution(
            template=HOLDOUT_TEMPLATE,
            player_count=4,
            enemy_count=12,
        )
        assert resolution.template_type == "hold_out"
        assert resolution.player_score == 4  # Start with 4 points

    def test_create_gauntlet_resolution(self):
        """Create a gauntlet resolution."""
        resolution = create_sitrep_resolution(
            template=GAUNTLET_TEMPLATE,
            player_count=4,
            enemy_count=8,
        )
        assert resolution.template_type == "gauntlet"
        assert resolution.max_rounds == 6

    def test_create_recon_resolution(self):
        """Create a recon resolution with multiple zones."""
        resolution = create_sitrep_resolution(
            template=RECON_TEMPLATE,
            player_count=4,
            enemy_count=6,
        )
        assert resolution.template_type == "recon"
        assert len(resolution.zone_states) == 4  # 4 potential objectives


class TestSpawnReserves:
    """Tests for spawn_reserves function."""

    def test_spawn_reserves_no_deployment(self):
        """Spawn reserves when deployment is None."""
        resolution = SitrepResolution(
            template_type="control",
            deployment=None,
        )
        updated, spawned = spawn_reserves(resolution)
        assert updated == resolution
        assert spawned == []

    def test_spawn_reserves_empty_pool(self):
        """Spawn reserves from empty pool."""
        deployment = SitrepDeployment(
            reserve_pool=[],
            reserves_spawned_per_round=1,
        )
        resolution = SitrepResolution(
            template_type="control",
            deployment=deployment,
            reserves_remaining=0,
        )
        updated, spawned = spawn_reserves(resolution)
        assert spawned == []

    def test_spawn_reserves_basic(self):
        """Spawn reserves from pool."""
        deployment = SitrepDeployment(
            reserve_pool=["grunt_1", "grunt_2", "grunt_3"],
            reserves_spawned_per_round=1,
            ingress_zones=[
                SitrepZone(zone_type="ingress", location="zone_a"),
                SitrepZone(zone_type="ingress", location="zone_b"),
            ],
        )
        resolution = SitrepResolution(
            template_type="escort",
            deployment=deployment,
            reserves_remaining=3,
            surviving_enemies=2,
        )
        updated, spawned = spawn_reserves(resolution, count=1)
        assert len(spawned) == 1
        assert updated.reserves_remaining == 2
        assert updated.surviving_enemies == 3
        assert updated.deployment is not None
        assert updated.deployment.last_ingress_zone is not None

    def test_spawn_reserves_rotation_rule(self):
        """Spawn reserves respects ingress zone rotation (no same zone twice)."""
        deployment = SitrepDeployment(
            reserve_pool=["grunt_1", "grunt_2", "grunt_3", "grunt_4"],
            reserves_spawned_per_round=1,
            last_ingress_zone="zone_a",
            ingress_zones=[
                SitrepZone(zone_type="ingress", location="zone_a"),
                SitrepZone(zone_type="ingress", location="zone_b"),
            ],
        )
        resolution = SitrepResolution(
            template_type="escort",
            deployment=deployment,
            reserves_remaining=4,
        )
        updated, spawned = spawn_reserves(resolution, count=1)
        assert updated.deployment.last_ingress_zone != "zone_a"

    def test_spawn_reserves_multiple(self):
        """Spawn multiple reserves at once."""
        deployment = SitrepDeployment(
            reserve_pool=["grunt_1", "grunt_2", "grunt_3"],
            reserves_spawned_per_round=2,
            ingress_zones=[
                SitrepZone(zone_type="ingress", location="zone_a"),
            ],
        )
        resolution = SitrepResolution(
            template_type="extract",
            deployment=deployment,
            reserves_remaining=3,
            surviving_enemies=2,
        )
        updated, spawned = spawn_reserves(resolution)
        assert len(spawned) == 2
        assert updated.reserves_remaining == 1
        assert updated.surviving_enemies == 4

    def test_spawn_reserves_with_seed(self):
        """Spawn reserves with seed for reproducibility."""
        deployment = SitrepDeployment(
            reserve_pool=["grunt_1", "grunt_2", "grunt_3", "grunt_4"],
            reserves_spawned_per_round=1,
            ingress_zones=[
                SitrepZone(zone_type="ingress", location="zone_a"),
                SitrepZone(zone_type="ingress", location="zone_b"),
            ],
        )
        resolution = SitrepResolution(
            template_type="escort",
            deployment=deployment,
            reserves_remaining=4,
        )
        updated1, spawned1 = spawn_reserves(resolution, seed=42)
        updated2, spawned2 = spawn_reserves(resolution, seed=42)
        assert spawned1 == spawned2


class TestUpdateZoneControl:
    """Tests for update_zone_control function."""

    def test_update_zone_to_player_controlled(self):
        """Update zone to player controlled increases score."""
        zone_states = {
            "zone_1": ZoneControlStateTracker(
                zone_id="zone_1",
                state="neutral",
                controlling_side=None,
            )
        }
        resolution = SitrepResolution(
            template_type="control",
            zone_states=zone_states,
            player_score=0,
            enemy_score=0,
        )
        updated = update_zone_control(
            resolution,
            "zone_1",
            "player_controlled",
            "players",
        )
        assert updated.zone_states["zone_1"].state == "player_controlled"
        assert updated.zone_states["zone_1"].controlling_side == "players"
        assert updated.player_score == 1

    def test_update_zone_to_enemy_controlled(self):
        """Update zone to enemy controlled increases enemy score."""
        zone_states = {
            "zone_1": ZoneControlStateTracker(
                zone_id="zone_1",
                state="neutral",
                controlling_side=None,
            )
        }
        resolution = SitrepResolution(
            template_type="control",
            zone_states=zone_states,
            player_score=0,
            enemy_score=0,
        )
        updated = update_zone_control(
            resolution,
            "zone_1",
            "enemy_controlled",
            "enemies",
        )
        assert updated.zone_states["zone_1"].state == "enemy_controlled"
        assert updated.enemy_score == 1

    def test_update_zone_to_contested(self):
        """Update zone to contested removes scores."""
        zone_states = {
            "zone_1": ZoneControlStateTracker(
                zone_id="zone_1",
                state="player_controlled",
                controlling_side="players",
            )
        }
        resolution = SitrepResolution(
            template_type="control",
            zone_states=zone_states,
            player_score=1,
            enemy_score=0,
        )
        updated = update_zone_control(
            resolution,
            "zone_1",
            "contested",
            None,
        )
        assert updated.zone_states["zone_1"].state == "contested"
        assert updated.player_score == 0

    def test_update_nonexistent_zone(self):
        """Update nonexistent zone returns unchanged resolution."""
        resolution = SitrepResolution(
            template_type="control",
            zone_states={},
        )
        updated = update_zone_control(
            resolution,
            "nonexistent",
            "player_controlled",
            "players",
        )
        assert updated == resolution

    def test_zone_control_transitions(self):
        """Test multiple zone control transitions."""
        zone_states = {
            "zone_1": ZoneControlStateTracker(
                zone_id="zone_1",
                state="neutral",
                controlling_side=None,
            )
        }
        resolution = SitrepResolution(
            template_type="control",
            zone_states=zone_states,
            player_score=0,
            enemy_score=0,
        )
        # Player takes zone
        updated = update_zone_control(
            resolution, "zone_1", "player_controlled", "players"
        )
        assert updated.player_score == 1
        # Enemy contests
        updated = update_zone_control(updated, "zone_1", "contested", None)
        assert updated.player_score == 0
        # Enemy takes zone
        updated = update_zone_control(updated, "zone_1", "enemy_controlled", "enemies")
        assert updated.player_score == 0
        assert updated.enemy_score == 1


class TestCheckExtractionProgress:
    """Tests for check_extraction_progress function."""

    def test_extraction_progress_free_action(self):
        """Free action adds 0.25 to extraction progress."""
        resolution = SitrepResolution(
            template_type="escort",
            extraction_progress=0.0,
        )
        updated = check_extraction_progress(resolution, "free")
        assert updated.extraction_progress == 0.25

    def test_extraction_progress_quick_action(self):
        """Quick action adds 0.5 to extraction progress."""
        resolution = SitrepResolution(
            template_type="extract",
            extraction_progress=0.0,
        )
        updated = check_extraction_progress(resolution, "quick")
        assert updated.extraction_progress == 0.5

    def test_extraction_progress_full_action(self):
        """Full action adds 1.0 to extraction progress."""
        resolution = SitrepResolution(
            template_type="escort",
            extraction_progress=0.0,
        )
        updated = check_extraction_progress(resolution, "full")
        assert updated.extraction_progress == 1.0

    def test_extraction_progress_complete(self):
        """Extraction complete when progress reaches 1.0."""
        resolution = SitrepResolution(
            template_type="escort",
            extraction_progress=0.75,
            victory_conditions=[
                SitrepVictoryCondition(
                    condition_type="extract_objective",
                    target_value=1,
                    current_value=75,
                    is_met=False,
                    description="Extract the objective",
                )
            ],
        )
        updated = check_extraction_progress(resolution, "full")
        assert updated.extraction_progress == 1.0
        assert updated.victory_conditions[0].is_met is True

    def test_extraction_progress_capped(self):
        """Extraction progress is capped at 1.0."""
        resolution = SitrepResolution(
            template_type="extract",
            extraction_progress=0.9,
        )
        updated = check_extraction_progress(resolution, "full")
        assert updated.extraction_progress == 1.0

    def test_extraction_progress_non_extraction_sitrep(self):
        """Non-extraction SITREPs are unaffected."""
        resolution = SitrepResolution(
            template_type="control",
            extraction_progress=0.0,
        )
        updated = check_extraction_progress(resolution, "full")
        assert updated.extraction_progress == 0.0


class TestCheckVictoryConditions:
    """Tests for check_victory_conditions function."""

    def test_control_zones_victory(self):
        """Control zones victory when threshold met."""
        zone_states = {
            "zone_1": ZoneControlStateTracker(
                zone_id="zone_1",
                state="player_controlled",
                controlling_side="players",
            ),
            "zone_2": ZoneControlStateTracker(
                zone_id="zone_2",
                state="player_controlled",
                controlling_side="players",
            ),
            "zone_3": ZoneControlStateTracker(
                zone_id="zone_3",
                state="player_controlled",
                controlling_side="players",
            ),
        }
        resolution = SitrepResolution(
            template_type="control",
            zone_states=zone_states,
            victory_conditions=[
                SitrepVictoryCondition(
                    condition_type="control_zones",
                    target_value=3,
                    current_value=0,
                    is_met=False,
                    description="Control 3 zones",
                )
            ],
        )
        updated = check_victory_conditions(resolution)
        assert updated.victory_conditions[0].is_met is True
        assert updated.victory_conditions[0].current_value == 3
        assert updated.outcome == "players_win"

    def test_extraction_victory(self):
        """Extraction victory when progress complete."""
        resolution = SitrepResolution(
            template_type="escort",
            extraction_progress=1.0,
            victory_conditions=[
                SitrepVictoryCondition(
                    condition_type="extract_objective",
                    target_value=1,
                    current_value=100,
                    is_met=False,
                    description="Extract the objective",
                )
            ],
        )
        updated = check_victory_conditions(resolution)
        assert updated.victory_conditions[0].is_met is True
        assert updated.outcome == "players_win"

    def test_no_victory_yet(self):
        """No victory when conditions not met."""
        zone_states = {
            "zone_1": ZoneControlStateTracker(
                zone_id="zone_1",
                state="player_controlled",
                controlling_side="players",
            ),
        }
        resolution = SitrepResolution(
            template_type="control",
            zone_states=zone_states,
            victory_conditions=[
                SitrepVictoryCondition(
                    condition_type="control_zones",
                    target_value=3,
                    current_value=1,
                    is_met=False,
                    description="Control 3 zones",
                )
            ],
        )
        updated = check_victory_conditions(resolution)
        assert updated.victory_conditions[0].is_met is False
        assert updated.outcome is None


class TestAdvanceSitrepRound:
    """Tests for advance_sitrep_round function."""

    def test_advance_round_basic(self):
        """Advance round increments round counter."""
        resolution = SitrepResolution(
            template_type="control",
            current_round=1,
            max_rounds=6,
            turn_limit_reached=False,
        )
        updated = advance_sitrep_round(resolution)
        assert updated.current_round == 2
        assert updated.turn_limit_reached is False

    def test_advance_round_turn_limit(self):
        """Advance round past max sets turn_limit_reached."""
        resolution = SitrepResolution(
            template_type="control",
            current_round=5,
            max_rounds=6,
            turn_limit_reached=False,
        )
        updated = advance_sitrep_round(resolution)
        assert updated.current_round == 6
        assert updated.turn_limit_reached is True

    def test_advance_round_escort_spawns_reserves(self):
        """ESCORT type spawns reserves with increasing pattern."""
        deployment = SitrepDeployment(
            reserve_pool=["grunt_1", "grunt_2"],
            reserves_spawned_per_round=1,
            reserve_pattern="increasing",
            ingress_zones=[
                SitrepZone(zone_type="ingress", location="zone_a"),
            ],
        )
        resolution = SitrepResolution(
            template_type="escort",
            current_round=1,
            max_rounds=6,
            deployment=deployment,
            surviving_enemies=2,
        )
        updated = advance_sitrep_round(resolution)
        assert updated.surviving_enemies == 3  # 1 reserve spawned


class TestResolveSitrep:
    """Tests for resolve_sitrep function."""

    def test_resolve_control_player_victory(self):
        """Control: player score > enemy score = player win."""
        resolution = SitrepResolution(
            template_type="control",
            player_score=3,
            enemy_score=1,
        )
        updated = resolve_sitrep(resolution)
        assert updated.outcome == "players_win"

    def test_resolve_control_enemy_victory(self):
        """Control: enemy score > player score = enemy win."""
        resolution = SitrepResolution(
            template_type="control",
            player_score=1,
            enemy_score=3,
        )
        updated = resolve_sitrep(resolution)
        assert updated.outcome == "enemies_win"

    def test_resolve_control_draw(self):
        """Control: tied score = draw."""
        resolution = SitrepResolution(
            template_type="control",
            player_score=2,
            enemy_score=2,
        )
        updated = resolve_sitrep(resolution)
        assert updated.outcome == "draw"

    def test_resolve_holdout_player_victory(self):
        """Holdout: score >= 1 = player win."""
        resolution = SitrepResolution(
            template_type="hold_out",
            player_score=2,
        )
        updated = resolve_sitrep(resolution)
        assert updated.outcome == "players_win"

    def test_resolve_holdout_enemy_victory(self):
        """Holdout: score < 1 = enemy win."""
        resolution = SitrepResolution(
            template_type="hold_out",
            player_score=0,
        )
        updated = resolve_sitrep(resolution)
        assert updated.outcome == "enemies_win"

    def test_resolve_escort_player_victory(self):
        """Escort: extraction complete = player win."""
        resolution = SitrepResolution(
            template_type="escort",
            extraction_progress=1.0,
            outcome="players_win",
        )
        updated = resolve_sitrep(resolution)
        assert updated.outcome == "players_win"

    def test_resolve_escort_enemy_victory(self):
        """Escort: time limit reached without extraction = enemy win."""
        resolution = SitrepResolution(
            template_type="escort",
            extraction_progress=0.5,
            turn_limit_reached=True,
            current_round=6,
            max_rounds=6,
        )
        updated = resolve_sitrep(resolution)
        assert updated.outcome == "enemies_win"

    def test_resolve_gauntlet_player_victory(self):
        """Gauntlet: more players than enemies = player win."""
        resolution = SitrepResolution(
            template_type="gauntlet",
            surviving_players=4,
            surviving_enemies=2,
        )
        updated = resolve_sitrep(resolution)
        assert updated.outcome == "players_win"

    def test_resolve_gauntlet_enemy_victory(self):
        """Gauntlet: fewer players than enemies = enemy win."""
        resolution = SitrepResolution(
            template_type="gauntlet",
            surviving_players=2,
            surviving_enemies=4,
        )
        updated = resolve_sitrep(resolution)
        assert updated.outcome == "enemies_win"

    def test_resolve_already_resolved(self):
        """Already resolved SITREP returns unchanged."""
        resolution = SitrepResolution(
            template_type="control",
            outcome="players_win",
        )
        updated = resolve_sitrep(resolution)
        assert updated.outcome == "players_win"


class TestIntegration:
    """Integration tests for end-to-end SITREP resolution flows."""

    def test_full_escort_mission_flow(self):
        """Test full escort mission from start to completion."""
        resolution = create_sitrep_resolution(
            template=ESCORT_TEMPLATE,
            player_count=4,
            reserve_ids=["grunt_1", "grunt_2"],
            enemy_count=4,
        )
        assert resolution.current_round == 1
        assert resolution.extraction_progress == 0.0

        # Round 1: Advance, spawn reserves
        resolution = advance_sitrep_round(resolution)
        assert resolution.current_round == 2

        # Extract with free actions
        resolution = check_extraction_progress(resolution, "free")
        assert resolution.extraction_progress == 0.25
        resolution = check_extraction_progress(resolution, "free")
        assert resolution.extraction_progress == 0.5
        resolution = check_extraction_progress(resolution, "full")
        assert resolution.extraction_progress == 1.0

        # Check victory
        resolution = check_victory_conditions(resolution)
        assert resolution.outcome == "players_win"

        # Final resolution
        resolution = resolve_sitrep(resolution)
        assert resolution.outcome == "players_win"

    def test_full_control_mission_flow(self):
        """Test full control mission with zone takedowns."""
        resolution = create_sitrep_resolution(
            template=CONTROL_TEMPLATE,
            player_count=4,
            enemy_count=6,
        )

        # Player takes first zone
        resolution = update_zone_control(
            resolution, "quadrant_nw", "player_controlled", "players"
        )
        assert resolution.player_score == 1

        # Player takes second zone
        resolution = update_zone_control(
            resolution, "quadrant_ne", "player_controlled", "players"
        )
        assert resolution.player_score == 2

        # Player takes third zone (win condition)
        resolution = update_zone_control(
            resolution, "quadrant_sw", "player_controlled", "players"
        )
        assert resolution.player_score == 3

        # Check victory - 3 zones controlled = player win
        resolution = check_victory_conditions(resolution)
        assert resolution.victory_conditions[0].is_met is True
        assert resolution.victory_conditions[0].current_value == 3
        assert resolution.outcome == "players_win"

    def test_gauntlet_mission_with_enemy_numbers(self):
        """Test gauntlet mission where enemies outnumber players."""
        resolution = create_sitrep_resolution(
            template=GAUNTLET_TEMPLATE,
            player_count=4,
            enemy_count=8,
        )

        # Enemy elimination simulation
        resolution = resolution.model_copy(update={"surviving_enemies": 2})

        # Check victory
        resolution = check_victory_conditions(resolution)
        assert resolution.outcome == "players_win"

        resolution = resolve_sitrep(resolution)
        assert resolution.outcome == "players_win"
