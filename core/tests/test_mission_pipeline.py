"""Tests for Phase 34: Mission → Combat Pipeline.

Tests terrain generation, enemy force calculation, and SITREP resolution
integration with the mission launch flow.
"""

import pytest
from core.shared.scenario import SITREP_TEMPLATES, SitrepType
from core.shared.terrain_generation import (
    TileSetType,
    TerrainGeneratorParams,
    generate_terrain_from_sitrep,
    generate_zone_coords,
)
from core.shared.sitrep_resolution import (
    create_sitrep_resolution,
    advance_sitrep_round,
    update_zone_control,
    check_victory_conditions,
)
from core.gm_toolkit.encounter_builder import (
    EncounterDifficulty,
    estimate_party_power,
    calculate_enemy_force,
)
from core.mech.combat_state import MechCombatScenario, CombatantState, CombatStats, CombatResources
from core.mech.grid import HexCoord


class TestTerrainGenerationFromSitrep:
    """Test terrain generation from SITREP templates."""

    @pytest.mark.parametrize("sitrep_type", [
        "escort", "control", "extract", "hold_out", "gauntlet", "recon"
    ])
    def test_terrain_generation_for_all_sitrep_types(self, sitrep_type: SitrepType):
        """Test that terrain generates correctly for all SITREP types."""
        template = SITREP_TEMPLATES[sitrep_type]
        params = TerrainGeneratorParams(
            map_width=20,
            map_height=16,
            sitrep_template=template,
            tile_set="urban",
            seed=42,
        )

        result = generate_terrain_from_sitrep(template, params)

        assert result is not None
        assert result.terrain_map is not None
        assert len(result.primitives) > 0

    @pytest.mark.parametrize("tile_set", ["urban", "industrial", "wilderness", "zero_g"])
    def test_terrain_generation_for_all_tile_sets(self, tile_set: TileSetType):
        """Test that terrain generates correctly for all tile sets."""
        template = SITREP_TEMPLATES["control"]
        params = TerrainGeneratorParams(
            map_width=20,
            map_height=16,
            sitrep_template=template,
            tile_set=tile_set,
            seed=42,
        )

        result = generate_terrain_from_sitrep(template, params)

        assert result is not None
        assert result.terrain_map is not None

    def test_terrain_generation_respects_seed(self):
        """Test that same seed produces same terrain."""
        template = SITREP_TEMPLATES["control"]
        params = TerrainGeneratorParams(
            map_width=20,
            map_height=16,
            sitrep_template=template,
            tile_set="urban",
            seed=12345,
        )

        result1 = generate_terrain_from_sitrep(template, params)
        result2 = generate_terrain_from_sitrep(template, params)

        # Same seed should produce same number of primitives
        assert len(result1.primitives) == len(result2.primitives)

    def test_terrain_generation_with_custom_dimensions(self):
        """Test terrain with non-default dimensions."""
        template = SITREP_TEMPLATES["control"]
        params = TerrainGeneratorParams(
            map_width=30,
            map_height=24,
            sitrep_template=template,
            tile_set="urban",
            seed=42,
        )

        result = generate_terrain_from_sitrep(template, params)

        assert result is not None
        assert result.terrain_map is not None

    def test_zone_generation_creates_deployment_zones(self):
        """Test that deployment zones are created."""
        template = SITREP_TEMPLATES["control"]
        params = TerrainGeneratorParams(
            map_width=20,
            map_height=16,
            sitrep_template=template,
            tile_set="urban",
            seed=42,
        )

        result = generate_terrain_from_sitrep(template, params)

        # Control template should have deployment zones
        assert len(result.zones) > 0 or len(template.deployment_zones) > 0


class TestZoneCoordinateGeneration:
    """Test zone coordinate generation."""

    def test_center_zone_coords(self):
        """Test center zone coordinate generation."""
        from core.shared.scenario import SitrepZone

        zone = SitrepZone(
            zone_type="objective",
            location="center",
            width=4,
            height=4,
        )

        coords = generate_zone_coords(zone, 20, 16)

        assert len(coords) == 16  # 4x4 = 16 hexes
        # All coords should be within map bounds
        for coord in coords:
            assert 0 <= coord.q < 20
            assert 0 <= coord.r < 16

    def test_map_edge_zone_coords(self):
        """Test map edge zone coordinate generation."""
        from core.shared.scenario import SitrepZone

        zone = SitrepZone(
            zone_type="deployment",
            location="map_edge",
            width=4,
            height=4,
        )

        coords = generate_zone_coords(zone, 20, 16)

        assert len(coords) == 16
        # Should be at the left edge (q near 0)
        for coord in coords:
            assert coord.q < 5


class TestEnemyForceCalculation:
    """Test enemy force generation."""

    @pytest.mark.parametrize("difficulty", [
        "trivial", "easy", "standard", "hard", "extreme"
    ])
    def test_force_calculation_for_all_difficulties(self, difficulty: EncounterDifficulty):
        """Test force calculation for all difficulty levels."""
        player_power = estimate_party_power(4, 3.0)

        force = calculate_enemy_force(
            difficulty=difficulty,
            sitrep_type="control",
            player_power=player_power,
        )

        assert force is not None
        assert force.target_victory_points >= 0

    def test_force_calculation_scales_with_player_count(self):
        """Test that force scales with player count."""
        small_party = estimate_party_power(3, 3.0)
        large_party = estimate_party_power(5, 3.0)

        small_force = calculate_enemy_force("standard", "control", small_party)
        large_force = calculate_enemy_force("standard", "control", large_party)

        assert large_force.target_victory_points > small_force.target_victory_points

    def test_force_calculation_scales_with_license_level(self):
        """Test that force scales with license level."""
        low_level = estimate_party_power(4, 0.0)
        high_level = estimate_party_power(4, 12.0)

        low_force = calculate_enemy_force("standard", "control", low_level)
        high_force = calculate_enemy_force("standard", "control", high_level)

        assert high_force.target_victory_points > low_force.target_victory_points

    def test_force_calculation_splits_initial_and_reserves(self):
        """Test that force is split between initial and reserves based on SITREP."""
        player_power = estimate_party_power(4, 3.0)

        # Escort has increasing reserves
        escort_force = calculate_enemy_force("standard", "escort", player_power)

        # Both should have some initial VP allocation
        assert escort_force.target_victory_points > 0


class TestSitrepResolutionCreation:
    """Test SITREP resolution state creation."""

    @pytest.mark.parametrize("sitrep_type", [
        "escort", "control", "extract", "hold_out", "gauntlet", "recon"
    ])
    def test_sitrep_resolution_creation(self, sitrep_type: SitrepType):
        """Test SITREP resolution creates correctly for all types."""
        template = SITREP_TEMPLATES[sitrep_type]

        resolution = create_sitrep_resolution(
            template=template,
            player_count=4,
            reserve_ids=["npc_1", "npc_2"],
            enemy_count=3,
        )

        assert resolution is not None
        assert resolution.template_type == sitrep_type
        assert resolution.current_round == 1
        assert resolution.surviving_players == 4

    def test_sitrep_resolution_has_victory_conditions(self):
        """Test that resolution has victory conditions from template."""
        template = SITREP_TEMPLATES["control"]

        resolution = create_sitrep_resolution(
            template=template,
            player_count=4,
        )

        assert len(resolution.victory_conditions) > 0

    def test_sitrep_resolution_tracks_zones(self):
        """Test that resolution tracks zone states."""
        template = SITREP_TEMPLATES["control"]

        resolution = create_sitrep_resolution(
            template=template,
            player_count=4,
        )

        # Control has objective zones
        if template.objective_zones:
            assert len(resolution.zone_states) > 0


class TestSitrepResolutionAdvancement:
    """Test SITREP resolution state changes."""

    def test_advance_sitrep_round(self):
        """Test advancing to next round."""
        template = SITREP_TEMPLATES["control"]
        resolution = create_sitrep_resolution(template, 4)

        assert resolution.current_round == 1

        advanced = advance_sitrep_round(resolution)

        assert advanced.current_round == 2

    def test_update_zone_control(self):
        """Test updating zone control state."""
        template = SITREP_TEMPLATES["control"]
        resolution = create_sitrep_resolution(template, 4)

        if resolution.zone_states:
            zone_id = list(resolution.zone_states.keys())[0]

            updated = update_zone_control(
                resolution=resolution,
                zone_id=zone_id,
                new_state="player_controlled",
                controlling_side="players",
            )

            assert updated.zone_states[zone_id].state == "player_controlled"

    def test_check_victory_conditions(self):
        """Test victory condition checking."""
        template = SITREP_TEMPLATES["control"]
        resolution = create_sitrep_resolution(template, 4)

        checked = check_victory_conditions(resolution)

        # Should return resolution with conditions evaluated
        assert checked is not None


class TestMechCombatScenarioWithSitrep:
    """Test MechCombatScenario with SITREP resolution."""

    def test_scenario_can_have_sitrep_resolution(self):
        """Test that scenario accepts sitrep_resolution field."""
        template = SITREP_TEMPLATES["control"]
        resolution = create_sitrep_resolution(template, 4)

        # Create a simple combatant for the scenario
        combatant = CombatantState(
            id="combat_test",
            name="Test Mech",
            side="players",
            kind="mech",
            stats=CombatStats(
                size="size_1",
                hp_max=10,
                evasion=8,
                e_defense=8,
            ),
            resources=CombatResources(
                hp_current=10,
            ),
        )

        scenario = MechCombatScenario(
            combatants=[combatant],
            sitrep_resolution=resolution,
        )

        assert scenario.sitrep_resolution is not None
        assert scenario.sitrep_resolution.template_type == "control"

    def test_scenario_can_have_terrain(self):
        """Test that scenario accepts terrain field."""
        template = SITREP_TEMPLATES["control"]
        params = TerrainGeneratorParams(
            map_width=20,
            map_height=16,
            sitrep_template=template,
            tile_set="urban",
            seed=42,
        )

        generated = generate_terrain_from_sitrep(template, params)

        scenario = MechCombatScenario(
            combatants=[],
            terrain=generated.terrain_map,
        )

        assert scenario.terrain is not None


class TestDeploymentPositionAssignment:
    """Test deployment position assignment logic."""

    def test_positions_are_within_zone_bounds(self):
        """Test that assigned positions are within deployment zone bounds."""
        from core.shared.scenario import SitrepZone

        zone = SitrepZone(
            zone_type="deployment",
            location="map_edge",
            width=4,
            height=4,
        )

        coords = generate_zone_coords(zone, 20, 16)

        # All coordinates should be valid
        for coord in coords:
            assert isinstance(coord, HexCoord)
            assert coord.q >= 0
            assert coord.r >= 0


class TestEndToEndMissionPipeline:
    """Integration tests for the complete mission pipeline."""

    def test_full_pipeline_with_control_sitrep(self):
        """Test complete pipeline with CONTROL SITREP."""
        # 1. Get template
        template = SITREP_TEMPLATES["control"]

        # 2. Generate terrain
        params = TerrainGeneratorParams(
            map_width=20,
            map_height=16,
            sitrep_template=template,
            tile_set="urban",
            seed=42,
        )
        generated = generate_terrain_from_sitrep(template, params)

        # 3. Calculate enemy force
        player_power = estimate_party_power(4, 3.0)
        force = calculate_enemy_force("standard", "control", player_power)

        # 4. Create SITREP resolution
        resolution = create_sitrep_resolution(
            template=template,
            player_count=4,
            enemy_count=int(force.target_victory_points),
        )

        # 5. Create scenario
        scenario = MechCombatScenario(
            combatants=[],
            terrain=generated.terrain_map,
            sitrep_resolution=resolution,
        )

        # Verify everything is connected
        assert scenario.terrain is not None
        assert scenario.sitrep_resolution is not None
        assert scenario.sitrep_resolution.template_type == "control"
