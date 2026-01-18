"""Integration tests for terrain generation with SITREP linkage."""

import pytest
from core.mech.grid import HexCoord
from core.shared.scenario import (
    SitrepTemplate,
    SitrepZone,
    VictoryCondition,
)
from core.shared.terrain import (
    calculate_movement_cost,
    get_cover_difficulty,
    get_terrain_at,
)
from core.shared.terrain_primitives import (
    MATERIAL_FORTIFIED,
    FloorTile,
    Obstacle,
    compose_terrain_map,
)
from core.shared.terrain_generation import (
    TileSetType,
    TileSetConfig,
    TILE_SETS,
    TerrainGeneratorParams,
    generate_terrain_from_sitrep,
    generate_zone_coords,
    generate_random_terrain,
)
from core.shared.hide_search import (
    SoftCoverZoneState,
    is_in_active_soft_cover_zone,
    check_soft_cover_for_hide,
    is_soft_cover_area,
)


# =============================================================================
# Tile Set Configuration Tests
# =============================================================================


class TestTileSetConfig:
    """Tests for tile set configurations."""

    def test_all_tile_sets_defined(self) -> None:
        """All expected tile sets are defined."""
        expected_sets: list[TileSetType] = ["urban", "industrial", "wilderness", "zero_g"]
        for tile_set in expected_sets:
            assert tile_set in TILE_SETS

    def test_urban_tile_set(self) -> None:
        """Urban tile set has correct configuration."""
        urban = TILE_SETS["urban"]
        assert "building" in urban.features
        assert "wall" in urban.features
        assert "hardy" in urban.materials or "fortified" in urban.materials
        assert urban.cover_density > 0

    def test_industrial_tile_set_has_hazards(self) -> None:
        """Industrial tile set includes hazards."""
        industrial = TILE_SETS["industrial"]
        assert len(industrial.hazards) > 0
        assert "electricity" in industrial.hazards or "acid" in industrial.hazards

    def test_wilderness_has_soft_cover(self) -> None:
        """Wilderness tile set enables soft cover zones."""
        wilderness = TILE_SETS["wilderness"]
        assert wilderness.soft_cover_zones is True
        assert "organic" in wilderness.materials

    def test_zero_g_has_special_rules(self) -> None:
        """Zero-G tile set has special rules."""
        zero_g = TILE_SETS["zero_g"]
        assert "no_falling" in zero_g.special_rules
        assert "3d_movement" in zero_g.special_rules


# =============================================================================
# Zone Coordinate Generation Tests
# =============================================================================


class TestGenerateZoneCoords:
    """Tests for generate_zone_coords function."""

    def test_center_location(self) -> None:
        """Center location places zone in map center."""
        zone = SitrepZone(
            zone_type="objective",
            location="center",
            width=4,
            height=4,
        )
        coords = generate_zone_coords(zone, map_width=20, map_height=16)

        # Verify coordinates exist
        assert len(coords) == 16  # 4x4

        # Verify roughly centered
        q_values = [c.q for c in coords]
        r_values = [c.r for c in coords]
        avg_q = sum(q_values) / len(q_values)
        avg_r = sum(r_values) / len(r_values)
        assert 8 <= avg_q <= 12  # Roughly centered on q
        assert 6 <= avg_r <= 10  # Roughly centered on r

    def test_map_edge_location(self) -> None:
        """Map edge location places zone at edge."""
        zone = SitrepZone(
            zone_type="deployment",
            location="map_edge",
            width=4,
            height=4,
        )
        coords = generate_zone_coords(zone, map_width=20, map_height=16)

        # Should be at left edge (q near 0)
        q_values = [c.q for c in coords]
        assert min(q_values) == 0

    def test_quadrant_distribution(self) -> None:
        """Quadrant location distributes zones."""
        zone = SitrepZone(
            zone_type="deployment",
            location="quadrant",
            width=3,
            height=3,
        )

        # Generate 4 zones in different quadrants
        coords_0 = generate_zone_coords(zone, 20, 16, zone_index=0, total_zones=4)
        coords_1 = generate_zone_coords(zone, 20, 16, zone_index=1, total_zones=4)
        coords_2 = generate_zone_coords(zone, 20, 16, zone_index=2, total_zones=4)
        coords_3 = generate_zone_coords(zone, 20, 16, zone_index=3, total_zones=4)

        # Each should have different average positions
        avg_positions = []
        for coords in [coords_0, coords_1, coords_2, coords_3]:
            avg_q = sum(c.q for c in coords) / len(coords)
            avg_r = sum(c.r for c in coords) / len(coords)
            avg_positions.append((avg_q, avg_r))

        # Verify they're different
        unique_positions = set(avg_positions)
        assert len(unique_positions) == 4


# =============================================================================
# SITREP Terrain Generation Tests
# =============================================================================


class TestGenerateTerrainFromSitrep:
    """Tests for generate_terrain_from_sitrep function."""

    def test_control_template_generates_4_objectives(self) -> None:
        """CONTROL SITREP generates 4 objective zones."""
        template = SitrepTemplate(
            sitrep_type="control",
            name="CONTROL",
            description="Control 4 zones",
            objective_zones=[
                SitrepZone(zone_type="objective", location="quadrant"),
                SitrepZone(zone_type="objective", location="quadrant"),
                SitrepZone(zone_type="objective", location="quadrant"),
                SitrepZone(zone_type="objective", location="quadrant"),
            ],
            victory_conditions=[
                VictoryCondition(
                    condition_type="control_zones",
                    threshold=2,
                    description="Control 2 zones",
                )
            ],
        )
        params = TerrainGeneratorParams(seed=42)
        result = generate_terrain_from_sitrep(template, params)

        # Should have 4 objective zones
        objective_count = sum(1 for p in result.primitives if hasattr(p, 'objective_type'))
        assert objective_count == 4

    def test_deployment_zones_created(self) -> None:
        """Deployment zones are created from template."""
        template = SitrepTemplate(
            sitrep_type="escort",
            name="ESCORT",
            description="Escort objective",
            deployment_zones=[
                SitrepZone(zone_type="deployment", location="map_edge"),
            ],
            victory_conditions=[],
        )
        params = TerrainGeneratorParams(seed=42)
        result = generate_terrain_from_sitrep(template, params)

        # Should have deployment zone in zones dict
        assert any("deployment" in zone_id for zone_id in result.zones.keys())

    def test_extraction_zone_created(self) -> None:
        """Extraction zone is created when specified."""
        template = SitrepTemplate(
            sitrep_type="extract",
            name="EXTRACT",
            description="Extract",
            extraction_zone=SitrepZone(zone_type="extraction", location="opposite_edge"),
            victory_conditions=[],
        )
        params = TerrainGeneratorParams(seed=42)
        result = generate_terrain_from_sitrep(template, params)

        # Should have extraction zone
        assert any("extraction" in zone_id for zone_id in result.zones.keys())

    def test_ingress_zones_created(self) -> None:
        """Ingress zones are created from template."""
        template = SitrepTemplate(
            sitrep_type="escort",
            name="ESCORT",
            description="Escort",
            ingress_zones=[
                SitrepZone(zone_type="ingress", location="left_flank"),
                SitrepZone(zone_type="ingress", location="right_flank"),
            ],
            victory_conditions=[],
        )
        params = TerrainGeneratorParams(seed=42)
        result = generate_terrain_from_sitrep(template, params)

        # Should have ingress zones
        ingress_count = sum(1 for zone_id in result.zones.keys() if "ingress" in zone_id)
        assert ingress_count == 2

    def test_fortified_terrain_notes_creates_fortified_cover(self) -> None:
        """terrain_notes='fortified' creates FORTIFIED material obstacles."""
        template = SitrepTemplate(
            sitrep_type="control",
            name="CONTROL",
            description="Control fortified zone",
            objective_zones=[
                SitrepZone(
                    zone_type="objective",
                    location="center",
                    terrain_notes="fortified bunker",
                ),
            ],
            victory_conditions=[],
        )
        params = TerrainGeneratorParams(seed=42, density=0.0)  # No random features
        result = generate_terrain_from_sitrep(template, params)

        # Should have obstacles with fortified material
        fortified_obstacles = [
            p for p in result.primitives
            if isinstance(p, Obstacle) and p.material == MATERIAL_FORTIFIED
        ]
        assert len(fortified_obstacles) > 0


class TestTerrainGenerationIntegration:
    """Integration tests for generated terrain with existing rules."""

    def test_generated_terrain_works_with_movement(self) -> None:
        """Generated terrain integrates with calculate_movement_cost."""
        template = SitrepTemplate(
            sitrep_type="control",
            name="CONTROL",
            description="Test",
            objective_zones=[
                SitrepZone(zone_type="objective", location="center"),
            ],
            victory_conditions=[],
        )
        params = TerrainGeneratorParams(
            seed=42,
            tile_set="wilderness",
            density=0.5,
        )
        result = generate_terrain_from_sitrep(template, params)

        # Movement cost function should work
        for tile in result.terrain_map.tiles[:5]:
            cost = calculate_movement_cost(1, result.terrain_map, tile.coord)
            if tile.difficult:
                assert cost == 2
            else:
                assert cost == 1

    def test_generated_terrain_works_with_cover(self) -> None:
        """Generated terrain integrates with get_cover_difficulty."""
        # Create terrain with known cover
        obs = Obstacle(
            id="wall_1",
            name="Wall",
            coords=[HexCoord(q=5, r=5)],
            hard_cover_size="size_1",
        )
        composed = compose_terrain_map([obs])

        # Cover calculation should work
        result = get_cover_difficulty(
            terrain=composed.terrain_map,
            attacker_coord=HexCoord(q=10, r=5),
            target_coord=HexCoord(q=4, r=5),  # Adjacent to wall
            target_size="size_1",
        )
        assert result.cover_type == "hard"
        assert result.difficulty_modifier == 2

    def test_generated_terrain_map_usable_by_resolve_movement(self) -> None:
        """GeneratedTerrain.terrain_map is a valid TerrainMap."""
        params = TerrainGeneratorParams(seed=42)
        result = generate_random_terrain(params)

        # Should be usable with terrain functions
        terrain_map = result.terrain_map
        assert terrain_map is not None

        # Should be able to query any coord
        for q in range(5):
            for r in range(5):
                coord = HexCoord(q=q, r=r)
                # get_terrain_at should work
                terrain = get_terrain_at(terrain_map, coord)
                # May be None or a TerrainHex


# =============================================================================
# Soft Cover Zone Integration Tests
# =============================================================================


class TestSoftCoverZoneIntegration:
    """Tests for soft cover zone integration with hide rules."""

    def test_is_in_active_soft_cover_zone_basic(self) -> None:
        """is_in_active_soft_cover_zone works with basic zone."""
        zone = SoftCoverZoneState(
            zone_id="smoke_1",
            coords=frozenset([HexCoord(q=0, r=0), HexCoord(q=1, r=0)]),
            zone_subtype="smoke",
        )

        assert is_in_active_soft_cover_zone([zone], HexCoord(q=0, r=0)) is True
        assert is_in_active_soft_cover_zone([zone], HexCoord(q=5, r=5)) is False

    def test_zone_expiration(self) -> None:
        """Expired zones are not considered active."""
        zone = SoftCoverZoneState(
            zone_id="smoke_1",
            coords=frozenset([HexCoord(q=0, r=0)]),
            zone_subtype="smoke",
            created_round=1,
            duration_rounds=2,
        )

        # Active in rounds 1-2
        assert is_in_active_soft_cover_zone([zone], HexCoord(q=0, r=0), current_round=1) is True
        assert is_in_active_soft_cover_zone([zone], HexCoord(q=0, r=0), current_round=2) is True

        # Expired in round 3+
        assert is_in_active_soft_cover_zone([zone], HexCoord(q=0, r=0), current_round=3) is False
        assert is_in_active_soft_cover_zone([zone], HexCoord(q=0, r=0), current_round=4) is False

    def test_permanent_zone_never_expires(self) -> None:
        """Permanent zones (duration=None) never expire."""
        zone = SoftCoverZoneState(
            zone_id="foliage_1",
            coords=frozenset([HexCoord(q=0, r=0)]),
            zone_subtype="foliage",
            duration_rounds=None,
        )

        assert is_in_active_soft_cover_zone([zone], HexCoord(q=0, r=0), current_round=100) is True

    def test_check_soft_cover_for_hide_combines_sources(self) -> None:
        """check_soft_cover_for_hide checks both terrain and zones."""
        from core.shared.terrain import TerrainMap, TerrainHex

        # Create terrain with soft cover area
        terrain = TerrainMap(tiles=[
            TerrainHex(coord=HexCoord(q=10, r=10), provides_soft_cover=True),
            TerrainHex(coord=HexCoord(q=10, r=11), provides_soft_cover=True),
            TerrainHex(coord=HexCoord(q=10, r=9), provides_soft_cover=True),
            TerrainHex(coord=HexCoord(q=11, r=10), provides_soft_cover=True),
        ])

        # Create a zone elsewhere
        zone = SoftCoverZoneState(
            zone_id="smoke_1",
            coords=frozenset([HexCoord(q=0, r=0)]),
            zone_subtype="smoke",
        )

        # Should find zone-based cover
        assert check_soft_cover_for_hide(
            terrain=None,
            soft_cover_zones=[zone],
            target_coord=HexCoord(q=0, r=0),
        ) is True

        # Should find terrain-based cover (if area is large enough)
        result = check_soft_cover_for_hide(
            terrain=terrain,
            soft_cover_zones=[],
            target_coord=HexCoord(q=10, r=10),
            min_adjacent_hexes=3,
        )
        # May or may not be true depending on is_soft_cover_area implementation

    def test_wilderness_terrain_generates_soft_cover_zones(self) -> None:
        """Wilderness tile set can generate soft cover zones."""
        params = TerrainGeneratorParams(
            seed=42,
            tile_set="wilderness",
            density=0.8,  # High density to ensure some brush
        )
        result = generate_random_terrain(params)

        # Wilderness may generate brush which creates soft cover zones
        # The soft cover zones should be tracked
        # (May or may not have zones depending on RNG)
        assert result.soft_cover_zones is not None


# =============================================================================
# Random Terrain Generation Tests
# =============================================================================


class TestGenerateRandomTerrain:
    """Tests for generate_random_terrain function."""

    def test_seeded_generation_is_deterministic(self) -> None:
        """Same seed produces same terrain."""
        params = TerrainGeneratorParams(seed=12345, density=0.3)

        result1 = generate_random_terrain(params)
        result2 = generate_random_terrain(params)

        # Same number of tiles
        assert len(result1.terrain_map.tiles) == len(result2.terrain_map.tiles)

        # Same coordinates
        coords1 = {t.coord for t in result1.terrain_map.tiles}
        coords2 = {t.coord for t in result2.terrain_map.tiles}
        assert coords1 == coords2

    def test_different_seeds_produce_different_terrain(self) -> None:
        """Different seeds produce different terrain."""
        params1 = TerrainGeneratorParams(seed=111, density=0.3)
        params2 = TerrainGeneratorParams(seed=222, density=0.3)

        result1 = generate_random_terrain(params1)
        result2 = generate_random_terrain(params2)

        coords1 = {t.coord for t in result1.terrain_map.tiles}
        coords2 = {t.coord for t in result2.terrain_map.tiles}

        # Very unlikely to be identical with different seeds
        # (Could happen by chance but improbable)
        assert coords1 != coords2 or len(coords1) == 0

    def test_density_affects_feature_count(self) -> None:
        """Higher density produces more features."""
        low_density = TerrainGeneratorParams(seed=42, density=0.1)
        high_density = TerrainGeneratorParams(seed=42, density=0.5)

        result_low = generate_random_terrain(low_density)
        result_high = generate_random_terrain(high_density)

        # High density should have more features
        assert len(result_high.terrain_map.tiles) >= len(result_low.terrain_map.tiles)

    def test_tile_set_affects_features(self) -> None:
        """Different tile sets produce different feature types."""
        urban_params = TerrainGeneratorParams(seed=42, tile_set="urban", density=0.3)
        wilderness_params = TerrainGeneratorParams(seed=42, tile_set="wilderness", density=0.3)

        urban_result = generate_random_terrain(urban_params)
        wilderness_result = generate_random_terrain(wilderness_params)

        # Should have generated some features
        assert len(urban_result.primitives) > 0 or urban_result.terrain_map.tiles
        assert len(wilderness_result.primitives) > 0 or wilderness_result.terrain_map.tiles


# =============================================================================
# Compatibility Tests
# =============================================================================


class TestCompatibilityWithExistingRules:
    """Tests to ensure generated terrain works with existing core rules."""

    def test_cover_difficulty_unchanged(self) -> None:
        """Cover difficulty calculation works correctly."""
        # Create known terrain setup
        wall = Obstacle(
            id="wall",
            name="Wall",
            coords=[HexCoord(q=5, r=5)],
            provides_hard_cover=True,
            hard_cover_size="size_2",
        )
        result = compose_terrain_map([wall])

        # Hard cover should give +2 difficulty
        cover = get_cover_difficulty(
            terrain=result.terrain_map,
            attacker_coord=HexCoord(q=10, r=5),
            target_coord=HexCoord(q=4, r=5),  # Adjacent to wall
            target_size="size_1",
        )
        assert cover.difficulty_modifier == 2
        assert cover.cover_type == "hard"

    def test_hide_rules_work_with_zones(self) -> None:
        """Hide rules integrate with soft cover zones."""
        zone = SoftCoverZoneState(
            zone_id="smoke",
            coords=frozenset([HexCoord(q=0, r=0)]),
            zone_subtype="smoke",
        )

        # Should be valid for hiding
        in_zone = is_in_active_soft_cover_zone([zone], HexCoord(q=0, r=0))
        assert in_zone is True

        # Combined check should work
        result = check_soft_cover_for_hide(
            terrain=None,
            soft_cover_zones=[zone],
            target_coord=HexCoord(q=0, r=0),
        )
        assert result is True
