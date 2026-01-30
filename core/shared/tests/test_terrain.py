"""Terrain resolution tests for Lancer combat."""

from core.shared.terrain import (
    get_terrain_at,
    get_terrain_effects_at,
    calculate_movement_cost,
    get_cover_difficulty,
    get_cover_for_footprint,
    check_soft_cover,
    check_hard_cover_available,
    resolve_dangerous_terrain,
    get_elevation_bonus,
    calculate_climb_cost,
)
from core.mech.terrain import TerrainHex, TerrainMap
from core.mech.grid import HexCoord


class TestGetTerrainAt:
    """Tests for get_terrain_at function."""

    def test_no_terrain_returns_none(self) -> None:
        """None terrain returns None."""
        result = get_terrain_at(None, HexCoord(q=0, r=0))
        assert result is None

    def test_empty_terrain_returns_none(self) -> None:
        """Empty terrain map returns None."""
        terrain = TerrainMap(tiles=[])
        result = get_terrain_at(terrain, HexCoord(q=0, r=0))
        assert result is None

    def test_finds_terrain_at_coord(self) -> None:
        """Finds terrain at specified coordinate."""
        terrain = TerrainMap(
            tiles=[TerrainHex(coord=HexCoord(q=2, r=3), difficult=True)]
        )
        result = get_terrain_at(terrain, HexCoord(q=2, r=3))
        assert result is not None
        assert result.difficult is True

    def test_returns_none_for_empty_coord(self) -> None:
        """Returns None for coordinate with no terrain."""
        terrain = TerrainMap(
            tiles=[TerrainHex(coord=HexCoord(q=0, r=0), difficult=True)]
        )
        result = get_terrain_at(terrain, HexCoord(q=5, r=5))
        assert result is None


class TestGetTerrainEffectsAt:
    """Tests for get_terrain_effects_at function."""

    def test_no_terrain_returns_defaults(self) -> None:
        """No terrain returns default effects."""
        result = get_terrain_effects_at(None, HexCoord(q=0, r=0))
        assert result.elevation == 0
        assert result.blocks_line_of_sight is False
        assert result.provides_soft_cover is False
        assert result.provides_hard_cover is False
        assert result.difficult is False
        assert result.dangerous is False

    def test_returns_all_terrain_properties(self) -> None:
        """Returns all terrain properties."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(
                    coord=HexCoord(q=1, r=0),
                    elevation=2,
                    blocks_line_of_sight=True,
                    provides_soft_cover=True,
                    provides_hard_cover=True,
                    hard_cover_size="size_2",
                    difficult=True,
                    dangerous=True,
                )
            ]
        )
        result = get_terrain_effects_at(terrain, HexCoord(q=1, r=0))
        assert result.coord == HexCoord(q=1, r=0)
        assert result.elevation == 2
        assert result.blocks_line_of_sight is True
        assert result.provides_soft_cover is True
        assert result.provides_hard_cover is True
        assert result.hard_cover_size == "size_2"
        assert result.difficult is True
        assert result.dangerous is True


class TestCalculateMovementCost:
    """Tests for calculate_movement_cost function."""

    def test_normal_terrain_cost(self) -> None:
        """Normal terrain costs 1:1."""
        terrain = TerrainMap(tiles=[])
        result = calculate_movement_cost(3, terrain, HexCoord(q=0, r=0))
        assert result == 3

    def test_no_terrain_cost(self) -> None:
        """No terrain costs 1:1."""
        result = calculate_movement_cost(3, None, HexCoord(q=0, r=0))
        assert result == 3

    def test_difficult_terrain_cost(self) -> None:
        """Difficult terrain costs 2:1."""
        terrain = TerrainMap(
            tiles=[TerrainHex(coord=HexCoord(q=0, r=0), difficult=True)]
        )
        result = calculate_movement_cost(3, terrain, HexCoord(q=0, r=0))
        assert result == 6

    def test_custom_difficult_cost(self) -> None:
        """Custom difficult terrain cost works."""
        terrain = TerrainMap(
            tiles=[TerrainHex(coord=HexCoord(q=0, r=0), difficult=True)]
        )
        result = calculate_movement_cost(
            2, terrain, HexCoord(q=0, r=0), difficult_cost=3
        )
        assert result == 6

    def test_single_space_difficult(self) -> None:
        """Single space through difficult terrain."""
        terrain = TerrainMap(
            tiles=[TerrainHex(coord=HexCoord(q=1, r=0), difficult=True)]
        )
        result = calculate_movement_cost(1, terrain, HexCoord(q=1, r=0))
        assert result == 2


class TestCheckSoftCover:
    """Tests for check_soft_cover function."""

    def test_no_terrain_no_soft_cover(self) -> None:
        """No terrain means no soft cover."""
        result = check_soft_cover(None, HexCoord(q=0, r=0), HexCoord(q=1, r=0))
        assert result is False

    def test_no_los_block_no_soft_cover(self) -> None:
        """Terrain without LOS block provides no soft cover."""
        terrain = TerrainMap(
            tiles=[TerrainHex(coord=HexCoord(q=0, r=0), provides_soft_cover=False)]
        )
        result = check_soft_cover(terrain, HexCoord(q=0, r=0), HexCoord(q=0, r=0))
        assert result is False

    def test_soft_cover_flag(self) -> None:
        """Soft cover flag returns True."""
        terrain = TerrainMap(
            tiles=[TerrainHex(coord=HexCoord(q=0, r=0), provides_soft_cover=True)]
        )
        result = check_soft_cover(terrain, HexCoord(q=0, r=0), HexCoord(q=0, r=0))
        assert result is True

    def test_los_block_provides_soft_cover(self) -> None:
        """Terrain that blocks LOS provides soft cover."""
        terrain = TerrainMap(
            tiles=[TerrainHex(coord=HexCoord(q=0, r=0), blocks_line_of_sight=True)]
        )
        result = check_soft_cover(terrain, HexCoord(q=0, r=0), HexCoord(q=0, r=0))
        assert result is True


class TestCheckHardCoverAvailable:
    """Tests for check_hard_cover_available function."""

    def test_no_terrain(self) -> None:
        """No terrain means no hard cover."""
        result = check_hard_cover_available(
            None, HexCoord(q=0, r=0), HexCoord(q=1, r=0), "size_1"
        )
        assert result.available is False

    def test_no_hard_cover_flag(self) -> None:
        """Terrain without hard cover flag returns False."""
        terrain = TerrainMap(
            tiles=[TerrainHex(coord=HexCoord(q=0, r=0), provides_hard_cover=False)]
        )
        result = check_hard_cover_available(
            terrain, HexCoord(q=0, r=0), HexCoord(q=1, r=0), "size_1"
        )
        assert result.available is False

    def test_size_mismatch(self) -> None:
        """Cover smaller than target provides no benefit."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(
                    coord=HexCoord(q=1, r=0),
                    provides_hard_cover=True,
                    hard_cover_size="size_half",
                )
            ]
        )
        result = check_hard_cover_available(
            terrain, HexCoord(q=0, r=0), HexCoord(q=2, r=0), "size_2"
        )
        assert result.available is False
        assert result.size_match is False

    def test_size_match(self) -> None:
        """Cover size equal to target works when adjacent."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(
                    coord=HexCoord(q=1, r=0),
                    provides_hard_cover=True,
                    hard_cover_size="size_1",
                )
            ]
        )
        result = check_hard_cover_available(
            terrain, HexCoord(q=0, r=0), HexCoord(q=2, r=0), "size_1"
        )
        assert result.available is True
        assert result.size_match is True

    def test_larger_cover(self) -> None:
        """Cover larger than target works when adjacent."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(
                    coord=HexCoord(q=1, r=0),
                    provides_hard_cover=True,
                    hard_cover_size="size_3",
                )
            ]
        )
        result = check_hard_cover_available(
            terrain, HexCoord(q=0, r=0), HexCoord(q=2, r=0), "size_1"
        )
        assert result.available is True

    def test_no_size_specified(self) -> None:
        """Hard cover without size specified returns False."""
        terrain = TerrainMap(
            tiles=[TerrainHex(coord=HexCoord(q=1, r=0), provides_hard_cover=True)]
        )
        result = check_hard_cover_available(
            terrain, HexCoord(q=0, r=0), HexCoord(q=2, r=0), "size_1"
        )
        assert result.available is False

    def test_target_on_cover_hex(self) -> None:
        """Target standing on hard cover hex doesn't count as adjacent."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(
                    coord=HexCoord(q=0, r=0),
                    provides_hard_cover=True,
                    hard_cover_size="size_1",
                )
            ]
        )
        result = check_hard_cover_available(
            terrain, HexCoord(q=0, r=0), HexCoord(q=0, r=0), "size_1"
        )
        assert result.available is False


class TestGetCoverDifficulty:
    """Tests for get_cover_difficulty function."""

    def test_no_cover(self) -> None:
        """No terrain means no cover."""
        result = get_cover_difficulty(
            None, HexCoord(q=0, r=0), HexCoord(q=1, r=0), "size_1"
        )
        assert result.cover_type == "none"
        assert result.difficulty_modifier == 0

    def test_soft_cover(self) -> None:
        """Soft cover gives +1 difficulty."""
        terrain = TerrainMap(
            tiles=[TerrainHex(coord=HexCoord(q=0, r=0), provides_soft_cover=True)]
        )
        result = get_cover_difficulty(
            terrain, HexCoord(q=0, r=0), HexCoord(q=0, r=0), "size_1"
        )
        assert result.cover_type == "soft"
        assert result.difficulty_modifier == 1

    def test_hard_cover(self) -> None:
        """Hard cover gives +2 difficulty."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(
                    coord=HexCoord(q=0, r=0),
                    provides_hard_cover=True,
                    hard_cover_size="size_2",
                ),
                TerrainHex(
                    coord=HexCoord(q=1, r=0),
                    provides_hard_cover=True,
                    hard_cover_size="size_2",
                ),
            ]
        )
        result = get_cover_difficulty(
            terrain, HexCoord(q=2, r=0), HexCoord(q=1, r=0), "size_1"
        )
        assert result.cover_type == "hard"
        assert result.difficulty_modifier == 2

    def test_custom_cover_difficulties(self) -> None:
        """Custom cover difficulty values work."""
        terrain = TerrainMap(
            tiles=[TerrainHex(coord=HexCoord(q=0, r=0), provides_soft_cover=True)]
        )
        result = get_cover_difficulty(
            terrain,
            HexCoord(q=0, r=0),
            HexCoord(q=0, r=0),
            "size_1",
            soft_cover_difficulty=2,
        )
        assert result.difficulty_modifier == 2


class TestResolveDangerousTerrain:
    """Tests for resolve_dangerous_terrain function."""

    def test_no_dangerous_terrain(self) -> None:
        """No dangerous terrain returns no check required."""
        terrain = TerrainMap(tiles=[])
        result = resolve_dangerous_terrain(terrain, HexCoord(q=0, r=0), skill_bonus=0)
        assert result.check_required is False

    def test_no_terrain(self) -> None:
        """No terrain returns no check required."""
        result = resolve_dangerous_terrain(None, HexCoord(q=0, r=0), skill_bonus=0)
        assert result.check_required is False

    def test_check_passed(self) -> None:
        """Successful check deals no damage."""
        terrain = TerrainMap(
            tiles=[TerrainHex(coord=HexCoord(q=0, r=0), dangerous=True)]
        )
        result = resolve_dangerous_terrain(terrain, HexCoord(q=0, r=0), skill_bonus=10)
        assert result.check_required is True
        assert result.check_passed is True
        assert result.damage_dealt == 0

    def test_check_failed(self) -> None:
        """Failed check deals damage. Use very low skill bonus to guarantee failure."""
        terrain = TerrainMap(
            tiles=[TerrainHex(coord=HexCoord(q=0, r=0), dangerous=True)]
        )
        result = resolve_dangerous_terrain(terrain, HexCoord(q=0, r=0), skill_bonus=-15)
        assert result.check_required is True
        assert result.check_passed is False
        assert result.damage_dealt == 5

    def test_custom_damage(self) -> None:
        """Custom damage value works."""
        terrain = TerrainMap(
            tiles=[TerrainHex(coord=HexCoord(q=0, r=0), dangerous=True)]
        )
        result = resolve_dangerous_terrain(
            terrain, HexCoord(q=0, r=0), skill_bonus=-15, damage=10
        )
        assert result.damage_dealt == 10

    def test_damage_type(self) -> None:
        """Damage type is set correctly."""
        terrain = TerrainMap(
            tiles=[TerrainHex(coord=HexCoord(q=0, r=0), dangerous=True)]
        )
        result = resolve_dangerous_terrain(
            terrain, HexCoord(q=0, r=0), skill_bonus=-15, damage_type="energy"
        )
        assert result.damage_type == "energy"

    def test_check_once_per_round(self) -> None:
        """Same round check is skipped when using shared state."""
        terrain = TerrainMap(
            tiles=[TerrainHex(coord=HexCoord(q=0, r=0), dangerous=True)]
        )
        rounds_checked: set[int] = set()
        result1 = resolve_dangerous_terrain(
            terrain,
            HexCoord(q=0, r=0),
            skill_bonus=-15,
            round_checked=1,
            rounds_already_checked=rounds_checked,
        )
        result2 = resolve_dangerous_terrain(
            terrain,
            HexCoord(q=0, r=0),
            skill_bonus=-15,
            round_checked=1,
            rounds_already_checked=rounds_checked,
        )
        assert result1.check_already_done_this_round is False
        assert result2.check_already_done_this_round is True
        assert result2.check_passed is None

    def test_different_rounds(self) -> None:
        """Different rounds both check."""
        terrain = TerrainMap(
            tiles=[TerrainHex(coord=HexCoord(q=0, r=0), dangerous=True)]
        )
        rounds_checked: set[int] = set()
        result1 = resolve_dangerous_terrain(
            terrain,
            HexCoord(q=0, r=0),
            skill_bonus=-15,
            round_checked=1,
            rounds_already_checked=rounds_checked,
        )
        result2 = resolve_dangerous_terrain(
            terrain,
            HexCoord(q=0, r=0),
            skill_bonus=-15,
            round_checked=2,
            rounds_already_checked=rounds_checked,
        )
        assert result1.check_already_done_this_round is False
        assert result2.check_already_done_this_round is False


class TestGetElevationBonus:
    """Tests for get_elevation_bonus function."""

    def test_higher_elevation(self) -> None:
        """Higher elevation gives +1."""
        result = get_elevation_bonus(attacker_elevation=2, target_elevation=0)
        assert result == 1

    def test_same_elevation(self) -> None:
        """Same elevation gives no bonus."""
        result = get_elevation_bonus(attacker_elevation=1, target_elevation=1)
        assert result == 0

    def test_lower_elevation(self) -> None:
        """Lower elevation gives no bonus."""
        result = get_elevation_bonus(attacker_elevation=0, target_elevation=2)
        assert result == 0

    def test_zero_elevation(self) -> None:
        """Zero elevation works correctly."""
        result = get_elevation_bonus(attacker_elevation=0, target_elevation=0)
        assert result == 0


class TestCalculateClimbCost:
    """Tests for calculate_climb_cost function."""

    def test_climb_cost(self) -> None:
        """Climbing costs 2:1 by default."""
        result = calculate_climb_cost(3)
        assert result == 6

    def test_custom_climb_cost(self) -> None:
        """Custom climb cost works."""
        result = calculate_climb_cost(2, climb_cost=3)
        assert result == 6

    def test_single_space(self) -> None:
        """Single space climb."""
        result = calculate_climb_cost(1)
        assert result == 2


class TestGetCoverForFootprint:
    """Tests for get_cover_for_footprint function for Size 2+ targets."""

    def test_size_1_target_uses_single_hex(self) -> None:
        """Size 1 target uses single hex, same as regular cover check."""
        terrain = TerrainMap(
            tiles=[TerrainHex(coord=HexCoord(q=1, r=0), provides_soft_cover=True)]
        )
        result = get_cover_for_footprint(
            attacker_coord=HexCoord(q=0, r=0),
            target_center=HexCoord(q=1, r=0),
            target_size="size_1",
            terrain=terrain,
        )
        assert result.cover_type == "soft"
        assert result.difficulty_modifier == 1

    def test_size_2_target_no_cover_any_hex(self) -> None:
        """Size 2 target with no cover at any footprint hex returns no cover."""
        terrain = TerrainMap(tiles=[])
        result = get_cover_for_footprint(
            attacker_coord=HexCoord(q=0, r=0),
            target_center=HexCoord(q=3, r=0),
            target_size="size_2",
            terrain=terrain,
        )
        assert result.cover_type == "none"
        assert result.difficulty_modifier == 0

    def test_size_2_target_partial_cover_uses_least(self) -> None:
        """Size 2 target with partial cover uses most exposed (least cover) hex."""
        # Soft cover at center (3,0) but no cover at edge hex (4,0)
        terrain = TerrainMap(
            tiles=[TerrainHex(coord=HexCoord(q=3, r=0), provides_soft_cover=True)]
        )
        result = get_cover_for_footprint(
            attacker_coord=HexCoord(q=0, r=0),
            target_center=HexCoord(q=3, r=0),
            target_size="size_2",
            terrain=terrain,
        )
        # Should return no cover since edge hexes have no cover
        assert result.cover_type == "none"

    def test_size_2_target_all_soft_cover(self) -> None:
        """Size 2 target with soft cover at all footprint hexes returns soft cover."""
        # Create soft cover at center and all 6 adjacent hexes for size 2
        tiles = [
            TerrainHex(coord=HexCoord(q=3, r=0), provides_soft_cover=True),
            TerrainHex(coord=HexCoord(q=4, r=0), provides_soft_cover=True),
            TerrainHex(coord=HexCoord(q=2, r=0), provides_soft_cover=True),
            TerrainHex(coord=HexCoord(q=3, r=1), provides_soft_cover=True),
            TerrainHex(coord=HexCoord(q=3, r=-1), provides_soft_cover=True),
            TerrainHex(coord=HexCoord(q=4, r=-1), provides_soft_cover=True),
            TerrainHex(coord=HexCoord(q=2, r=1), provides_soft_cover=True),
        ]
        terrain = TerrainMap(tiles=tiles)
        result = get_cover_for_footprint(
            attacker_coord=HexCoord(q=0, r=0),
            target_center=HexCoord(q=3, r=0),
            target_size="size_2",
            terrain=terrain,
        )
        assert result.cover_type == "soft"
        assert result.difficulty_modifier == 1

    def test_size_2_mixed_coverage_returns_least(self) -> None:
        """Size 2 target with mixed coverage returns most favorable for attacker."""
        # Soft cover at some hexes, but not all - should return no cover
        # (because attacker targets the most exposed hex)
        tiles = [
            TerrainHex(coord=HexCoord(q=3, r=0), provides_soft_cover=True),
            TerrainHex(coord=HexCoord(q=4, r=0), provides_soft_cover=True),
            TerrainHex(coord=HexCoord(q=2, r=0), provides_soft_cover=True),
            # Missing: (3, 1), (3, -1), (4, -1), (2, 1) - these have no cover
        ]
        terrain = TerrainMap(tiles=tiles)
        result = get_cover_for_footprint(
            attacker_coord=HexCoord(q=0, r=0),
            target_center=HexCoord(q=3, r=0),
            target_size="size_2",
            terrain=terrain,
        )
        # Should return no cover since some edge hexes have no cover
        assert result.cover_type == "none"
        assert result.difficulty_modifier == 0

    def test_size_2_exposed_corner(self) -> None:
        """Size 2 target with one exposed corner returns no cover."""
        # Cover everywhere except one footprint hex
        tiles = [
            TerrainHex(coord=HexCoord(q=3, r=0), provides_soft_cover=True),
            TerrainHex(coord=HexCoord(q=2, r=0), provides_soft_cover=True),
            TerrainHex(coord=HexCoord(q=3, r=1), provides_soft_cover=True),
            TerrainHex(coord=HexCoord(q=3, r=-1), provides_soft_cover=True),
            TerrainHex(coord=HexCoord(q=4, r=-1), provides_soft_cover=True),
            TerrainHex(coord=HexCoord(q=2, r=1), provides_soft_cover=True),
            # Missing (4,0) - exposed corner
        ]
        terrain = TerrainMap(tiles=tiles)
        result = get_cover_for_footprint(
            attacker_coord=HexCoord(q=0, r=0),
            target_center=HexCoord(q=3, r=0),
            target_size="size_2",
            terrain=terrain,
        )
        # Should return no cover since (4,0) has no cover
        assert result.cover_type == "none"
        assert result.difficulty_modifier == 0

    def test_no_terrain_returns_no_cover(self) -> None:
        """No terrain returns no cover."""
        result = get_cover_for_footprint(
            attacker_coord=HexCoord(q=0, r=0),
            target_center=HexCoord(q=3, r=0),
            target_size="size_2",
            terrain=None,
        )
        assert result.cover_type == "none"
        assert result.difficulty_modifier == 0
