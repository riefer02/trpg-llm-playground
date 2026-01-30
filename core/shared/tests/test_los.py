"""Line of sight resolution tests for Lancer combat."""

from core.mech.grid import HexCoord, HexPosition
from core.mech.terrain import TerrainHex, TerrainMap
from core.shared.los import (
    LOSResult,
    LOSCheckRequest,
    check_line_of_sight,
    check_obscured_los,
    check_clear_los,
    check_path_clear,
    check_los_with_cover,
    check_elevation_blocks_los,
    get_los_blocking_hexes,
    get_los_obscuring_hexes,
    check_line_of_sight_to_footprint,
)


class TestLOSResult:
    """Tests for LOSResult model."""

    def test_default_values(self) -> None:
        """Default values are correctly set."""
        result = LOSResult()
        assert result.has_los is False
        assert result.los_type == "blocked"
        assert result.blocked_by == []
        assert result.obscured_by == []
        assert result.reason == ""

    def test_clear_los_result(self) -> None:
        """Clear LOS result has expected values."""
        result = LOSResult(
            has_los=True,
            los_type="clear",
            reason="Clear line of sight",
        )
        assert result.has_los is True
        assert result.los_type == "clear"


class TestLOSCheckRequest:
    """Tests for LOSCheckRequest model."""

    def test_default_values(self) -> None:
        """Default values are correctly set."""
        request = LOSCheckRequest(
            attacker_pos=HexPosition(coord=HexCoord(q=0, r=0)),
            target_pos=HexPosition(coord=HexCoord(q=1, r=0)),
        )
        assert request.terrain is None
        assert request.check_elevation is True

    def test_with_terrain(self) -> None:
        """Request with terrain is correctly stored."""
        terrain = TerrainMap(tiles=[])
        request = LOSCheckRequest(
            attacker_pos=HexPosition(coord=HexCoord(q=0, r=0)),
            target_pos=HexPosition(coord=HexCoord(q=1, r=0)),
            terrain=terrain,
            check_elevation=False,
        )
        assert request.terrain == terrain
        assert request.check_elevation is False


class TestCheckLineOfSight:
    """Tests for check_line_of_sight function."""

    def test_same_position(self) -> None:
        """Same position returns clear LOS."""
        pos = HexPosition(coord=HexCoord(q=0, r=0))
        request = LOSCheckRequest(
            attacker_pos=pos,
            target_pos=pos,
        )
        result = check_line_of_sight(request)
        assert result.has_los is True
        assert result.los_type == "clear"

    def test_no_terrain_clear(self) -> None:
        """No terrain means clear LOS."""
        request = LOSCheckRequest(
            attacker_pos=HexPosition(coord=HexCoord(q=0, r=0)),
            target_pos=HexPosition(coord=HexCoord(q=3, r=0)),
            terrain=None,
        )
        result = check_line_of_sight(request)
        assert result.has_los is True
        assert result.los_type == "clear"

    def test_empty_terrain_map(self) -> None:
        """Empty terrain map means clear LOS."""
        terrain = TerrainMap(tiles=[])
        request = LOSCheckRequest(
            attacker_pos=HexPosition(coord=HexCoord(q=0, r=0)),
            target_pos=HexPosition(coord=HexCoord(q=3, r=0)),
            terrain=terrain,
        )
        result = check_line_of_sight(request)
        assert result.has_los is True
        assert result.los_type == "clear"

    def test_blocked_by_terrain(self) -> None:
        """Terrain that blocks LOS returns blocked result."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=1, r=0), blocks_line_of_sight=True),
            ]
        )
        request = LOSCheckRequest(
            attacker_pos=HexPosition(coord=HexCoord(q=0, r=0)),
            target_pos=HexPosition(coord=HexCoord(q=2, r=0)),
            terrain=terrain,
        )
        result = check_line_of_sight(request)
        assert result.has_los is False
        assert result.los_type == "blocked"
        assert len(result.blocked_by) == 1
        assert result.blocked_by[0] == HexCoord(q=1, r=0)

    def test_blocked_by_multiple_hexes(self) -> None:
        """Multiple blocking hexes are all reported."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=1, r=0), blocks_line_of_sight=True),
                TerrainHex(coord=HexCoord(q=2, r=0), blocks_line_of_sight=True),
            ]
        )
        request = LOSCheckRequest(
            attacker_pos=HexPosition(coord=HexCoord(q=0, r=0)),
            target_pos=HexPosition(coord=HexCoord(q=3, r=0)),
            terrain=terrain,
        )
        result = check_line_of_sight(request)
        assert result.has_los is False
        assert result.los_type == "blocked"
        assert len(result.blocked_by) == 2

    def test_obscured_by_soft_cover(self) -> None:
        """Soft cover terrain returns obscured result."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=1, r=0), provides_soft_cover=True),
            ]
        )
        request = LOSCheckRequest(
            attacker_pos=HexPosition(coord=HexCoord(q=0, r=0)),
            target_pos=HexPosition(coord=HexCoord(q=2, r=0)),
            terrain=terrain,
        )
        result = check_line_of_sight(request)
        assert result.has_los is True
        assert result.los_type == "obscured"
        assert len(result.obscured_by) == 1
        assert result.obscured_by[0] == HexCoord(q=1, r=0)

    def test_obscured_by_smoke_cloud(self) -> None:
        """Smoke cloud provides soft cover."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=1, r=0), provides_soft_cover=True),
                TerrainHex(coord=HexCoord(q=2, r=0), provides_soft_cover=True),
            ]
        )
        request = LOSCheckRequest(
            attacker_pos=HexPosition(coord=HexCoord(q=0, r=0)),
            target_pos=HexPosition(coord=HexCoord(q=3, r=0)),
            terrain=terrain,
        )
        result = check_line_of_sight(request)
        assert result.has_los is True
        assert result.los_type == "obscured"
        assert len(result.obscured_by) == 2

    def test_blocked_takes_precedence(self) -> None:
        """Blocking terrain takes precedence over soft cover."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(
                    coord=HexCoord(q=1, r=0),
                    blocks_line_of_sight=True,
                    provides_soft_cover=True,
                ),
            ]
        )
        request = LOSCheckRequest(
            attacker_pos=HexPosition(coord=HexCoord(q=0, r=0)),
            target_pos=HexPosition(coord=HexCoord(q=2, r=0)),
            terrain=terrain,
        )
        result = check_line_of_sight(request)
        assert result.has_los is False
        assert result.los_type == "blocked"

    def test_adjacent_positions(self) -> None:
        """Adjacent positions have clear LOS."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=0, r=0)),
            ]
        )
        request = LOSCheckRequest(
            attacker_pos=HexPosition(coord=HexCoord(q=0, r=0)),
            target_pos=HexPosition(coord=HexCoord(q=1, r=0)),
            terrain=terrain,
        )
        result = check_line_of_sight(request)
        assert result.has_los is True
        assert result.los_type == "clear"

    def test_diagonal_path(self) -> None:
        """Diagonal path correctly checks LOS."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=1, r=0), blocks_line_of_sight=True),
            ]
        )
        request = LOSCheckRequest(
            attacker_pos=HexPosition(coord=HexCoord(q=0, r=0)),
            target_pos=HexPosition(coord=HexCoord(q=1, r=1)),
            terrain=terrain,
        )
        result = check_line_of_sight(request)
        assert result.has_los is True
        assert result.los_type == "clear"

    def test_complex_path(self) -> None:
        """Complex path with multiple hexes is correctly checked."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=1, r=0)),
                TerrainHex(coord=HexCoord(q=2, r=0)),
                TerrainHex(coord=HexCoord(q=3, r=0)),
            ]
        )
        request = LOSCheckRequest(
            attacker_pos=HexPosition(coord=HexCoord(q=0, r=0)),
            target_pos=HexPosition(coord=HexCoord(q=4, r=0)),
            terrain=terrain,
        )
        result = check_line_of_sight(request)
        assert result.has_los is True
        assert result.los_type == "clear"


class TestCheckObscuredLOS:
    """Tests for check_obscured_los function."""

    def test_no_obscuring_terrain(self) -> None:
        """No obscuring terrain returns False."""
        terrain = TerrainMap(tiles=[])
        request = LOSCheckRequest(
            attacker_pos=HexPosition(coord=HexCoord(q=0, r=0)),
            target_pos=HexPosition(coord=HexCoord(q=3, r=0)),
            terrain=terrain,
        )
        result = check_obscured_los(request)
        assert result is False

    def test_obscuring_terrain(self) -> None:
        """Obscuring terrain returns True."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=1, r=0), provides_soft_cover=True),
            ]
        )
        request = LOSCheckRequest(
            attacker_pos=HexPosition(coord=HexCoord(q=0, r=0)),
            target_pos=HexPosition(coord=HexCoord(q=2, r=0)),
            terrain=terrain,
        )
        result = check_obscured_los(request)
        assert result is True

    def test_blocked_not_obscuring(self) -> None:
        """Blocked LOS is not considered obscured."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=1, r=0), blocks_line_of_sight=True),
            ]
        )
        request = LOSCheckRequest(
            attacker_pos=HexPosition(coord=HexCoord(q=0, r=0)),
            target_pos=HexPosition(coord=HexCoord(q=2, r=0)),
            terrain=terrain,
        )
        result = check_obscured_los(request)
        assert result is False


class TestCheckClearLOS:
    """Tests for check_clear_los function."""

    def test_clear_path(self) -> None:
        """Clear path returns True."""
        terrain = TerrainMap(tiles=[])
        request = LOSCheckRequest(
            attacker_pos=HexPosition(coord=HexCoord(q=0, r=0)),
            target_pos=HexPosition(coord=HexCoord(q=3, r=0)),
            terrain=terrain,
        )
        result = check_clear_los(request)
        assert result is True

    def test_obscured_not_clear(self) -> None:
        """Obscured LOS is not considered clear."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=1, r=0), provides_soft_cover=True),
            ]
        )
        request = LOSCheckRequest(
            attacker_pos=HexPosition(coord=HexCoord(q=0, r=0)),
            target_pos=HexPosition(coord=HexCoord(q=2, r=0)),
            terrain=terrain,
        )
        result = check_clear_los(request)
        assert result is False

    def test_blocked_not_clear(self) -> None:
        """Blocked LOS is not considered clear."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=1, r=0), blocks_line_of_sight=True),
            ]
        )
        request = LOSCheckRequest(
            attacker_pos=HexPosition(coord=HexCoord(q=0, r=0)),
            target_pos=HexPosition(coord=HexCoord(q=2, r=0)),
            terrain=terrain,
        )
        result = check_clear_los(request)
        assert result is False


class TestCheckPathClear:
    """Tests for check_path_clear function (seeking/arcing weapons)."""

    def test_no_terrain_path_clear(self) -> None:
        """No terrain means path is clear."""
        path_exists, blocking = check_path_clear(
            None, HexCoord(q=0, r=0), HexCoord(q=3, r=0)
        )
        assert path_exists is True
        assert blocking == []

    def test_blocked_path(self) -> None:
        """Blocking terrain blocks path."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=1, r=0), blocks_line_of_sight=True),
            ]
        )
        path_exists, blocking = check_path_clear(
            terrain, HexCoord(q=0, r=0), HexCoord(q=2, r=0)
        )
        assert path_exists is False
        assert len(blocking) == 1
        assert blocking[0] == HexCoord(q=1, r=0)

    def test_clear_with_soft_cover(self) -> None:
        """Soft cover does not block path (for seeking weapons)."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=1, r=0), provides_soft_cover=True),
            ]
        )
        path_exists, blocking = check_path_clear(
            terrain, HexCoord(q=0, r=0), HexCoord(q=2, r=0)
        )
        assert path_exists is True
        assert blocking == []

    def test_multiple_blocking_hexes(self) -> None:
        """Multiple blocking hexes are all reported."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=1, r=0), blocks_line_of_sight=True),
                TerrainHex(coord=HexCoord(q=2, r=0), blocks_line_of_sight=True),
            ]
        )
        path_exists, blocking = check_path_clear(
            terrain, HexCoord(q=0, r=0), HexCoord(q=3, r=0)
        )
        assert path_exists is False
        assert len(blocking) == 2

    def test_adjacent_positions(self) -> None:
        """Adjacent positions have clear path."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=0, r=0)),
            ]
        )
        path_exists, blocking = check_path_clear(
            terrain, HexCoord(q=0, r=0), HexCoord(q=1, r=0)
        )
        assert path_exists is True
        assert blocking == []


class TestCheckLOSWithCover:
    """Tests for check_los_with_cover convenience function."""

    def test_clear_los_no_cover(self) -> None:
        """Clear LOS returns correct result."""
        terrain = TerrainMap(tiles=[])
        result = check_los_with_cover(
            HexPosition(coord=HexCoord(q=0, r=0)),
            HexPosition(coord=HexCoord(q=3, r=0)),
            terrain,
        )
        assert result.has_los is True
        assert result.los_type == "clear"

    def test_obscured_soft_cover(self) -> None:
        """Obscured LOS indicates soft cover."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=1, r=0), provides_soft_cover=True),
            ]
        )
        result = check_los_with_cover(
            HexPosition(coord=HexCoord(q=0, r=0)),
            HexPosition(coord=HexCoord(q=2, r=0)),
            terrain,
        )
        assert result.has_los is True
        assert result.los_type == "obscured"


class TestCheckElevationBlocksLOS:
    """Tests for check_elevation_blocks_los function."""

    def test_same_elevation(self) -> None:
        """Same elevation never blocks."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=1, r=0), blocks_line_of_sight=True),
            ]
        )
        result = check_elevation_blocks_los(
            HexPosition(coord=HexCoord(q=0, r=0)),
            HexPosition(coord=HexCoord(q=2, r=0)),
            terrain,
        )
        assert result is False

    def test_higher_attacker_with_blocking(self) -> None:
        """Higher attacker with blocking terrain at target position."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=2, r=0), blocks_line_of_sight=True),
            ]
        )
        result = check_elevation_blocks_los(
            HexPosition(coord=HexCoord(q=0, r=0), elevation=2),
            HexPosition(coord=HexCoord(q=2, r=0), elevation=0),
            terrain,
        )
        assert result is True

    def test_no_terrain_elevation_matters(self) -> None:
        """No terrain means elevation doesn't block."""
        result = check_elevation_blocks_los(
            HexPosition(coord=HexCoord(q=0, r=0), elevation=2),
            HexPosition(coord=HexCoord(q=2, r=0), elevation=0),
            None,
        )
        assert result is False


class TestGetLOSBlockingHexes:
    """Tests for get_los_blocking_hexes function."""

    def test_no_blocking_hexes(self) -> None:
        """No blocking hexes returns empty list."""
        terrain = TerrainMap(tiles=[])
        result = get_los_blocking_hexes(terrain, HexCoord(q=0, r=0), HexCoord(q=3, r=0))
        assert result == []

    def test_one_blocking_hex(self) -> None:
        """One blocking hex is correctly identified."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=1, r=0), blocks_line_of_sight=True),
            ]
        )
        result = get_los_blocking_hexes(terrain, HexCoord(q=0, r=0), HexCoord(q=2, r=0))
        assert len(result) == 1
        assert result[0] == HexCoord(q=1, r=0)

    def test_multiple_blocking_hexes(self) -> None:
        """Multiple blocking hexes are all identified."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=1, r=0), blocks_line_of_sight=True),
                TerrainHex(coord=HexCoord(q=2, r=0), blocks_line_of_sight=True),
            ]
        )
        result = get_los_blocking_hexes(terrain, HexCoord(q=0, r=0), HexCoord(q=3, r=0))
        assert len(result) == 2


class TestGetLOSObscuringHexes:
    """Tests for get_los_obscuring_hexes function."""

    def test_no_obscuring_hexes(self) -> None:
        """No obscuring hexes returns empty list."""
        terrain = TerrainMap(tiles=[])
        result = get_los_obscuring_hexes(
            terrain, HexCoord(q=0, r=0), HexCoord(q=3, r=0)
        )
        assert result == []

    def test_one_obscuring_hex(self) -> None:
        """One obscuring hex is correctly identified."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=1, r=0), provides_soft_cover=True),
            ]
        )
        result = get_los_obscuring_hexes(
            terrain, HexCoord(q=0, r=0), HexCoord(q=2, r=0)
        )
        assert len(result) == 1
        assert result[0] == HexCoord(q=1, r=0)

    def test_multiple_obscuring_hexes(self) -> None:
        """Multiple obscuring hexes are all identified."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=1, r=0), provides_soft_cover=True),
                TerrainHex(coord=HexCoord(q=2, r=0), provides_soft_cover=True),
            ]
        )
        result = get_los_obscuring_hexes(
            terrain, HexCoord(q=0, r=0), HexCoord(q=3, r=0)
        )
        assert len(result) == 2


class TestLOSEdgeCases:
    """Edge case tests for LOS resolution."""

    def test_no_terrain_path(self) -> None:
        """Path with no terrain along it."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=5, r=0), blocks_line_of_sight=True),
            ]
        )
        request = LOSCheckRequest(
            attacker_pos=HexPosition(coord=HexCoord(q=0, r=0)),
            target_pos=HexPosition(coord=HexCoord(q=3, r=0)),
            terrain=terrain,
        )
        result = check_line_of_sight(request)
        assert result.has_los is True
        assert result.los_type == "clear"

    def test_long_distance(self) -> None:
        """Long distance LOS check."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=5, r=0), provides_soft_cover=True),
                TerrainHex(coord=HexCoord(q=10, r=0), blocks_line_of_sight=True),
            ]
        )
        request = LOSCheckRequest(
            attacker_pos=HexPosition(coord=HexCoord(q=0, r=0)),
            target_pos=HexPosition(coord=HexCoord(q=15, r=0)),
            terrain=terrain,
        )
        result = check_line_of_sight(request)
        assert result.has_los is False
        assert result.los_type == "blocked"
        assert len(result.blocked_by) == 1

    def test_soft_cover_not_blocking(self) -> None:
        """Soft cover does not block LOS, just obscures."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=1, r=0), provides_soft_cover=True),
                TerrainHex(coord=HexCoord(q=2, r=0), provides_soft_cover=True),
            ]
        )
        request = LOSCheckRequest(
            attacker_pos=HexPosition(coord=HexCoord(q=0, r=0)),
            target_pos=HexPosition(coord=HexCoord(q=3, r=0)),
            terrain=terrain,
        )
        result = check_line_of_sight(request)
        assert result.has_los is True
        assert result.los_type == "obscured"
        assert len(result.obscured_by) == 2

    def test_elevation_difference_clear(self) -> None:
        """Elevation difference doesn't block if no blocking terrain."""
        request = LOSCheckRequest(
            attacker_pos=HexPosition(coord=HexCoord(q=0, r=0), elevation=3),
            target_pos=HexPosition(coord=HexCoord(q=3, r=0), elevation=0),
            terrain=None,
            check_elevation=True,
        )
        result = check_line_of_sight(request)
        assert result.has_los is True
        assert result.los_type == "clear"

    def test_hex_not_in_terrain_map(self) -> None:
        """Hex not in terrain map has no terrain effects."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=10, r=10), blocks_line_of_sight=True),
            ]
        )
        request = LOSCheckRequest(
            attacker_pos=HexPosition(coord=HexCoord(q=0, r=0)),
            target_pos=HexPosition(coord=HexCoord(q=3, r=0)),
            terrain=terrain,
        )
        result = check_line_of_sight(request)
        assert result.has_los is True
        assert result.los_type == "clear"


class TestLOSIntegration:
    """Integration tests for LOS with other systems."""

    def test_los_with_terrain_terrain_types(self) -> None:
        """LOS works correctly with all terrain types."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(
                    coord=HexCoord(q=1, r=0),
                    blocks_line_of_sight=True,
                    provides_soft_cover=True,
                ),
                TerrainHex(coord=HexCoord(q=2, r=0), difficult=True),
                TerrainHex(coord=HexCoord(q=3, r=0), dangerous=True),
            ]
        )
        request = LOSCheckRequest(
            attacker_pos=HexPosition(coord=HexCoord(q=0, r=0)),
            target_pos=HexPosition(coord=HexCoord(q=4, r=0)),
            terrain=terrain,
        )
        result = check_line_of_sight(request)
        assert result.has_los is False
        assert result.los_type == "blocked"

    def test_path_clear_with_difficult_terrain(self) -> None:
        """Path clear ignores difficult terrain (for seeking weapons)."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=1, r=0), difficult=True),
                TerrainHex(coord=HexCoord(q=2, r=0), dangerous=True),
            ]
        )
        path_exists, blocking = check_path_clear(
            terrain, HexCoord(q=0, r=0), HexCoord(q=3, r=0)
        )
        assert path_exists is True
        assert blocking == []

    def test_seeking_weapon_path(self) -> None:
        """Seeking weapon only needs a path, not clear LOS."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=1, r=0), provides_soft_cover=True),
                TerrainHex(coord=HexCoord(q=2, r=0), provides_soft_cover=True),
            ]
        )
        path_exists, blocking = check_path_clear(
            terrain, HexCoord(q=0, r=0), HexCoord(q=3, r=0)
        )
        assert path_exists is True

        los_result = check_line_of_sight(
            LOSCheckRequest(
                attacker_pos=HexPosition(coord=HexCoord(q=0, r=0)),
                target_pos=HexPosition(coord=HexCoord(q=3, r=0)),
                terrain=terrain,
            )
        )
        assert los_result.los_type == "obscured"


class TestCheckLineOfSightToFootprint:
    """Tests for check_line_of_sight_to_footprint function for Size 2+ targets."""

    def test_size_1_target_single_hex(self) -> None:
        """Size 1 target has single hex footprint, behaves like regular LOS."""
        result = check_line_of_sight_to_footprint(
            attacker_pos=HexPosition(coord=HexCoord(q=0, r=0)),
            target_center=HexCoord(q=3, r=0),
            target_size="size_1",
            terrain=None,
        )
        assert result.has_los is True
        assert result.los_type == "clear"

    def test_size_2_target_clear_to_center(self) -> None:
        """Size 2 target with clear LOS to center returns clear."""
        result = check_line_of_sight_to_footprint(
            attacker_pos=HexPosition(coord=HexCoord(q=0, r=0)),
            target_center=HexCoord(q=3, r=0),
            target_size="size_2",
            terrain=None,
        )
        assert result.has_los is True
        assert result.los_type == "clear"

    def test_size_2_target_center_blocked_edge_clear(self) -> None:
        """Size 2 target with blocked center but clear LOS to footprint edge."""
        # Block the center hex (3,0) but leave edge hex (4,0) visible
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=2, r=0), blocks_line_of_sight=True),
            ]
        )
        # Attacker at (0,0), target center at (3,0)
        # Path to center (3,0) goes through blocked (2,0)
        # But path to footprint edge (4,0) or (3,1) etc might be clear
        result = check_line_of_sight_to_footprint(
            attacker_pos=HexPosition(coord=HexCoord(q=0, r=0)),
            target_center=HexCoord(q=3, r=0),
            target_size="size_2",
            terrain=terrain,
        )
        # Should find LOS to one of the edge hexes
        assert result.has_los is True

    def test_size_2_target_all_footprint_blocked(self) -> None:
        """Size 2 target with all footprint hexes blocked has no LOS."""
        # Block all paths to a size 2 target at (3,0)
        # Size 2 has radius 1, so footprint is: (3,0), (4,0), (2,0), (3,-1), (3,1), (4,-1), (2,1)
        # Need a wall that blocks all of them from (0,0)
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=1, r=0), blocks_line_of_sight=True),
                TerrainHex(coord=HexCoord(q=1, r=-1), blocks_line_of_sight=True),
                TerrainHex(coord=HexCoord(q=1, r=1), blocks_line_of_sight=True),
            ]
        )
        result = check_line_of_sight_to_footprint(
            attacker_pos=HexPosition(coord=HexCoord(q=0, r=0)),
            target_center=HexCoord(q=3, r=0),
            target_size="size_2",
            terrain=terrain,
        )
        assert result.has_los is False
        assert result.los_type == "blocked"

    def test_size_2_target_obscured_better_than_blocked(self) -> None:
        """Size 2 target returns obscured if that's best available LOS."""
        # Block center path, obscure edge path
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=2, r=0), blocks_line_of_sight=True),
                TerrainHex(coord=HexCoord(q=2, r=1), provides_soft_cover=True),
            ]
        )
        result = check_line_of_sight_to_footprint(
            attacker_pos=HexPosition(coord=HexCoord(q=0, r=0)),
            target_center=HexCoord(q=3, r=0),
            target_size="size_2",
            terrain=terrain,
        )
        # Should find at least obscured LOS to some edge hex
        assert result.has_los is True

    def test_size_3_target_larger_footprint(self) -> None:
        """Size 3 target has larger footprint (radius 2)."""
        # Block a line but size 3 footprint extends beyond
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=2, r=0), blocks_line_of_sight=True),
            ]
        )
        result = check_line_of_sight_to_footprint(
            attacker_pos=HexPosition(coord=HexCoord(q=0, r=0)),
            target_center=HexCoord(q=5, r=0),
            target_size="size_3",
            terrain=terrain,
        )
        # Size 3 has radius 2, so should have clear LOS to edge hexes
        assert result.has_los is True

    def test_returns_best_los_type(self) -> None:
        """Returns the best (most favorable for attacker) LOS type found."""
        # Create terrain where some footprint hexes are blocked,
        # some obscured, and at least one clear
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=3, r=-1), provides_soft_cover=True),
            ]
        )
        result = check_line_of_sight_to_footprint(
            attacker_pos=HexPosition(coord=HexCoord(q=0, r=0)),
            target_center=HexCoord(q=3, r=0),
            target_size="size_2",
            terrain=terrain,
        )
        # Should find clear LOS to center (3,0) or other clear footprint hex
        assert result.los_type == "clear"
