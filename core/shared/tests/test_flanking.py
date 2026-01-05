"""Flanking detection tests for Lancer combat."""

import pytest
from core.shared.flanking import (
    is_on_row,
    is_line_clear_of_hard_cover,
    get_adjacent_hard_covers,
    check_flanking,
    get_cover_difficulty_with_flanking,
    FlankingResult,
)
from core.mech.terrain import TerrainHex, TerrainMap
from core.mech.grid import HexCoord


class TestIsOnRow:
    """Tests for is_on_row function."""

    def test_attacker_on_exact_row(self) -> None:
        """Attacker exactly on the line from cover through target."""
        cover = HexCoord(q=0, r=0)
        target = HexCoord(q=1, r=0)
        attacker = HexCoord(q=2, r=0)
        result = is_on_row(cover, target, attacker)
        assert result is True

    def test_attacker_beyond_target_on_row(self) -> None:
        """Attacker beyond target on the same row."""
        cover = HexCoord(q=0, r=0)
        target = HexCoord(q=1, r=0)
        attacker = HexCoord(q=3, r=0)
        result = is_on_row(cover, target, attacker)
        assert result is True

    def test_attacker_opposite_side(self) -> None:
        """Attacker on the opposite side of target from cover."""
        cover = HexCoord(q=0, r=0)
        target = HexCoord(q=1, r=0)
        attacker = HexCoord(q=0, r=1)
        result = is_on_row(cover, target, attacker)
        assert result is False

    def test_attacker_diagonal_not_on_row(self) -> None:
        """Attacker at diagonal position not on row."""
        cover = HexCoord(q=0, r=0)
        target = HexCoord(q=1, r=0)
        attacker = HexCoord(q=1, r=1)
        result = is_on_row(cover, target, attacker)
        assert result is False

    def test_same_hex(self) -> None:
        """Attacker and target at same position."""
        cover = HexCoord(q=0, r=0)
        target = HexCoord(q=1, r=0)
        attacker = HexCoord(q=1, r=0)
        result = is_on_row(cover, target, attacker)
        assert result is True

    def test_row_with_different_direction(self) -> None:
        """Row extending in different direction from cover."""
        cover = HexCoord(q=2, r=2)
        target = HexCoord(q=1, r=1)
        attacker = HexCoord(q=0, r=0)
        result = is_on_row(cover, target, attacker)
        assert result is True

    def test_not_on_row_diagonal(self) -> None:
        """Attacker not on the row (diagonal offset)."""
        cover = HexCoord(q=0, r=0)
        target = HexCoord(q=1, r=0)
        attacker = HexCoord(q=2, r=1)
        result = is_on_row(cover, target, attacker)
        assert result is False


class TestIsLineClearOfHardCover:
    """Tests for is_line_clear_of_hard_cover function."""

    def test_no_terrain_clear(self) -> None:
        """No terrain means line is clear."""
        result = is_line_clear_of_hard_cover(
            None, HexCoord(q=0, r=0), HexCoord(q=2, r=0)
        )
        assert result is True

    def test_adjacent_hexes_clear(self) -> None:
        """Adjacent hexes with no hard cover in between."""
        terrain = TerrainMap(tiles=[])
        result = is_line_clear_of_hard_cover(
            terrain, HexCoord(q=0, r=0), HexCoord(q=1, r=0)
        )
        assert result is True

    def test_clear_path_no_hard_cover(self) -> None:
        """Path with only soft cover or difficult terrain."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=1, r=0), difficult=True),
                TerrainHex(coord=HexCoord(q=2, r=0), provides_soft_cover=True),
            ]
        )
        result = is_line_clear_of_hard_cover(
            terrain, HexCoord(q=0, r=0), HexCoord(q=3, r=0)
        )
        assert result is True

    def test_blocked_by_hard_cover(self) -> None:
        """Path blocked by hard cover."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(
                    coord=HexCoord(q=1, r=0),
                    provides_hard_cover=True,
                    hard_cover_size="size_1",
                )
            ]
        )
        result = is_line_clear_of_hard_cover(
            terrain, HexCoord(q=0, r=0), HexCoord(q=2, r=0)
        )
        assert result is False

    def test_blocked_by_multiple_hard_covers(self) -> None:
        """Path blocked by multiple hard covers."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(
                    coord=HexCoord(q=1, r=0),
                    provides_hard_cover=True,
                    hard_cover_size="size_1",
                ),
                TerrainHex(
                    coord=HexCoord(q=2, r=0),
                    provides_hard_cover=True,
                    hard_cover_size="size_1",
                ),
            ]
        )
        result = is_line_clear_of_hard_cover(
            terrain, HexCoord(q=0, r=0), HexCoord(q=3, r=0)
        )
        assert result is False

    def test_clear_path_long_distance(self) -> None:
        """Clear path at longer distance."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(coord=HexCoord(q=2, r=0), difficult=True),
                TerrainHex(coord=HexCoord(q=3, r=0), elevation=1),
            ]
        )
        result = is_line_clear_of_hard_cover(
            terrain, HexCoord(q=0, r=0), HexCoord(q=5, r=0)
        )
        assert result is True


class TestGetAdjacentHardCovers:
    """Tests for get_adjacent_hard_covers function."""

    def test_no_terrain(self) -> None:
        """No terrain returns empty list."""
        result = get_adjacent_hard_covers(None, HexCoord(q=0, r=0))
        assert result == []

    def test_no_hard_cover(self) -> None:
        """Terrain without hard cover returns empty list."""
        terrain = TerrainMap(
            tiles=[TerrainHex(coord=HexCoord(q=1, r=0), provides_soft_cover=True)]
        )
        result = get_adjacent_hard_covers(terrain, HexCoord(q=0, r=0))
        assert result == []

    def test_single_adjacent_hard_cover(self) -> None:
        """Returns single adjacent hard cover."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(
                    coord=HexCoord(q=1, r=0),
                    provides_hard_cover=True,
                    hard_cover_size="size_2",
                )
            ]
        )
        result = get_adjacent_hard_covers(terrain, HexCoord(q=0, r=0))
        assert len(result) == 1
        assert result[0] == HexCoord(q=1, r=0)

    def test_multiple_adjacent_hard_covers(self) -> None:
        """Returns multiple adjacent hard covers."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(
                    coord=HexCoord(q=1, r=0),
                    provides_hard_cover=True,
                    hard_cover_size="size_2",
                ),
                TerrainHex(
                    coord=HexCoord(q=0, r=1),
                    provides_hard_cover=True,
                    hard_cover_size="size_2",
                ),
                TerrainHex(
                    coord=HexCoord(q=-1, r=1),
                    provides_hard_cover=False,
                ),
            ]
        )
        result = get_adjacent_hard_covers(terrain, HexCoord(q=0, r=0))
        assert len(result) == 2
        assert HexCoord(q=1, r=0) in result
        assert HexCoord(q=0, r=1) in result

    def test_hard_cover_not_adjacent(self) -> None:
        """Hard cover not adjacent to target is not returned."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(
                    coord=HexCoord(q=2, r=0),
                    provides_hard_cover=True,
                    hard_cover_size="size_2",
                )
            ]
        )
        result = get_adjacent_hard_covers(terrain, HexCoord(q=0, r=0))
        assert result == []


class TestCheckFlanking:
    """Tests for check_flanking function."""

    def test_no_terrain(self) -> None:
        """No terrain means cannot flank."""
        result = check_flanking(None, HexCoord(q=1, r=0), HexCoord(q=0, r=0))
        assert result.is_flanked is False
        assert "No terrain" in result.reason

    def test_not_adjacent(self) -> None:
        """Attacker not adjacent to target cannot flank."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(
                    coord=HexCoord(q=1, r=0),
                    provides_hard_cover=True,
                    hard_cover_size="size_1",
                )
            ]
        )
        result = check_flanking(terrain, HexCoord(q=3, r=0), HexCoord(q=0, r=0))
        assert result.is_flanked is False
        assert "not adjacent" in result.reason

    def test_no_hard_cover(self) -> None:
        """Target without hard cover cannot be flanked."""
        terrain = TerrainMap(
            tiles=[TerrainHex(coord=HexCoord(q=1, r=0), provides_soft_cover=True)]
        )
        result = check_flanking(terrain, HexCoord(q=1, r=0), HexCoord(q=0, r=0))
        assert result.is_flanked is False
        assert "no adjacent hard cover" in result.reason

    def test_flanking_success(self) -> None:
        """Attacker flanks target relative to hard cover."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(
                    coord=HexCoord(q=1, r=0),
                    provides_hard_cover=True,
                    hard_cover_size="size_2",
                )
            ]
        )
        result = check_flanking(terrain, HexCoord(q=-1, r=0), HexCoord(q=0, r=0))
        assert result.is_flanked is True
        assert len(result.flanked_cover_hexes) == 1
        assert result.flanked_cover_hexes[0] == HexCoord(q=1, r=0)

    def test_not_on_row(self) -> None:
        """Attacker adjacent but not on row."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(
                    coord=HexCoord(q=1, r=0),
                    provides_hard_cover=True,
                    hard_cover_size="size_2",
                )
            ]
        )
        result = check_flanking(terrain, HexCoord(q=0, r=1), HexCoord(q=0, r=0))
        assert result.is_flanked is False
        assert "not on the same row" in result.reason

    def test_on_row_but_adjacent(self) -> None:
        """Attacker on row and adjacent to target."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(
                    coord=HexCoord(q=1, r=0),
                    provides_hard_cover=True,
                    hard_cover_size="size_2",
                ),
            ]
        )
        result = check_flanking(terrain, HexCoord(q=-1, r=0), HexCoord(q=0, r=0))
        assert result.is_flanked is True

    def test_multiple_covers_one_flanked(self) -> None:
        """Multiple covers, one is flanked."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(
                    coord=HexCoord(q=1, r=0),
                    provides_hard_cover=True,
                    hard_cover_size="size_2",
                ),
                TerrainHex(
                    coord=HexCoord(q=0, r=1),
                    provides_hard_cover=True,
                    hard_cover_size="size_2",
                ),
            ]
        )
        result = check_flanking(terrain, HexCoord(q=-1, r=0), HexCoord(q=0, r=0))
        assert result.is_flanked is True
        assert HexCoord(q=1, r=0) in result.flanked_cover_hexes
        assert HexCoord(q=0, r=1) not in result.flanked_cover_hexes

    def test_multiple_covers_one_is_flanked(self) -> None:
        """Only one of multiple covers is flankable."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(
                    coord=HexCoord(q=1, r=0),
                    provides_hard_cover=True,
                    hard_cover_size="size_2",
                ),
                TerrainHex(
                    coord=HexCoord(q=2, r=0),
                    provides_hard_cover=True,
                    hard_cover_size="size_2",
                ),
            ]
        )
        result = check_flanking(terrain, HexCoord(q=-1, r=0), HexCoord(q=0, r=0))
        assert result.is_flanked is True
        assert len(result.flanked_cover_hexes) == 1
        assert HexCoord(q=1, r=0) in result.flanked_cover_hexes

    def test_attacker_beyond_target(self) -> None:
        """Attacker beyond target on same row."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(
                    coord=HexCoord(q=1, r=0),
                    provides_hard_cover=True,
                    hard_cover_size="size_2",
                )
            ]
        )
        result = check_flanking(terrain, HexCoord(q=-1, r=0), HexCoord(q=0, r=0))
        assert result.is_flanked is True

    def test_extended_line_beyond_target(self) -> None:
        """Attacker on row extended beyond target."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(
                    coord=HexCoord(q=1, r=0),
                    provides_hard_cover=True,
                    hard_cover_size="size_2",
                )
            ]
        )
        result = check_flanking(terrain, HexCoord(q=-1, r=0), HexCoord(q=0, r=0))
        assert result.is_flanked is True


class TestGetCoverDifficultyWithFlanking:
    """Tests for get_cover_difficulty_with_flanking function."""

    def test_no_cover(self) -> None:
        """No cover returns none."""
        result = get_cover_difficulty_with_flanking(
            None, HexCoord(q=0, r=0), HexCoord(q=1, r=0), "size_1"
        )
        assert result.cover_type == "none"
        assert result.difficulty_modifier == 0

    def test_hard_cover_not_flanked(self) -> None:
        """Hard cover when not flanked."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(
                    coord=HexCoord(q=1, r=0),
                    provides_hard_cover=True,
                    hard_cover_size="size_2",
                )
            ]
        )
        result = get_cover_difficulty_with_flanking(
            terrain, HexCoord(q=0, r=1), HexCoord(q=0, r=0), "size_1"
        )
        assert result.cover_type == "hard"
        assert result.difficulty_modifier == 2

    def test_hard_cover_flanked_negated(self) -> None:
        """Hard cover negated by flanking."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(
                    coord=HexCoord(q=1, r=0),
                    provides_hard_cover=True,
                    hard_cover_size="size_2",
                )
            ]
        )
        result = get_cover_difficulty_with_flanking(
            terrain, HexCoord(q=-1, r=0), HexCoord(q=0, r=0), "size_1"
        )
        assert result.cover_type == "none"
        assert result.difficulty_modifier == 0
        assert "negated by flanking" in result.reason

    def test_hard_cover_flanked_fallthrough_soft(self) -> None:
        """Hard cover negated by flanking, soft cover applies."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(
                    coord=HexCoord(q=1, r=0),
                    provides_hard_cover=True,
                    provides_soft_cover=True,
                    hard_cover_size="size_2",
                ),
                TerrainHex(
                    coord=HexCoord(q=0, r=0),
                    provides_soft_cover=True,
                ),
            ]
        )
        result = get_cover_difficulty_with_flanking(
            terrain, HexCoord(q=-1, r=0), HexCoord(q=0, r=0), "size_1"
        )
        assert result.cover_type == "soft"
        assert result.difficulty_modifier == 1
        assert "negated by flanking" in result.reason

    def test_soft_cover_only(self) -> None:
        """Soft cover only (no hard cover available)."""
        terrain = TerrainMap(
            tiles=[TerrainHex(coord=HexCoord(q=0, r=0), provides_soft_cover=True)]
        )
        result = get_cover_difficulty_with_flanking(
            terrain, HexCoord(q=1, r=0), HexCoord(q=0, r=0), "size_1"
        )
        assert result.cover_type == "soft"
        assert result.difficulty_modifier == 1

    def test_custom_difficulty_values(self) -> None:
        """Custom difficulty values are respected."""
        terrain = TerrainMap(
            tiles=[TerrainHex(coord=HexCoord(q=0, r=0), provides_soft_cover=True)]
        )
        result = get_cover_difficulty_with_flanking(
            terrain,
            HexCoord(q=1, r=0),
            HexCoord(q=0, r=0),
            "size_1",
            soft_cover_difficulty=2,
            hard_cover_difficulty=3,
        )
        assert result.cover_type == "soft"
        assert result.difficulty_modifier == 2


class TestFlankingResultModel:
    """Tests for FlankingResult model behavior."""

    def test_default_values(self) -> None:
        """FlankingResult has correct defaults."""
        result = FlankingResult()
        assert result.is_flanked is False
        assert result.flanked_cover_hexes == []
        assert result.reason == ""

    def test_immutable(self) -> None:
        """FlankingResult is immutable."""
        from pydantic import ValidationError

        result = FlankingResult(is_flanked=True, reason="test")
        with pytest.raises((TypeError, ValidationError)):
            result.is_flanked = False


class TestComplexFlankingScenarios:
    """Tests for complex real-world flanking scenarios."""

    def test_partial_blockage(self) -> None:
        """Partial blockage where some paths are clear."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(
                    coord=HexCoord(q=1, r=0),
                    provides_hard_cover=True,
                    hard_cover_size="size_2",
                ),
                TerrainHex(
                    coord=HexCoord(q=2, r=1),
                    provides_hard_cover=True,
                    hard_cover_size="size_2",
                ),
            ]
        )
        result = check_flanking(terrain, HexCoord(q=-1, r=0), HexCoord(q=0, r=0))
        assert result.is_flanked is True

    def test_fully_surrounded(self) -> None:
        """Target surrounded by hard cover on all sides."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(
                    coord=HexCoord(q=1, r=0),
                    provides_hard_cover=True,
                    hard_cover_size="size_2",
                ),
                TerrainHex(
                    coord=HexCoord(q=0, r=1),
                    provides_hard_cover=True,
                    hard_cover_size="size_2",
                ),
                TerrainHex(
                    coord=HexCoord(q=-1, r=1),
                    provides_hard_cover=True,
                    hard_cover_size="size_2",
                ),
                TerrainHex(
                    coord=HexCoord(q=-1, r=0),
                    provides_hard_cover=True,
                    hard_cover_size="size_2",
                ),
                TerrainHex(
                    coord=HexCoord(q=0, r=-1),
                    provides_hard_cover=True,
                    hard_cover_size="size_2",
                ),
                TerrainHex(
                    coord=HexCoord(q=1, r=-1),
                    provides_hard_cover=True,
                    hard_cover_size="size_2",
                ),
            ]
        )
        result = check_flanking(terrain, HexCoord(q=-1, r=0), HexCoord(q=0, r=0))
        assert result.is_flanked is True
        assert len(result.flanked_cover_hexes) == 1

    def test_hard_cover_blocking_own_line(self) -> None:
        """Hard cover that blocks its own line to attacker."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(
                    coord=HexCoord(q=1, r=0),
                    provides_hard_cover=True,
                    hard_cover_size="size_3",
                ),
            ]
        )
        result = check_flanking(terrain, HexCoord(q=0, r=1), HexCoord(q=0, r=0))
        assert result.is_flanked is False
        assert "not on the same row" in result.reason

    def test_no_adjacency_for_hard_cover(self) -> None:
        """Hard cover exists but not adjacent to target."""
        terrain = TerrainMap(
            tiles=[
                TerrainHex(
                    coord=HexCoord(q=2, r=0),
                    provides_hard_cover=True,
                    hard_cover_size="size_2",
                ),
                TerrainHex(
                    coord=HexCoord(q=3, r=0),
                    provides_hard_cover=True,
                    hard_cover_size="size_2",
                ),
            ]
        )
        result = check_flanking(terrain, HexCoord(q=1, r=0), HexCoord(q=0, r=0))
        assert result.is_flanked is False
        assert "no adjacent hard cover" in result.reason

    def test_different_row_directions(self) -> None:
        """Flanking works in different row directions."""
        for dq, dr in [(1, 0), (1, -1), (0, 1), (-1, 0), (-1, 1), (0, -1)]:
            cover = HexCoord(q=0, r=0)
            target = HexCoord(q=cover.q + dq, r=cover.r + dr)
            attacker = HexCoord(q=cover.q + dq * 2, r=cover.r + dr * 2)

            terrain = TerrainMap(
                tiles=[
                    TerrainHex(
                        coord=cover,
                        provides_hard_cover=True,
                        hard_cover_size="size_2",
                    )
                ]
            )

            result = check_flanking(terrain, attacker, target)
            assert result.is_flanked is True, f"Failed for direction ({dq}, {dr})"
