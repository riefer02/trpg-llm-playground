"""Tests for hex grid primitives and adjacency calculations.

Tests the size-aware adjacency helpers per PR2 rules:
- Size represents "area of influence"
- Larger units engage/interact at extended range
"""

from core.mech.grid import (
    HexCoord,
    adjacency_distance,
    is_adjacent_by_size,
)


class TestAdjacencyDistance:
    """Tests for adjacency_distance() helper."""

    def test_size_1_vs_size_1(self):
        """Size 1 vs size 1 has adjacency distance 1."""
        assert adjacency_distance("size_1", "size_1") == 1

    def test_size_half_vs_size_1(self):
        """Size 1/2 vs size 1 has adjacency distance 1."""
        assert adjacency_distance("size_half", "size_1") == 1

    def test_size_2_vs_size_1(self):
        """Size 2 vs size 1 uses the larger size (2)."""
        assert adjacency_distance("size_2", "size_1") == 2

    def test_size_1_vs_size_2(self):
        """Size 1 vs size 2 uses the larger size (2)."""
        assert adjacency_distance("size_1", "size_2") == 2

    def test_size_2_vs_size_2(self):
        """Size 2 vs size 2 has adjacency distance 2."""
        assert adjacency_distance("size_2", "size_2") == 2

    def test_size_3_vs_size_1(self):
        """Size 3 vs size 1 uses the larger size (3)."""
        assert adjacency_distance("size_3", "size_1") == 3

    def test_size_3_vs_size_2(self):
        """Size 3 vs size 2 uses the larger size (3)."""
        assert adjacency_distance("size_3", "size_2") == 3

    def test_size_4_vs_size_1(self):
        """Size 4 vs size 1 uses the larger size (4)."""
        assert adjacency_distance("size_4", "size_1") == 4

    def test_size_5_vs_size_3(self):
        """Size 5 vs size 3 uses the larger size (5)."""
        assert adjacency_distance("size_5", "size_3") == 5


class TestIsAdjacentBySize:
    """Tests for is_adjacent_by_size() helper."""

    def test_same_position_not_adjacent(self):
        """Same position is never adjacent."""
        coord = HexCoord(q=0, r=0)
        assert not is_adjacent_by_size(coord, coord, "size_1", "size_1")
        assert not is_adjacent_by_size(coord, coord, "size_2", "size_2")

    def test_distance_1_size_1_is_adjacent(self):
        """Distance 1 is adjacent for size 1 units."""
        a = HexCoord(q=0, r=0)
        b = HexCoord(q=1, r=0)
        assert is_adjacent_by_size(a, b, "size_1", "size_1")

    def test_distance_2_size_1_not_adjacent(self):
        """Distance 2 is not adjacent for size 1 units."""
        a = HexCoord(q=0, r=0)
        b = HexCoord(q=2, r=0)
        assert not is_adjacent_by_size(a, b, "size_1", "size_1")

    def test_distance_2_size_2_is_adjacent(self):
        """Distance 2 is adjacent when one unit is size 2."""
        a = HexCoord(q=0, r=0)
        b = HexCoord(q=2, r=0)
        assert is_adjacent_by_size(a, b, "size_2", "size_1")
        assert is_adjacent_by_size(a, b, "size_1", "size_2")

    def test_distance_3_size_2_not_adjacent(self):
        """Distance 3 is not adjacent for size 2 units."""
        a = HexCoord(q=0, r=0)
        b = HexCoord(q=3, r=0)
        assert not is_adjacent_by_size(a, b, "size_2", "size_1")

    def test_distance_3_size_3_is_adjacent(self):
        """Distance 3 is adjacent when one unit is size 3."""
        a = HexCoord(q=0, r=0)
        b = HexCoord(q=3, r=0)
        assert is_adjacent_by_size(a, b, "size_3", "size_1")
        assert is_adjacent_by_size(a, b, "size_1", "size_3")

    def test_all_hex_directions_at_boundary(self):
        """Size 2 adjacency works in all hex directions."""
        origin = HexCoord(q=0, r=0)
        # Distance 2 in each axial direction
        distance_2_coords = [
            HexCoord(q=2, r=0),   # East
            HexCoord(q=2, r=-2),  # Northeast
            HexCoord(q=0, r=-2),  # Northwest
            HexCoord(q=-2, r=0),  # West
            HexCoord(q=-2, r=2),  # Southwest
            HexCoord(q=0, r=2),   # Southeast
        ]
        for coord in distance_2_coords:
            assert is_adjacent_by_size(origin, coord, "size_2", "size_1"), (
                f"Size 2 should be adjacent at distance 2: {coord}"
            )

    def test_diagonal_hex_distance_2(self):
        """Hex distance accounts for diagonal movement correctly."""
        a = HexCoord(q=0, r=0)
        # (1, -1) is distance 1 in hex coords
        b = HexCoord(q=1, r=-1)
        assert a.distance_to(b) == 1
        assert is_adjacent_by_size(a, b, "size_1", "size_1")


class TestSizeAwareEngagementScenarios:
    """Integration scenarios for size-aware engagement."""

    def test_drake_engages_at_distance_2(self):
        """Size 2 Drake frame engages hostiles at distance 2."""
        drake_pos = HexCoord(q=0, r=0)
        hostile_pos = HexCoord(q=2, r=0)  # 2 hexes away

        # Drake is size 2, hostile is size 1
        assert is_adjacent_by_size(drake_pos, hostile_pos, "size_2", "size_1")

    def test_size_1_mech_vs_size_3_npc(self):
        """Size 1 player mech engages size 3 NPC at distance 3."""
        player_pos = HexCoord(q=0, r=0)
        npc_pos = HexCoord(q=3, r=0)  # 3 hexes away

        # Size 3 NPC extends engagement range to 3
        assert is_adjacent_by_size(player_pos, npc_pos, "size_1", "size_3")

    def test_mine_trigger_extended_for_size_2(self):
        """Size 2 mech triggers mines from 2 hexes away."""
        mine_pos = HexCoord(q=5, r=5)
        mech_pos = HexCoord(q=3, r=5)  # 2 hexes away

        # Mech is size 2, mine is effectively size 1
        assert is_adjacent_by_size(mine_pos, mech_pos, "size_1", "size_2")

    def test_standard_size_1_engagement(self):
        """Standard size 1 vs size 1 only at distance 1."""
        pos1 = HexCoord(q=0, r=0)
        pos2 = HexCoord(q=1, r=0)
        pos3 = HexCoord(q=2, r=0)

        # Distance 1: adjacent
        assert is_adjacent_by_size(pos1, pos2, "size_1", "size_1")
        # Distance 2: not adjacent
        assert not is_adjacent_by_size(pos1, pos3, "size_1", "size_1")
