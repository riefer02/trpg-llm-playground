"""Flanking detection for Lancer combat.

Per PR2 ~4058-4062:
- A character that is adjacent to a target and on the same 'row' as that target
  is considered to be 'flanking' that target.
- If a character is flanking a target, hard cover does not apply from that cover.
- The 'same row' means one of the spaces of your mech must occupy a space in the
  same row (grid or hex map) that's totally clear of that hard cover.

This module provides type-safe helpers for:
- Determining if an attacker is flanking a target relative to hard cover
- Checking if the line from cover to attacker is clear of hard cover
- Integration with terrain cover difficulty calculations
"""

from __future__ import annotations

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.enums import SizeClass
from core.mech.terrain import TerrainMap, TerrainHex, terrain_index
from core.mech.grid import HexCoord, hex_line


class FlankingResult(FrozenModel):
    """Result of flanking detection check.

    Attributes:
        is_flanked: True if attacker flanks the target relative to any adjacent hard cover
        flanked_cover_hexes: List of adjacent hard cover hexes that are flanked
        on_row_hexes: Hexes on the same row as target relative to each cover
        line_clear_hexes: Hexes that are clear of hard cover between cover and attacker
        reason: Detailed explanation of the flanking result
    """

    is_flanked: bool = False
    flanked_cover_hexes: list[HexCoord] = Field(default_factory=list)
    on_row_hexes: dict[HexCoord, bool] = Field(default_factory=dict)
    line_clear_hexes: dict[HexCoord, bool] = Field(default_factory=dict)
    reason: str = ""


def is_on_row(
    cover_hex: HexCoord,
    target: HexCoord,
    attacker: HexCoord,
) -> bool:
    """Check if attacker is on the same row as target relative to cover.

    Per PR2: "one of the spaces of your mech must occupy a space in the same row
    (grid or hex map) that's totally clear of that hard cover"

    The 'row' extends from the cover hex through the target hex and beyond.
    Uses cross product to verify collinearity and dot product to verify
    the attacker is beyond the target on the same line.

    Args:
        cover_hex: The hard cover hex the target is adjacent to
        target: The target's position
        attacker: The attacker's position to check

    Returns:
        True if attacker is on the same row as target relative to cover
    """
    v_cover_target = (target.q - cover_hex.q, target.r - cover_hex.r)
    v_cover_attacker = (attacker.q - cover_hex.q, attacker.r - cover_hex.r)

    cross = (
        v_cover_target[0] * v_cover_attacker[1]
        - v_cover_target[1] * v_cover_attacker[0]
    )

    if cross != 0:
        return False

    dot = (
        v_cover_target[0] * v_cover_attacker[0]
        + v_cover_target[1] * v_cover_attacker[1]
    )
    return dot > 0


def is_line_clear_of_hard_cover(
    terrain: TerrainMap | None,
    from_hex: HexCoord,
    to_hex: HexCoord,
) -> bool:
    """Check if the line from one hex to another is totally clear of hard cover.

    Per PR2: "that's totally clear of that hard cover"

    Checks all hexes along the line from from_hex to to_hex (exclusive of endpoints)
    to verify no hard cover exists on the path.

    Args:
        terrain: The terrain map
        from_hex: Starting hex (typically the hard cover hex)
        to_hex: Ending hex (typically the attacker's position)

    Returns:
        True if no hard cover exists between from_hex and to_hex
    """
    if terrain is None:
        return True

    line = hex_line(from_hex, to_hex)
    hexes_to_check = line[1:-1]
    if not hexes_to_check:
        return True

    terrain_idx = terrain_index(terrain)
    for hex_coord in hexes_to_check:
        if hex_coord in terrain_idx:
            hex_terrain = terrain_idx[hex_coord]
            if hex_terrain.provides_hard_cover:
                return False

    return True


def get_adjacent_hard_covers(
    terrain: TerrainMap | None,
    target_coord: HexCoord,
) -> list[HexCoord]:
    """Get all adjacent hexes that provide hard cover.

    Args:
        terrain: The terrain map
        target_coord: The target's position

    Returns:
        List of adjacent hex coordinates that provide hard cover
    """
    if terrain is None:
        return []

    adjacent_coords = target_coord.neighbors()
    hard_covers: list[HexCoord] = []

    for adj_coord in adjacent_coords:
        hex_terrain = get_terrain_at(terrain, adj_coord)
        if hex_terrain and hex_terrain.provides_hard_cover:
            hard_covers.append(adj_coord)

    return hard_covers


def check_flanking(
    terrain: TerrainMap | None,
    attacker_coord: HexCoord,
    target_coord: HexCoord,
) -> FlankingResult:
    """Check if attacker flanks target relative to adjacent hard cover.

    Per PR2 ~4058-4062:
    "If a character is adjacent to hard cover, it benefits from that cover as long
    as you cannot flank that character. Flanking means one of the spaces of your
    mech must occupy a space in the same row (grid or hex map) that's totally clear
    of that hard cover."

    Flanking requires:
    1. Attacker is adjacent to target
    2. Attacker is on the same row as target relative to some hard cover
    3. The line from that cover to attacker is totally clear of hard cover

    Args:
        terrain: The terrain map
        attacker_coord: Attacker's position
        target_coord: Target's position

    Returns:
        FlankingResult with is_flanked status and detailed breakdown
    """
    if terrain is None:
        return FlankingResult(
            is_flanked=False,
            reason="No terrain - cannot check for hard cover",
        )

    if not target_coord.is_adjacent(attacker_coord):
        return FlankingResult(
            is_flanked=False,
            reason="Attacker is not adjacent to target",
        )

    hard_covers = get_adjacent_hard_covers(terrain, target_coord)

    if not hard_covers:
        return FlankingResult(
            is_flanked=False,
            reason="Target has no adjacent hard cover",
        )

    flanked_covers: list[HexCoord] = []
    on_row_results: dict[HexCoord, bool] = {}
    line_clear_results: dict[HexCoord, bool] = {}

    for cover_hex in hard_covers:
        on_row = is_on_row(cover_hex, target_coord, attacker_coord)
        on_row_results[cover_hex] = on_row

        if on_row:
            line_clear = is_line_clear_of_hard_cover(terrain, cover_hex, attacker_coord)
            line_clear_results[cover_hex] = line_clear

            if line_clear:
                flanked_covers.append(cover_hex)

    is_flanked = len(flanked_covers) > 0

    if is_flanked:
        if len(flanked_covers) == 1:
            cover_desc = f"hard cover at ({flanked_covers[0].q}, {flanked_covers[0].r})"
        else:
            coords = ", ".join(f"({c.q}, {c.r})" for c in flanked_covers)
            cover_desc = f"hard covers at {coords}"
        reason = f"Attacker flanks target relative to {cover_desc}"
    else:
        if hard_covers:
            if not any(on_row_results.values()):
                reason = "Attacker is not on the same row as target relative to any adjacent hard cover"
            else:
                reason = "Attacker is on the same row but line from cover to attacker is blocked by hard cover"
        else:
            reason = "Target has no adjacent hard cover"

    return FlankingResult(
        is_flanked=is_flanked,
        flanked_cover_hexes=flanked_covers,
        on_row_hexes=on_row_results,
        line_clear_hexes=line_clear_results,
        reason=reason,
    )


def get_cover_difficulty_with_flanking(
    terrain: TerrainMap | None,
    attacker_coord: HexCoord,
    target_coord: HexCoord,
    target_size: SizeClass,
    soft_cover_difficulty: int = 1,
    hard_cover_difficulty: int = 2,
) -> "CoverDifficultyResult":
    """Calculate cover difficulty with flanking detection.

    This is a convenience function that combines get_cover_difficulty with
    flanking detection. If the attacker flanks the target, hard cover is negated
    and the result falls through to soft cover if available.

    Per PR2:
    - Soft Cover: +1 Difficulty
    - Hard Cover: +2 Difficulty (requires adjacency)
    - You have one or the other (they don't stack)
    - Flanking negates hard cover

    Args:
        terrain: The terrain map
        attacker_coord: Attacker's position
        target_coord: Target's position
        target_size: Target's size class
        soft_cover_difficulty: Difficulty from soft cover (default 1)
        hard_cover_difficulty: Difficulty from hard cover (default 2)

    Returns:
        CoverDifficultyResult with cover type and modifier
    """
    from core.shared.terrain import (
        get_cover_difficulty,
        check_soft_cover,
        CoverDifficultyResult,
    )

    flanking = check_flanking(terrain, attacker_coord, target_coord)

    hard_cover = check_hard_cover_available_with_size(
        terrain=terrain,
        attacker_coord=attacker_coord,
        target_coord=target_coord,
        target_size=target_size,
    )

    if hard_cover.available and not flanking.is_flanked:
        return CoverDifficultyResult(
            cover_type="hard",
            difficulty_modifier=hard_cover_difficulty,
            reason=f"Target has hard cover (adjacent to size {hard_cover.cover_size or 'N/A'})",
        )

    soft_cover = check_soft_cover(terrain, attacker_coord, target_coord)

    if soft_cover:
        if flanking.is_flanked and hard_cover.available:
            return CoverDifficultyResult(
                cover_type="soft",
                difficulty_modifier=soft_cover_difficulty,
                reason="Hard cover negated by flanking, target has soft cover",
            )
        return CoverDifficultyResult(
            cover_type="soft",
            difficulty_modifier=soft_cover_difficulty,
            reason="Target has soft cover (line of sight obscured)",
        )

    if flanking.is_flanked and hard_cover.available:
        return CoverDifficultyResult(
            cover_type="none",
            difficulty_modifier=0,
            reason="Hard cover negated by flanking, no soft cover",
        )

    return CoverDifficultyResult(
        cover_type="none",
        difficulty_modifier=0,
        reason="No cover",
    )


def check_hard_cover_available_with_size(
    terrain: TerrainMap | None,
    attacker_coord: HexCoord,
    target_coord: HexCoord,
    target_size: SizeClass,
) -> "HardCoverAvailabilityResult":
    """Check if target has hard cover available (with size matching).

    This is a convenience wrapper that matches the existing terrain.py interface.

    Args:
        terrain: The terrain map
        attacker_coord: Attacker's position
        target_coord: Target's position
        target_size: Target's size class

    Returns:
        HardCoverAvailabilityResult with availability details
    """
    from core.shared.terrain import (
        check_hard_cover_available,
        HardCoverAvailabilityResult,
    )

    return check_hard_cover_available(
        terrain=terrain,
        attacker_coord=attacker_coord,
        target_coord=target_coord,
        target_size=target_size,
        requires_adjacency=True,
        requires_size_match=True,
        hard_cover_size=None,
    )


def get_terrain_at(
    terrain: TerrainMap | None,
    coord: HexCoord,
) -> TerrainHex | None:
    """Get terrain at a specific coordinate.

    Args:
        terrain: The terrain map (None for no terrain)
        coord: The coordinate to check

    Returns:
        TerrainHex at that coordinate, or None if no terrain/invalid
    """
    if terrain is None:
        return None

    idx = terrain_index(terrain)
    return idx.get(coord)
