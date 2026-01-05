"""Line of sight resolution for Lancer combat.

Per PR2 ~4030-4068:
- Draw line from center of attacker to center of target
- If line is "mostly unbroken" = clear shot (no cover)
- If line is "significantly obscured or broken up" = soft cover (+1 difficulty)
- If solid obstruction blocks line = may grant hard cover (+2 difficulty, requires adjacency)

This module provides type-safe helpers for:
- Checking line of sight between two positions
- Detecting soft cover (obscured LOS)
- Detecting blocking terrain (hard cover potential)
- Path clearance for seeking/arcing weapons
"""

from __future__ import annotations

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel
from core.mech.grid import HexCoord, HexPosition
from core.mech.terrain import TerrainMap, TerrainHex, terrain_index


class LOSResult(FrozenModel):
    """Result of line of sight check.

    Attributes:
        has_los: Whether attacker has line of sight to target
        los_type: LOS quality - clear, obscured (soft cover), or blocked
        blocked_by: List of hex coordinates that block LOS entirely
        obscured_by: List of hex coordinates that cause soft cover
        reason: Detailed explanation of LOS result
    """

    has_los: bool = False
    los_type: Literal["clear", "obscured", "blocked"] = "blocked"
    blocked_by: list[HexCoord] = Field(default_factory=list)
    obscured_by: list[HexCoord] = Field(default_factory=list)
    reason: str = ""


class LOSCheckRequest(FrozenModel):
    """Request for line of sight check.

    Attributes:
        attacker_pos: Attacker's position
        target_pos: Target's position
        terrain: Terrain map for checking obstructions
        check_elevation: Whether to account for elevation in LOS check
    """

    attacker_pos: HexPosition
    target_pos: HexPosition
    terrain: TerrainMap | None = None
    check_elevation: bool = True


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
    return idx.get((coord.q, coord.r))


def check_line_of_sight(request: LOSCheckRequest) -> LOSResult:
    """Check line of sight between two positions.

    Per PR2 ~4056:
    "Draw a line from the center of one character to the center of another
    character. If that line can be drawn mostly unbroken it's a clear shot.
    If the line is significantly obscured or broken up... the target has soft cover."

    Args:
        request: LOS check request with positions and terrain

    Returns:
        LOSResult with has_los status, los_type, and detailed breakdown
    """
    from core.mech.grid import hex_line

    attacker_coord = request.attacker_pos.coord
    target_coord = request.target_pos.coord

    path = hex_line(attacker_coord, target_coord)

    if len(path) <= 1:
        return LOSResult(
            has_los=True,
            los_type="clear",
            reason="Attacker and target are at the same position",
        )

    blocked_hexes: list[HexCoord] = []
    obscured_hexes: list[HexCoord] = []

    for hex_coord in path[1:-1]:
        hex_terrain = get_terrain_at(request.terrain, hex_coord)

        if hex_terrain is None:
            continue

        if hex_terrain.blocks_line_of_sight:
            blocked_hexes.append(hex_coord)

        if hex_terrain.provides_soft_cover:
            obscured_hexes.append(hex_coord)

    if blocked_hexes:
        blocked_coords = ", ".join(f"({c.q}, {c.r})" for c in blocked_hexes)
        return LOSResult(
            has_los=False,
            los_type="blocked",
            blocked_by=blocked_hexes,
            obscured_by=obscured_hexes,
            reason=f"Line of sight blocked by terrain at {blocked_coords}",
        )

    if obscured_hexes:
        return LOSResult(
            has_los=True,
            los_type="obscured",
            blocked_by=blocked_hexes,
            obscured_by=obscured_hexes,
            reason="Line of sight obscured by soft cover terrain",
        )

    return LOSResult(
        has_los=True,
        los_type="clear",
        reason="Clear line of sight",
    )


def check_obscured_los(request: LOSCheckRequest) -> bool:
    """Check if line of sight is obscured (soft cover present).

    Per PR2 ~4039-4048:
    "If your shot is obscured or obstructed somehow, your target has soft cover."

    This is a simplified check that returns True if there is soft cover
    terrain that would give the target soft cover.

    Args:
        request: LOS check request with positions and terrain

    Returns:
        True if LOS is obscured (soft cover applies)
    """
    los_result = check_line_of_sight(request)
    return los_result.los_type == "obscured"


def check_clear_los(request: LOSCheckRequest) -> bool:
    """Check if line of sight is clear (no cover).

    Per PR2 ~4056:
    "If that line can be drawn mostly unbroken it's a clear shot."

    Args:
        request: LOS check request with positions and terrain

    Returns:
        True if LOS is clear (no cover)
    """
    los_result = check_line_of_sight(request)
    return los_result.los_type == "clear"


def check_path_clear(
    terrain: TerrainMap | None,
    attacker_coord: HexCoord,
    target_coord: HexCoord,
) -> tuple[bool, list[HexCoord]]:
    """Check if a clear path exists (for seeking/arcing weapons).

    Per PR2 ~4028-4030:
    "Weapons with the powerful seeking tag totally ignore cover and line of sight,
    as long as they could draw a path to their target."
    "The arcing tag can still attack... as long as it can trace a path to its target."

    This checks if any path exists, ignoring cover/LOS but checking for
    actual obstructions that would block a physical path.

    Args:
        terrain: The terrain map
        attacker_coord: Attacker's position
        target_coord: Target's position

    Returns:
        Tuple of (path_exists, blocking_hexes)
    """
    from core.mech.grid import hex_line

    path = hex_line(attacker_coord, target_coord)

    if len(path) <= 1:
        return True, []

    blocking_hexes: list[HexCoord] = []

    for hex_coord in path[1:-1]:
        hex_terrain = get_terrain_at(terrain, hex_coord)

        if hex_terrain is not None and hex_terrain.blocks_line_of_sight:
            blocking_hexes.append(hex_coord)

    if blocking_hexes:
        return False, blocking_hexes

    return True, []


def check_los_with_cover(
    attacker_pos: HexPosition,
    target_pos: HexPosition,
    terrain: TerrainMap | None,
) -> LOSResult:
    """Check line of sight and identify cover type.

    Convenience function that combines LOS check with cover identification.

    Per PR2:
    - Clear LOS = no cover bonus
    - Obscured LOS = soft cover (+1 difficulty)
    - Blocked LOS = may have hard cover (if adjacent)

    Args:
        attacker_pos: Attacker's position
        target_pos: Target's position
        terrain: Terrain map for checking cover

    Returns:
        LOSResult with cover-related information
    """
    request = LOSCheckRequest(
        attacker_pos=attacker_pos,
        target_pos=target_pos,
        terrain=terrain,
        check_elevation=True,
    )

    return check_line_of_sight(request)


def check_elevation_blocks_los(
    attacker_pos: HexPosition,
    target_pos: HexPosition,
    terrain: TerrainMap | None,
) -> bool:
    """Check if elevation difference blocks line of sight.

    Per PR2 flight rules ~3909-3917:
    - Flying at altitude ignores obstructions
    - Terrain elevation can block shots

    This is a simplified check - if attacker is significantly higher
    or lower than target, and terrain at target blocks LOS, the
    elevation may affect targeting.

    Args:
        attacker_pos: Attacker's position
        target_pos: Target's position
        terrain: Terrain map

    Returns:
        True if elevation would block LOS
    """
    elevation_delta = abs(attacker_pos.elevation - target_pos.elevation)

    if elevation_delta == 0:
        return False

    target_terrain = get_terrain_at(terrain, target_pos.coord)

    if target_terrain is None:
        return False

    if attacker_pos.elevation > target_pos.elevation:
        if target_terrain.blocks_line_of_sight:
            return True

    return False


def get_los_blocking_hexes(
    terrain: TerrainMap | None,
    attacker_coord: HexCoord,
    target_coord: HexCoord,
) -> list[HexCoord]:
    """Get all hex coordinates that block line of sight.

    Args:
        terrain: The terrain map
        attacker_coord: Attacker's position
        target_coord: Target's position

    Returns:
        List of hex coordinates that block LOS
    """
    from core.mech.grid import hex_line

    path = hex_line(attacker_coord, target_coord)
    blocking_hexes: list[HexCoord] = []

    for hex_coord in path[1:-1]:
        hex_terrain = get_terrain_at(terrain, hex_coord)

        if hex_terrain is not None and hex_terrain.blocks_line_of_sight:
            blocking_hexes.append(hex_coord)

    return blocking_hexes


def get_los_obscuring_hexes(
    terrain: TerrainMap | None,
    attacker_coord: HexCoord,
    target_coord: HexCoord,
) -> list[HexCoord]:
    """Get all hex coordinates that obscure line of sight (soft cover).

    Args:
        terrain: The terrain map
        attacker_coord: Attacker's position
        target_coord: Target's position

    Returns:
        List of hex coordinates that cause soft cover
    """
    from core.mech.grid import hex_line

    path = hex_line(attacker_coord, target_coord)
    obscuring_hexes: list[HexCoord] = []

    for hex_coord in path[1:-1]:
        hex_terrain = get_terrain_at(terrain, hex_coord)

        if hex_terrain is not None and hex_terrain.provides_soft_cover:
            obscuring_hexes.append(hex_coord)

    return obscuring_hexes
