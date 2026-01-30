"""Geometry validation helpers for mech combat.

Covers line of sight, cover, movement paths, and engagement geometry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from core.mech.combat_rules import DEFAULT_MECH_COMBAT_RULES, LineOfSightRules
from core.mech.combat_state import CombatantState
from core.mech.grid import HexCoord, hexes_between
from core.mech.terrain import TerrainHex

if TYPE_CHECKING:

    from core.mech.combat_state import ActionUse
    from core.mech.combat_actions import ActionRule
    from core.mech.grid import HexPosition


__all__ = [
    "line_of_sight_blocked",
    "cover_between",
    "movement_path_valid",
    "is_adjacent",
    "_adjacency_distance",
    "_size_value",
    "_hostiles_for",
    "_surface_elevation",
    "_movement_segments",
    "_is_engaged",
    "_line_of_sight_clear",
    "_path_clear",
    "_cover_between",
]


def _adjacent_hard_cover_coords(
    tiles: dict[HexCoord, TerrainHex],
    target_coord: HexCoord,
    target_size: str | None,
) -> set[HexCoord]:
    adjacent: set[HexCoord] = set()
    for neighbor in target_coord.neighbors():
        tile = tiles.get(neighbor)
        if (
            tile
            and tile.provides_hard_cover
            and _cover_size_allows_target(tile.hard_cover_size, target_size)
        ):
            adjacent.add(neighbor)
    return adjacent


def _cover_size_allows_target(cover_size: str | None, target_size: str | None) -> bool:
    if cover_size is None or target_size is None:
        return True
    return _size_value(cover_size) >= _size_value(target_size)


def _is_flanking(
    tiles: dict[HexCoord, TerrainHex],
    attacker_coord: HexCoord,
    target_coord: HexCoord,
) -> bool:
    line = hexes_between(attacker_coord, target_coord, include_endpoints=False)
    if not line:
        return True
    cover_coord = line[-1]
    tile = tiles.get(cover_coord)
    return not (tile and tile.provides_hard_cover)


def _cover_between(
    tiles: dict[HexCoord, TerrainHex],
    start_coord: HexCoord,
    end_coord: HexCoord,
    target_size: str | None,
) -> Literal["hard", "soft", "none"]:
    cover_rules = DEFAULT_MECH_COMBAT_RULES.cover_rules
    line = hexes_between(start_coord, end_coord, include_endpoints=False)
    hard_between = any(
        tiles.get(coord) and tiles[coord].provides_hard_cover for coord in line
    )
    hard_between_size_ok = any(
        tiles.get(coord)
        and tiles[coord].provides_hard_cover
        and _cover_size_allows_target(tiles[coord].hard_cover_size, target_size)
        for coord in line
    )
    soft_between = any(
        tiles.get(coord) and tiles[coord].provides_soft_cover for coord in line
    )

    if not cover_rules.hard_cover_requires_adjacency and hard_between:
        if cover_rules.hard_cover_requires_size_match and not hard_between_size_ok:
            return "soft" if hard_between or soft_between else "none"
        return "hard"

    size_check = target_size if cover_rules.hard_cover_requires_size_match else None
    hard_adjacent = _adjacent_hard_cover_coords(tiles, end_coord, size_check)
    if hard_adjacent:
        hard_cover = True
        if cover_rules.hard_cover_flanking_negates and _is_flanking(
            tiles, start_coord, end_coord
        ):
            hard_cover = False
        if cover_rules.hard_cover_requires_adjacency and hard_cover:
            return "hard"
        if not cover_rules.hard_cover_requires_adjacency and hard_between:
            return "hard"

    if soft_between or hard_between:
        return "soft"

    return "none"


def _line_of_sight_clear(
    tiles: dict[HexCoord, TerrainHex],
    start_coord: HexCoord,
    end_coord: HexCoord,
    start_elevation: int,
    end_elevation: int,
    line_of_sight_rules: LineOfSightRules,
    target_coord: HexCoord | None = None,
) -> bool:
    for coord in hexes_between(
        start_coord,
        end_coord,
        include_endpoints=False,
    ):
        tile = tiles.get(coord)
        if tile and tile.blocks_line_of_sight:
            if (
                line_of_sight_rules.adjacent_cover_does_not_block_los
                and target_coord
                and tile.provides_hard_cover
                and coord.distance_to(target_coord) == 1
            ):
                continue
            if tile.elevation >= min(start_elevation, end_elevation):
                return False
    return True


def _path_clear(
    tiles: dict[HexCoord, TerrainHex],
    start_coord: HexCoord,
    end_coord: HexCoord,
    line_of_sight_rules: LineOfSightRules,
    target_coord: HexCoord | None = None,
) -> bool:
    for coord in hexes_between(
        start_coord,
        end_coord,
        include_endpoints=False,
    ):
        tile = tiles.get(coord)
        if tile and tile.blocks_line_of_sight:
            if (
                line_of_sight_rules.adjacent_cover_does_not_block_los
                and target_coord
                and tile.provides_hard_cover
                and coord.distance_to(target_coord) == 1
            ):
                continue
            return False
    return True


def _terrain_elevation(
    tiles: dict[HexCoord, TerrainHex],
    coord: HexCoord,
) -> int:
    tile = tiles.get(coord)
    return tile.elevation if tile else 0


def _blocked_area_coords_by_los(
    tiles: dict[HexCoord, TerrainHex],
    origin: HexPosition,
    coords: list[HexCoord],
    line_of_sight_rules: LineOfSightRules,
) -> set[HexCoord]:
    blocked: set[HexCoord] = set()
    for coord in coords:
        end_elevation = _terrain_elevation(tiles, coord)
        if not _line_of_sight_clear(
            tiles,
            origin.coord,
            coord,
            origin.elevation,
            end_elevation,
            line_of_sight_rules,
            coord,
        ):
            blocked.add(coord)
    return blocked


def _blocked_area_coords_by_path(
    tiles: dict[HexCoord, TerrainHex],
    origin: HexPosition,
    coords: list[HexCoord],
    line_of_sight_rules: LineOfSightRules,
) -> set[HexCoord]:
    blocked: set[HexCoord] = set()
    for coord in coords:
        if not _path_clear(
            tiles,
            origin.coord,
            coord,
            line_of_sight_rules,
            coord,
        ):
            blocked.add(coord)
    return blocked


def _adjacency_distance(attacker: CombatantState, target: CombatantState) -> int:
    size_to_radius: dict[str, int] = {
        "size_half": 1,
        "size_1": 1,
        "size_2": 2,
        "size_3": 3,
        "size_4": 4,
        "size_5": 5,
    }
    return max(size_to_radius[attacker.stats.size], size_to_radius[target.stats.size])


def _size_value(size: str) -> float:
    size_values: dict[str, float] = {
        "size_half": 0.5,
        "size_1": 1.0,
        "size_2": 2.0,
        "size_3": 3.0,
        "size_4": 4.0,
        "size_5": 5.0,
    }
    return size_values.get(size, 1.0)


def _hostiles_for(
    actor: CombatantState, combatants: dict[str, CombatantState]
) -> list[CombatantState]:
    if actor.side == "players":
        return [c for c in combatants.values() if c.side == "hostiles"]
    if actor.side == "hostiles":
        return [c for c in combatants.values() if c.side == "players"]
    return []


def _surface_elevation(
    terrain_tiles: dict[HexCoord, TerrainHex],
    position: HexPosition,
) -> int:
    tile = terrain_tiles.get(position.coord)
    return tile.elevation if tile else 0


def _movement_segments(path: list[HexPosition]) -> int:
    if len(path) < 2:
        return 0
    directions: list[tuple[int, int, int]] = []
    for prev, curr in zip(path, path[1:]):
        dq = curr.coord.q - prev.coord.q
        dr = curr.coord.r - prev.coord.r
        de = curr.elevation - prev.elevation
        if dq == 0 and dr == 0 and de == 0:
            continue
        directions.append((dq, dr, de))
    if not directions:
        return 0
    segments = 1
    for prev_dir, curr_dir in zip(directions, directions[1:]):
        if curr_dir != prev_dir:
            segments += 1
    return segments


def _is_engaged(actor: CombatantState, hostiles: list[CombatantState]) -> bool:
    if not actor.position:
        return False
    for hostile in hostiles:
        if not hostile.position:
            continue
        distance = actor.position.distance_3d(hostile.position)
        if distance <= _adjacency_distance(actor, hostile):
            return True
    return False


def _effective_ignores_los(
    action: ActionUse, line_of_sight_rules: LineOfSightRules
) -> bool:
    tags = set(action.weapon_tags)
    if action.ignores_line_of_sight:
        return True
    if "seeking" in tags and line_of_sight_rules.seeking_ignores_los:
        return True
    if "arcing" in tags and line_of_sight_rules.arcing_allows_no_los:
        return True
    return False


def _effective_ignores_cover(
    action: ActionUse,
    rule: ActionRule | None,
    line_of_sight_rules: LineOfSightRules,
) -> bool:
    tags = set(action.weapon_tags)
    rule_ignores = bool(rule and rule.attack and rule.attack.ignores_cover)
    if action.ignores_cover or rule_ignores:
        return True
    if "seeking" in tags and line_of_sight_rules.seeking_ignores_cover:
        return True
    return False


def line_of_sight_blocked(
    tiles: dict[HexCoord, TerrainHex],
    start_coord: HexCoord,
    end_coord: HexCoord,
    start_elevation: int,
    end_elevation: int,
    line_of_sight_rules: LineOfSightRules | None = None,
) -> bool:
    """Public wrapper for line of sight check.

    Args:
        tiles: Terrain map indexed by HexCoord
        start_coord: Starting hex coordinates
        end_coord: Ending hex coordinates
        start_elevation: Starting elevation
        end_elevation: Ending elevation
        line_of_sight_rules: LOS rules (uses default if None)

    Returns:
        True if line of sight is blocked
    """
    rules = line_of_sight_rules or DEFAULT_MECH_COMBAT_RULES.line_of_sight_rules
    return not _line_of_sight_clear(
        tiles, start_coord, end_coord, start_elevation, end_elevation, rules
    )


def cover_between(
    tiles: dict[HexCoord, TerrainHex],
    start_coord: HexCoord,
    end_coord: HexCoord,
    target_size: str | None = None,
) -> Literal["hard", "soft", "none"]:
    """Public wrapper for cover calculation.

    Args:
        tiles: Terrain map indexed by HexCoord
        start_coord: Attacker hex coordinates
        end_coord: Target hex coordinates
        target_size: Target size for cover size checks

    Returns:
        "hard", "soft", or "none"
    """
    return _cover_between(tiles, start_coord, end_coord, target_size)


def movement_path_valid(
    path: list[HexPosition],
    max_segments: int | None = None,
) -> bool:
    """Check if a movement path is valid.

    Args:
        path: List of positions in movement order
        max_segments: Maximum allowed direction changes (None for unlimited)

    Returns:
        True if path is valid
    """
    if len(path) < 2:
        return True

    for prev, curr in zip(path, path[1:]):
        if prev.distance_3d(curr) > 1:
            return False

    if max_segments is not None:
        return _movement_segments(path) <= max_segments

    return True


def is_adjacent(
    actor: CombatantState,
    target: CombatantState,
    allow_same_hex: bool = True,
) -> bool:
    """Check if two combatants are adjacent.

    Args:
        actor: First combatant
        target: Second combatant
        allow_same_hex: Consider same hex as adjacent (for large mechs)

    Returns:
        True if combatants are within adjacency distance
    """
    if not actor.position or not target.position:
        return False

    distance = actor.position.distance_3d(target.position)
    max_distance = _adjacency_distance(actor, target)

    if allow_same_hex and distance == 0:
        return True

    return distance <= max_distance
