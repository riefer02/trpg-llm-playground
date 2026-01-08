"""Terrain resolution helpers for Lancer combat.

This module provides type-safe helpers for terrain effects including:
- Movement cost calculation (difficult terrain)
- Cover difficulty modifiers (soft/hard cover)
- Hard cover availability (adjacency + size matching)
- Dangerous terrain resolution (engineering check)
- Elevation bonuses (for talents like Solar Backdrop)

Terrain Rules (per PR2 ~3851-3860):
- Difficult Terrain: 1 space costs 2 movement
- Dangerous Terrain: Engineering check on entry, 5 damage on failure
- Cover: Soft (+1 difficulty) or Hard (+2 difficulty, requires adjacency)
- Elevation: Higher elevation grants +1 accuracy (talent-dependent)

Terrain types are defined here (moved from core/mech/terrain.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.enums import DamageType, SizeClass
from core.shared.dice import roll_dice, round_up
from core.mech.grid import HexCoord

if TYPE_CHECKING:
    from core.mech.grid import HexPosition


class TerrainHex(FrozenModel):
    """Terrain entry for a single hex."""

    coord: HexCoord
    elevation: int = Field(default=0, ge=0)
    blocks_line_of_sight: bool = False
    provides_soft_cover: bool = False
    provides_hard_cover: bool = False
    hard_cover_size: SizeClass | None = None
    difficult: bool = False
    dangerous: bool = False


class TerrainMap(FrozenModel):
    """Sparse terrain map for combat scenarios."""

    tiles: list[TerrainHex] = Field(default_factory=list)


def terrain_index(terrain: TerrainMap | None) -> dict[HexCoord, TerrainHex]:
    """Build a lookup table for terrain by axial coordinates.

    Args:
        terrain: The terrain map (None for empty dict)

    Returns:
        Dict mapping HexCoord to TerrainHex
    """
    if terrain is None:
        return {}
    return {tile.coord: tile for tile in terrain.tiles}


def terrain_at(terrain: TerrainMap | None, coord: HexCoord) -> TerrainHex | None:
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


class TerrainEffectResult(FrozenModel):
    """All terrain effects at a specific coordinate."""

    coord: HexCoord
    elevation: int = 0
    blocks_line_of_sight: bool = False
    provides_soft_cover: bool = False
    provides_hard_cover: bool = False
    hard_cover_size: SizeClass | None = None
    difficult: bool = False
    dangerous: bool = False


class CoverDifficultyResult(FrozenModel):
    """Result of cover difficulty calculation."""

    cover_type: Literal["none", "soft", "hard"] = "none"
    difficulty_modifier: int = 0
    reason: str = ""


class HardCoverAvailabilityResult(FrozenModel):
    """Result of hard cover availability check."""

    available: bool = False
    adjacent_cover_hex: HexCoord | None = None
    cover_size: SizeClass | None = None
    size_match: bool = False
    reason: str = ""


class DangerousTerrainResult(FrozenModel):
    """Result of dangerous terrain check resolution."""

    check_required: bool = False
    check_already_done_this_round: bool = False
    check_passed: bool | None = None
    roll_result: int | None = None
    damage_dealt: int = 0
    damage_type: DamageType = "kinetic"
    reason: str = ""


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
    return terrain_at(terrain, coord)


def get_terrain_effects_at(
    terrain: TerrainMap | None,
    coord: HexCoord,
) -> TerrainEffectResult:
    """Get all terrain effects at a coordinate.

    Args:
        terrain: The terrain map (None for no terrain)
        coord: The coordinate to check

    Returns:
        TerrainEffectResult with all terrain properties
    """
    hex_terrain = get_terrain_at(terrain, coord)

    if hex_terrain is None:
        return TerrainEffectResult(coord=coord)

    return TerrainEffectResult(
        coord=coord,
        elevation=hex_terrain.elevation,
        blocks_line_of_sight=hex_terrain.blocks_line_of_sight,
        provides_soft_cover=hex_terrain.provides_soft_cover,
        provides_hard_cover=hex_terrain.provides_hard_cover,
        hard_cover_size=hex_terrain.hard_cover_size,
        difficult=hex_terrain.difficult,
        dangerous=hex_terrain.dangerous,
    )


def calculate_movement_cost(
    spaces: int,
    terrain: TerrainMap | None,
    coord: HexCoord,
    difficult_cost: int = 2,
) -> int:
    """Calculate movement cost for a space, accounting for difficult terrain.

    Per PR2: "1 space of movement through Difficult Terrain costs 2 spaces
    worth of movement speed."

    Args:
        spaces: Number of spaces to move through this terrain
        terrain: The terrain map
        coord: The coordinate being moved through
        difficult_cost: Movement cost per difficult terrain space (default 2)

    Returns:
        Total movement cost for these spaces
    """
    hex_terrain = get_terrain_at(terrain, coord)

    if hex_terrain is None or not hex_terrain.difficult:
        return spaces

    return spaces * difficult_cost


def get_cover_difficulty(
    terrain: TerrainMap | None,
    attacker_coord: HexCoord,
    target_coord: HexCoord,
    target_size: SizeClass,
    soft_cover_difficulty: int = 1,
    hard_cover_difficulty: int = 2,
    hard_cover_requires_adjacency: bool = True,
    hard_cover_flanking_negates: bool = True,
    hard_cover_requires_size_match: bool = True,
    hard_cover_size: SizeClass | None = None,
) -> CoverDifficultyResult:
    """Calculate cover difficulty modifier for a ranged attack.

    Per PR2:
    - Soft Cover: +1 Difficulty
    - Hard Cover: +2 Difficulty (requires adjacency)
    - Cover types do not stack
    - Flanking negates hard cover

    Args:
        terrain: The terrain map
        attacker_coord: Attacker's position
        target_coord: Target's position
        target_size: Target's size class
        soft_cover_difficulty: Difficulty from soft cover (default 1)
        hard_cover_difficulty: Difficulty from hard cover (default 2)
        hard_cover_requires_adjacency: Hard cover needs adjacency (default True)
        hard_cover_flanking_negates: Flanking removes hard cover (default True)
        hard_cover_requires_size_match: Hard cover needs size match (default True)
        hard_cover_size: Specific hard cover size (defaults to terrain hard_cover_size)

    Returns:
        CoverDifficultyResult with cover type and modifier
    """
    hard_cover = check_hard_cover_available(
        terrain=terrain,
        attacker_coord=attacker_coord,
        target_coord=target_coord,
        target_size=target_size,
        requires_adjacency=hard_cover_requires_adjacency,
        requires_size_match=hard_cover_requires_size_match,
        hard_cover_size=hard_cover_size,
    )

    is_flanked = False
    if hard_cover.available and hard_cover_flanking_negates:
        from core.shared.flanking import check_flanking

        flanking_result = check_flanking(terrain, attacker_coord, target_coord)
        is_flanked = flanking_result.is_flanked

    if hard_cover.available and not is_flanked:
        return CoverDifficultyResult(
            cover_type="hard",
            difficulty_modifier=hard_cover_difficulty,
            reason=f"Target has hard cover (adjacent to size {hard_cover.cover_size or 'N/A'})",
        )

    soft_cover = check_soft_cover(terrain, attacker_coord, target_coord)

    if soft_cover:
        if is_flanked and hard_cover.available:
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

    if is_flanked and hard_cover.available:
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


def check_soft_cover(
    terrain: TerrainMap | None,
    attacker_coord: HexCoord,
    target_coord: HexCoord,
) -> bool:
    """Check if target has soft cover (obscured line of sight).

    Per PR2: "If that line [attacker to target] is significantly obscured or
    broken up by a smoke cloud, trees, fencing, or objects that would give
    hard cover (but the target isn't adjacent to that cover), the target has
    soft cover."

    This is a simplified check that returns True if there are any terrain
    hexes that block line of sight between attacker and target.

    Args:
        terrain: The terrain map
        attacker_coord: Attacker's position
        target_coord: Target's position

    Returns:
        True if soft cover is present
    """
    if terrain is None:
        return False

    los_terrain = get_terrain_effects_at(terrain, target_coord)

    if los_terrain.blocks_line_of_sight or los_terrain.provides_soft_cover:
        return True

    return False


def check_hard_cover_available(
    terrain: TerrainMap | None,
    attacker_coord: HexCoord,
    target_coord: HexCoord,
    target_size: SizeClass,
    requires_adjacency: bool = True,
    requires_size_match: bool = True,
    hard_cover_size: SizeClass | None = None,
) -> HardCoverAvailabilityResult:
    """Check if target has hard cover available.

    Per PR2: "A character needs something to be at least the same size as it
    is to benefit from hard cover... and must be physically adjacent to that
    cover to benefit from it."

    Note: This checks hard cover availability only. Flanking negation is
    handled separately by the caller (deferred to future priority).

    Args:
        terrain: The terrain map
        attacker_coord: Attacker's position
        target_coord: Target's position
        target_size: Target's size class
        requires_adjacency: Hard cover needs adjacency (default True)
        requires_size_match: Hard cover needs size match (default True)
        hard_cover_size: Specific hard cover size (defaults to terrain hard_cover_size)

    Returns:
        HardCoverAvailabilityResult with availability details
    """
    if terrain is None:
        return HardCoverAvailabilityResult(
            available=False,
            reason="No terrain",
        )

    size_order: dict[SizeClass, int] = {
        "size_half": 0,
        "size_1": 1,
        "size_2": 2,
        "size_3": 3,
        "size_4": 4,
    }
    target_size_val = size_order.get(target_size, 0)

    if requires_adjacency:
        adjacent_coords = target_coord.neighbors()
        best_adjacent_cover: HardCoverAvailabilityResult | None = None

        for adj_coord in adjacent_coords:
            adj_hex = get_terrain_at(terrain, adj_coord)
            if adj_hex and adj_hex.provides_hard_cover:
                adj_cover_size = hard_cover_size or adj_hex.hard_cover_size

                if adj_cover_size is None:
                    result = HardCoverAvailabilityResult(
                        available=False,
                        adjacent_cover_hex=adj_coord,
                        cover_size=None,
                        size_match=False,
                        reason="Adjacent hard cover has no size specified",
                    )
                    if best_adjacent_cover is None:
                        best_adjacent_cover = result
                    continue

                size_match = True
                if requires_size_match:
                    cover_size_val = size_order.get(adj_cover_size, 0)
                    if cover_size_val < target_size_val:
                        size_match = False

                if size_match:
                    return HardCoverAvailabilityResult(
                        available=True,
                        adjacent_cover_hex=adj_coord,
                        cover_size=adj_cover_size,
                        size_match=True,
                        reason="Hard cover available",
                    )
                else:
                    result = HardCoverAvailabilityResult(
                        available=False,
                        adjacent_cover_hex=adj_coord,
                        cover_size=adj_cover_size,
                        size_match=False,
                        reason=f"Cover size ({adj_cover_size}) smaller than target ({target_size})",
                    )
                    if best_adjacent_cover is None:
                        best_adjacent_cover = result

        if best_adjacent_cover is not None:
            return best_adjacent_cover

        return HardCoverAvailabilityResult(
            available=False,
            adjacent_cover_hex=None,
            cover_size=None,
            size_match=False,
            reason="No adjacent hard cover found",
        )

    return HardCoverAvailabilityResult(
        available=False,
        adjacent_cover_hex=None,
        cover_size=None,
        size_match=False,
        reason="Adjacency check disabled but no cover at target location",
    )


def resolve_dangerous_terrain(
    terrain: TerrainMap | None,
    coord: HexCoord,
    skill_bonus: int,
    damage: int = 5,
    damage_type: DamageType = "kinetic",
    check_skill: Literal["engineering"] = "engineering",
    check_once_per_round: bool = True,
    round_checked: int | None = None,
    rounds_already_checked: set[int] | None = None,
) -> DangerousTerrainResult:
    """Resolve dangerous terrain effects.

    Per PR2: "Dangerous Terrain prompts an engineering check to navigate the
    first time on a turn an actor enters it on their turn, or if they start
    their turn there. Should a player fail that check, they take 5 kinetic,
    energy, explosive, or burn damage on failure."

    Args:
        terrain: The terrain map
        coord: The coordinate being entered/occupied
        skill_bonus: Bonus to the engineering check (skill rank + triggers)
        damage: Damage on failed check (default 5)
        damage_type: Damage type on failed check (default kinetic)
        check_skill: Skill for check (default engineering)
        check_once_per_round: Only one check per round (default True)
        round_checked: Current round number for tracking
        rounds_already_checked: Set of rounds already checked (for tracking)

    Returns:
        DangerousTerrainResult with check outcome
    """
    hex_terrain = get_terrain_at(terrain, coord)

    if hex_terrain is None or not hex_terrain.dangerous:
        return DangerousTerrainResult(
            check_required=False,
            reason="No dangerous terrain at coordinate",
        )

    if check_once_per_round and round_checked is not None:
        already_checked = rounds_already_checked or set()
        if round_checked in already_checked:
            return DangerousTerrainResult(
                check_required=True,
                check_already_done_this_round=True,
                check_passed=None,
                damage_dealt=0,
                damage_type=damage_type,
                reason=f"Already checked dangerous terrain in round {round_checked}",
            )

    roll = roll_dice("1d20")
    total = roll + skill_bonus

    if total >= 10:
        if (
            check_once_per_round
            and round_checked is not None
            and rounds_already_checked is not None
        ):
            rounds_already_checked.add(round_checked)
        return DangerousTerrainResult(
            check_required=True,
            check_passed=True,
            roll_result=roll,
            damage_dealt=0,
            damage_type=damage_type,
            reason=f"Dangerous terrain check passed ({roll}+{skill_bonus} vs DC 10)",
        )

    if (
        check_once_per_round
        and round_checked is not None
        and rounds_already_checked is not None
    ):
        rounds_already_checked.add(round_checked)
    return DangerousTerrainResult(
        check_required=True,
        check_passed=False,
        roll_result=roll,
        damage_dealt=damage,
        damage_type=damage_type,
        reason=f"Dangerous terrain check failed ({roll}+{skill_bonus} < 10), {damage} {damage_type} damage",
    )


def get_elevation_bonus(
    attacker_elevation: int,
    target_elevation: int,
) -> int:
    """Calculate elevation accuracy bonus.

    Per PR2 (Solar Backdrop talent): "gain +1 Accuracy on any ranged attack
    if you are standing or flying at a higher elevation than your target."

    Note: This is currently only referenced by talents, not a general rule.
    This helper enables talents that reference elevation.

    Args:
        attacker_elevation: Attacker's elevation
        target_elevation: Target's elevation

    Returns:
        +1 if attacker is higher, 0 otherwise
    """
    if attacker_elevation > target_elevation:
        return 1
    return 0


def calculate_climb_cost(
    spaces: int,
    climb_cost: int = 2,
) -> int:
    """Calculate movement cost for climbing.

    Per PR2: "Climbing like difficult terrain, costs 2 spaces of movement
    for every space moved."

    Args:
        spaces: Number of spaces being climbed
        climb_cost: Movement cost per space (default 2)

    Returns:
        Total movement cost for climbing
    """
    return spaces * climb_cost
