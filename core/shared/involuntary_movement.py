"""Involuntary movement resolution helpers for Lancer combat.

This module provides type-safe helpers for involuntary movement including:
- Push, pull, shove, and knockback resolution
- Lifting and dragging mechanics
- Path validation for straight-line movement
- Obstruction handling

Involuntary Movement Rules (per PR2 ~3846-3849):
- Push, pull, or shove forces movement in a direct line
- Does NOT provoke reactions
- Ignores engagement during movement
- Must still obey obstructions

Lifting and Dragging (per PR2 ~3862-3865):
- Drag: up to 2x size, becomes slowed
- Lift: up to own size, becomes immobilized
- Cannot take reactions while dragging/lifting
- Pilots: cannot drag/lift > size 1/2

Knockback Rules (per PR2 various):
- Knockback moves target in straight line directly away
- Blocked by obstructions
- Some effects grant bonus knockback spaces
"""

from __future__ import annotations

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.enums import SizeClass, StatusType
from core.shared.conditions import apply_condition, ConditionApplicationResult
from core.mech.grid import HexCoord
from core.mech.terrain import TerrainMap, terrain_index


SIZE_ORDER: dict[SizeClass, int] = {
    "size_half": 1,
    "size_1": 1,
    "size_2": 2,
    "size_3": 3,
    "size_4": 4,
    "size_5": 5,
}


class InvoluntaryMovementType(FrozenModel):
    """Type of involuntary movement."""

    type: Literal["push", "pull", "shove", "knockback", "drag", "lift"]
    spaces: int = Field(default=1, ge=0)
    direction: HexCoord | None = None


class InvoluntaryMovementPath(FrozenModel):
    """Path validation for involuntary movement."""

    start: HexCoord
    end: HexCoord | None = None
    spaces_moved: int = 0
    spaces_requested: int = 0
    path_clear: bool = True
    obstructions: list[HexCoord] = Field(default_factory=list)
    straight_line: bool = True


class InvoluntaryMovementResult(FrozenModel):
    """Result of involuntary movement resolution."""

    movement_type: str
    start: HexCoord
    end: HexCoord | None
    spaces_moved: int
    path_clear: bool
    obstructed: bool = False
    obstruction_coord: HexCoord | None = None
    provoked_reactions: bool = False
    ignored_engagement: bool = True
    reason: str = ""


class PushResult(FrozenModel):
    """Push movement result."""

    spaces_pushed: int
    end_position: HexCoord
    obstructed: bool
    obstruction_coord: HexCoord | None = None
    reason: str = ""


class PullResult(FrozenModel):
    """Pull movement result."""

    spaces_pulled: int
    end_position: HexCoord
    obstructed: bool
    obstruction_coord: HexCoord | None = None
    reason: str = ""


class KnockbackResult(FrozenModel):
    """Knockback movement result."""

    spaces_knocked: int
    end_position: HexCoord | None
    obstructed: bool
    obstruction_coord: HexCoord | None = None
    direction: HexCoord
    reason: str = ""


class ShoveResult(FrozenModel):
    """Shove movement result."""

    spaces_shoved: int
    end_position: HexCoord
    obstructed: bool
    obstruction_coord: HexCoord | None = None
    reason: str = ""


class DragResult(FrozenModel):
    """Drag movement result."""

    dragger_size: SizeClass
    dragged_size: SizeClass
    can_drag: bool
    max_drag_size: int
    slowed_applied: bool = False
    spaces_moved: int = 0
    end_position: HexCoord | None = None
    obstructed: bool = False
    reason: str = ""


class LiftResult(FrozenModel):
    """Lift movement result."""

    lifter_size: SizeClass
    lifted_size: SizeClass
    can_lift: bool
    immobilized_applied: bool = False
    lifted_overhead: bool = False
    reason: str = ""


def _is_coord_occupied(
    coord: HexCoord,
    terrain: TerrainMap | None,
    occupied_hexes: dict[HexCoord, bool] | None = None,
) -> bool:
    """Check if a coordinate is occupied or blocked.

    Args:
        coord: Coordinate to check
        terrain: Terrain map for terrain-based blocking
        occupied_hexes: Map of explicitly occupied hexes

    Returns:
        True if coordinate is occupied or blocked
    """
    if occupied_hexes and coord in occupied_hexes:
        return True

    return False


def validate_straight_line_path(
    start: HexCoord,
    direction: HexCoord,
    spaces: int,
    terrain: TerrainMap | None = None,
    occupied_hexes: dict[HexCoord, bool] | None = None,
) -> InvoluntaryMovementPath:
    """Validate a straight-line path for involuntary movement.

    Per PR2 ~3847-3849:
    "Involuntary movement... forces them to move in a direct line in a
    direction specified by the triggering action or attack."

    Args:
        start: Starting coordinate
        direction: Direction to move (vector)
        spaces: Number of spaces to move
        terrain: Terrain map for obstructions
        occupied_hexes: Map of occupied hexes

    Returns:
        InvoluntaryMovementPath with validation results
    """
    if spaces <= 0:
        return InvoluntaryMovementPath(
            start=start,
            spaces_moved=0,
            spaces_requested=spaces,
            path_clear=True,
            straight_line=True,
        )

    obstructions: list[HexCoord] = []
    current = start
    moved = 0

    for i in range(spaces):
        next_q = current.q + direction.q
        next_r = current.r + direction.r
        next_coord = HexCoord(q=next_q, r=next_r)

        if _is_coord_occupied(next_coord, terrain, occupied_hexes):
            obstructions.append(next_coord)
            return InvoluntaryMovementPath(
                start=start,
                end=current,
                spaces_moved=moved,
                spaces_requested=spaces,
                path_clear=False,
                obstructions=obstructions,
                straight_line=True,
            )

        current = next_coord
        moved += 1

    return InvoluntaryMovementPath(
        start=start,
        end=current,
        spaces_moved=moved,
        spaces_requested=spaces,
        path_clear=True,
        obstructions=obstructions,
        straight_line=True,
    )


def validate_pull_path(
    source: HexCoord,
    target: HexCoord,
    spaces: int,
    terrain: TerrainMap | None = None,
    occupied_hexes: dict[HexCoord, bool] | None = None,
) -> InvoluntaryMovementPath:
    """Validate path for pulling a target toward source.

    Args:
        source: Position of the pulling character
        target: Current position of target
        spaces: Number of spaces to pull
        terrain: Terrain map for obstructions
        occupied_hexes: Map of occupied hexes

    Returns:
        InvoluntaryMovementPath with validation results
    """
    dq = source.q - target.q
    dr = source.r - target.r

    if dq == 0 and dr == 0:
        direction = HexCoord(q=0, r=0)
    elif abs(dq) >= abs(dr):
        sign = 1 if dq > 0 else -1
        direction = HexCoord(q=sign, r=0)
    else:
        sign = 1 if dr > 0 else -1
        direction = HexCoord(q=0, r=sign)

    return validate_straight_line_path(
        target, direction, spaces, terrain, occupied_hexes
    )


def resolve_push(
    start: HexCoord,
    direction: HexCoord,
    spaces: int,
    terrain: TerrainMap | None = None,
    occupied_hexes: dict[HexCoord, bool] | None = None,
) -> PushResult:
    """Resolve a push movement.

    Per PR2: Push forces movement in a direct line.

    Args:
        start: Starting coordinate
        direction: Direction to push
        spaces: Number of spaces to push
        terrain: Terrain map for obstructions
        occupied_hexes: Map of occupied hexes

    Returns:
        PushResult with resolution details
    """
    path = validate_straight_line_path(
        start, direction, spaces, terrain, occupied_hexes
    )

    if not path.path_clear:
        return PushResult(
            spaces_pushed=path.spaces_moved,
            end_position=path.end or start,
            obstructed=True,
            obstruction_coord=path.obstructions[0] if path.obstructions else None,
            reason=f"Push blocked by obstruction after {path.spaces_moved} spaces",
        )

    return PushResult(
        spaces_pushed=spaces,
        end_position=path.end or start,
        obstructed=False,
        reason=f"Push successful: moved {spaces} spaces",
    )


def resolve_pull(
    source: HexCoord,
    target: HexCoord,
    spaces: int,
    terrain: TerrainMap | None = None,
    occupied_hexes: dict[HexCoord, bool] | None = None,
) -> PullResult:
    """Resolve a pull movement toward source.

    Per PR2: Pull forces movement toward the source.

    Args:
        source: Position of the pulling character
        target: Starting position of target
        spaces: Number of spaces to pull
        terrain: Terrain map for obstructions
        occupied_hexes: Map of occupied hexes

    Returns:
        PullResult with resolution details
    """
    path = validate_pull_path(source, target, spaces, terrain, occupied_hexes)

    if not path.path_clear:
        return PullResult(
            spaces_pulled=path.spaces_moved,
            end_position=path.end or target,
            obstructed=True,
            obstruction_coord=path.obstructions[0] if path.obstructions else None,
            reason=f"Pull blocked by obstruction after {path.spaces_moved} spaces",
        )

    return PullResult(
        spaces_pulled=spaces,
        end_position=path.end or target,
        obstructed=False,
        reason=f"Pull successful: moved {spaces} spaces toward source",
    )


def resolve_knockback(
    source: HexCoord,
    target: HexCoord,
    spaces: int,
    terrain: TerrainMap | None = None,
    occupied_hexes: dict[HexCoord, bool] | None = None,
) -> KnockbackResult:
    """Resolve knockback movement directly away from source.

    Per PR2 ~5027-5028:
    "Knockback X - On hit, you may knock back a target X spaces in a straight line
    directly away from the point of origin (unless specified, this is your mech)."

    Args:
        source: Position of the knockback source
        target: Starting position of target
        spaces: Number of spaces to knock back
        terrain: Terrain map for obstructions
        occupied_hexes: Map of occupied hexes

    Returns:
        KnockbackResult with resolution details
    """
    dq = target.q - source.q
    dr = target.r - source.r

    if dq == 0 and dr == 0:
        direction = HexCoord(q=0, r=0)
    elif abs(dq) >= abs(dr):
        sign = 1 if dq > 0 else -1
        direction = HexCoord(q=sign, r=0)
    else:
        sign = 1 if dr > 0 else -1
        direction = HexCoord(q=0, r=sign)

    path = validate_straight_line_path(
        target, direction, spaces, terrain, occupied_hexes
    )

    if not path.path_clear:
        return KnockbackResult(
            spaces_knocked=path.spaces_moved,
            end_position=path.end or target,
            obstructed=True,
            obstruction_coord=path.obstructions[0] if path.obstructions else None,
            direction=direction,
            reason=f"Knockback blocked by obstruction after {path.spaces_moved} spaces",
        )

    return KnockbackResult(
        spaces_knocked=spaces,
        end_position=path.end or target,
        obstructed=False,
        direction=direction,
        reason=f"Knockback successful: moved {spaces} spaces",
    )


def resolve_shove(
    start: HexCoord,
    direction: HexCoord,
    spaces: int,
    terrain: TerrainMap | None = None,
    occupied_hexes: dict[HexCoord, bool] | None = None,
) -> ShoveResult:
    """Resolve a shove movement.

    Shove is similar to push but typically has different conditions/effects.

    Args:
        start: Starting coordinate
        direction: Direction to shove
        spaces: Number of spaces to shove
        terrain: Terrain map for obstructions
        occupied_hexes: Map of occupied hexes

    Returns:
        ShoveResult with resolution details
    """
    path = validate_straight_line_path(
        start, direction, spaces, terrain, occupied_hexes
    )

    if not path.path_clear:
        return ShoveResult(
            spaces_shoved=path.spaces_moved,
            end_position=path.end or start,
            obstructed=True,
            obstruction_coord=path.obstructions[0] if path.obstructions else None,
            reason=f"Shove blocked by obstruction after {path.spaces_moved} spaces",
        )

    return ShoveResult(
        spaces_shoved=spaces,
        end_position=path.end or start,
        obstructed=False,
        reason=f"Shove successful: moved {spaces} spaces",
    )


def can_drag(
    dragger_size: SizeClass,
    target_size: SizeClass,
    is_pilot: bool = False,
) -> tuple[bool, str, int]:
    """Check if a character can drag another.

    Per PR2 ~3862-3863:
    "A mech can drag another character or object up to 2x its size, but is slowed
    while doing so... Pilots... cannot drag or lift objects larger than size 1/2."

    Args:
        dragger_size: Size of the dragging character
        target_size: Size of the character/object to drag
        is_pilot: Whether the dragger is a pilot (not a mech)

    Returns:
        Tuple of (can_drag, reason, max_drag_size)
    """
    if is_pilot:
        if target_size != "size_half":
            return False, "Pilots cannot drag objects larger than size 1/2", 0
        return (
            True,
            "Pilot can drag size 1/2 or smaller",
            1,
        )

    max_drag_size = SIZE_ORDER.get(dragger_size, 0) * 2
    target_size_val = SIZE_ORDER.get(target_size, 0)

    if target_size_val > max_drag_size:
        return (
            False,
            f"Cannot drag size {target_size} (max drag size: {max_drag_size})",
            max_drag_size,
        )

    return True, f"Can drag size {target_size} (max: {max_drag_size})", max_drag_size


def can_lift(
    lifter_size: SizeClass,
    target_size: SizeClass,
    is_pilot: bool = False,
) -> tuple[bool, str]:
    """Check if a character can lift another.

    Per PR2 ~3863-3864:
    "lift a character or object overhead that's its size or smaller, but is immobilized
    while doing so... Pilots... cannot drag or lift objects larger than size 1/2."

    Args:
        lifter_size: Size of the lifting character
        target_size: Size of the character/object to lift
        is_pilot: Whether the lifter is a pilot (not a mech)

    Returns:
        Tuple of (can_lift, reason)
    """
    if is_pilot:
        if target_size != "size_half":
            return False, "Pilots cannot lift objects larger than size 1/2"
        return True, "Pilot can lift size 1/2 or smaller"

    lifter_val = SIZE_ORDER.get(lifter_size, 0)
    target_val = SIZE_ORDER.get(target_size, 0)

    if target_val > lifter_val:
        return False, f"Cannot lift size {target_size} (lifter is size {lifter_size})"

    return True, f"Can lift size {target_size}"


def resolve_drag(
    dragger_size: SizeClass,
    dragged_size: SizeClass,
    target_coord: HexCoord,
    terrain: TerrainMap | None = None,
    occupied_hexes: dict[HexCoord, bool] | None = None,
    is_pilot: bool = False,
) -> DragResult:
    """Resolve dragging another character.

    Per PR2 ~3862:
    "A mech can drag another character or object up to 2x its size, but is slowed
    while doing so, and... While dragging or lifting... a mech cannot take reactions."

    Args:
        dragger_size: Size of the dragging character
        dragged_size: Size of the character being dragged
        target_coord: Target position for the drag
        terrain: Terrain map for obstructions
        occupied_hexes: Map of occupied hexes
        is_pilot: Whether the dragger is a pilot

    Returns:
        DragResult with resolution details
    """
    can, reason, max_drag = can_drag(dragger_size, dragged_size, is_pilot)

    if not can:
        return DragResult(
            dragger_size=dragger_size,
            dragged_size=dragged_size,
            can_drag=False,
            max_drag_size=max_drag,
            reason=reason,
        )

    return DragResult(
        dragger_size=dragger_size,
        dragged_size=dragged_size,
        can_drag=True,
        max_drag_size=max_drag,
        slowed_applied=True,
        end_position=target_coord,
        reason=reason,
    )


def resolve_lift(
    lifter_size: SizeClass,
    lifted_size: SizeClass,
    is_pilot: bool = False,
) -> LiftResult:
    """Resolve lifting another character overhead.

    Per PR2 ~3863-3864:
    "lift a character or object overhead that's its size or smaller, but is immobilized
    while doing so... While dragging or lifting... a mech cannot take reactions."

    Args:
        lifter_size: Size of the lifting character
        lifted_size: Size of the character being lifted
        is_pilot: Whether the lifter is a pilot

    Returns:
        LiftResult with resolution details
    """
    can, reason = can_lift(lifter_size, lifted_size, is_pilot)

    if not can:
        return LiftResult(
            lifter_size=lifter_size,
            lifted_size=lifted_size,
            can_lift=False,
            reason=reason,
        )

    return LiftResult(
        lifter_size=lifter_size,
        lifted_size=lifted_size,
        can_lift=True,
        immobilized_applied=True,
        lifted_overhead=True,
        reason=reason,
    )


def apply_drag_penalty(
    conditions: list[StatusType],
) -> ConditionApplicationResult:
    """Apply slowed condition for dragging.

    Args:
        conditions: Current conditions list to modify

    Returns:
        ConditionApplicationResult with application details
    """
    return apply_condition(conditions, "slowed")


def apply_lift_penalty(
    conditions: list[StatusType],
) -> ConditionApplicationResult:
    """Apply immobilized condition for lifting.

    Args:
        conditions: Current conditions list to modify

    Returns:
        ConditionApplicationResult with application details
    """
    return apply_condition(conditions, "immobilized")


def get_involuntary_movement_result(
    movement_type: str,
    start: HexCoord,
    end: HexCoord | None,
    spaces_moved: int,
    path_clear: bool,
    obstructed: bool,
    obstruction_coord: HexCoord | None = None,
) -> InvoluntaryMovementResult:
    """Create a standardized InvoluntaryMovementResult.

    Args:
        movement_type: Type of movement (push, pull, knockback, etc.)
        start: Starting coordinate
        end: Ending coordinate (None if movement failed)
        spaces_moved: Number of spaces moved
        path_clear: Whether path was clear
        obstructed: Whether movement was blocked
        obstruction_coord: Coordinate of obstruction if any

    Returns:
        InvoluntaryMovementResult with all details
    """
    return InvoluntaryMovementResult(
        movement_type=movement_type,
        start=start,
        end=end,
        spaces_moved=spaces_moved,
        path_clear=path_clear,
        obstructed=obstructed,
        obstruction_coord=obstruction_coord,
        provoked_reactions=False,
        ignored_engagement=True,
        reason=f"{movement_type} {'blocked' if obstructed else 'completed'} after {spaces_moved} spaces",
    )


def is_involuntary_movement(
    movement_type: str,
) -> bool:
    """Check if a movement type is considered involuntary.

    Args:
        movement_type: Type of movement to check

    Returns:
        True if movement is involuntary
    """
    return movement_type in ["push", "pull", "shove", "knockback", "drag", "lift"]


def breaks_grapple(movement_type: str) -> bool:
    """Check if a movement type breaks grapple.

    Per PR2 ~4166:
    "The grapple breaks if either target breaks adjacency (is knocked back for example)"

    Args:
        movement_type: Type of movement

    Returns:
        True if this movement breaks grapple
    """
    return movement_type in ["knockback", "shove", "push"]
