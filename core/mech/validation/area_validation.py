"""Area attack validation helpers for mech combat.

Covers area pattern geometry, origin resolution, and target identification."""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.mech.grid import (
    hex_cone,
    hex_cone_centered,
    hex_line_from_direction,
    hexes_in_radius,
    normalize_hex_direction,
)

if TYPE_CHECKING:
    from typing import Any

    from core.mech.combat_state import ActionUse, CombatantState, HexPosition


__all__ = [
    "resolve_area_origin",
    "get_action_targets",
    "validate_area_geometry",
    "calculate_area_coords",
]


def _resolve_area_origin(
    action: ActionUse,
    actor_position: HexPosition | None,
    target_position: HexPosition | None,
) -> HexPosition | None:
    if not action.area_pattern:
        return None
    pattern = action.area_pattern.pattern
    if pattern == "burst":
        return actor_position
    if action.area_origin:
        return action.area_origin
    if pattern == "blast":
        return action.target_position or target_position
    if pattern in ("line", "cone"):
        return actor_position
    return None


def _action_targets(
    action: ActionUse,
    combatants_by_id: dict[str, CombatantState],
) -> list[tuple[CombatantState | None, HexPosition | None]]:
    targets: list[tuple[CombatantState | None, HexPosition | None]] = []
    for target_id in action.target_ids:
        target = combatants_by_id.get(target_id)
        targets.append((target, target.position if target else None))
    targets.extend((None, pos) for pos in action.target_positions)
    if action.target_id or action.target_position:
        target = combatants_by_id.get(action.target_id) if action.target_id else None
        targets.append(
            (target, action.target_position or (target.position if target else None))
        )
    return targets


def _validate_area_geometry(
    action: ActionUse,
    origin: HexPosition | None,
    area_coords: set[tuple[int, int]] | None,
    issues: list[Any],
) -> None:
    if not action.area_pattern:
        return
    pattern = action.area_pattern.pattern
    size = action.area_pattern.size

    if pattern in ("line", "cone") and not action.area_direction:
        issues.append(
            {
                "code": "area_direction_missing",
                "message": f"Action {action.action_id} uses {pattern} but has no direction specified.",
                "severity": "warning",
            }
        )

    if action.area_affected:
        if not origin:
            issues.append(
                {
                    "code": "area_origin_missing",
                    "message": f"Action {action.action_id} uses area pattern but has no origin.",
                    "severity": "warning",
                }
            )
            return
        if area_coords is None:
            return
        for coord in action.area_affected:
            if (coord.q, coord.r) not in area_coords:
                issues.append(
                    {
                        "code": "area_affected_not_in_shape",
                        "message": (
                            f"Action {action.action_id} includes hex {coord.q},{coord.r} "
                            f"outside {pattern} size {size}."
                        ),
                        "severity": "warning",
                    }
                )


def _area_coords_for_action(
    action: ActionUse,
    origin: HexPosition,
    issues: list[Any],
) -> set[tuple[int, int]] | None:
    if not action.area_pattern:
        return None
    pattern = action.area_pattern.pattern
    size = action.area_pattern.size

    if pattern in ("line", "cone"):
        if not action.area_direction:
            return None
        step = normalize_hex_direction(action.area_direction)
        if not step:
            issues.append(
                {
                    "code": "area_direction_invalid",
                    "message": (
                        f"Action {action.action_id} uses {pattern} with a non-axial direction."
                    ),
                    "severity": "warning",
                }
            )
            return None
        if pattern == "line":
            coords = hex_line_from_direction(origin.coord, step, size)
        else:
            if action.area_pattern.cone_mode == "axis":
                coords = hex_cone_centered(origin.coord, step, size)
            else:
                coords = hex_cone(origin.coord, step, size)
        return {(coord.q, coord.r) for coord in coords}

    if pattern in ("blast", "burst"):
        coords = hexes_in_radius(origin.coord, size)
        return {(coord.q, coord.r) for coord in coords}

    return None


def resolve_area_origin(
    action: ActionUse,
    actor_position: HexPosition | None,
    target_position: HexPosition | None,
) -> HexPosition | None:
    """Public wrapper for area origin resolution.

    Args:
        action: The action with area pattern
        actor_position: Current position of the acting combatant
        target_position: Target position if applicable

    Returns:
        The origin hex for the area pattern, or None if no area
    """
    return _resolve_area_origin(action, actor_position, target_position)


def get_action_targets(
    action: ActionUse,
    combatants_by_id: dict[str, CombatantState],
) -> list[tuple[CombatantState | None, HexPosition | None]]:
    """Get all targets for an action.

    Args:
        action: The action being taken
        combatants_by_id: Map of combatant IDs to states

    Returns:
        List of (combatant, position) tuples for all targets
    """
    return _action_targets(action, combatants_by_id)


def validate_area_geometry(
    action: ActionUse,
    origin: HexPosition | None,
    area_coords: set[tuple[int, int]] | None,
) -> list[dict]:
    """Validate that area affected hexes match the pattern.

    Args:
        action: The action with area pattern
        origin: Origin position for the area
        area_coords: Calculated hexes for the area pattern

    Returns:
        List of validation issue dicts (empty if valid)
    """
    issues: list[dict] = []
    _validate_area_geometry(action, origin, area_coords, issues)
    return issues


def calculate_area_coords(
    action: ActionUse,
    origin: HexPosition,
) -> set[tuple[int, int]] | None:
    """Calculate hex coordinates for an area pattern.

    Args:
        action: The action with area pattern
        origin: Origin position for the area

    Returns:
        Set of (q, r) coordinates for affected hexes, or None if invalid
    """
    issues: list[dict] = []
    return _area_coords_for_action(action, origin, issues)
