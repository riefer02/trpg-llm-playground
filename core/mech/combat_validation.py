"""Validation helpers for mech combat scenarios."""

from __future__ import annotations

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel

from core.mech.combat_rules import DEFAULT_MECH_COMBAT_RULES, LineOfSightRules
from core.mech.combat_state import MechCombatScenario, CombatTurn, CombatantState, ActionUse
from core.mech.combat_actions import ACTION_RULES_BY_ID, ActionRule
from core.mech.grid import (
    HexCoord,
    hexes_between,
    hex_cone,
    hex_cone_centered,
    hex_line_from_direction,
    hexes_in_radius,
    normalize_hex_direction,
)
from core.mech.grid import HexPosition
from core.mech.terrain import terrain_index, TerrainHex
from core.mech.statuses import STATUS_DEFINITIONS_BY_ID
from core.shared.enums import CoverType
from core.shared.effects import (
    AttackContextCondition,
    CheckContextCondition,
    ConditionGroup,
    EffectCondition,
    SizeCondition,
    SpatialCondition,
    PerTargetCounter,
)


class CombatValidationIssue(FrozenModel):
    """A combat validation issue."""

    code: str
    message: str
    severity: Literal["error", "warning"] = "error"



class CombatValidation(FrozenModel):
    """Validation result for a combat scenario."""

    valid: bool
    issues: list[CombatValidationIssue] = Field(default_factory=list)


STRICT_AOE_WARNING_CODES = {
    "area_origin_missing",
    "area_origin_line_of_sight_blocked",
    "area_origin_path_blocked",
    "area_line_of_sight_blocked",
    "area_path_blocked",
}


def _promote_strict_warnings(
    issues: list[CombatValidationIssue],
) -> list[CombatValidationIssue]:
    promoted: list[CombatValidationIssue] = []
    for issue in issues:
        if issue.severity == "warning" and issue.code in STRICT_AOE_WARNING_CODES:
            promoted.append(
                CombatValidationIssue(
                    code=issue.code,
                    message=issue.message,
                    severity="error",
                )
            )
        else:
            promoted.append(issue)
    return promoted


def _condition_matches_statuses(
    condition: EffectCondition | None,
    statuses: set[str],
) -> bool:
    if condition is None:
        return True
    if isinstance(condition, str):
        return condition in statuses
    if isinstance(condition, ConditionGroup):
        if condition.all_of and not all(
            _condition_matches_statuses(item, statuses) for item in condition.all_of
        ):
            return False
        if condition.any_of and not any(
            _condition_matches_statuses(item, statuses) for item in condition.any_of
        ):
            return False
        if condition.none_of and any(
            _condition_matches_statuses(item, statuses) for item in condition.none_of
        ):
            return False
        return True
    if isinstance(
        condition,
        (SpatialCondition, AttackContextCondition, CheckContextCondition, SizeCondition),
    ):
        return False
    return False



def _adjacent_hard_cover_coords(
    tiles: dict[tuple[int, int], TerrainHex],
    target_coord: HexCoord,
    target_size: str | None,
) -> set[tuple[int, int]]:
    adjacent: set[tuple[int, int]] = set()
    for neighbor in target_coord.neighbors():
        tile = tiles.get((neighbor.q, neighbor.r))
        if tile and tile.provides_hard_cover and _cover_size_allows_target(tile.hard_cover_size, target_size):
            adjacent.add((neighbor.q, neighbor.r))
    return adjacent


def _cover_size_allows_target(cover_size: str | None, target_size: str | None) -> bool:
    if cover_size is None or target_size is None:
        return True
    return _size_value(cover_size) >= _size_value(target_size)


def _is_flanking(
    tiles: dict[tuple[int, int], TerrainHex],
    attacker_coord: HexCoord,
    target_coord: HexCoord,
) -> bool:
    line = hexes_between(attacker_coord, target_coord, include_endpoints=False)
    if not line:
        return True
    cover_coord = line[-1]
    tile = tiles.get((cover_coord.q, cover_coord.r))
    return not (tile and tile.provides_hard_cover)


def _cover_between(
    tiles: dict[tuple[int, int], TerrainHex],
    start_coord: tuple[int, int],
    end_coord: tuple[int, int],
    target_size: str | None,
) -> CoverType:
    cover_rules = DEFAULT_MECH_COMBAT_RULES.cover_rules
    start = HexCoord(q=start_coord[0], r=start_coord[1])
    end = HexCoord(q=end_coord[0], r=end_coord[1])
    line = hexes_between(start, end, include_endpoints=False)
    hard_between = any(
        tiles.get((coord.q, coord.r)) and tiles[(coord.q, coord.r)].provides_hard_cover
        for coord in line
    )
    hard_between_size_ok = any(
        tiles.get((coord.q, coord.r))
        and tiles[(coord.q, coord.r)].provides_hard_cover
        and _cover_size_allows_target(tiles[(coord.q, coord.r)].hard_cover_size, target_size)
        for coord in line
    )
    soft_between = any(
        tiles.get((coord.q, coord.r)) and tiles[(coord.q, coord.r)].provides_soft_cover
        for coord in line
    )

    if not cover_rules.hard_cover_requires_adjacency and hard_between:
        if cover_rules.hard_cover_requires_size_match and not hard_between_size_ok:
            return "soft" if hard_between or soft_between else "none"
        return "hard"

    size_check = target_size if cover_rules.hard_cover_requires_size_match else None
    hard_adjacent = _adjacent_hard_cover_coords(tiles, end, size_check)
    if hard_adjacent:
        hard_cover = True
        if cover_rules.hard_cover_flanking_negates and _is_flanking(tiles, start, end):
            hard_cover = False
        if cover_rules.hard_cover_requires_adjacency and hard_cover:
            return "hard"
        if not cover_rules.hard_cover_requires_adjacency and hard_between:
            return "hard"

    if soft_between or hard_between:
        return "soft"

    return "none"


def _line_of_sight_clear(
    tiles: dict[tuple[int, int], TerrainHex],
    start_coord: tuple[int, int],
    end_coord: tuple[int, int],
    start_elevation: int,
    end_elevation: int,
    line_of_sight_rules: LineOfSightRules,
    target_coord: HexCoord | None = None,
) -> bool:
    for coord in hexes_between(
        HexCoord(q=start_coord[0], r=start_coord[1]),
        HexCoord(q=end_coord[0], r=end_coord[1]),
        include_endpoints=False,
    ):
        tile = tiles.get((coord.q, coord.r))
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
    tiles: dict[tuple[int, int], TerrainHex],
    start_coord: tuple[int, int],
    end_coord: tuple[int, int],
    line_of_sight_rules: LineOfSightRules,
    target_coord: HexCoord | None = None,
) -> bool:
    for coord in hexes_between(
        HexCoord(q=start_coord[0], r=start_coord[1]),
        HexCoord(q=end_coord[0], r=end_coord[1]),
        include_endpoints=False,
    ):
        tile = tiles.get((coord.q, coord.r))
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
    tiles: dict[tuple[int, int], TerrainHex],
    coord: HexCoord,
) -> int:
    tile = tiles.get((coord.q, coord.r))
    return tile.elevation if tile else 0


def _blocked_area_coords_by_los(
    tiles: dict[tuple[int, int], TerrainHex],
    origin: HexPosition,
    coords: list[HexCoord],
    line_of_sight_rules: LineOfSightRules,
) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    start_coord = (origin.coord.q, origin.coord.r)
    for coord in coords:
        end_coord = (coord.q, coord.r)
        end_elevation = _terrain_elevation(tiles, coord)
        if not _line_of_sight_clear(
            tiles,
            start_coord,
            end_coord,
            origin.elevation,
            end_elevation,
            line_of_sight_rules,
            coord,
        ):
            blocked.add(end_coord)
    return blocked


def _blocked_area_coords_by_path(
    tiles: dict[tuple[int, int], TerrainHex],
    origin: HexPosition,
    coords: list[HexCoord],
    line_of_sight_rules: LineOfSightRules,
) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    start_coord = (origin.coord.q, origin.coord.r)
    for coord in coords:
        end_coord = (coord.q, coord.r)
        if not _path_clear(
            tiles,
            start_coord,
            end_coord,
            line_of_sight_rules,
            coord,
        ):
            blocked.add(end_coord)
    return blocked


def _adjacency_distance(attacker: CombatantState, target: CombatantState) -> int:
    size_to_radius = {
        "size_half": 1,
        "size_1": 1,
        "size_2": 2,
        "size_3": 3,
        "size_4": 4,
        "size_5": 5,
    }
    return max(size_to_radius[attacker.stats.size], size_to_radius[target.stats.size])


def _size_value(size: str) -> float:
    size_values = {
        "size_half": 0.5,
        "size_1": 1.0,
        "size_2": 2.0,
        "size_3": 3.0,
        "size_4": 4.0,
        "size_5": 5.0,
    }
    return size_values.get(size, 1.0)


def _hostiles_for(actor: CombatantState, combatants: dict[str, CombatantState]) -> list[CombatantState]:
    if actor.side == "players":
        return [c for c in combatants.values() if c.side == "hostiles"]
    if actor.side == "hostiles":
        return [c for c in combatants.values() if c.side == "players"]
    return []


def _surface_elevation(
    terrain_tiles: dict[tuple[int, int], TerrainHex],
    position: HexPosition,
) -> int:
    tile = terrain_tiles.get((position.coord.q, position.coord.r))
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


def _effective_ignores_los(action: ActionUse, line_of_sight_rules: LineOfSightRules) -> bool:
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
        targets.append((target, action.target_position or (target.position if target else None)))
    return targets


def _index_per_target_counters(
    combatant: CombatantState,
) -> tuple[dict[tuple[str, str], PerTargetCounter], dict[str, PerTargetCounter]]:
    counters_by_target: dict[tuple[str, str], PerTargetCounter] = {}
    templates_by_effect: dict[str, PerTargetCounter] = {}
    for counter in combatant.per_target_counters.values():
        if counter.target_id:
            counters_by_target[(counter.effect_id, counter.target_id)] = counter
        else:
            templates_by_effect[counter.effect_id] = counter
    return counters_by_target, templates_by_effect


def _apply_per_target_effects(
    action: ActionUse,
    actor_id: str,
    combatants_by_id: dict[str, CombatantState],
    counters_by_target: dict[tuple[str, str], PerTargetCounter],
    templates_by_effect: dict[str, PerTargetCounter],
    issues: list[CombatValidationIssue],
) -> None:
    for applied in action.applied_per_target_effects:
        if applied.target_id not in combatants_by_id:
            issues.append(
                CombatValidationIssue(
                    code="per_target_unknown_target",
                    message=(
                        f"Action {action.action_id} applies {applied.effect_id} "
                        f"to unknown target {applied.target_id}."
                    ),
                )
            )
            continue

        key = (applied.effect_id, applied.target_id)
        counter = counters_by_target.get(key)
        if counter is None:
            template = templates_by_effect.get(applied.effect_id)
            if template:
                counter = template.model_copy(
                    update={"target_id": applied.target_id, "current_count": template.current_count}
                )
            else:
                if applied.max_count is None:
                    issues.append(
                        CombatValidationIssue(
                            code="per_target_limit_unknown",
                            message=(
                                f"Action {action.action_id} applies {applied.effect_id} "
                                f"to {applied.target_id} without a max_count."
                            ),
                            severity="warning",
                        )
                    )
                    continue
                counter = PerTargetCounter(
                    effect_id=applied.effect_id,
                    max_count=applied.max_count,
                    reset_on=applied.reset_on or "scene_end",
                    target_id=applied.target_id,
                )

        new_count = counter.current_count + applied.count
        if new_count > counter.max_count:
            issues.append(
                CombatValidationIssue(
                    code="per_target_limit_exceeded",
                    message=(
                        f"{actor_id} applies {applied.effect_id} to {applied.target_id} "
                        f"{new_count} times (max {counter.max_count})."
                    ),
                )
            )

        counters_by_target[key] = counter.model_copy(update={"current_count": new_count})


def _validate_area_geometry(
    action: ActionUse,
    origin: HexPosition | None,
    area_coords: set[tuple[int, int]] | None,
    issues: list[CombatValidationIssue],
) -> None:
    if not action.area_pattern:
        return
    pattern = action.area_pattern.pattern
    size = action.area_pattern.size

    if pattern in ("line", "cone") and not action.area_direction:
        issues.append(
            CombatValidationIssue(
                code="area_direction_missing",
                message=f"Action {action.action_id} uses {pattern} but has no direction specified.",
                severity="warning",
            )
        )

    if action.area_affected:
        if not origin:
            issues.append(
                CombatValidationIssue(
                    code="area_origin_missing",
                    message=f"Action {action.action_id} uses area pattern but has no origin.",
                    severity="warning",
                )
            )
            return
        if area_coords is None:
            return
        for coord in action.area_affected:
            if (coord.q, coord.r) not in area_coords:
                issues.append(
                    CombatValidationIssue(
                        code="area_affected_not_in_shape",
                        message=(
                            f"Action {action.action_id} includes hex {coord.q},{coord.r} "
                            f"outside {pattern} size {size}."
                        ),
                        severity="warning",
                    )
                )


def _area_coords_for_action(
    action: ActionUse,
    origin: HexPosition,
    issues: list[CombatValidationIssue],
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
                CombatValidationIssue(
                    code="area_direction_invalid",
                    message=(
                        f"Action {action.action_id} uses {pattern} with a non-axial direction."
                    ),
                    severity="warning",
                )
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


def _validate_turn(
    turn: CombatTurn,
    issues: list[CombatValidationIssue],
    combatants_by_id: dict[str, CombatantState],
    terrain_tiles: dict[tuple[int, int], TerrainHex],
    environment: str,
    per_target_state_by_actor: dict[
        str, tuple[dict[tuple[str, str], PerTargetCounter], dict[str, PerTargetCounter]]
    ],
) -> None:
    economy = DEFAULT_MECH_COMBAT_RULES.turn_actions.action_economy
    max_quick = economy.quick_actions_per_turn
    max_full = economy.full_actions_per_turn
    overcharge_used = any(action.action_id == "overcharge" for action in turn.actions)
    overcharge_used = overcharge_used or any(action.granted_by_overcharge for action in turn.actions)
    overcharge_count = sum(1 for action in turn.actions if action.action_id == "overcharge")

    if overcharge_used:
        max_quick += 1

    quick_count = 0
    full_count = 0
    reaction_count = 0
    non_free_counts: dict[str, int] = {}
    actor = combatants_by_id.get(turn.actor_id)
    per_target_state = per_target_state_by_actor.get(turn.actor_id)
    per_target_counters = per_target_state[0] if per_target_state else {}
    per_target_templates = per_target_state[1] if per_target_state else {}

    if not actor:
        issues.append(
            CombatValidationIssue(
                code="unknown_actor",
                message=f"Unknown actor id: {turn.actor_id}.",
            )
        )

    actor_statuses: list[str] = []
    mode_effects = actor.active_mode_effects if actor else []
    base_statuses = list(actor.statuses) + list(actor.conditions) if actor else []
    mode_statuses = [
        grant.status
        for mode in mode_effects
        for grant in mode.effects.status_grants
        if _condition_matches_statuses(grant.condition, set(base_statuses))
    ]
    if actor:
        actor_statuses = base_statuses + mode_statuses
    hostiles = _hostiles_for(actor, combatants_by_id) if actor else []
    engaged = False
    if actor:
        engaged = "engaged" in actor_statuses or _is_engaged(actor, hostiles)
    actor_restrictions = [
        STATUS_DEFINITIONS_BY_ID[status].effects.action_restrictions
        for status in actor_statuses
        if status in STATUS_DEFINITIONS_BY_ID
    ]
    actor_movement_restrictions = [
        STATUS_DEFINITIONS_BY_ID[status].effects.movement_restrictions
        for status in actor_statuses
        if status in STATUS_DEFINITIONS_BY_ID
    ]
    actor_status_set = set(actor_statuses)
    mode_action_restrictions = [
        restriction
        for mode in mode_effects
        for restriction in mode.effects.action_restrictions
        if restriction.target == "self"
        and _condition_matches_statuses(restriction.condition, actor_status_set)
    ]
    mode_tech_restrictions = [
        restriction
        for mode in mode_effects
        for restriction in mode.effects.tech_restrictions
        if restriction.target == "self"
        and _condition_matches_statuses(restriction.condition, actor_status_set)
    ]

    if actor_movement_restrictions:
        if turn.move_used and any(restriction.max_voluntary_speed == 0 for restriction in actor_movement_restrictions):
            issues.append(
                CombatValidationIssue(
                    code="movement_disallowed",
                    message=f"{turn.actor_id} cannot take regular movement due to status restrictions.",
                )
            )

    if actor_restrictions:
        quick_caps = [r.max_quick_actions for r in actor_restrictions if r.max_quick_actions is not None]
        if quick_caps:
            max_quick = min(max_quick, *quick_caps)
        if any(r.disallow_full_actions for r in actor_restrictions):
            max_full = 0
        if turn.move_used and any(r.disallow_move for r in actor_restrictions):
            issues.append(
                CombatValidationIssue(
                    code="movement_disallowed",
                    message=f"{turn.actor_id} cannot take regular movement due to action restrictions.",
                )
            )

    if actor and turn.movement_path:
        start_position = actor.position
        if start_position and turn.movement_path[0] != start_position:
            issues.append(
                CombatValidationIssue(
                    code="movement_start_mismatch",
                    message=f"{turn.actor_id} movement path does not start at current position.",
                    severity="warning",
                )
            )

        if turn.movement_mode != "teleport":
            for prev, curr in zip(turn.movement_path, turn.movement_path[1:]):
                if prev.distance_3d(curr) > 1:
                    issues.append(
                        CombatValidationIssue(
                            code="movement_step_invalid",
                            message=f"{turn.actor_id} movement path includes non-adjacent step.",
                            severity="warning",
                        )
                    )
                    break

        if turn.movement_mode == "ground":
            for step in turn.movement_path:
                surface = _surface_elevation(terrain_tiles, step)
                if step.elevation != surface:
                    issues.append(
                        CombatValidationIssue(
                            code="ground_movement_off_surface",
                            message=(
                                f"{turn.actor_id} moves through elevation {step.elevation} "
                                f"but surface is {surface}."
                            ),
                            severity="warning",
                        )
                    )
                    break

        if turn.movement_mode == "teleport":
            if len(turn.movement_path) < 2:
                issues.append(
                    CombatValidationIssue(
                        code="teleport_path_missing",
                        message=f"{turn.actor_id} teleports without a start/end path.",
                        severity="warning",
                    )
                )
            if actor_movement_restrictions and any(
                restriction.max_voluntary_speed == 0 for restriction in actor_movement_restrictions
            ):
                issues.append(
                    CombatValidationIssue(
                        code="teleport_immobilized",
                        message=f"{turn.actor_id} teleports while immobilized.",
                    )
                )
            if turn.movement_path:
                start = turn.movement_path[0]
                end = turn.movement_path[-1]
                start_surface = _surface_elevation(terrain_tiles, start)
                end_surface = _surface_elevation(terrain_tiles, end)
                if start.elevation != start_surface or end.elevation != end_surface:
                    issues.append(
                        CombatValidationIssue(
                            code="teleport_requires_surface",
                            message=f"{turn.actor_id} teleports to or from mid-air.",
                        )
                    )

        if turn.movement_mode in ("flight", "hover"):
            flight_rules = DEFAULT_MECH_COMBAT_RULES.flight
            is_hover = turn.movement_mode == "hover"
            if (
                environment != "zero_g"
                and flight_rules.must_move_min_spaces > 0
                and not (is_hover and flight_rules.hover_allows_stationary)
                and len(turn.movement_path) < 2
            ):
                issues.append(
                    CombatValidationIssue(
                        code="flight_requires_movement",
                        message=f"{turn.actor_id} is flying but does not move this turn.",
                        severity="warning",
                    )
                )
            if flight_rules.cannot_be_prone and actor and "prone" in actor_statuses:
                issues.append(
                    CombatValidationIssue(
                        code="flight_prone_invalid",
                        message=f"{turn.actor_id} is prone while flying.",
                        severity="warning",
                    )
                )
            movement_actions = 0
            for action in turn.actions:
                rule = ACTION_RULES_BY_ID.get(action.action_id)
                if rule and rule.movement and not action.used_as_reaction:
                    movement_actions += 1
            allowed_segments = movement_actions + (1 if turn.move_used else 0)
            segments = _movement_segments(turn.movement_path)
            if allowed_segments == 0 and segments > 0:
                issues.append(
                    CombatValidationIssue(
                        code="movement_without_action",
                        message=f"{turn.actor_id} moves without a movement allowance.",
                        severity="warning",
                    )
                )
            elif (
                flight_rules.straight_line_per_movement
                and not (is_hover and flight_rules.hover_ignores_straight_line)
                and segments > allowed_segments
            ):
                issues.append(
                    CombatValidationIssue(
                        code="flight_path_not_straight",
                        message=(
                            f"{turn.actor_id} flight path changes direction "
                            f"{segments - 1} times without enough movement segments."
                        ),
                        severity="warning",
                    )
                )
            if environment != "zero_g" and turn.movement_path:
                end = turn.movement_path[-1]
                surface = _surface_elevation(terrain_tiles, end)
                limit = flight_rules.combat_altitude_limit
                if end.elevation > surface + limit:
                    has_non_movement_action = any(
                        ACTION_RULES_BY_ID.get(action.action_id)
                        and not ACTION_RULES_BY_ID[action.action_id].movement
                        for action in turn.actions
                    )
                    if has_non_movement_action:
                        issues.append(
                            CombatValidationIssue(
                                code="flight_above_altitude_limit",
                                message=(
                                    f"{turn.actor_id} is above altitude limit {limit} "
                                    "and takes non-movement actions."
                                ),
                                severity="warning",
                            )
                        )

    if actor and turn.movement_path:
        for index, step in enumerate(turn.movement_path):
            for hostile in hostiles:
                if not hostile.position:
                    continue
                distance = step.distance_3d(hostile.position)
                if distance > _adjacency_distance(actor, hostile):
                    continue
                if _size_value(hostile.stats.size) >= _size_value(actor.stats.size):
                    if DEFAULT_MECH_COMBAT_RULES.engagement.stop_on_engage_same_size_or_larger:
                        if index < len(turn.movement_path) - 1:
                            issues.append(
                                CombatValidationIssue(
                                    code="movement_should_stop_on_engage",
                                    message=(
                                        f"{turn.actor_id} should stop moving after engaging "
                                        f"{hostile.id} of equal or larger size."
                                    ),
                                    severity="warning",
                                )
                            )
                            break

    for action in turn.actions:
        rule = ACTION_RULES_BY_ID.get(action.action_id)
        if not rule:
            issues.append(
                CombatValidationIssue(
                    code="unknown_action",
                    message=f"Unknown action id: {action.action_id}.",
                    severity="warning",
                )
            )
        elif rule.action_type != action.action_type and action.action_type not in rule.alternate_action_types:
            issues.append(
                CombatValidationIssue(
                    code="action_type_mismatch",
                    message=(
                        f"Action {action.action_id} uses {action.action_type} "
                        f"but rule expects {rule.action_type}."
                    ),
                    severity="warning",
                )
            )

        if rule:
            if actor and actor.ai_controlled and rule.scope == "pilot":
                issues.append(
                    CombatValidationIssue(
                        code="ai_pilot_action_disallowed",
                        message=(
                            f"{turn.actor_id} is AI-controlled and cannot take pilot action "
                            f"{action.action_id}."
                        ),
                    )
                )

        if actor_restrictions:
            disallow_actions = any(r.disallow_actions for r in actor_restrictions)
            disallow_full = any(r.disallow_full_actions for r in actor_restrictions)
            disallow_free = any(r.disallow_free_actions for r in actor_restrictions)
            disallow_reactions = any(r.disallow_reactions for r in actor_restrictions)
            disallow_move = any(r.disallow_move for r in actor_restrictions)
            disallow_overcharge = any(r.disallow_overcharge for r in actor_restrictions)
            disallow_boost = any(r.disallow_boost for r in actor_restrictions)
            disallow_tech = any(r.disallow_tech_actions for r in actor_restrictions)

            allowed_action_ids = {
                action_id
                for restriction in actor_restrictions
                for action_id in restriction.allowed_action_ids
            }
            allowed_attack_ids = {
                action_id
                for restriction in actor_restrictions
                for action_id in restriction.allowed_attack_action_ids
            }

            if disallow_actions:
                is_attack = rule.category == "attack"
                if action.action_id not in allowed_action_ids and not (
                    is_attack and action.action_id in allowed_attack_ids
                ):
                    issues.append(
                        CombatValidationIssue(
                            code="action_disallowed",
                            message=(
                                f"{turn.actor_id} cannot take action {action.action_id} "
                                "due to status restrictions."
                            ),
                        )
                    )

            if disallow_free and (action.used_as_free_action or action.action_type == "free"):
                issues.append(
                    CombatValidationIssue(
                        code="free_action_disallowed",
                        message=(
                            f"{turn.actor_id} cannot take free action {action.action_id} "
                            "due to status restrictions."
                        ),
                    )
                )

            if disallow_full and action.action_type == "full":
                issues.append(
                    CombatValidationIssue(
                        code="full_action_disallowed",
                        message=(
                            f"{turn.actor_id} cannot take full action {action.action_id} "
                            "due to status restrictions."
                        ),
                    )
                )

            if disallow_reactions and (action.used_as_reaction or action.action_type == "reaction"):
                issues.append(
                    CombatValidationIssue(
                        code="reaction_disallowed",
                        message=(
                            f"{turn.actor_id} cannot take reaction {action.action_id} "
                            "due to status restrictions."
                        ),
                    )
                )

            if disallow_move and rule.movement:
                issues.append(
                    CombatValidationIssue(
                        code="movement_action_disallowed",
                        message=(
                            f"{turn.actor_id} cannot take movement action {action.action_id} "
                            "due to status restrictions."
                        ),
                    )
                )

            if disallow_overcharge and (action.action_id == "overcharge" or action.granted_by_overcharge):
                issues.append(
                    CombatValidationIssue(
                        code="overcharge_disallowed",
                        message=(
                            f"{turn.actor_id} cannot overcharge due to status restrictions."
                        ),
                    )
                )

                if disallow_boost and rule.movement and rule.movement.counts_as_boost:
                    issues.append(
                        CombatValidationIssue(
                            code="boost_disallowed",
                            message=(
                                f"{turn.actor_id} cannot boost due to status restrictions."
                            ),
                        )
                    )

                if disallow_tech and rule.category == "tech":
                    issues.append(
                        CombatValidationIssue(
                            code="tech_action_disallowed",
                            message=(
                                f"{turn.actor_id} cannot take tech action {action.action_id} "
                                "due to status restrictions."
                            ),
                        )
                    )

        if mode_action_restrictions:
            for restriction in mode_action_restrictions:
                if restriction.action_ids and action.action_id in restriction.action_ids:
                    issues.append(
                        CombatValidationIssue(
                            code="mode_action_disallowed",
                            message=(
                                f"{turn.actor_id} cannot take action {action.action_id} "
                                "due to mode restrictions."
                            ),
                        )
                    )
                if rule and restriction.action_categories and rule.category in restriction.action_categories:
                    issues.append(
                        CombatValidationIssue(
                            code="mode_action_category_disallowed",
                            message=(
                                f"{turn.actor_id} cannot take {rule.category} actions "
                                "due to mode restrictions."
                            ),
                        )
                    )
                if rule and restriction.disallow_attack_rolls and (rule.attack or (rule.tech and rule.tech.is_attack)):
                    issues.append(
                        CombatValidationIssue(
                            code="mode_attack_roll_disallowed",
                            message=(
                                f"{turn.actor_id} cannot make attack rolls "
                                "due to mode restrictions."
                            ),
                        )
                    )
                if restriction.disallow_heat_generation:
                    heat_generated = action.heat_generated or 0
                    if action.action_id == "overcharge" or heat_generated > 0:
                        issues.append(
                            CombatValidationIssue(
                                code="mode_heat_generation_disallowed",
                                message=(
                                    f"{turn.actor_id} cannot take action {action.action_id} "
                                    "that generates heat while in current mode."
                                ),
                            )
                        )

        if mode_tech_restrictions and rule and rule.category == "tech":
            if any(restriction.disallow_tech_actions for restriction in mode_tech_restrictions):
                issues.append(
                    CombatValidationIssue(
                        code="mode_tech_action_disallowed",
                        message=(
                            f"{turn.actor_id} cannot take tech action {action.action_id} "
                            "due to mode restrictions."
                        ),
                    )
                )

        if actor_movement_restrictions and rule.movement:
            if any(restriction.max_voluntary_speed == 0 for restriction in actor_movement_restrictions):
                issues.append(
                    CombatValidationIssue(
                        code="movement_disallowed",
                        message=(
                            f"{turn.actor_id} cannot take movement action {action.action_id} "
                            "due to status restrictions."
                        ),
                    )
                )
            if rule.movement.counts_as_boost and any(
                restriction.only_regular_move for restriction in actor_movement_restrictions
            ):
                issues.append(
                    CombatValidationIssue(
                        code="boost_disallowed",
                        message=(
                            f"{turn.actor_id} cannot boost due to status restrictions."
                        ),
                    )
                )

        if rule.requires_target and not action.target_id and not action.target_position:
            issues.append(
                CombatValidationIssue(
                    code="missing_target",
                    message=f"Action {action.action_id} requires a target.",
                )
            )

        targets = _action_targets(action, combatants_by_id)
        range_targets_present = any(position for _, position in targets)
        if action.target_id and not combatants_by_id.get(action.target_id):
            issues.append(
                CombatValidationIssue(
                    code="unknown_target",
                    message=f"Action {action.action_id} targets unknown id {action.target_id}.",
                )
            )
        for target_id in action.target_ids:
            if target_id and not combatants_by_id.get(target_id):
                issues.append(
                    CombatValidationIssue(
                        code="unknown_target",
                        message=f"Action {action.action_id} targets unknown id {target_id}.",
                    )
                )

        if actor and action.applied_per_target_effects:
            _apply_per_target_effects(
                action,
                turn.actor_id,
                combatants_by_id,
                per_target_counters,
                per_target_templates,
                issues,
            )

        actor_position = actor.position if actor else None
        primary_target = combatants_by_id.get(action.target_id) if action.target_id else None
        target_position = action.target_position or (primary_target.position if primary_target else None)
        area_origin = _resolve_area_origin(action, actor_position, target_position)
        area_coords = None
        if action.area_pattern and area_origin:
            area_coords = _area_coords_for_action(action, area_origin, issues)
        _validate_area_geometry(action, area_origin, area_coords, issues)

        if action.consumes_lock_on:
            if not primary_target:
                issues.append(
                    CombatValidationIssue(
                        code="lock_on_consume_missing_target",
                        message=f"Action {action.action_id} consumes lock on but has no target.",
                    )
                )
            else:
                target_statuses = list(primary_target.statuses) + list(primary_target.conditions)
                if "lock_on" not in target_statuses:
                    issues.append(
                        CombatValidationIssue(
                            code="lock_on_not_present",
                            message=(
                                f"Action {action.action_id} consumes lock on but target "
                                f"{primary_target.id} is not locked on."
                            ),
                            severity="warning",
                        )
                    )
                if actor and primary_target and actor.side == primary_target.side:
                    issues.append(
                        CombatValidationIssue(
                            code="lock_on_consume_friendly",
                            message=(
                                f"Action {action.action_id} consumes lock on against a non-hostile target."
                            ),
                            severity="warning",
                        )
                    )
                if rule and not (rule.attack or (rule.tech and rule.tech.is_attack)):
                    issues.append(
                        CombatValidationIssue(
                            code="lock_on_consume_non_attack",
                            message=(
                                f"Action {action.action_id} consumes lock on but is not an attack."
                            ),
                            severity="warning",
                        )
                    )

        for target, position in targets:
            if target and turn.actor_id == target.id:
                if not DEFAULT_MECH_COMBAT_RULES.valid_target_rules.allow_self:
                    issues.append(
                        CombatValidationIssue(
                            code="self_target_not_allowed",
                            message=(
                                f"Action {action.action_id} targets {target.id} but self-targeting is disallowed."
                            ),
                        )
                    )

            if target:
                if target.kind == "object" and not DEFAULT_MECH_COMBAT_RULES.valid_target_rules.allow_objects:
                    issues.append(
                        CombatValidationIssue(
                            code="object_target_not_allowed",
                            message=(
                                f"Action {action.action_id} targets object {target.id} but objects are disallowed."
                            ),
                        )
                    )
                if target.kind != "object" and not DEFAULT_MECH_COMBAT_RULES.valid_target_rules.allow_characters:
                    issues.append(
                        CombatValidationIssue(
                            code="character_target_not_allowed",
                            message=(
                                f"Action {action.action_id} targets character {target.id} but characters are disallowed."
                            ),
                        )
                    )
                target_mode_effects = target.active_mode_effects
                target_mode_statuses = [
                    grant.status
                    for mode in target_mode_effects
                    for grant in mode.effects.status_grants
                ]
                target_statuses = list(target.statuses) + list(target.conditions) + target_mode_statuses
                target_mode_tech_restrictions = [
                    restriction
                    for mode in target_mode_effects
                    for restriction in mode.effects.tech_restrictions
                    if restriction.target == "self"
                ]
                target_tag_immunities = [
                    immunity
                    for mode in target_mode_effects
                    for immunity in mode.effects.tag_immunities
                ]
                for status in target_statuses:
                    status_def = STATUS_DEFINITIONS_BY_ID.get(status)
                    if not status_def:
                        continue
                    targeting = status_def.effects.targeting_restrictions
                    if targeting.cannot_be_targeted:
                        issues.append(
                            CombatValidationIssue(
                                code="targeting_restricted",
                                message=(
                                    f"Action {action.action_id} targets {target.name} "
                                    f"but targeting is restricted by {status}."
                                ),
                                severity="warning",
                            )
                        )
                    if targeting.miss_chance:
                        issues.append(
                            CombatValidationIssue(
                                code="targeting_miss_chance",
                                message=(
                                    f"Action {action.action_id} targets {target.name} "
                                    f"with {status} miss chance {targeting.miss_chance:.0%}."
                                ),
                                severity="warning",
                            )
                        )
                if rule and rule.category == "tech":
                    immune_to_tech = any(
                        STATUS_DEFINITIONS_BY_ID.get(status)
                        and STATUS_DEFINITIONS_BY_ID[status].effects.immune_to_tech
                        for status in target_statuses
                    )
                    immune_to_tech = immune_to_tech or any(
                        restriction.immune_to_tech for restriction in target_mode_tech_restrictions
                    )
                    if immune_to_tech:
                        issues.append(
                            CombatValidationIssue(
                                code="target_immune_to_tech",
                                message=(
                                    f"Action {action.action_id} targets {target.name} "
                                    "but the target is immune to tech effects."
                                ),
                                severity="warning",
                            )
                        )
                if action.weapon_tags and target_tag_immunities:
                    immune_tags = {tag for immunity in target_tag_immunities for tag in immunity.tags}
                    if immune_tags.intersection(action.weapon_tags):
                        issues.append(
                            CombatValidationIssue(
                                code="target_tag_immune",
                                message=(
                                    f"Action {action.action_id} targets {target.name} "
                                    "but the target is immune to the action's tags."
                                ),
                                severity="warning",
                            )
                        )

            if position and not DEFAULT_MECH_COMBAT_RULES.valid_target_rules.allow_points:
                if not target:
                    issues.append(
                        CombatValidationIssue(
                            code="point_target_not_allowed",
                            message=f"Action {action.action_id} targets a point but points are disallowed.",
                        )
                    )
        if rule.requires_adjacent_target:
            if not actor_position:
                issues.append(
                    CombatValidationIssue(
                        code="adjacency_unknown",
                        message=f"Action {action.action_id} requires adjacency but actor position is missing.",
                        severity="warning",
                    )
                )
            else:
                valid_adjacent = False
                for target, position in targets:
                    if not position or not target:
                        continue
                    distance = actor_position.distance_3d(position)
                    allowed = _adjacency_distance(actor, target)
                    if distance <= allowed:
                        valid_adjacent = True
                        break
                if not valid_adjacent and targets:
                    issues.append(
                        CombatValidationIssue(
                            code="target_not_adjacent",
                            message=f"Action {action.action_id} requires adjacency but no target is adjacent.",
                        )
                    )

        if rule.search and action.action_id == "search":
            if rule.search.requires_hidden_target and primary_target:
                target_statuses = list(primary_target.statuses) + list(primary_target.conditions)
                if "hidden" not in target_statuses:
                    issues.append(
                        CombatValidationIssue(
                            code="search_target_not_hidden",
                            message=(
                                f"Action {action.action_id} targets {primary_target.id} "
                                "but the target is not hidden."
                            ),
                            severity="warning",
                        )
                    )
            if action.contested_check is None:
                issues.append(
                    CombatValidationIssue(
                        code="search_contested_check_missing",
                        message=f"Action {action.action_id} requires a contested check.",
                    )
                )

        if rule.hide and action.action_id == "hide":
            if rule.hide.disallow_if_engaged and engaged:
                issues.append(
                    CombatValidationIssue(
                        code="hide_while_engaged",
                        message=f"{turn.actor_id} cannot hide while engaged.",
                    )
                )
            if not actor_position:
                issues.append(
                    CombatValidationIssue(
                        code="hide_position_unknown",
                        message=f"{turn.actor_id} hides but has no position set.",
                        severity="warning",
                    )
                )
            else:
                can_hide = False
                if rule.hide.allow_without_cover_if_invisible and "invisible" in actor_statuses:
                    can_hide = True
                else:
                    can_hide = True
                    for hostile in hostiles:
                        if not hostile.position:
                            continue
                        start_coord = (hostile.position.coord.q, hostile.position.coord.r)
                        end_coord = (actor_position.coord.q, actor_position.coord.r)
                        los_clear = _line_of_sight_clear(
                            terrain_tiles,
                            start_coord,
                            end_coord,
                            hostile.position.elevation,
                            actor_position.elevation,
                            DEFAULT_MECH_COMBAT_RULES.line_of_sight_rules,
                            actor_position.coord,
                        )
                        if not los_clear and rule.hide.allow_without_cover_if_no_los:
                            continue
                        cover = _cover_between(
                            terrain_tiles,
                            start_coord,
                            end_coord,
                            actor.stats.size if actor else None,
                        )
                        if cover == "none":
                            can_hide = False
                            break
                if not can_hide:
                    issues.append(
                        CombatValidationIssue(
                            code="hide_without_cover",
                            message=f"{turn.actor_id} attempts to hide without sufficient cover.",
                            severity="warning",
                        )
                    )

        if rule.uses_sensor_range:
            if rule.search and actor and actor.kind == "pilot":
                if not actor_position:
                    issues.append(
                        CombatValidationIssue(
                            code="pilot_search_range_unknown",
                            message=f"Action {action.action_id} uses pilot search but actor position is missing.",
                            severity="warning",
                        )
                    )
                else:
                    for target, position in targets:
                        if not position:
                            continue
                        distance = actor_position.distance_3d(position)
                        if distance > rule.search.pilot_range:
                            issues.append(
                                CombatValidationIssue(
                                    code="pilot_search_range_exceeded",
                                    message=(
                                        f"Action {action.action_id} targets at {distance} "
                                        f"beyond pilot search range {rule.search.pilot_range}."
                                    ),
                                )
                            )
            else:
                if not actor_position:
                    issues.append(
                        CombatValidationIssue(
                            code="sensor_range_unknown",
                            message=f"Action {action.action_id} uses sensors but actor position is missing.",
                            severity="warning",
                        )
                    )
                else:
                    for target, position in targets:
                        if not position:
                            continue
                        distance = actor_position.distance_3d(position)
                        if distance > actor.stats.sensor_range:
                            issues.append(
                                CombatValidationIssue(
                                    code="sensor_range_exceeded",
                                    message=(
                                        f"Action {action.action_id} targets at {distance} "
                                        f"beyond sensors {actor.stats.sensor_range}."
                                    ),
                                )
                            )

        if action.range_spaces is not None:
            if not actor_position:
                issues.append(
                    CombatValidationIssue(
                        code="range_unknown",
                        message=f"Action {action.action_id} specifies range but actor position is missing.",
                        severity="warning",
                    )
                )
            else:
                for target, position in targets:
                    if not position:
                        continue
                    range_anchor = position
                    if action.area_pattern and action.area_pattern.pattern in ("line", "cone", "blast"):
                        if area_origin is None:
                            issues.append(
                                CombatValidationIssue(
                                    code="area_origin_missing",
                                    message=(
                                        f"Action {action.action_id} uses {action.area_pattern.pattern} "
                                        "but has no origin specified."
                                    ),
                                    severity="warning",
                                )
                            )
                        else:
                            range_anchor = area_origin
                    distance = actor_position.distance_3d(range_anchor)
                    if distance > action.range_spaces:
                        issues.append(
                            CombatValidationIssue(
                                code="range_exceeded",
                                message=(
                                    f"Action {action.action_id} targets at {distance} "
                                    f"beyond range {action.range_spaces}."
                                ),
                            )
                        )
                if (
                    action.area_pattern
                    and action.area_pattern.pattern in ("line", "cone", "blast")
                    and not range_targets_present
                ):
                    if area_origin is None:
                        issues.append(
                            CombatValidationIssue(
                                code="area_origin_missing",
                                message=(
                                    f"Action {action.action_id} uses {action.area_pattern.pattern} "
                                    "but has no origin specified."
                                ),
                                severity="warning",
                            )
                        )
                    else:
                        distance = actor_position.distance_3d(area_origin)
                        if distance > action.range_spaces:
                            issues.append(
                                CombatValidationIssue(
                                    code="area_origin_range_exceeded",
                                    message=(
                                        f"Action {action.action_id} places origin at {distance} "
                                        f"beyond range {action.range_spaces}."
                                    ),
                                )
                            )

        line_of_sight_rules = DEFAULT_MECH_COMBAT_RULES.line_of_sight_rules
        if rule.requires_line_of_sight and not _effective_ignores_los(
            action,
            line_of_sight_rules,
        ):
            if not actor_position:
                issues.append(
                    CombatValidationIssue(
                        code="line_of_sight_unknown",
                        message=f"Action {action.action_id} requires line of sight but actor position is missing.",
                        severity="warning",
                    )
                )
            else:
                origin_position = area_origin or actor_position
                if not origin_position:
                    issues.append(
                        CombatValidationIssue(
                            code="line_of_sight_unknown",
                            message=f"Action {action.action_id} requires line of sight but origin is missing.",
                            severity="warning",
                        )
                    )
                else:
                    for target, position in targets:
                        if not position:
                            continue
                        start_coord = (origin_position.coord.q, origin_position.coord.r)
                        end_coord = (position.coord.q, position.coord.r)
                        if not _line_of_sight_clear(
                            terrain_tiles,
                            start_coord,
                            end_coord,
                            origin_position.elevation,
                            position.elevation,
                            line_of_sight_rules,
                            position.coord,
                        ):
                            issues.append(
                                CombatValidationIssue(
                                    code="line_of_sight_blocked",
                                    message=f"Action {action.action_id} lacks line of sight to target.",
                                    severity="warning",
                                )
                            )

        if (
            action.area_pattern
            and action.area_pattern.pattern in ("line", "cone", "blast")
            and actor_position
            and area_origin
            and rule.requires_line_of_sight
            and not _effective_ignores_los(action, line_of_sight_rules)
        ):
            start_coord = (actor_position.coord.q, actor_position.coord.r)
            end_coord = (area_origin.coord.q, area_origin.coord.r)
            if not _line_of_sight_clear(
                terrain_tiles,
                start_coord,
                end_coord,
                actor_position.elevation,
                area_origin.elevation,
                line_of_sight_rules,
                area_origin.coord,
            ):
                issues.append(
                    CombatValidationIssue(
                        code="area_origin_line_of_sight_blocked",
                        message=(
                            f"Action {action.action_id} places origin without line of sight."
                        ),
                        severity="warning",
                    )
                )

        tags = set(action.weapon_tags)
        requires_path_check = (
            ("arcing" in tags and line_of_sight_rules.arcing_requires_path_clear)
            or ("seeking" in tags and line_of_sight_rules.seeking_requires_path_clear)
        )
        if requires_path_check and actor_position:
            origin_position = area_origin or actor_position
            if origin_position:
                for target, position in targets:
                    if not position:
                        continue
                    start_coord = (origin_position.coord.q, origin_position.coord.r)
                    end_coord = (position.coord.q, position.coord.r)
                    if not _path_clear(
                        terrain_tiles,
                        start_coord,
                        end_coord,
                        line_of_sight_rules,
                        position.coord,
                    ):
                        issues.append(
                            CombatValidationIssue(
                                code="path_blocked",
                                message=(
                                    f"Action {action.action_id} uses arcing/seeking but path is blocked."
                                ),
                                severity="warning",
                            )
                        )
                if action.area_pattern and area_origin:
                    if action.area_pattern.pattern in ("line", "cone", "blast"):
                        start_coord = (actor_position.coord.q, actor_position.coord.r)
                        end_coord = (area_origin.coord.q, area_origin.coord.r)
                        if not _path_clear(
                            terrain_tiles,
                            start_coord,
                            end_coord,
                            line_of_sight_rules,
                            area_origin.coord,
                        ):
                            issues.append(
                                CombatValidationIssue(
                                    code="area_origin_path_blocked",
                                    message=(
                                        f"Action {action.action_id} uses arcing/seeking "
                                    "but the area origin path is blocked."
                                ),
                                severity="warning",
                            )
                        )

        if action.area_pattern and area_origin and action.area_pattern.pattern in ("line", "cone"):
            area_check_coords: list[HexCoord]
            if action.area_affected:
                area_check_coords = action.area_affected
            elif area_coords is not None:
                area_check_coords = [HexCoord(q=q, r=r) for q, r in area_coords]
            else:
                area_check_coords = []

            if (
                area_check_coords
                and rule.requires_line_of_sight
                and not _effective_ignores_los(action, line_of_sight_rules)
            ):
                blocked = _blocked_area_coords_by_los(
                    terrain_tiles,
                    area_origin,
                    area_check_coords,
                    line_of_sight_rules,
                )
                if blocked:
                    issues.append(
                        CombatValidationIssue(
                            code="area_line_of_sight_blocked",
                            message=(
                                f"Action {action.action_id} affects {len(blocked)} hexes "
                                "without line of sight."
                            ),
                            severity="warning",
                        )
                    )

            if area_check_coords and requires_path_check:
                blocked = _blocked_area_coords_by_path(
                    terrain_tiles,
                    area_origin,
                    area_check_coords,
                    line_of_sight_rules,
                )
                if blocked:
                    issues.append(
                        CombatValidationIssue(
                            code="area_path_blocked",
                            message=(
                                f"Action {action.action_id} affects {len(blocked)} hexes "
                                "with blocked arcing/seeking paths."
                            ),
                            severity="warning",
                        )
                    )

        if rule.attack and actor_position:
            origin_position = area_origin or actor_position
            if not origin_position:
                issues.append(
                    CombatValidationIssue(
                        code="cover_unknown",
                        message=f"Action {action.action_id} requires cover check but origin is missing.",
                        severity="warning",
                    )
                )
            else:
                for target, position in targets:
                    if not position:
                        continue
                    start_coord = (origin_position.coord.q, origin_position.coord.r)
                    end_coord = (position.coord.q, position.coord.r)
                    cover = _cover_between(
                        terrain_tiles,
                        start_coord,
                        end_coord,
                        target.stats.size if target else None,
                    )
                    attack_type = action.attack_type_override or rule.attack.attack_type
                    if (
                        cover != "none"
                        and attack_type != "melee"
                        and not _effective_ignores_cover(
                            action,
                            rule,
                            DEFAULT_MECH_COMBAT_RULES.line_of_sight_rules,
                        )
                    ):
                        issues.append(
                            CombatValidationIssue(
                                code="cover_applies",
                                message=(
                                    f"Action {action.action_id} has {cover} cover between attacker and target."
                                ),
                                severity="warning",
                            )
                        )

        if action.area_pattern and target_position:
            origin_position = area_origin or actor_position
            if not origin_position:
                issues.append(
                    CombatValidationIssue(
                        code="area_origin_missing",
                        message=f"Action {action.action_id} uses area pattern but has no origin.",
                        severity="warning",
                    )
                )
            else:
                if area_coords is not None:
                    coord = target_position.coord
                    if (coord.q, coord.r) not in area_coords:
                        issues.append(
                            CombatValidationIssue(
                                code="area_out_of_bounds",
                                message=(
                                    f"Action {action.action_id} targets {coord.q},{coord.r} "
                                    f"outside {action.area_pattern.pattern} size {action.area_pattern.size}."
                                ),
                                severity="warning",
                            )
                        )

        if rule.attack and rule.attack.uses_weapon:
            if action.weapon_count is not None:
                expected = rule.attack.weapon_count
                if action.uses_superheavy and rule.attack.allow_superheavy:
                    if action.weapon_count != 1:
                        issues.append(
                            CombatValidationIssue(
                                code="superheavy_weapon_count",
                                message=(
                                    f"Action {action.action_id} uses superheavy but weapon_count is "
                                    f"{action.weapon_count} (expected 1)."
                                ),
                            )
                        )
                elif action.weapon_count != expected:
                    issues.append(
                        CombatValidationIssue(
                            code="weapon_count_mismatch",
                            message=(
                                f"Action {action.action_id} uses {action.weapon_count} weapons "
                                f"(expected {expected})."
                            ),
                        )
                    )

            if action.uses_superheavy and not rule.attack.allow_superheavy:
                issues.append(
                    CombatValidationIssue(
                        code="superheavy_disallowed",
                        message=f"Action {action.action_id} cannot use a superheavy weapon.",
                    )
                )

            if action.uses_aux_bonus_attack and not rule.attack.allow_aux_bonus_attack:
                issues.append(
                    CombatValidationIssue(
                        code="aux_bonus_disallowed",
                        message=f"Action {action.action_id} cannot use an aux bonus attack.",
                    )
                )

        if rule.stabilize and action.action_id == "stabilize":
            if not action.stabilize_primary or not action.stabilize_secondary:
                issues.append(
                    CombatValidationIssue(
                        code="stabilize_selection_missing",
                        message="Stabilize requires one primary and one secondary option.",
                    )
                )
            else:
                if action.stabilize_primary not in rule.stabilize.primary_options:
                    issues.append(
                        CombatValidationIssue(
                            code="stabilize_primary_invalid",
                            message=f"Invalid stabilize primary option {action.stabilize_primary}.",
                        )
                    )
                if action.stabilize_secondary not in rule.stabilize.secondary_options:
                    issues.append(
                        CombatValidationIssue(
                            code="stabilize_secondary_invalid",
                            message=f"Invalid stabilize secondary option {action.stabilize_secondary}.",
                    )
                )

        if rule.attack and engaged:
            attack_type = action.attack_type_override or rule.attack.attack_type
            if attack_type == "ranged":
                issues.append(
                    CombatValidationIssue(
                        code="ranged_attack_while_engaged",
                        message=(
                            f"{turn.actor_id} makes a ranged attack while engaged; "
                            "apply engagement difficulty."
                        ),
                        severity="warning",
                    )
                )

        if action.action_id == "overwatch" and action.range_spaces is None:
            issues.append(
                CombatValidationIssue(
                    code="overwatch_range_missing",
                    message="Overwatch action missing threat/range value.",
                    severity="warning",
                )
            )
        if action.action_id == "overwatch" and rule and rule.overwatch:
            allowed_triggers = {rule.overwatch.trigger}
            if actor:
                for trigger in actor.reaction_triggers:
                    if trigger.reaction_id == "overwatch":
                        allowed_triggers.update(trigger.trigger_events)
            if action.reaction_trigger:
                if action.reaction_trigger not in allowed_triggers:
                    issues.append(
                        CombatValidationIssue(
                            code="overwatch_trigger_invalid",
                            message=(
                                f"{turn.actor_id} uses overwatch with trigger "
                                f"{action.reaction_trigger}, which is not permitted."
                            ),
                        )
                    )
            elif action.used_as_reaction or action.action_type == "reaction":
                issues.append(
                    CombatValidationIssue(
                        code="overwatch_trigger_missing",
                        message=(
                            f"{turn.actor_id} uses overwatch as a reaction without a trigger event."
                        ),
                        severity="warning",
                    )
                )
        is_free = action.used_as_free_action or action.action_type == "free"
        is_reaction = action.used_as_reaction or action.action_type == "reaction"

        if is_reaction:
            reaction_count += 1
            continue

        if not is_free:
            non_free_counts[action.action_id] = non_free_counts.get(action.action_id, 0) + 1

        if action.action_type == "quick":
            quick_count += 1
        elif action.action_type == "full":
            full_count += 1

    if full_count > 0 and quick_count > 0:
        issues.append(
            CombatValidationIssue(
                code="mixed_action_economy",
                message="Turn includes both full and quick actions.",
            )
        )

    if full_count > max_full:
        issues.append(
            CombatValidationIssue(
                code="too_many_full_actions",
                message=f"Full actions {full_count} exceed allowed {max_full}.",
            )
        )

    if quick_count > max_quick:
        issues.append(
            CombatValidationIssue(
                code="too_many_quick_actions",
                message=f"Quick actions {quick_count} exceed allowed {max_quick}.",
            )
        )

    if reaction_count > DEFAULT_MECH_COMBAT_RULES.reaction_rules.max_reactions_per_turn:
        issues.append(
            CombatValidationIssue(
                code="too_many_reactions",
                message=f"Reactions {reaction_count} exceed per-turn limit.",
            )
        )

    for action_id, count in non_free_counts.items():
        if count > 1 and not economy.allows_duplicate_actions:
            issues.append(
                CombatValidationIssue(
                    code="duplicate_action",
                    message=f"Action {action_id} used multiple times without free-action allowance.",
                )
            )

    if any(action.granted_by_overcharge for action in turn.actions) and not overcharge_used:
        issues.append(
            CombatValidationIssue(
                code="overcharge_missing",
                message="Action granted by overcharge but overcharge not used.",
                severity="warning",
            )
        )

    if overcharge_count > DEFAULT_MECH_COMBAT_RULES.turn_actions.overcharge_limit_per_turn:
        issues.append(
            CombatValidationIssue(
                code="overcharge_limit",
                message=(
                    f"Overcharge used {overcharge_count} times "
                    f"but limit is {DEFAULT_MECH_COMBAT_RULES.turn_actions.overcharge_limit_per_turn}."
                ),
            )
        )


def validate_combat_scenario(
    scenario: MechCombatScenario,
    strict: bool = False,
) -> CombatValidation:
    """Validate a combat scenario against action economy and targeting rules."""
    issues: list[CombatValidationIssue] = []
    combatants_by_id = {combatant.id: combatant for combatant in scenario.combatants}
    terrain_tiles = terrain_index(scenario.terrain)
    per_target_state_by_actor = {
        combatant.id: _index_per_target_counters(combatant)
        for combatant in scenario.combatants
    }

    for round_ in scenario.rounds:
        for turn in round_.turns:
            _validate_turn(
                turn,
                issues,
                combatants_by_id,
                terrain_tiles,
                scenario.environment,
                per_target_state_by_actor,
            )

    if strict:
        issues = _promote_strict_warnings(issues)

    return CombatValidation(valid=not any(i.severity == "error" for i in issues), issues=issues)
