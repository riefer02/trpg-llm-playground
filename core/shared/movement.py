"""Unified Movement Resolution Module

Per PR2 3729-3930: Comprehensive movement system for mechs, drones, and pilots.

Movement Rules:
- Regular move: Speed spaces without action
- Difficult terrain: 1 space costs 2 movement
- Obstructions: Block passage (smaller don't)
- Engagement: Must stop when adjacent to same/larger hostile
- Flight: Min 1 space, straight lines, altitude limits
- Teleport: Surface to surface only, ignores obstructions/LOS

Resolution Pattern:
1. Create MovementInput with all parameters
2. Call resolve_movement() returning MovementResult
3. Caller applies state changes based on result
"""

from __future__ import annotations

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.enums import SizeClass
from core.shared.terrain import TerrainMap, get_terrain_at, calculate_movement_cost
from core.mech.grid import HexCoord, HexPosition, hex_line, adjacency_distance
from core.mech.combat_state import MechCombatScenario


MovementMode = Literal["ground", "flight", "hover", "teleport"]


class MovementInput(FrozenModel):
    """Unified input for movement resolution per Lancer rules.

    Per PR2 3729-3930:
    - Movement costs 1 space per space moved
    - Difficult terrain costs 2 spaces
    - Engagement stops movement with same/larger hostile
    - Obstructions block path (smaller don't)
    - Flight/teleport have special rules
    """

    entity_id: str = Field(..., description="ID of entity to move")
    destination: HexPosition = Field(..., description="Target position")
    current_scenario: MechCombatScenario = Field(
        ..., description="Current combat scenario"
    )
    speed: int = Field(..., ge=0, description="Movement speed in spaces")
    mode: MovementMode = Field(
        default="ground",
        description="Movement mode: ground/flight/hover/teleport",
    )
    ignore_engagement: bool = Field(
        default=False,
        description="Ignore engagement rules (e.g., Disengage action)",
    )
    ignore_reactions: bool = Field(
        default=False,
        description="Ignore reactions (movement doesn't provoke)",
    )
    force_movement_cost: int | None = Field(
        default=None,
        description="Force total cost for testing",
    )
    force_path: list[HexCoord] | None = Field(
        default=None,
        description="Force specific path for testing",
    )


class MovementResult(FrozenModel):
    """Result of movement resolution."""

    movement_successful: bool = Field(..., description="Whether movement completed")
    path_clear: bool = Field(..., description="Whether path is clear")
    spaces_moved: int = Field(..., description="Spaces actually moved")
    total_movement_cost: int = Field(..., description="Total movement cost")
    new_position: HexPosition | None = Field(
        default=None,
        description="Final position if movement succeeded",
    )
    terrain_costs: list[int] = Field(
        default_factory=list,
        description="Per-hex terrain costs",
    )
    terrain_encountered: list[str] = Field(
        default_factory=list,
        description="Terrain types encountered",
    )
    engagement_stopped: bool = Field(
        default=False,
        description="Movement stopped by engagement rules",
    )
    obstruction_blocked: bool = Field(
        default=False,
        description="Movement blocked by obstruction",
    )
    flight_validation_passed: bool = Field(
        default=True,
        description="Flight rules validated successfully",
    )
    teleport_validation_passed: bool = Field(
        default=True,
        description="Teleport rules validated successfully",
    )
    reason: str = Field(default="", description="Explanation of result")


def hex_line_simple(start: HexCoord, end: HexCoord) -> list[HexCoord]:
    """Simple hex line between two coordinates.

    Wrapper around grid.hex_line() for convenient access from movement module.
    Per PR2 3676+: Movement uses hex grid for positioning.
    """
    return hex_line(start, end)


def cube_round(cube: tuple[float, float, float]) -> tuple[int, int, int]:
    """Round fractional cube coordinates to nearest hex.

    Per hex grid mathematics for smooth path interpolation.
    """
    rx, ry, rz = cube
    x_round = round(rx)
    y_round = round(ry)
    z_round = round(rz)

    x_diff = abs(x_round - rx)
    y_diff = abs(y_round - ry)
    z_diff = abs(z_round - rz)

    if x_diff > y_diff and x_diff > z_diff:
        x_round = -y_round - z_round
    elif y_diff > z_diff:
        y_round = -x_round - z_round
    else:
        z_round = -x_round - y_round

    return (x_round, y_round, z_round)


def _find_entity_position(
    entity_id: str,
    scenario: MechCombatScenario,
) -> HexPosition | None:
    """Find entity's current position in scenario.

    Checks combatants first, then deployables.
    Returns None if not found.
    """
    for combatant in scenario.combatants:
        if combatant.id == entity_id:
            return combatant.position
    for deployable in scenario.deployables.values():
        if deployable.id == entity_id:
            return deployable.position
    return None


def _size_value(size: SizeClass) -> float:
    """Convert size class to numeric value for comparison."""
    size_values: dict[SizeClass, float] = {
        "size_half": 0.5,
        "size_1": 1.0,
        "size_2": 2.0,
        "size_3": 3.0,
        "size_4": 4.0,
        "size_5": 5.0,
    }
    return size_values.get(size, 1.0)


def _get_entity_size(
    entity_id: str,
    scenario: MechCombatScenario,
) -> SizeClass:
    """Get entity's size class."""
    for combatant in scenario.combatants:
        if combatant.id == entity_id:
            return combatant.stats.size
    for deployable in scenario.deployables.values():
        if deployable.id == entity_id:
            return deployable.size
    return "size_1"


def _is_hostile(
    entity_id: str,
    target_id: str,
    scenario: MechCombatScenario,
) -> bool:
    """Check if target is hostile to entity."""
    entity = None
    target = None
    for combatant in scenario.combatants:
        if combatant.id == entity_id:
            entity = combatant
        if combatant.id == target_id:
            target = combatant
    if entity is None or target is None:
        return False
    return entity.side != target.side


def _is_ally(
    entity_id: str,
    target_id: str,
    scenario: MechCombatScenario,
) -> bool:
    """Check if target is ally of entity."""
    entity = None
    target = None
    for combatant in scenario.combatants:
        if combatant.id == entity_id:
            entity = combatant
        if combatant.id == target_id:
            target = combatant
    if entity is None or target is None:
        return False
    return entity.side == target.side


def _surface_elevation(
    terrain: TerrainMap | None,
    coord: HexCoord,
) -> int:
    """Get surface elevation at a coordinate."""
    if terrain is None:
        return 0
    terrain_hex = get_terrain_at(terrain, coord)
    return terrain_hex.elevation if terrain_hex else 0


def check_engagement_stop(
    entity_id: str,
    entity_size: SizeClass,
    path: list[HexCoord],
    scenario: MechCombatScenario,
    ignore_engagement: bool = False,
) -> tuple[bool, HexCoord | None]:
    """Check if movement should stop due to engagement rules.

    Per PR2 3818-3819:
    "If you move adjacent to a hostile character, you become engaged.
    If you become engaged with a target the same size or larger, you must stop."

    Args:
        entity_id: Moving entity ID
        entity_size: Size of moving entity
        path: Movement path coordinates
        scenario: Current combat scenario
        ignore_engagement: Whether to skip engagement check

    Returns:
        (should_stop, stop_position) - True if must stop, and where
    """
    if ignore_engagement:
        return False, None

    for i, coord in enumerate(path[1:], start=1):
        for combatant in scenario.combatants:
            if combatant.id == entity_id:
                continue
            if not _is_hostile(entity_id, combatant.id, scenario):
                continue
            distance = coord.distance_to(combatant.position.coord)
            adj_dist = adjacency_distance(entity_size, combatant.stats.size)
            if distance <= adj_dist:
                if _size_value(combatant.stats.size) >= _size_value(entity_size):
                    return True, coord
    return False, None


def check_obstructions(
    path: list[HexCoord],
    entity_id: str,
    entity_size: SizeClass,
    scenario: MechCombatScenario,
    mode: MovementMode,
) -> tuple[bool, int]:
    """Check if path is blocked by obstructions.

    Per PR2 3812-3815:
    "Obstructions block passage. Obstructions smaller than the moving
    object do not block movement, and can be passed over freely.
    Friendly NPCs or allied players never cause obstruction."

    Args:
        path: Movement path coordinates
        entity_id: Moving entity ID
        entity_size: Size of moving entity
        scenario: Current combat scenario
        mode: Movement mode (flight/hover/teleport ignore obstructions)

    Returns:
        (is_blocked, blocked_index) - True if blocked, and first blocked hex index
    """
    if mode in ("flight", "hover", "teleport"):
        return False, -1

    for i, coord in enumerate(path[1:], start=1):
        for combatant in scenario.combatants:
            if combatant.id == entity_id:
                continue
            if _is_ally(entity_id, combatant.id, scenario):
                continue
            if combatant.position.coord == coord:
                if _size_value(combatant.stats.size) < _size_value(entity_size):
                    continue
                return True, i
    return False, -1


def validate_flight(
    path: list[HexPosition],
    mode: MovementMode,
    terrain: TerrainMap | None,
) -> tuple[bool, str]:
    """Validate flight movement per PR2 3919-3922.

    Flight Rules:
    - Must move at least 1 space or fall (if not immobilized/stunned)
    - Movement must start/end along straight line per segment
    - Cannot exceed altitude limit (default 3 spaces above surface)
    - Cannot be prone while flying

    Args:
        path: Movement path
        mode: Flight mode (flight/hover)
        terrain: Terrain map for surface elevation

    Returns:
        (is_valid, reason) - False if invalid, with explanation
    """
    if mode not in ("flight", "hover"):
        return True, ""

    if len(path) < 2 and mode != "hover":
        return False, "Must move at least 1 space while flying or fall"

    max_altitude = 3

    for pos in path:
        surface = _surface_elevation(terrain, pos.coord)
        if pos.elevation > surface + max_altitude:
            return False, f"Cannot exceed altitude {max_altitude}"

    return True, ""


def validate_teleport(
    start: HexPosition,
    end: HexPosition,
    entity_id: str,
    scenario: MechCombatScenario,
) -> tuple[bool, str]:
    """Validate teleport movement per PR2 3897-3902.

    Teleport Rules:
    - Must start and end on a surface (cannot teleport mid-air)
    - Cannot teleport if immobilized (counts as moving 1 space)
    - Teleport ignores obstructions, LOS, engagement
    - Fails if destination occupied

    Args:
        start: Starting position
        end: Destination position
        entity_id: Teleporting entity ID
        scenario: Current combat scenario

    Returns:
        (is_valid, reason) - False if invalid, with explanation
    """
    terrain = scenario.terrain

    start_surface = _surface_elevation(terrain, start.coord)
    end_surface = _surface_elevation(terrain, end.coord)

    if start.elevation != start_surface:
        return False, "Cannot teleport from mid-air"
    if end.elevation != end_surface:
        return False, "Cannot teleport to mid-air"

    for combatant in scenario.combatants:
        if combatant.id == entity_id:
            continue
        if combatant.position.coord == end.coord:
            return False, "Destination space occupied"

    return True, ""


def _generate_reason(
    spaces_moved: int,
    total_cost: int,
    speed: int,
    engagement_stopped: bool,
    obstruction_blocked: bool,
) -> str:
    """Generate human-readable reason string."""
    parts = []
    if engagement_stopped:
        parts.append("stopped by engagement")
    elif obstruction_blocked:
        parts.append("blocked by obstruction")
    else:
        parts.append(f"moved {spaces_moved} spaces")
    parts.append(f"cost {total_cost}/{speed}")
    return ", ".join(parts)


def _failed_result(reason: str) -> MovementResult:
    """Create a failed movement result."""
    return MovementResult(
        movement_successful=False,
        path_clear=False,
        spaces_moved=0,
        total_movement_cost=0,
        new_position=None,
        reason=reason,
    )


def resolve_movement(input: MovementInput) -> MovementResult:
    """Resolve movement for any entity following Lancer rules.

    Per PR2 3729-3930:
    - Regular move: Speed spaces without action
    - Difficult terrain: 1 space costs 2 movement
    - Engagement: Must stop when adjacent to same/larger hostile
    - Obstructions: Block passage (smaller don't)
    - Flight: Min 1 space, straight lines, altitude limits
    - Teleport: Surface to surface only, ignores obstructions/LOS

    Args:
        input: MovementInput with all required parameters

    Returns:
        MovementResult describing what SHOULD happen.
        Caller applies state changes based on result.
    """
    start_pos = _find_entity_position(input.entity_id, input.current_scenario)
    if start_pos is None:
        return _failed_result(f"Entity {input.entity_id} not found in scenario")

    path = input.force_path or hex_line_simple(start_pos.coord, input.destination.coord)

    teleport_valid, teleport_reason = validate_teleport(
        start_pos, input.destination, input.entity_id, input.current_scenario
    )
    if not teleport_valid:
        return MovementResult(
            movement_successful=False,
            path_clear=False,
            spaces_moved=0,
            total_movement_cost=0,
            new_position=None,
            teleport_validation_passed=False,
            reason=teleport_reason,
        )

    flight_valid, flight_reason = validate_flight(
        path, input.mode, input.current_scenario.terrain
    )
    if not flight_valid:
        return MovementResult(
            movement_successful=False,
            path_clear=False,
            spaces_moved=0,
            total_movement_cost=0,
            new_position=None,
            flight_validation_passed=False,
            reason=flight_reason,
        )

    entity_size = _get_entity_size(input.entity_id, input.current_scenario)
    blocked, blocked_idx = check_obstructions(
        path, input.entity_id, entity_size, input.current_scenario, input.mode
    )
    if blocked:
        return MovementResult(
            movement_successful=False,
            path_clear=False,
            spaces_moved=0,
            total_movement_cost=0,
            new_position=None,
            obstruction_blocked=True,
            reason=f"Path blocked at hex {blocked_idx}",
        )

    terrain_costs: list[int] = []
    terrain_encountered: list[str] = []
    total_cost = 0
    spaces_moved = 0
    engagement_stopped = False

    for i, coord in enumerate(path[1:], start=1):
        if i > input.speed:
            break

        cost = calculate_movement_cost(
            spaces=1,
            terrain=input.current_scenario.terrain,
            coord=coord,
        )
        terrain_costs.append(cost)
        total_cost += cost
        spaces_moved += 1

        hex_terrain = get_terrain_at(input.current_scenario.terrain, coord)
        if hex_terrain:
            if hex_terrain.difficult:
                terrain_encountered.append("difficult")
            if hex_terrain.dangerous:
                terrain_encountered.append("dangerous")

        stop_needed, stop_at = check_engagement_stop(
            input.entity_id,
            entity_size,
            path[: i + 1],
            input.current_scenario,
            input.ignore_engagement,
        )
        if stop_needed:
            engagement_stopped = True
            spaces_moved = i - 1
            break

    if input.force_movement_cost:
        total_cost = input.force_movement_cost

    reached_destination = spaces_moved == len(path) - 1
    movement_successful = (
        reached_destination
        and total_cost <= input.speed
        and not engagement_stopped
        and not blocked
    )
    path_clear = len(path) - 1 <= input.speed

    return MovementResult(
        movement_successful=movement_successful,
        path_clear=path_clear,
        spaces_moved=spaces_moved,
        total_movement_cost=total_cost,
        new_position=input.destination if movement_successful else None,
        terrain_costs=terrain_costs,
        terrain_encountered=terrain_encountered,
        engagement_stopped=engagement_stopped,
        obstruction_blocked=blocked,
        flight_validation_passed=flight_valid,
        teleport_validation_passed=teleport_valid,
        reason=_generate_reason(
            spaces_moved, total_cost, input.speed, engagement_stopped, blocked
        ),
    )


class DroneMovementInput(FrozenModel):
    """Input for drone movement following normal Lancer rules.

    DEPRECATED: Use MovementInput with speed parameter instead.
    This class maintained for backward compatibility.
    """

    drone_id: str = Field(..., description="ID of drone to move")
    destination: HexPosition = Field(..., description="Target position")
    current_scenario: MechCombatScenario = Field(
        ..., description="Current combat scenario"
    )
    drone_speed: int = Field(default=4, description="Drone movement speed")
    force_movement_cost: int | None = Field(
        default=None, description="Forced total cost for testing"
    )
    force_path: list[HexCoord] | None = Field(
        default=None, description="Forced path for testing"
    )


class DroneMovementResult(FrozenModel):
    """Result of drone movement attempt.

    DEPRECATED: Use MovementResult instead.
    This class maintained for backward compatibility.
    """

    movement_successful: bool = Field(..., description="Whether movement completed")
    path_clear: bool = Field(..., description="Whether path is clear")
    spaces_moved: int = Field(..., description="Number of spaces moved")
    total_movement_cost: int = Field(..., description="Total movement cost")
    new_position: HexPosition | None = Field(default=None, description="Final position")
    terrain_costs: list[int] = Field(
        default_factory=list, description="Per-space terrain costs"
    )
    terrain_encountered: list[str] = Field(
        default_factory=list, description="Terrain types encountered"
    )
    reason: str = Field(default="", description="Explanation of result")


def resolve_drone_movement(input: DroneMovementInput) -> DroneMovementResult:
    """Resolve drone movement following normal Lancer rules.

    DEPRECATED: Use resolve_movement() with MovementInput instead.
    This function maintained for backward compatibility.

    Per PR2:
    - Movement costs 1 space per space moved
    - Difficult terrain costs 2 spaces per space
    - Movement doesn't provoke reactions
    - Flying ignores obstructions at altitude 1+

    Returns what SHOULD happen - caller applies state changes.
    """
    movement_input = MovementInput(
        entity_id=input.drone_id,
        destination=input.destination,
        current_scenario=input.current_scenario,
        speed=input.drone_speed,
        mode="ground",
        force_movement_cost=input.force_movement_cost,
        force_path=input.force_path,
    )
    result = resolve_movement(movement_input)
    return DroneMovementResult(
        movement_successful=result.movement_successful,
        path_clear=result.path_clear,
        spaces_moved=result.spaces_moved,
        total_movement_cost=result.total_movement_cost,
        new_position=result.new_position,
        terrain_costs=result.terrain_costs,
        terrain_encountered=result.terrain_encountered,
        reason=result.reason,
    )
