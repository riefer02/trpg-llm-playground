"""
Drone Turn Integration Module

Implements turn-phase integration for all drone types per PR2 rules:

Turn Integration:
- Drones act on their owner's turn (PR2 5071)
- Drones can only make a regular move unless specified (PR2 10249)
- Drone abilities activate based on conditions during other actions

Turn Phase Handlers:
- Start of owner's turn: Apply active mode effects, take heat, check priming
- During owner's turn: Move drones (regular move), activate abilities
- End of owner's turn: Prime drones, apply end-of-turn effects
- During other turns: Check reaction triggers (Turret, Restock)

Drone Types and Turn Behavior:
- Turret: Reaction when ally hits within range 10 (always ready)
- Restock: Primes at end of owner's turn, activates when ally moves adjacent
- Latch: Mount (quick action) OR active mode (start of turn heat + buffs)
- ICEOUT: Zone persists, moved as quick action
- Tracking: Tech attack on owner's turn
- Hive: Zone persists, moved as quick action

Resolution Pattern:
1. Create Input model with phase and context
2. Call resolve_* function returning Result model
3. Caller applies state changes based on result
"""

from __future__ import annotations

from typing import Literal, Union
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.enums import StatusType
from core.shared.id_helpers import CombatantIdField
from core.mech.grid import HexCoord, HexPosition, hexes_in_radius
from core.mech.combat_state import DeployableState, MechCombatScenario


DroneTurnPhase = Literal["start", "movement", "end"]

DroneType = Literal["turret", "restock", "latch", "iceout", "tracking", "hive"]

DroneReactionTrigger = Literal[
    "ally_hit_within_range", "ally_adjacent_start", "ally_adjacent_move"
]


class DroneTurnStartInput(FrozenModel):
    """Input for drone start-of-turn processing per PR2 rules.

    Drones act on owner's turn. Some drones have start-of-turn effects:
    - Latch (active mode): Takes 1 heat, grants buffs to target
    - Tracking: Info refreshed on owner's turn
    """

    owner_id: str = Field(..., description="ID of drone owner")
    deployed_drones: dict[str, DeployableState] = Field(
        ..., description="All deployed drones belonging to owner"
    )
    current_turn: int = Field(..., ge=1, description="Current turn number")
    latch_drone_active: bool = Field(
        default=False, description="Whether any latch drone is in active mode"
    )
    latch_drone_target_id: CombatantIdField | None = Field(
        default=None, description="Target ID of active latch drone"
    )
    tier: int = Field(default=1, ge=1, le=3, description="NPC tier for scaling")


class DroneTurnStartResult(FrozenModel):
    """Result of drone start-of-turn processing."""

    drones_ready_to_act: list[str] = Field(
        default_factory=list, description="Drone IDs that can act this turn"
    )
    drones_needing_movement: list[str] = Field(
        default_factory=list, description="Drone IDs that can move this turn"
    )
    heat_to_owner: int = Field(default=0, description="Heat to add to owner")
    latch_active_effects: dict[str, bool] = Field(
        default_factory=dict, description="Active latch effects applied"
    )
    conditions_granted: list[str] = Field(
        default_factory=list, description="Conditions to grant to latch target"
    )
    conditions_immunized: list[StatusType] = Field(
        default_factory=list, description="Conditions immune for latch target"
    )
    accuracy_bonus: int = Field(default=0, description="Accuracy bonus to grant")
    drones_to_deactivate: list[str] = Field(
        default_factory=list, description="Drone IDs to deactivate (e.g., destroyed)"
    )
    reason: str = Field(default="", description="Explanation of result")


def resolve_drone_turn_start(input: DroneTurnStartInput) -> DroneTurnStartResult:
    """Process start-of-turn effects for all deployed drones.

    Per PR2: Drones act on owner's turn. Active latch drones take 1 heat
    at start of owner's turn and grant buffs to their target.

    Returns what SHOULD happen - caller applies state changes.
    """
    drones_ready: list[str] = []
    drones_needing_movement: list[str] = []
    heat_to_owner = 0
    latch_effects: dict[str, bool] = {}
    conditions_granted: list[str] = []
    conditions_immunized: list[StatusType] = []
    accuracy_bonus = 0
    drones_to_deactivate: list[str] = []

    for drone_id, drone in input.deployed_drones.items():
        if drone.is_destroyed:
            drones_to_deactivate.append(drone_id)
            continue

        if not drone.is_active:
            continue

        if drone.can_act:
            drones_ready.append(drone_id)

        if drone.can_move and drone.acts_on_owner_turn:
            drones_needing_movement.append(drone_id)

    if input.latch_drone_active and input.latch_drone_target_id:
        heat_to_owner = 1
        latch_effects = {
            "accuracy_bonus": True,
            "condition_immunity": True,
        }
        conditions_immunized = [
            "impaired",
            "jammed",
            "slowed",
            "shredded",
            "immobilized",
        ]
        accuracy_bonus = 1

    return DroneTurnStartResult(
        drones_ready_to_act=drones_ready,
        drones_needing_movement=drones_needing_movement,
        heat_to_owner=heat_to_owner,
        latch_active_effects=latch_effects,
        conditions_granted=conditions_granted,
        conditions_immunized=conditions_immunized,
        accuracy_bonus=accuracy_bonus,
        drones_to_deactivate=drones_to_deactivate,
        reason=f"Start of turn: {len(drones_ready)} drones ready, {len(drones_needing_movement)} can move, latch heat={heat_to_owner}",
    )


class DroneTurnEndInput(FrozenModel):
    """Input for drone end-of-turn processing per PR2 rules.

    End-of-turn effects:
    - Restock drone: Primes after owner's turn ends (PR2 7840)
    - Latch drone: Active mode ends if owner or target stunned
    """

    owner_id: str = Field(..., description="ID of drone owner")
    deployed_drones: dict[str, DeployableState] = Field(
        ..., description="All deployed drones belonging to owner"
    )
    current_turn: int = Field(..., ge=1, description="Current turn number")
    owner_is_stunned: bool = Field(
        default=False, description="Whether owner is stunned"
    )
    latch_drone_active: bool = Field(
        default=False, description="Whether any latch drone is in active mode"
    )
    latch_drone_target_id: CombatantIdField | None = Field(
        default=None, description="Target ID of active latch drone"
    )
    tier: int = Field(default=1, ge=1, le=3, description="NPC tier for scaling")


class DroneTurnEndResult(FrozenModel):
    """Result of drone end-of-turn processing."""

    drones_to_prime: list[str] = Field(
        default_factory=list, description="Drone IDs that will prime at end of turn"
    )
    latch_mode_end: bool = Field(
        default=False, description="Whether active latch mode ends"
    )
    drones_to_deactivate: list[str] = Field(
        default_factory=list, description="Drone IDs to deactivate (e.g., consumed)"
    )
    drones_to_recall: list[str] = Field(
        default_factory=list, description="Drone IDs to recall this turn"
    )
    reason: str = Field(default="", description="Explanation of result")


def resolve_drone_turn_end(input: DroneTurnEndInput) -> DroneTurnEndResult:
    """Process end-of-turn effects for all deployed drones.

    Per PR2 7840: "After your turn ends, the drone primes." - Restock drone
    Per PR2 7829: Active latch mode ends early if owner or target stunned.

    Returns what SHOULD happen - caller applies state changes.
    """
    drones_to_prime: list[str] = []
    latch_mode_ends = False
    drones_to_deactivate: list[str] = []
    drones_to_recall: list[str] = []

    for drone_id, drone in input.deployed_drones.items():
        if drone.is_destroyed:
            drones_to_deactivate.append(drone_id)
            continue

        if not drone.is_active:
            continue

        if drone.kind == "drone" and not drone.is_armed:
            if drone_id not in input.deployed_drones:
                continue
            drone_state = input.deployed_drones.get(drone_id)
            if drone_state and drone_state.kind == "drone":
                drones_to_prime.append(drone_id)

    if input.latch_drone_active:
        if input.owner_is_stunned:
            latch_mode_ends = True

    return DroneTurnEndResult(
        drones_to_prime=drones_to_prime,
        latch_mode_end=latch_mode_ends,
        drones_to_deactivate=drones_to_deactivate,
        drones_to_recall=drones_to_recall,
        reason=f"End of turn: {len(drones_to_prime)} drones to prime, latch ends={latch_mode_ends}",
    )


from core.shared.movement import (
    DroneMovementInput,
    DroneMovementResult,
    resolve_drone_movement,
)

__all__ = [
    "DroneMovementInput",
    "DroneMovementResult",
    "resolve_drone_movement",
]


class DroneReactionCheckInput(FrozenModel):
    """Input for checking drone reaction triggers during other actions.

    Some drones activate based on conditions during OTHER actions:
    - Turret: When ally hits within range 10
    - Restock: When ally moves adjacent or starts turn adjacent
    """

    trigger_type: DroneReactionTrigger = Field(
        ..., description="Type of reaction trigger"
    )
    drone_id: str = Field(..., description="ID of drone to check")
    deployed_drones: dict[str, DeployableState] = Field(
        ..., description="All deployed drones"
    )
    current_scenario: MechCombatScenario = Field(
        ..., description="Current combat scenario"
    )
    ally_id: str | None = Field(
        default=None, description="ID of ally who triggered condition"
    )
    ally_position: HexPosition | None = Field(
        default=None, description="Position of triggering ally"
    )
    target_id: str | None = Field(
        default=None, description="ID of attack target (for turret)"
    )
    target_position: HexPosition | None = Field(
        default=None, description="Position of attack target"
    )
    ally_attack_hit: bool = Field(
        default=False, description="Whether ally attack hit (for turret)"
    )
    tier: int = Field(default=1, ge=1, le=3, description="NPC tier for scaling")


class DroneReactionCheckResult(FrozenModel):
    """Result of drone reaction trigger check."""

    reaction_available: bool = Field(
        ..., description="Whether drone can react to trigger"
    )
    drone_id: str = Field(..., description="ID of drone")
    drone_type: DroneType | None = Field(
        default=None, description="Type of drone if known"
    )
    activation_ready: bool = Field(
        default=False, description="Whether drone is primed/ready"
    )
    range_check: bool | None = Field(
        default=None, description="Whether target is in range (if applicable)"
    )
    conditions_met: list[str] = Field(
        default_factory=list, description="Conditions that are met"
    )
    conditions_failed: list[str] = Field(
        default_factory=list, description="Conditions that are not met"
    )
    reason: str = Field(default="", description="Explanation of result")


def resolve_drone_reaction_check(
    input: DroneReactionCheckInput,
) -> DroneReactionCheckResult:
    """Check if a drone can react to a trigger condition.

    Per PR2 rules:
    - Turret: Reaction when ally hits within range 10
    - Restock: Activates when ally moves adjacent OR starts turn adjacent

    Returns what SHOULD happen - caller applies state changes.
    """
    drone = input.deployed_drones.get(input.drone_id)

    if drone is None:
        return DroneReactionCheckResult(
            reaction_available=False,
            drone_id=input.drone_id,
            conditions_failed=["drone_not_found"],
            reason=f"Drone {input.drone_id} not found in deployed drones",
        )

    if drone.is_destroyed:
        return DroneReactionCheckResult(
            reaction_available=False,
            drone_id=input.drone_id,
            conditions_failed=["drone_destroyed"],
            reason=f"Drone {input.drone_id} is destroyed",
        )

    if not drone.is_active:
        return DroneReactionCheckResult(
            reaction_available=False,
            drone_id=input.drone_id,
            conditions_failed=["drone_inactive"],
            reason=f"Drone {input.drone_id} is inactive",
        )

    conditions_met: list[str] = []
    conditions_failed: list[str] = []
    range_check: bool | None = None

    if input.trigger_type == "ally_hit_within_range":
        if not input.ally_attack_hit:
            conditions_failed.append("ally_attack_missed")
            return DroneReactionCheckResult(
                reaction_available=False,
                drone_id=input.drone_id,
                conditions_failed=conditions_failed,
                reason="Ally attack missed, turret cannot react",
            )

        if input.ally_position is None or input.target_position is None:
            conditions_failed.append("positions_required")
            return DroneReactionCheckResult(
                reaction_available=False,
                drone_id=input.drone_id,
                conditions_failed=conditions_failed,
                reason="Positions required for range check",
            )

        ally_coord = input.ally_position.coord
        drone_pos = drone.position
        drone_coord = drone_pos.coord

        if ally_coord.distance_to(drone_coord) <= 10:
            conditions_met.append("ally_within_range_10")
            range_check = True
        else:
            conditions_failed.append("ally_out_of_range")
            range_check = False

    elif input.trigger_type == "ally_adjacent_start":
        if input.ally_position is None:
            conditions_failed.append("position_required")
            return DroneReactionCheckResult(
                reaction_available=False,
                drone_id=input.drone_id,
                conditions_failed=conditions_failed,
                reason="Ally position required for adjacency check",
            )

        ally_coord = input.ally_position.coord
        drone_coord = drone.position.coord

        if ally_coord.distance_to(drone_coord) <= 1:
            conditions_met.append("ally_adjacent")
            range_check = True
        else:
            conditions_failed.append("ally_not_adjacent")
            range_check = False

    elif input.trigger_type == "ally_adjacent_move":
        if input.ally_position is None:
            conditions_failed.append("position_required")
            return DroneReactionCheckResult(
                reaction_available=False,
                drone_id=input.drone_id,
                conditions_failed=conditions_failed,
                reason="Ally position required for adjacency check",
            )

        ally_coord = input.ally_position.coord
        drone_coord = drone.position.coord

        if ally_coord.distance_to(drone_coord) <= 1:
            conditions_met.append("ally_adjacent")
            range_check = True
        else:
            conditions_failed.append("ally_not_adjacent")
            range_check = False

    reaction_available = (
        len(conditions_met) > 0
        and len(conditions_failed) == 0
        and (range_check is None or range_check)
    )

    drone_type: DroneType | None = None
    drone_name_lower = drone.name.lower()
    if "turret" in drone_name_lower:
        drone_type = "turret"
    elif "restock" in drone_name_lower:
        drone_type = "restock"
    elif "latch" in drone_name_lower:
        drone_type = "latch"
    elif "iceout" in drone_name_lower:
        drone_type = "iceout"
    elif "tracking" in drone_name_lower:
        drone_type = "tracking"
    elif "hive" in drone_name_lower:
        drone_type = "hive"

    return DroneReactionCheckResult(
        reaction_available=reaction_available,
        drone_id=input.drone_id,
        drone_type=drone_type,
        activation_ready=drone.is_armed
        if input.trigger_type.startswith("ally_adjacent")
        else True,
        range_check=range_check,
        conditions_met=conditions_met,
        conditions_failed=conditions_failed,
        reason=f"Reaction check: {'available' if reaction_available else 'not available'} for {input.trigger_type}",
    )


class DroneTurnInput(FrozenModel):
    """Comprehensive input for full drone turn processing.

    Combines start, movement, and end phases into single input.
    """

    owner_id: str = Field(..., description="ID of drone owner")
    deployed_drones: dict[str, DeployableState] = Field(
        ..., description="All deployed drones belonging to owner"
    )
    current_turn: int = Field(..., ge=1, description="Current turn number")
    tier: int = Field(default=1, ge=1, le=3, description="NPC tier for scaling")
    owner_is_stunned: bool = Field(
        default=False, description="Whether owner is stunned"
    )
    latch_drone_active: bool = Field(
        default=False, description="Whether any latch drone is in active mode"
    )
    latch_drone_target_id: CombatantIdField | None = Field(
        default=None, description="Target ID of active latch drone"
    )


class DroneTurnResult(FrozenModel):
    """Comprehensive result of full drone turn processing."""

    start_result: DroneTurnStartResult = Field(
        ..., description="Start-of-turn processing result"
    )
    movement_results: dict[str, DroneMovementResult] = Field(
        default_factory=dict, description="Movement results by drone ID"
    )
    end_result: DroneTurnEndResult = Field(
        ..., description="End-of-turn processing result"
    )
    total_heat_to_owner: int = Field(
        default=0, description="Total heat to add to owner"
    )
    drones_ready: list[str] = Field(
        default_factory=list, description="Drones that can act"
    )
    drones_can_move: list[str] = Field(
        default_factory=list, description="Drones that can move"
    )
    reason: str = Field(default="", description="Explanation of result")


def resolve_drone_turn(input: DroneTurnInput) -> DroneTurnResult:
    """Process complete drone turn for an owner.

    Handles start-of-turn, movement, and end-of-turn phases per PR2 rules.
    Drones act on owner's turn, can only make regular move.

    Returns what SHOULD happen - caller applies state changes.
    """
    start_input = DroneTurnStartInput(
        owner_id=input.owner_id,
        deployed_drones=input.deployed_drones,
        current_turn=input.current_turn,
        latch_drone_active=input.latch_drone_active,
        latch_drone_target_id=input.latch_drone_target_id,
        tier=input.tier,
    )

    start_result = resolve_drone_turn_start(start_input)

    end_input = DroneTurnEndInput(
        owner_id=input.owner_id,
        deployed_drones=input.deployed_drones,
        current_turn=input.current_turn,
        owner_is_stunned=input.owner_is_stunned,
        latch_drone_active=input.latch_drone_active,
        latch_drone_target_id=input.latch_drone_target_id,
        tier=input.tier,
    )

    end_result = resolve_drone_turn_end(end_input)

    total_heat = start_result.heat_to_owner

    return DroneTurnResult(
        start_result=start_result,
        movement_results={},
        end_result=end_result,
        total_heat_to_owner=total_heat,
        drones_ready=start_result.drones_ready_to_act,
        drones_can_move=start_result.drones_needing_movement,
        reason=f"Drone turn processed: {len(start_result.drones_ready_to_act)} ready, {len(start_result.drones_needing_movement)} can move",
    )


class DroneZoneManagementInput(FrozenModel):
    """Input for managing persistent drone zones (ICEOUT, Hive).

    Per PR2: Zones persist until drone is destroyed or scene ends.
    Zones can be moved as quick actions.
    """

    drone_id: str = Field(..., description="ID of zone-creating drone")
    deployed_drones: dict[str, DeployableState] = Field(
        ..., description="All deployed drones"
    )
    current_scenario: MechCombatScenario = Field(
        ..., description="Current combat scenario"
    )
    new_position: HexPosition | None = Field(
        default=None, description="New position for zone (if moving)"
    )
    tier: int = Field(default=1, ge=1, le=3, description="NPC tier for scaling")


class DroneZoneManagementResult(FrozenModel):
    """Result of drone zone management."""

    zone_type: DroneType | None = Field(
        default=None, description="Type of zone (iceout or hive)"
    )
    zone_active: bool = Field(default=False, description="Whether zone is active")
    zone_position: HexPosition | None = Field(
        default=None, description="Current zone position"
    )
    zone_radius: int = Field(default=0, description="Zone radius in hexes")
    effects_applied: list[str] = Field(
        default_factory=list, description="Effects currently applied"
    )
    moved: bool = Field(default=False, description="Whether zone was moved")
    reason: str = Field(default="", description="Explanation of result")


def resolve_drone_zone_management(
    input: DroneZoneManagementInput,
) -> DroneZoneManagementResult:
    """Manage persistent zones from ICEOUT and Hive drones.

    Per PR2:
    - ICEOUT: Burst 1 zone, tech action immunity, can move as quick action
    - Hive: Burst 2 zone, soft cover + entry damage, can move as quick action

    Returns what SHOULD happen - caller applies state changes.
    """
    drone = input.deployed_drones.get(input.drone_id)

    if drone is None:
        return DroneZoneManagementResult(
            zone_active=False,
            reason=f"Drone {input.drone_id} not found",
        )

    if drone.is_destroyed:
        return DroneZoneManagementResult(
            zone_active=False,
            reason=f"Drone {input.drone_id} is destroyed",
        )

    zone_type: DroneType | None = None
    zone_radius = 0
    effects: list[str] = []

    drone_name_lower = drone.name.lower()
    if "iceout" in drone_name_lower:
        zone_type = "iceout"
        zone_radius = 1
        effects = ["tech_immunity"]
    elif "hive" in drone_name_lower:
        zone_type = "hive"
        zone_radius = 2
        effects = ["soft_cover", "entry_damage"]

    if zone_type is None:
        return DroneZoneManagementResult(
            zone_active=False,
            reason=f"Drone {input.drone_id} is not a zone-creating drone",
        )

    position = input.new_position if input.new_position else drone.position
    moved = input.new_position is not None and input.new_position != drone.position

    return DroneZoneManagementResult(
        zone_type=zone_type,
        zone_active=True,
        zone_position=position,
        zone_radius=zone_radius,
        effects_applied=effects,
        moved=moved,
        reason=f"Zone {zone_type} at radius {zone_radius}, moved={moved}",
    )
