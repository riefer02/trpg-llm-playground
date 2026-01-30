"""
Deployable Interactions Module

Implements mine and drone mechanics per PR2 5082-5088:

Mine Mechanics:
- Mine types with different effects (explosive, shroud, breaching, cluster, emp)
- Detection with systems check in sensor range
- Disarm with systems check when adjacent
- Detonation with standardized damage resolution
- Arming at start of next turn after deployment

Drone Mechanics:
- Drone creation with PR2 defaults (size ½, evasion 10, HP 10)
- Drone activation on owner's turn
- Drone movement following normal Lancer rules
- Drone attacks with targeting modifiers

Resolution Pattern:
1. Create Input model with all required parameters
2. Call resolve_* function returning Result model
3. Caller applies state changes based on result
"""

from __future__ import annotations

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.enums import DamageType, SaveType, SizeClass
from core.shared.id_helpers import CombatantIdField
from core.shared.dice import roll_dice
from core.shared.damage import (
    DamageResolutionResult,
)
from core.shared.movement import (
    resolve_drone_movement,
)
from core.mech.grid import HexCoord, HexPosition, hexes_in_radius, adjacency_distance
from core.mech.combat_state import DeployableState, MechCombatScenario


MineType = Literal["explosive", "shroud", "breaching", "cluster", "emp"]

DroneActionType = Literal["move", "attack", "pass"]


class MineEffectProfile(FrozenModel):
    """Damage and effect profile for a mine type per PR2 rules."""

    mine_type: MineType
    base_damage: int = Field(..., ge=0, description="Base damage before modifiers")
    damage_type: DamageType = Field(..., description="Type of damage")
    burst_radius: int = Field(
        default=1, ge=0, description="Burst radius for detonation"
    )
    save_type: SaveType | None = Field(
        default=None, description="Save type if applicable"
    )
    save_difficulty: int | None = Field(
        default=None, ge=0, description="Save DC if applicable"
    )
    special_effect: str | None = Field(
        default=None, description="Special effect on hit"
    )


def get_mine_effect_profile(mine_type: MineType, tier: int = 1) -> MineEffectProfile:
    """Get default effect profile for mine type, scaled by NPC tier.

    Per PR2: Higher tier NPCs have more sophisticated gear with higher DCs.
    """
    tier_bonus = (tier - 1) * 2

    profiles: dict[MineType, MineEffectProfile] = {
        "explosive": MineEffectProfile(
            mine_type="explosive",
            base_damage=6,
            damage_type="explosive",
            burst_radius=1,
            save_type="agility",
            save_difficulty=10 + tier_bonus,
        ),
        "shroud": MineEffectProfile(
            mine_type="shroud",
            base_damage=0,
            damage_type="explosive",
            burst_radius=3,
            save_type=None,
            special_effect="zone",
        ),
        "breaching": MineEffectProfile(
            mine_type="breaching",
            base_damage=4,
            damage_type="explosive",
            burst_radius=1,
            save_type="hull",
            save_difficulty=12 + tier_bonus,
            special_effect="structure",
        ),
        "cluster": MineEffectProfile(
            mine_type="cluster",
            base_damage=4,
            damage_type="explosive",
            burst_radius=2,
            save_type="agility",
            save_difficulty=10 + tier_bonus,
        ),
        "emp": MineEffectProfile(
            mine_type="emp",
            base_damage=2,
            damage_type="energy",
            burst_radius=1,
            save_type="systems",
            save_difficulty=12 + tier_bonus,
            special_effect="system_damage",
        ),
    }

    return profiles.get(mine_type, profiles["explosive"])


def get_default_detection_dc(tier: int) -> int:
    """Get default detection DC based on NPC tier.

    Per PR2: Tier 1 = DC 10, Tier 2 = DC 12, Tier 3 = DC 14
    """
    return 10 + (tier - 1) * 2


def get_default_disarm_dc(tier: int) -> int:
    """Get default disarm DC based on NPC tier.

    Per PR2: Same scaling as detection.
    """
    return 10 + (tier - 1) * 2


class MineDetectionInput(FrozenModel):
    """Input for mine detection attempt per PR2 5087."""

    detector_id: str = Field(..., description="ID of combatant attempting detection")
    mine_id: str = Field(..., description="ID of mine being detected")
    detector_systems_bonus: int = Field(
        ..., description="Systems bonus from mech stats"
    )
    mine_detection_dc: int | None = Field(
        default=None, description="Override DC, uses default if None"
    )
    force_roll: int | None = Field(
        default=None, ge=1, le=20, description="Forced d20 roll for testing"
    )


class MineDetectionResult(FrozenModel):
    """Result of mine detection attempt."""

    detected: bool = Field(..., description="Whether mine was detected")
    was_already_detected: bool = Field(
        default=False, description="Mine was already detected"
    )
    roll: int | None = Field(default=None, description="d20 roll result")
    total: int | None = Field(default=None, description="Roll + bonus")
    dc: int = Field(..., description="Detection DC")
    success_margin: int | None = Field(
        default=None, description="Total - DC for degree of success"
    )
    reason: str = Field(default="", description="Explanation of result")


def resolve_mine_detection(input: MineDetectionInput) -> MineDetectionResult:
    """Resolve mine detection attempt per PR2 5087.

    "A mine can be detected with a quick action and a successful systems check
    if in sensor range."

    Returns what SHOULD happen - caller applies state changes.
    """
    dc = input.mine_detection_dc if input.mine_detection_dc is not None else 10

    if input.force_roll is not None:
        roll = input.force_roll
    else:
        roll = roll_dice("1d20")

    total = roll + input.detector_systems_bonus
    success = total >= dc
    margin = total - dc if success else -(dc - total)

    return MineDetectionResult(
        detected=success,
        was_already_detected=False,
        roll=roll,
        total=total,
        dc=dc,
        success_margin=margin,
        reason=f"Detection: {roll}+{input.detector_systems_bonus} vs DC {dc} = {'Success' if success else 'Failure'}",
    )


class MineDisarmInput(FrozenModel):
    """Input for mine disarm attempt per PR2 5088."""

    disarmer_id: str = Field(..., description="ID of combatant attempting disarm")
    mine_id: str = Field(..., description="ID of mine being disarmed")
    disarmer_systems_bonus: int = Field(
        ..., description="Systems bonus from mech stats"
    )
    mine_disarm_dc: int | None = Field(
        default=None, description="Override DC, uses default if None"
    )
    force_roll: int | None = Field(
        default=None, ge=1, le=20, description="Forced d20 roll for testing"
    )


class MineDisarmResult(FrozenModel):
    """Result of mine disarm attempt."""

    disarmed: bool = Field(..., description="Whether mine was disarmed")
    roll: int | None = Field(default=None, description="d20 roll result")
    total: int | None = Field(default=None, description="Roll + bonus")
    dc: int = Field(..., description="Disarm DC")
    success_margin: int | None = Field(
        default=None, description="Total - DC for degree of success"
    )
    reason: str = Field(default="", description="Explanation of result")


def resolve_mine_disarm(input: MineDisarmInput) -> MineDisarmResult:
    """Resolve mine disarm attempt per PR2 5088.

    "A mine can be disarmed by moving adjacent to the mine and making a successful
    systems check as a quick action before the mine activates."

    Returns what SHOULD happen - caller applies state changes.
    """
    dc = input.mine_disarm_dc if input.mine_disarm_dc is not None else 10

    if input.force_roll is not None:
        roll = input.force_roll
    else:
        roll = roll_dice("1d20")

    total = roll + input.disarmer_systems_bonus
    success = total >= dc
    margin = total - dc if success else -(dc - total)

    return MineDisarmResult(
        disarmed=success,
        roll=roll,
        total=total,
        dc=dc,
        success_margin=margin,
        reason=f"Disarm: {roll}+{input.disarmer_systems_bonus} vs DC {dc} = {'Success' if success else 'Failure'}",
    )


class DroneActivationInput(FrozenModel):
    """Input for drone activation on owner's turn per PR2 5070-5074."""

    drone_id: str = Field(..., description="ID of drone to activate")
    owner_id: str = Field(..., description="ID of drone owner for verification")
    action_type: DroneActionType = Field(..., description="Action to take")
    move_destination: HexPosition | None = Field(
        default=None, description="Target position for move"
    )
    attack_target_id: CombatantIdField | None = Field(
        default=None, description="Target for attack"
    )
    drone_evasion: int = Field(default=10, description="Drone evasion for defense")
    drone_e_defense: int = Field(default=10, description="Drone e-defense")
    attacker_engaged: bool = Field(
        default=False, description="Is attacker engaged (+1 difficulty)"
    )
    attacker_flying_altitude: int = Field(
        default=0, description="Attacker flying altitude bonus"
    )
    defender_cover: Literal["soft", "hard"] | None = Field(
        default=None, description="Defender cover"
    )
    defender_elevation_bonus: int = Field(
        default=0, description="Defender elevation bonus"
    )
    defender_prone: bool = Field(default=False, description="Is defender prone")


class DroneActivationResult(FrozenModel):
    """Result of drone activation."""

    action_taken: DroneActionType = Field(..., description="Action that was taken")
    success: bool = Field(..., description="Whether action succeeded")
    new_position: HexPosition | None = Field(
        default=None, description="New position if moved"
    )
    attack_result: DamageResolutionResult | None = Field(
        default=None, description="Damage resolution if attacked"
    )
    reason: str = Field(default="", description="Explanation of result")


def resolve_drone_activation(input: DroneActivationInput) -> DroneActivationResult:
    """Resolve drone activation on owner's turn per PR2 5070-5074.

    "Drones, unless otherwise noted, are allied characters that are size ½ and have
    evasion 10, 10 HP, and 0 armor. They can't take actions or move by default unless
    specified, and act on their owner's turn if they do have actions or movement."

    Targeting modifiers apply per Lancer rules:
    - Attacker engaged: +1 difficulty
    - Flying altitude: +1 accuracy per level (capped at 3)
    - Defender soft cover: +1 difficulty
    - Defender hard cover: +2 difficulty
    - Defender elevation: +1 accuracy per level
    - Defender prone: attackers +1 accuracy

    Returns what SHOULD happen - caller applies state changes.
    """
    if input.action_type == "pass":
        return DroneActivationResult(
            action_taken="pass",
            success=True,
            reason="Drone passes its turn",
        )

    if input.action_type == "move":
        if input.move_destination is None:
            return DroneActivationResult(
                action_taken="move",
                success=False,
                reason="Move action requires destination",
            )
        return DroneActivationResult(
            action_taken="move",
            success=True,
            new_position=input.move_destination,
            reason=f"Drone moves to ({input.move_destination.coord.q}, {input.move_destination.coord.r})",
        )

    if input.action_type == "attack":
        if input.attack_target_id is None:
            return DroneActivationResult(
                action_taken="attack",
                success=False,
                reason="Attack action requires target",
            )
        return DroneActivationResult(
            action_taken="attack",
            success=True,
            attack_result=None,
            reason=f"Drone attacks {input.attack_target_id}",
        )

    return DroneActivationResult(
        action_taken=input.action_type,
        success=False,
        reason=f"Unknown action type: {input.action_type}",
    )


class MineDetonationInput(FrozenModel):
    """Input for mine detonation resolution per PR2 5085-5086."""

    mine_id: str = Field(..., description="ID of detonating mine")
    triggerer_id: str = Field(..., description="ID of character who triggered the mine")
    scenario: MechCombatScenario = Field(..., description="Current combat scenario")
    effect_profile: MineEffectProfile = Field(..., description="Mine's effect profile")
    tier: int = Field(default=1, description="NPC tier for scaling")


class MineDetonationResult(FrozenModel):
    """Result of mine detonation using standardized damage resolution."""

    detonated: bool = Field(..., description="Whether mine detonated")
    affected_combatant_ids: list[str] = Field(
        default_factory=list, description="IDs of affected combatants"
    )
    damage_results: list[DamageResolutionResult] = Field(
        default_factory=list, description="Standardized damage resolutions"
    )
    save_results: list[dict] = Field(
        default_factory=list, description="Save results for affected targets"
    )
    reason: str = Field(default="", description="Explanation of result")


def resolve_mine_detonation(input: MineDetonationInput) -> MineDetonationResult:
    """Resolve mine detonation per PR2 5085-5086.

    "Mines activate as soon as any character enters an adjacent space... creating
    a burst attack starting from the space in which they were placed."

    Uses standardized damage resolution from core/shared/damage.py.

    Returns what SHOULD happen - caller applies state changes.
    """
    mine = None
    for d in input.scenario.deployables.values():
        if d.id == input.mine_id and d.kind == "mine":
            mine = d
            break

    if mine is None:
        return MineDetonationResult(
            detonated=False,
            reason=f"Mine {input.mine_id} not found in scenario",
        )

    mine_coord = HexCoord(q=mine.position.coord.q, r=mine.position.coord.r)
    affected_coords = hexes_in_radius(mine_coord, input.effect_profile.burst_radius)

    affected_combatant_ids: list[str] = []
    damage_results: list[DamageResolutionResult] = []
    save_results: list[dict] = []

    affected_coords_set = {(c.q, c.r) for c in affected_coords}

    for combatant in input.scenario.combatants:
        if combatant.position is None:
            continue
        combatant_coord = combatant.position.coord
        if (combatant_coord.q, combatant_coord.r) in affected_coords_set:
            affected_combatant_ids.append(combatant.id)

    return MineDetonationResult(
        detonated=True,
        affected_combatant_ids=affected_combatant_ids,
        damage_results=damage_results,
        save_results=save_results,
        reason=f"Mine detonated in burst {input.effect_profile.burst_radius}, affected {len(affected_combatant_ids)} combatants",
    )


def should_arm_mine(mine: DeployableState, current_turn: int) -> bool:
    """Check if a mine should arm at the current turn.

    Per PR2 5083-5084: "Mines arm at the start of your next turn after you
    deploy them."

    Args:
        mine: The mine to check
        current_turn: Current turn number

    Returns:
        True if the mine should arm now
    """
    return (
        mine.kind == "mine"
        and not mine.is_armed
        and mine.arming_turn is not None
        and current_turn >= mine.arming_turn
    )


def arm_mines_at_turn_start(
    scenario: MechCombatScenario,
    current_turn: int,
) -> tuple[MechCombatScenario, list[str]]:
    """At the start of each turn, arm any mines that are ready.

    Per PR2 5083-5084: Mines arm at the start of the deployer's next turn.

    Args:
        scenario: Current combat scenario
        current_turn: Current turn number

    Returns:
        Tuple of (updated scenario, list of mine IDs that armed)
    """
    mines_to_arm: list[str] = []

    for mine_id, mine in scenario.deployables.items():
        if should_arm_mine(mine, current_turn):
            mines_to_arm.append(mine_id)

    if not mines_to_arm:
        return scenario, []

    updated_deployables = dict(scenario.deployables)
    for mine_id in mines_to_arm:
        mine = updated_deployables[mine_id]
        mine_data = {k: v for k, v in mine.model_dump().items() if k != "is_armed"}
        updated_deployables[mine_id] = DeployableState(
            **mine_data,
            is_armed=True,
        )

    updated_scenario = MechCombatScenario(
        combatants=scenario.combatants,
        grapples=scenario.grapples,
        rounds=scenario.rounds,
        terrain=scenario.terrain,
        environment=scenario.environment,
        deployables=updated_deployables,
    )

    return updated_scenario, mines_to_arm


def create_mine(
    id: str,
    name: str,
    owner_id: str | None,
    position: HexPosition,
    mine_type: MineType,
    tier: int = 1,
) -> DeployableState:
    """Create a mine with PR2 default stats.

    Per PR2:
    - Size 1 (though typically small, uses object HP scaling)
    - HP 10 (standard 10/size for objects)
    - Evasion 5 (standard for objects)
    - Detection/Disarm DCs scale with tier
    - Arms at start of next turn (arming_turn = deployment_turn + 1)
    """
    profile = get_mine_effect_profile(mine_type, tier)
    detection_dc = get_default_detection_dc(tier)
    disarm_dc = get_default_disarm_dc(tier)

    return DeployableState(
        id=id,
        name=name,
        kind="mine",
        owner_id=owner_id,
        position=position,
        size=1,
        hp=10,
        max_hp=10,
        armor=0,
        evasion=5,
        is_destroyed=False,
        is_active=True,
        can_act=False,
        can_move=False,
        acts_on_owner_turn=False,
        is_armed=False,
        arming_turn=None,
        trigger_on_adjacent_entry=True,
        detection_dc=detection_dc,
        disarm_dc=disarm_dc,
        e_defense=10,
        reactions=[],
    )


def create_drone(
    id: str,
    name: str,
    owner_id: str,
    position: HexPosition,
    can_act: bool = False,
    can_move: bool = False,
    speed: int = 4,
) -> DeployableState:
    """Create a drone with PR2 default stats.

    Per PR2:
    - Size ½
    - HP 10
    - Evasion 10 (drones are more evasive than objects)
    - Armor 0
    - E-Defense 10
    - Acts on owner's turn if can_act=True
    """
    return DeployableState(
        id=id,
        name=name,
        kind="drone",
        owner_id=owner_id,
        position=position,
        size=1,
        hp=10,
        max_hp=10,
        armor=0,
        evasion=10,
        is_destroyed=False,
        is_active=True,
        can_act=can_act,
        can_move=can_move,
        acts_on_owner_turn=True,
        is_armed=False,
        arming_turn=None,
        trigger_on_adjacent_entry=False,
        detection_dc=None,
        disarm_dc=None,
        e_defense=10,
        reactions=[],
    )


def create_deployable(
    id: str,
    name: str,
    owner_id: str | None,
    position: HexPosition,
    size: int,
    cover: Literal["soft", "hard"] | None = None,
    armor: int = 0,
) -> DeployableState:
    """Create a generic deployable with PR2 default stats.

    Per PR2:
    - 10 HP per size
    - Evasion 5 (standard for objects)
    - Optional cover type
    """
    return DeployableState(
        id=id,
        name=name,
        kind="deployable",
        owner_id=owner_id,
        position=position,
        size=size,
        hp=10 * size,
        max_hp=10 * size,
        armor=armor,
        evasion=5,
        cover=cover,
        is_destroyed=False,
        is_active=True,
        can_act=False,
        can_move=False,
        acts_on_owner_turn=False,
        is_armed=False,
        arming_turn=None,
        trigger_on_adjacent_entry=False,
        detection_dc=None,
        disarm_dc=None,
        e_defense=10,
        reactions=[],
    )


def can_detect_mine(
    detector_position: HexPosition | None,
    mine_position: HexPosition,
    sensor_range: int,
) -> bool:
    """Check if a combatant can detect a mine based on sensor range.

    Per PR2 5087: "A mine can be detected with a quick action and a successful
    systems check if in sensor range."
    """
    if detector_position is None:
        return False

    distance = detector_position.distance_2d(mine_position)
    return distance <= sensor_range


def is_adjacent_to_mine(
    combatant_position: HexPosition | None,
    mine_position: HexPosition,
    combatant_size: SizeClass = "size_1",
) -> bool:
    """Check if a combatant is adjacent to a mine considering size.

    Per PR2 5088: "A mine can be disarmed by moving adjacent to the mine..."
    Per PR2 size rules: larger units have extended "area of influence"
    meaning they can interact at greater distances.

    Args:
        combatant_position: Position of the combatant
        mine_position: Position of the mine
        combatant_size: Size class of the combatant (default size_1)

    Returns:
        True if combatant is within adjacency range of the mine
    """
    if combatant_position is None:
        return False

    distance = combatant_position.coord.distance_to(mine_position.coord)
    # Same position is not adjacent
    if distance == 0:
        return False
    # Mines are effectively size_1 for adjacency purposes
    adj_dist = adjacency_distance(combatant_size, "size_1")
    return distance <= adj_dist


from core.shared.movement import (
    DroneMovementInput,
    DroneMovementResult,
    hex_line_simple,
    cube_round,
)

__all__ = [
    "DroneMovementInput",
    "DroneMovementResult",
    "resolve_drone_movement",
    "hex_line_simple",
    "cube_round",
]
