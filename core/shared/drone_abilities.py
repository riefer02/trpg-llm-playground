"""
Drone Abilities Module

Implements ability resolution for all drone types per PR2 5070-5088:

Drone Types:
- Turret Drone: Reaction attack when ally hits within range 10
- Restock Drone: Cool 1d6 heat, reload weapon, clear condition
- Latch Drone: Mount attack with heal OR active buff + condition immunity
- ICEOUT Drone: Burst 1 zone with tech action immunity
- Tracking Drone: Tech attack revealing target info, negating hide/invis
- Hive Drone: Burst 2 zone with soft cover and entry damage

Resolution Pattern:
1. Create Input model with all required parameters
2. Call resolve_* function returning Result model
3. Caller applies state changes based on result

Drone Default Stats (per PR2):
- Size: 1/2
- HP: 10
- Evasion: 10
- E-Defense: 10
- Armor: 0
- Acts on owner's turn if can_act=True
"""

from __future__ import annotations

from typing import Literal, Any, Union
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.enums import DamageType
from core.shared.dice import roll_dice
from core.mech.grid import HexPosition, HexCoord, hexes_in_radius
from core.mech.combat_state import MechCombatScenario


DroneAbilityType = Literal["turret", "restock", "latch", "iceout", "tracking", "hive"]

RestockAction = Literal["cool", "reload", "clear_condition"]

LatchMode = Literal["mount", "active"]


class TurretDroneInput(FrozenModel):
    """Input for Turret Drone ability resolution per PR2 7344-7358."""

    ability_type: Literal["turret"] = "turret"
    drone_id: str = Field(..., description="ID of the turret drone")
    owner_id: str = Field(..., description="ID of the drone's owner")
    tier: int = Field(default=1, ge=1, le=3, description="NPC tier for scaling")
    ally_attack_hit: bool = Field(
        ..., description="Whether the allied attack hit (trigger condition)"
    )
    target_id: str = Field(..., description="ID of the target of the allied attack")
    ally_position: HexPosition = Field(
        ..., description="Position of the ally who made the attack"
    )
    turret_position: HexPosition = Field(
        ..., description="Position of the turret drone"
    )
    turret_range: int = Field(default=10, description="Range of turret drone attack")
    base_damage: int = Field(default=3, description="Base damage before tier scaling")


class TurretDroneResult(FrozenModel):
    """Result of Turret Drone ability resolution."""

    ability_type: Literal["turret"] = "turret"
    success: bool = Field(..., description="Whether the ability succeeded")
    damage_dealt: bool = Field(
        ..., description="Whether damage was dealt by the turret"
    )
    damage_amount: int = Field(default=0, description="Amount of kinetic damage dealt")
    damage_type: Literal["kinetic"] = Field(
        default="kinetic", description="Type of damage dealt"
    )
    target_ids: list[str] = Field(
        default_factory=list, description="IDs of targets affected"
    )
    range_check: bool = Field(
        ..., description="Whether targets were within turret range"
    )
    reason: str = Field(default="", description="Explanation of the result")


class RestockDroneInput(FrozenModel):
    """Input for Restock Drone ability resolution per PR2 7833-7843."""

    ability_type: Literal["restock"] = "restock"
    drone_id: str = Field(..., description="ID of the restock drone")
    owner_id: str = Field(..., description="ID of the drone's owner")
    tier: int = Field(default=1, ge=1, le=3, description="NPC tier for scaling")
    activating_combatant_id: str = Field(
        ..., description="ID of the combatant activating the drone"
    )
    activating_combatant_position: HexPosition = Field(
        ..., description="Position of the activating combatant"
    )
    drone_position: HexPosition = Field(..., description="Position of the drone")
    action_choice: RestockAction = Field(
        ..., description="Action to perform: cool, reload, or clear_condition"
    )
    is_primed: bool = Field(
        default=False,
        description="Whether the drone has primed (after owner's turn ends)",
    )
    current_heat: int | None = Field(
        default=None, description="Current heat level of activating combatant"
    )
    force_cool_roll: int | None = Field(
        default=None, ge=1, le=6, description="Forced d6 roll for testing"
    )


class RestockDroneResult(FrozenModel):
    """Result of Restock Drone ability resolution."""

    ability_type: Literal["restock"] = "restock"
    success: bool = Field(..., description="Whether the ability succeeded")
    heat_cooled: int | None = Field(
        default=None, description="Amount of heat cooled (1d6)"
    )
    weapon_reloaded: bool = Field(
        default=False, description="Whether a weapon was reloaded"
    )
    condition_cleared: str | None = Field(
        default=None, description="Name of condition that was cleared"
    )
    drone_consumed: bool = Field(
        ..., description="Whether the drone was consumed after activation"
    )
    can_activate: bool = Field(..., description="Whether the activation was allowed")
    reason: str = Field(default="", description="Explanation of the result")


class LatchDroneInput(FrozenModel):
    """Input for Latch Drone ability resolution per PR2 7813-7831."""

    ability_type: Literal["latch"] = "latch"
    drone_id: str = Field(..., description="ID of the latch drone")
    owner_id: str = Field(..., description="ID of the drone's owner")
    tier: int = Field(default=1, ge=1, le=3, description="NPC tier for scaling")
    mode: LatchMode = Field(..., description="Mode of latch drone operation")
    target_id: str = Field(..., description="ID of the target mech")
    target_position: HexPosition = Field(..., description="Position of the target mech")
    shooter_id: str = Field(..., description="ID of the shooter/owner")
    shooter_position: HexPosition = Field(..., description="Position of the shooter")
    shooter_systems_bonus: int = Field(..., description="Systems bonus for attack roll")
    has_core_power: bool = Field(
        default=False, description="Whether owner has 1 core power for active mode"
    )
    force_roll: int | None = Field(
        default=None, ge=1, le=20, description="Forced d20 roll for testing"
    )
    base_attack_dc: int = Field(default=8, description="Base attack DC (vs Evasion)")


class LatchDroneResult(FrozenModel):
    """Result of Latch Drone ability resolution."""

    ability_type: Literal["latch"] = "latch"
    success: bool = Field(..., description="Whether the ability succeeded")
    attack_roll: int | None = Field(default=None, description="d20 attack roll result")
    attack_total: int | None = Field(default=None, description="Roll + bonus")
    hit: bool = Field(..., description="Whether the attack hit")
    core_power_spent: bool = Field(
        default=False, description="Whether core power was spent"
    )
    repair_spent: bool = Field(default=False, description="Whether a repair was spent")
    hp_healed: int | None = Field(default=None, description="Amount of HP healed")
    buffs_granted: list[str] = Field(
        default_factory=list, description="Buffs granted to target"
    )
    immunities_granted: list[str] = Field(
        default_factory=list, description="Condition immunities granted"
    )
    heat_to_owner: int | None = Field(
        default=None, description="Heat taken by owner at start of turn"
    )
    reason: str = Field(default="", description="Explanation of the result")


class ICEOUTDroneInput(FrozenModel):
    """Input for ICEOUT Drone ability resolution per PR2 8645-8658."""

    ability_type: Literal["iceout"] = "iceout"
    drone_id: str = Field(..., description="ID of the ICEOUT drone")
    owner_id: str = Field(..., description="ID of the drone's owner")
    tier: int = Field(default=1, ge=1, le=3, description="NPC tier for scaling")
    drone_position: HexPosition = Field(..., description="Position of the ICEOUT drone")
    affected_combatant_ids: list[str] = Field(
        default_factory=list, description="IDs of combatants in the burst 1 zone"
    )
    zone_size: int = Field(default=1, ge=1, description="Size of the zone (burst)")
    is_moving: bool = Field(
        default=False, description="Whether this is a move action (not initial deploy)"
    )


class ICEOUTDroneResult(FrozenModel):
    """Result of ICEOUT Drone ability resolution."""

    ability_type: Literal["iceout"] = "iceout"
    success: bool = Field(..., description="Whether the ability succeeded")
    zone_created: bool = Field(..., description="Whether the zone was created/moved")
    zone_size: int = Field(default=1, description="Size of the zone (burst)")
    affected_combatant_ids: list[str] = Field(
        default_factory=list, description="IDs of combatants in the zone"
    )
    tech_immunity_granted: list[str] = Field(
        default_factory=list, description="Combatants granted tech immunity"
    )
    statuses_cleared: list[str] = Field(
        default_factory=list, description="Tech statuses cleared"
    )
    can_move: bool = Field(
        default=True, description="Drone can still be moved this turn"
    )
    reason: str = Field(default="", description="Explanation of the result")


class TrackingDroneInput(FrozenModel):
    """Input for Tracking Drone ability resolution per PR2 8778-8789."""

    ability_type: Literal["tracking"] = "tracking"
    drone_id: str = Field(..., description="ID of the tracking drone")
    owner_id: str = Field(..., description="ID of the drone's owner")
    tier: int = Field(default=1, ge=1, le=3, description="NPC tier for scaling")
    target_id: str = Field(..., description="ID of the target to track")
    shooter_id: str = Field(..., description="ID of the shooter")
    shooter_systems_bonus: int = Field(..., description="Systems bonus for tech attack")
    target_systems_save_bonus: int = Field(
        default=0, description="Systems save bonus for target (if applicable)"
    )
    force_roll: int | None = Field(
        default=None, ge=1, le=20, description="Forced d20 roll for testing"
    )
    base_dc: int = Field(default=12, description="Base DC for remove check")


class TrackingDroneResult(FrozenModel):
    """Result of Tracking Drone ability resolution."""

    ability_type: Literal["tracking"] = "tracking"
    success: bool = Field(..., description="Whether the ability succeeded")
    attack_roll: int | None = Field(default=None, description="d20 tech attack roll")
    attack_total: int | None = Field(default=None, description="Roll + bonus")
    hit: bool = Field(..., description="Whether the tech attack hit")
    information_revealed: dict[str, Any] = Field(
        default_factory=dict, description="Information revealed on hit"
    )
    hide_negated: bool = Field(
        default=False, description="Whether hide condition is negated"
    )
    invisibility_ignored: bool = Field(
        default=False, description="Whether invisibility is ignored"
    )
    drone_attached: bool = Field(
        ..., description="Whether the drone attached to target"
    )
    remove_dc: int = Field(default=12, description="DC for Engineering check to remove")
    reason: str = Field(default="", description="Explanation of the result")


class HiveDroneInput(FrozenModel):
    """Input for Hive Drone ability resolution per PR2 9745-9759."""

    ability_type: Literal["hive"] = "hive"
    drone_id: str = Field(..., description="ID of the Hive drone")
    owner_id: str = Field(..., description="ID of the drone's owner")
    tier: int = Field(default=1, ge=1, le=3, description="NPC tier for scaling")
    drone_position: HexPosition = Field(..., description="Position of the Hive drone")
    enemy_ids: list[str] = Field(
        default_factory=list, description="IDs of enemies potentially in the zone"
    )
    ally_ids: list[str] = Field(
        default_factory=list, description="IDs of allies potentially in the zone"
    )
    zone_size: int = Field(default=2, ge=1, description="Size of the zone (burst)")
    base_damage: int = Field(default=1, description="Base damage per enemy")
    is_move: bool = Field(
        default=False, description="Whether this is a move action (not initial deploy)"
    )


class HiveDroneResult(FrozenModel):
    """Result of Hive Drone ability resolution."""

    ability_type: Literal["hive"] = "hive"
    success: bool = Field(..., description="Whether the ability succeeded")
    zone_created: bool = Field(..., description="Whether the zone was created/moved")
    zone_size: int = Field(default=2, description="Size of the zone (burst)")
    allies_covered: list[str] = Field(
        default_factory=list, description="Allies receiving soft cover"
    )
    enemies_damaged: list[str] = Field(
        default_factory=list, description="Enemies taking damage"
    )
    soft_cover_granted: list[str] = Field(
        default_factory=list, description="Allies granted soft cover"
    )
    damage_per_target: int = Field(default=1, description="Damage per enemy target")
    damage_type: Literal["kinetic"] = Field(
        default="kinetic", description="Type of damage dealt"
    )
    can_move: bool = Field(
        default=True, description="Drone can still be moved this turn"
    )
    reason: str = Field(default="", description="Explanation of the result")


DroneAbilityInput = Union[
    TurretDroneInput,
    RestockDroneInput,
    LatchDroneInput,
    ICEOUTDroneInput,
    TrackingDroneInput,
    HiveDroneInput,
]

DroneAbilityResult = Union[
    TurretDroneResult,
    RestockDroneResult,
    LatchDroneResult,
    ICEOUTDroneResult,
    TrackingDroneResult,
    HiveDroneResult,
]


def get_damage_for_tier(base_damage: int, tier: int) -> int:
    """Get damage value scaled by NPC tier.

    Per PR2: Higher tier NPCs have more sophisticated gear with higher damage.
    Standard scaling: +2 damage per tier level.
    """
    return base_damage + (tier - 1) * 2


def get_dc_for_tier(base_dc: int, tier: int) -> int:
    """Get DC value scaled by NPC tier.

    Per PR2: Higher tier NPCs have more sophisticated gear with higher DCs.
    Standard scaling: +2 per tier level.
    """
    return base_dc + (tier - 1) * 2


def resolve_turret_drone(input: TurretDroneInput) -> TurretDroneResult:
    """Resolve Turret Drone reaction attack per PR2 7344-7358.

    "This system fires a turret drone that attaches to any object or surface within
    sensor range. While attached, you can make the following reaction once for each
    turret per round. When an allied character hits with an attack within range 10
    of a turret, you can cause the turret to deal 3 kinetic damage to that target."

    Tier scaling: +2 damage per tier (3 → 5 → 7).

    Returns what SHOULD happen - caller applies state changes.
    """
    if not input.ally_attack_hit:
        return TurretDroneResult(
            ability_type="turret",
            success=True,
            damage_dealt=False,
            damage_amount=0,
            target_ids=[],
            range_check=True,
            reason="Turret did not trigger: allied attack missed",
        )

    distance = input.ally_position.distance_2d(input.turret_position)
    within_range = distance <= input.turret_range

    if not within_range:
        return TurretDroneResult(
            ability_type="turret",
            success=True,
            damage_dealt=False,
            damage_amount=0,
            target_ids=[],
            range_check=False,
            reason=f"Turret did not trigger: target at distance {distance}, beyond range {input.turret_range}",
        )

    damage_amount = get_damage_for_tier(input.base_damage, input.tier)

    return TurretDroneResult(
        ability_type="turret",
        success=True,
        damage_dealt=True,
        damage_amount=damage_amount,
        damage_type="kinetic",
        target_ids=[input.target_id],
        range_check=True,
        reason=f"Turret dealt {damage_amount} kinetic damage to {input.target_id}",
    )


def resolve_restock_drone(input: RestockDroneInput) -> RestockDroneResult:
    """Resolve Restock Drone activation per PR2 7833-7843.

    "After your turn ends, the drone primes. Any allied character that moves adjacent
    to the drone or starts their turn adjacent to it can activate it as a quick action.
    That character can then: Cool 1d6 heat; Reload one weapon with the loading tag;
    End one condition affecting it. The drone is then consumed."

    Returns what SHOULD happen - caller applies state changes.
    """
    distance = input.activating_combatant_position.distance_2d(input.drone_position)
    is_adjacent = distance <= 1

    if not is_adjacent:
        return RestockDroneResult(
            ability_type="restock",
            success=False,
            can_activate=False,
            drone_consumed=False,
            reason=f"Cannot activate: combatant at distance {distance}, not adjacent",
        )

    if not input.is_primed:
        return RestockDroneResult(
            ability_type="restock",
            success=False,
            can_activate=False,
            drone_consumed=False,
            reason="Cannot activate: drone has not yet primed (primes after owner's turn ends)",
        )

    if input.action_choice == "cool":
        if input.force_cool_roll is not None:
            cool_roll = input.force_cool_roll
        else:
            cool_roll = roll_dice("1d6")

        return RestockDroneResult(
            ability_type="restock",
            success=True,
            heat_cooled=cool_roll,
            weapon_reloaded=False,
            condition_cleared=None,
            drone_consumed=True,
            can_activate=True,
            reason=f"Restock drone cooled {cool_roll} heat from activating combatant",
        )

    if input.action_choice == "reload":
        return RestockDroneResult(
            ability_type="restock",
            success=True,
            heat_cooled=None,
            weapon_reloaded=True,
            condition_cleared=None,
            drone_consumed=True,
            can_activate=True,
            reason="Restock drone reloaded one loading weapon for activating combatant",
        )

    if input.action_choice == "clear_condition":
        return RestockDroneResult(
            ability_type="restock",
            success=True,
            heat_cooled=None,
            weapon_reloaded=False,
            condition_cleared="any",
            drone_consumed=True,
            can_activate=True,
            reason="Restock drone cleared one condition from activating combatant",
        )

    return RestockDroneResult(
        ability_type="restock",
        success=False,
        can_activate=False,
        drone_consumed=False,
        reason=f"Unknown action choice: {input.action_choice}",
    )


def resolve_latch_drone(input: LatchDroneInput) -> LatchDroneResult:
    """Resolve Latch Drone ability per PR2 7813-7831.

    Mount Mode:
    "Make a ranged attack vs evasion 8 and target any friendly mech in range. On hit,
    you or your target can spend a repair to heal half your target's HP."

    Active Mode (requires 1 Core Power):
    "You fire your drone at a friendly mech in range, where it clamps onto the target.
    For the rest of this scene, you take 1 heat at the start of your turn, but the
    targeted mech gains: +1 Accuracy on all attacks/checks/saves; Immune to impaired,
    jammed, slowed, shredded, and immobilized conditions from characters other than
    itself. This effect ends early if you or the targeted mech is STUNNED."

    Tier scaling: +2 DC per tier (8 → 10 → 12).

    Returns what SHOULD happen - caller applies state changes.
    """
    if input.mode == "mount":
        if input.force_roll is not None:
            roll = input.force_roll
        else:
            roll = roll_dice("1d20")

        attack_total = roll + input.shooter_systems_bonus
        dc = get_dc_for_tier(input.base_attack_dc, input.tier)
        hit = attack_total >= dc

        if hit:
            hp_healed = 5  # Half of 10 HP, typical mech HP

            return LatchDroneResult(
                ability_type="latch",
                success=True,
                attack_roll=roll,
                attack_total=attack_total,
                hit=True,
                repair_spent=False,
                hp_healed=hp_healed,
                reason=f"Latch drone mount hit: {roll}+{input.shooter_systems_bonus} vs DC {dc} = Hit",
            )

        return LatchDroneResult(
            ability_type="latch",
            success=True,
            attack_roll=roll,
            attack_total=attack_total,
            hit=False,
            reason=f"Latch drone mount miss: {roll}+{input.shooter_systems_bonus} vs DC {dc} = Miss",
        )

    if input.mode == "active":
        if not input.has_core_power:
            return LatchDroneResult(
                ability_type="latch",
                success=False,
                hit=False,
                reason="Active mode requires 1 core power",
            )

        buffs = ["+1 Accuracy on attacks/checks/saves"]
        immunities = [
            "immune to impaired",
            "immune to jammed",
            "immune to slowed",
            "immune to shredded",
            "immune to immobilized",
        ]

        return LatchDroneResult(
            ability_type="latch",
            success=True,
            hit=True,
            core_power_spent=True,
            buffs_granted=buffs,
            immunities_granted=immunities,
            heat_to_owner=1,
            reason="Latch drone active: clamped to target, grants +1 Accuracy and condition immunities, owner takes 1 heat at turn start",
        )

    return LatchDroneResult(
        ability_type="latch",
        success=False,
        hit=False,
        reason=f"Unknown latch mode: {input.mode}",
    )


def resolve_iceout_drone(input: ICEOUTDroneInput) -> ICEOUTDroneResult:
    """Resolve ICEOUT Drone zone creation per PR2 8645-8658.

    "You fire an ICEOUT drone at a point in sensor range, where it hovers in place.
    Once fired, the drone creates a BURST 1 zone around itself. Any character at least
    partially covered by the zone is: Immune to all tech actions; Cannot make or benefit
    from any tech actions; Any negative statuses caused by tech actions immediately end.

    The drone deactivates at the end of the current scene or when destroyed, and cannot
    be re-used. You can move it again to a point in your sensor range as a quick action."

    Returns what SHOULD happen - caller applies state changes.
    """
    return ICEOUTDroneResult(
        ability_type="iceout",
        success=True,
        zone_created=True,
        zone_size=input.zone_size,
        affected_combatant_ids=input.affected_combatant_ids,
        tech_immunity_granted=input.affected_combatant_ids,
        statuses_cleared=[],
        can_move=not input.is_moving,
        reason=f"ICEOUT drone created burst {input.zone_size} zone affecting {len(input.affected_combatant_ids)} combatants",
    )


def resolve_tracking_drone(input: TrackingDroneInput) -> TrackingDroneResult:
    """Resolve Tracking Drone tech attack per PR2 8778-8789.

    "Make a tech attack against a target in your sensor range. On a hit, you know the
    target's exact location, HP, Structure, Heat, and speed, no matter where it is. It
    cannot hide and your attacks against it ignore invisibility until the drone is
    removed from them. It takes a quick action and a successful engineering check from
    the targeted mech to remove a tracking drone."

    Tier scaling: +2 DC per tier for remove check (12 → 14 → 16).

    Returns what SHOULD happen - caller applies state changes.
    """
    if input.force_roll is not None:
        roll = input.force_roll
    else:
        roll = roll_dice("1d20")

    attack_total = roll + input.shooter_systems_bonus
    remove_dc = get_dc_for_tier(input.base_dc, input.tier)

    hit = attack_total >= remove_dc

    if hit:
        return TrackingDroneResult(
            ability_type="tracking",
            success=True,
            attack_roll=roll,
            attack_total=attack_total,
            hit=True,
            information_revealed={
                "location": "exact",
                "hp": "visible",
                "structure": "visible",
                "heat": "visible",
                "speed": "visible",
            },
            hide_negated=True,
            invisibility_ignored=True,
            drone_attached=True,
            remove_dc=remove_dc,
            reason=f"Tracking drone attached: revealed target info, negates hide, ignores invisibility (remove DC {remove_dc})",
        )

    return TrackingDroneResult(
        ability_type="tracking",
        success=True,
        attack_roll=roll,
        attack_total=attack_total,
        hit=False,
        drone_attached=False,
        remove_dc=remove_dc,
        reason=f"Tracking drone miss: {roll}+{input.shooter_systems_bonus} vs DC {remove_dc}",
    )


def resolve_hive_drone(input: HiveDroneInput) -> HiveDroneResult:
    """Resolve Hive Drone zone creation per PR2 9745-9759.

    "You can fire this drone to an empty space in sensor range as a quick action. While
    it's active, it emits a BURST 2 area around it that grants: Soft cover to any allied
    mech at least partially covered by the zone; Any hostile target that starts its turn
    in the area or enters it for the first time on their turn takes 1 AP kinetic damage.

    You can move it to a different space in your sensor range by repeating this action."

    Tier scaling: +1 damage per tier (1 → 2 → 3).

    Returns what SHOULD happen - caller applies state changes.
    """
    damage_per_target = get_damage_for_tier(input.base_damage, input.tier)

    return HiveDroneResult(
        ability_type="hive",
        success=True,
        zone_created=True,
        zone_size=input.zone_size,
        allies_covered=input.ally_ids,
        enemies_damaged=input.enemy_ids,
        soft_cover_granted=input.ally_ids,
        damage_per_target=damage_per_target,
        damage_type="kinetic",
        can_move=not input.is_move,
        reason=f"Hive drone created burst {input.zone_size} zone: {len(input.ally_ids)} allies get soft cover, {len(input.enemy_ids)} enemies take {damage_per_target} kinetic on entry/start",
    )


def resolve_drone_ability(input: DroneAbilityInput) -> DroneAbilityResult:
    """Dispatch drone ability resolution to appropriate handler per PR2 5070-5088.

    Per PR2 drone rules:
    "Drones, unless otherwise noted, are allied characters that are size ½ and have
    evasion 10, 10 HP, and 0 armor. They can't take actions or move by default unless
    specified, and act on their owner's turn if they do have actions or movement."

    Returns what SHOULD happen - caller applies state changes.
    """
    if isinstance(input, TurretDroneInput):
        return resolve_turret_drone(input)
    if isinstance(input, RestockDroneInput):
        return resolve_restock_drone(input)
    if isinstance(input, LatchDroneInput):
        return resolve_latch_drone(input)
    if isinstance(input, ICEOUTDroneInput):
        return resolve_iceout_drone(input)
    if isinstance(input, TrackingDroneInput):
        return resolve_tracking_drone(input)
    if isinstance(input, HiveDroneInput):
        return resolve_hive_drone(input)

    return TurretDroneResult(
        ability_type="turret",
        success=False,
        damage_dealt=False,
        range_check=False,
        reason=f"Unknown drone ability type: {type(input)}",
    )
