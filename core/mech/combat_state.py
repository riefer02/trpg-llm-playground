"""Combat state models for mech combat."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel

from core.shared.enums import StatusType, SizeClass, ActionType, AttackType
from core.shared.effects import (
    MechanicalEffect,
    ModeEffect,
    ReactionTriggerEffect,
    ReactionTriggerEvent,
    ProgressionState,
    PerTargetCounter,
    CooldownState,
    ProgressionResetTrigger,
)
from core.shared.sitrep_resolution import SitrepResolution
from core.shared.decisions import PendingDecision
from core.shared.scenario import MissionObjective
from core.shared.campaign.campaign import ReservePlanEntry
from core.shared.combat.statistics import CombatStatistics
from core.shared.rolls import ContestedCheck
from core.shared.dice import DiceExpression
from core.mech.grid import HexPosition, HexCoord
from core.mech.terrain import TerrainMap
from core.mech.weapon import WeaponTagType
from core.mech.mounts import MountSlotType
from core.mech.combat_rules import AttackPatternDefinition
from core.mech.timing import PreparedActionState
from core.shared.id_helpers import (
    DeployableIdField,
    CombatantIdField,
    WeaponIdField,
    SystemIdField,
    EffectIdField,
    ActionIdField,
    NpcIdField,
)

if TYPE_CHECKING:
    from core.shared.heat import MeltdownState
    from core.shared.protocols import ProtocolState
    from core.shared.turn_end import TurnEndEffectState
    from core.mech.statuses import StatusInstance
    from core.shared.flying import FlyingStatus


CombatSide = Literal["players", "hostiles", "neutral"]
CombatantKind = Literal["mech", "pilot", "npc", "object"]
CombatEnvironment = Literal["standard", "zero_g", "underwater"]
DeployableKind = Literal["drone", "mine", "deployable", "other"]


class DeployableState(FrozenModel):
    """Trackable deployable object in combat.

    Models drones, mines, and other deployable objects per PR2 rules.
    Drones: size ½, evasion 10, HP 10, armor 0, act on owner's turn
    Mines: arm at start of next turn, trigger on adjacent entry
    Deployables: 10 HP/size, evasion 5, default armor 0
    """

    id: DeployableIdField
    name: str
    kind: DeployableKind
    owner_id: CombatantIdField | None = None
    position: HexPosition
    size: int = Field(..., ge=1)
    hp: int = Field(..., ge=0)
    max_hp: int = Field(..., ge=1)
    armor: int = Field(default=0, ge=0)
    evasion: int = Field(default=5, ge=0)
    cover: Literal["soft", "hard"] | None = None
    is_destroyed: bool = False
    is_active: bool = True
    can_act: bool = False
    can_move: bool = False
    acts_on_owner_turn: bool = True
    is_armed: bool = False
    arming_turn: int | None = None
    trigger_on_adjacent_entry: bool = False
    detection_dc: int | None = None
    disarm_dc: int | None = None
    e_defense: int = Field(default=10, ge=0)
    reactions: list[str] = Field(default_factory=list)


class CombatStats(FrozenModel):
    """Combat-relevant stats for a combatant."""

    size: SizeClass
    hp_max: int = Field(..., ge=0)
    evasion: int = Field(..., ge=0)
    e_defense: int = Field(..., ge=0)
    armor: int = Field(default=0, ge=0)
    speed: int = Field(default=0, ge=0)
    sensor_range: int = Field(default=0, ge=0)
    tech_attack: int = Field(default=0)
    grit: int = Field(default=0, ge=0)
    engineering_skill: int = Field(
        default=0, ge=0, le=6, description="Engineering skill for saves"
    )


class CombatResources(FrozenModel):
    """Resource tracks for a combatant."""

    hp_current: int = Field(..., ge=0)
    heat_current: int = Field(default=0, ge=0)
    heat_cap: int = Field(default=0, ge=0)
    structure_current: int = Field(default=0, ge=0)
    stress_current: int = Field(default=0, ge=0)
    repairs_remaining: int = Field(default=0, ge=0)
    burn_marked: int = Field(default=0, ge=0, description="Accumulated burn damage")


class WeaponState(FrozenModel):
    """Weapon state for a mounted weapon."""

    weapon_id: WeaponIdField
    tags: list[WeaponTagType] = Field(default_factory=list)
    destroyed: bool = False
    limited_charges_remaining: int | None = Field(default=None, ge=0)
    needs_reload: bool = Field(
        default=False,
        description="True after firing a loading weapon, cleared by Stabilize",
    )
    thrown_coord: HexCoord | None = Field(
        default=None,
        description="Coordinate where the weapon was thrown/dropped (None if carried)",
    )


class WeaponMountState(FrozenModel):
    """Mount slot state and installed weapons."""

    mount_index: int = Field(..., ge=0)
    slot_type: MountSlotType | None = None
    weapons: list[WeaponState] = Field(default_factory=list)
    destroyed: bool = False


class MechSystemState(FrozenModel):
    """System state for a mech."""

    system_id: SystemIdField
    destroyed: bool = False
    limited_charges_remaining: int | None = Field(default=None, ge=0)


class MechInventory(FrozenModel):
    """Inventory state for mounts and systems."""

    mounts: list[WeaponMountState] = Field(default_factory=list)
    systems: list[MechSystemState] = Field(default_factory=list)


class OverchargeState(FrozenModel):
    """Tracks overcharge escalation state for a combatant.

    Overcharge costs escalate: 1 heat, then 1d3, 1d6, 1d6+4
    Resets on full repair per PR2 rules.
    """

    current_level: int = Field(default=0, ge=0, le=3)
    uses_this_turn: int = Field(default=0, ge=0)

    @property
    def can_overcharge(self) -> bool:
        """Check if overcharge is available this turn (once per turn)."""
        return self.uses_this_turn < 1

    @property
    def next_cost(self) -> int | DiceExpression:
        """Get the heat cost for the next overcharge."""
        from core.mech.rules import DEFAULT_OVERCHARGE_RULES

        rules = DEFAULT_OVERCHARGE_RULES
        return rules.costs[self.current_level]


class CombatantState(FrozenModel):
    """State for a combatant in mech combat.

    For NPCs, use the companion `core.npc` module to create NPCState
    and convert to CombatantState using the helper functions.
    """

    id: CombatantIdField
    name: str
    side: CombatSide
    kind: CombatantKind
    stats: CombatStats
    resources: CombatResources
    position: HexPosition | None = None
    statuses: list[StatusType] = Field(default_factory=list)
    conditions: list[StatusType] = Field(default_factory=list)
    status_instances: list["StatusInstance"] = Field(
        default_factory=list,
        description="Tracked status instances with duration metadata for automatic expiration",
    )
    inventory: MechInventory | None = None
    ai_controlled: bool = False
    ai_type: Literal["compcon", "nhp", "llm"] | None = Field(
        default=None, description="Type of AI installed (compcon, nhp, or llm)"
    )
    ai_control_state: Literal["pilot", "cede", "cede_remote", "unshackled"] = Field(
        default="pilot", description="Current control state of the mech"
    )
    nhp_behavior: (
        Literal["ignore_pilot", "overrule_pilot", "illogical", "remove_pilot"] | None
    ) = Field(default=None, description="NHP behavior when unshackled")
    unshackle_count: int = Field(
        default=0, ge=0, description="Number of times NHP has unshackled"
    )
    npc_role: Literal["striker", "defender", "controller", "supporter"] | None = Field(
        default=None,
        description="NPC role for AI behavior selection",
    )
    cede_turns_remaining: int = Field(
        default=0, ge=0, description="Turns remaining in cede state"
    )
    active_mode_effects: list[ModeEffect] = Field(default_factory=list)
    reaction_triggers: list[ReactionTriggerEffect] = Field(default_factory=list)
    progression_states: dict[str, ProgressionState] = Field(default_factory=dict)
    per_target_counters: dict[str, PerTargetCounter] = Field(default_factory=dict)
    cooldown_states: dict[str, CooldownState] = Field(default_factory=dict)
    prepared_action: PreparedActionState | None = Field(
        default=None, description="Held prepared action, if any"
    )
    per_round_reactions: dict[str, int] = Field(
        default_factory=dict, description="Per-round reaction usage {action_id: count}"
    )
    dangerous_terrain_last_check_round: int | None = Field(
        default=None,
        ge=1,
        description="Last round this combatant resolved a dangerous terrain check",
    )
    overcharge_state: OverchargeState | None = Field(
        default=None, description="Overcharge escalation state"
    )
    meltdown_state: "MeltdownState | None" = Field(
        default=None, description="Active meltdown countdown state, if any"
    )
    active_protocols: dict[str, "ProtocolState"] = Field(
        default_factory=dict, description="Active protocols by protocol_id"
    )
    turn_end_effects: dict[str, "TurnEndEffectState"] = Field(
        default_factory=dict, description="Effects expiring at turn boundaries"
    )
    # Pilot talent effects (Phase 32)
    talent_effects: list[MechanicalEffect] = Field(
        default_factory=list,
        description="Aggregated mechanical effects from pilot talents",
    )
    # Frame trait and core power effects (Phase 33)
    frame_trait_effects: list[MechanicalEffect] = Field(
        default_factory=list,
        description="Passive mechanical effects from frame traits",
    )
    core_power_available: bool = Field(
        default=True,
        description="Whether core power can be activated (once per mission)",
    )
    core_power_active: bool = Field(
        default=False, description="Whether core power is currently active"
    )
    core_power_effects: MechanicalEffect | None = Field(
        default=None, description="Active effects from core power when activated"
    )
    # Mount/Dismount/Eject state
    piloting_mech_id: str | None = Field(
        default=None, description="ID of mech this pilot is piloting (pilot only)"
    )
    mounted_pilot_id: str | None = Field(
        default=None, description="ID of pilot mounted in this mech (mech only)"
    )
    eject_used: bool = Field(
        default=False, description="Whether eject has been used this combat (mech only)"
    )
    # Flying state tracking (Phase 52)
    flying_status: "FlyingStatus | None" = Field(
        default=None, description="Current flying status (altitude, hover mode, etc.)"
    )
    falling_from_altitude: int | None = Field(
        default=None,
        ge=0,
        description="If falling, the altitude level falling from (for damage calc)",
    )

    def in_danger_zone(
        self, danger_zone_fraction: float = 0.5, rounding: Literal["up", "down"] = "up"
    ) -> bool:
        """Check if mech is in danger zone based on current heat.

        Per PR2: When a mech has 1/2 of its total heat capacity filled (rounded up),
        it's in the danger zone.

        Args:
            danger_zone_fraction: Fraction of heat cap for danger zone (default 0.5)
            rounding: How to round the threshold ("up" or "down")

        Returns:
            True if mech is in danger zone
        """
        if self.resources.heat_current is None or self.resources.heat_cap <= 0:
            return False

        if rounding == "up":
            threshold = int((self.resources.heat_cap * danger_zone_fraction) + 0.999)
        else:
            threshold = int(self.resources.heat_cap * danger_zone_fraction)

        return self.resources.heat_current >= threshold


class GrappleLink(FrozenModel):
    """Link between grappling combatants."""

    grappler_id: CombatantIdField
    target_id: CombatantIdField
    grappler_total_size: int = Field(default=1, ge=0)
    target_total_size: int = Field(default=1, ge=0)


PerTargetEffectSource = Literal["save_check", "triggered_effect", "direct"]


class AppliedPerTargetEffect(FrozenModel):
    """Applied per-target effect metadata for action resolution."""

    effect_id: EffectIdField
    target_id: CombatantIdField
    count: int = Field(default=1, ge=1)
    max_count: int | None = Field(default=None, ge=1)
    reset_on: ProgressionResetTrigger | None = None
    source: PerTargetEffectSource = "direct"


ActionLogEffectType = Literal[
    "weapon_thrown",
    "retrieve_thrown_weapon",
    "status_applied",
]


class ActionLogEffect(FrozenModel):
    """Small effect summary for combat log display."""

    type: ActionLogEffectType
    status: StatusType | None = None
    target_id: CombatantIdField | None = None
    weapon_id: WeaponIdField | None = None


class ActionUse(FrozenModel):
    """An action taken during a combat turn."""

    action_id: ActionIdField
    action_type: ActionType
    target_id: CombatantIdField | None = None
    target_position: HexPosition | None = None
    target_ids: list[str] = Field(default_factory=list)
    target_positions: list[HexPosition] = Field(default_factory=list)
    range_spaces: int | None = Field(default=None, ge=0)
    attack_type_override: AttackType | None = None
    weapon_tags: list[WeaponTagType] = Field(default_factory=list)
    area_pattern: AttackPatternDefinition | None = None
    area_origin: HexPosition | None = None
    area_direction: HexCoord | None = None
    area_affected: list[HexCoord] = Field(default_factory=list)
    weapon_count: int | None = Field(default=None, ge=0)
    uses_superheavy: bool | None = None
    uses_aux_bonus_attack: bool | None = None
    stabilize_primary: Literal["cool_heat", "spend_repair_full_hp"] | None = None
    stabilize_secondary: (
        Literal["reload_loading", "clear_burn", "clear_condition"] | None
    ) = None
    ignores_line_of_sight: bool = False
    ignores_cover: bool = False
    used_as_free_action: bool = False
    used_as_reaction: bool = False
    granted_by_overcharge: bool = False
    heat_generated: int | None = Field(default=None, ge=0)
    reaction_trigger: ReactionTriggerEvent | None = None
    contested_check: ContestedCheck | None = None
    consumes_lock_on: bool = False
    applied_per_target_effects: list[AppliedPerTargetEffect] = Field(
        default_factory=list
    )
    log_effects: list[ActionLogEffect] = Field(default_factory=list)


class CombatTurn(FrozenModel):
    """A single combat turn."""

    actor_id: CombatantIdField
    move_used: bool = False
    movement_mode: Literal["ground", "flight", "hover", "teleport"] = "ground"
    movement_path: list[HexPosition] = Field(default_factory=list)
    actions: list[ActionUse] = Field(default_factory=list)
    has_moved_or_acted: bool = Field(
        default=False,
        description="True after any non-protocol action/movement (blocks ordnance)",
    )


class CombatRound(FrozenModel):
    """A combat round."""

    round_index: int = Field(..., ge=1)
    turns: list[CombatTurn] = Field(default_factory=list)
    reaction_counts_by_actor: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="Per-round reaction tracking {actor_id: {action_id: count}}",
    )


class MechCombatScenario(FrozenModel):
    """Full combat scenario for evaluation."""

    combatants: list[CombatantState] = Field(default_factory=list)
    grapples: list[GrappleLink] = Field(default_factory=list)
    rounds: list[CombatRound] = Field(default_factory=list)
    terrain: TerrainMap | None = None
    environment: CombatEnvironment = "standard"
    deployables: dict[str, DeployableState] = Field(
        default_factory=dict, description="Trackable deployables {deployable_id: state}"
    )
    # Phase 34: SITREP mission state tracking
    sitrep_resolution: SitrepResolution | None = Field(
        default=None,
        description="Active SITREP mission state tracking (zones, scores, victory conditions)",
    )
    # Pending player decisions (saves, system trauma selection)
    pending_decisions: list[PendingDecision] = Field(
        default_factory=list,
        description="Pending decisions awaiting player input (saves, trauma selection)",
    )
    # Mission objectives from campaign lobby (for display during combat)
    objectives: list[MissionObjective] = Field(
        default_factory=list,
        description="Mission objectives from campaign lobby for tracking during combat",
    )
    # Mission reserves from campaign lobby (available for spending during combat)
    mission_reserves: list[ReservePlanEntry] = Field(
        default_factory=list,
        description="Reserves from campaign lobby available for spending during combat",
    )
    # Combat statistics tracking for mission debrief
    statistics: CombatStatistics = Field(
        default_factory=CombatStatistics,
        description="Combat statistics tracking for post-mission debriefing",
    )


def create_npc_combatant(
    npc_id: NpcIdField,
    npc_name: str,
    npc_side: CombatSide,
    npc_size: str,
    npc_hp: int,
    npc_evasion: int,
    npc_e_defense: int,
    npc_armor: int,
    npc_speed: int,
    npc_sensor_range: int,
    npc_structure: int,
    npc_tech_attack: int = 0,
) -> CombatantState:
    """Create a CombatantState for an NPC.

    This is a simple helper for creating NPC combatants without
    using the full NPC template system. For more complex NPCs,
    use the core.npc module.

    Args:
        npc_id: Unique identifier for this NPC
        npc_name: Display name for this NPC
        npc_side: Which side the NPC is on ("players", "hostiles", "neutral")
        npc_size: Size class ("size_1", "size_2", etc.)
        npc_hp: Maximum HP
        npc_evasion: Evasion value
        npc_e_defense: E-Defense value
        npc_armor: Armor value
        npc_speed: Speed value
        npc_sensor_range: Sensor range
        npc_structure: Structure points (1 for T1, 2 for T2, 3 for T3)
        npc_tech_attack: Tech attack bonus (default 0)

    Returns:
        A new CombatantState suitable for use in combat
    """

    return CombatantState(
        id=npc_id,
        name=npc_name,
        side=npc_side,
        kind="npc",
        stats=CombatStats(
            size=npc_size,  # type: ignore - SizeClass is a Literal type
            hp_max=npc_hp,
            evasion=npc_evasion,
            e_defense=npc_e_defense,
            armor=npc_armor,
            speed=npc_speed,
            sensor_range=npc_sensor_range,
            tech_attack=npc_tech_attack,
        ),
        resources=CombatResources(
            hp_current=npc_hp,
            structure_current=npc_structure,
        ),
    )


# Rebuild CombatantState to resolve forward references
# This must be done after CombatantState is defined and types are available
try:
    from core.shared.heat import MeltdownState
    from core.shared.protocols import ProtocolState
    from core.shared.turn_end import TurnEndEffectState
    from core.mech.statuses import StatusInstance
    from core.shared.flying import FlyingStatus

    CombatantState.model_rebuild(
        _types_namespace={
            "MeltdownState": MeltdownState,
            "ProtocolState": ProtocolState,
            "TurnEndEffectState": TurnEndEffectState,
            "StatusInstance": StatusInstance,
            "FlyingStatus": FlyingStatus,
        }
    )
except ImportError:
    pass  # Types not yet available during initial import
