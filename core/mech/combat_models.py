"""Combat execution input/output models.

This module contains the data models used for combat action inputs and results.
These are pure data containers with no combat logic dependencies.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from core.shared.models import FrozenModel
from core.shared.damage import DamageBreakdown
from core.shared.enums import ActionType, StatusType
from core.shared.full_tech import FullTechOptionSelection
from core.mech.grid import HexCoord, HexPosition
from core.mech.action_economy import ActionEconomyState
from core.mech.combat_state import ActionUse


# =============================================================================
# Type Aliases
# =============================================================================

StabilizePrimary = Literal["cool_heat", "spend_repair_full_hp", "cancel_meltdown"]
StabilizeSecondary = Literal["reload_loading", "clear_burn", "clear_condition"]


# =============================================================================
# Input Models
# =============================================================================


class ActionExecutionInput(FrozenModel):
    """Input for executing a combat action."""

    actor_id: str = Field(..., description="ID of the combatant taking action")
    action_id: str = Field(..., description="Action identifier (e.g., 'skirmish', 'barrage')")
    action_type: ActionType = Field(..., description="Type of action")
    target_ids: list[str] = Field(default_factory=list, description="Target combatant IDs")
    target_position: HexPosition | None = Field(default=None, description="Target position for area/movement")
    weapon_id: str | None = Field(default=None, description="Weapon being used")
    weapon_profile_id: str | None = Field(
        default=None,
        description="Weapon profile to use for weapons with multiple profiles (e.g., kinetic/energy/explosive)",
    )
    system_id: str | None = Field(default=None, description="System being activated")
    full_tech_first: FullTechOptionSelection | None = Field(
        default=None, description="First Full Tech option selection"
    )
    full_tech_second: FullTechOptionSelection | None = Field(
        default=None, description="Second Full Tech option selection"
    )
    movement_path: list[HexPosition] = Field(default_factory=list, description="Movement path for move actions")
    is_overcharge: bool = Field(default=False, description="Whether this action uses overcharge")
    granted_by_overcharge: bool = Field(default=False, description="Whether this action was granted by overcharge")
    # Stabilize options (PR2 4275-4286)
    stabilize_primary: StabilizePrimary | None = Field(
        default=None, description="Primary stabilize option: cool heat OR spend repair for full HP"
    )
    stabilize_secondary: StabilizeSecondary | None = Field(
        default=None, description="Secondary stabilize option: reload/clear burn/clear condition"
    )
    # Ram knockback preference
    apply_knockback: bool = Field(default=True, description="Whether to apply knockback on successful ram")
    # Thrown melee preference
    use_thrown: bool = Field(
        default=False,
        description="Whether to treat a melee weapon attack as thrown (uses thrown range, applies cover, disarms weapon)",
    )
    # Eject direction (for eject action)
    eject_direction: HexCoord | None = Field(
        default=None, description="Direction for eject (pilot flies 6 spaces in this direction)"
    )
    # Deploy action parameters (PR2 5070-5088)
    deploy_kind: Literal["drone", "mine", "deployable"] | None = Field(
        default=None, description="Kind of deployable to create"
    )
    deploy_name: str | None = Field(
        default=None, description="Name for the deployed entity"
    )
    mine_type: Literal["explosive", "shroud", "breaching", "cluster", "emp"] | None = Field(
        default=None, description="Type of mine being deployed"
    )


class ReactionInput(FrozenModel):
    """Input for declaring a reaction."""

    reactor_id: str = Field(..., description="ID of the reacting combatant")
    reaction_type: Literal["brace", "overwatch"] = Field(..., description="Type of reaction")
    trigger_action_id: str | None = Field(default=None, description="Action that triggered this reaction")
    target_ids: list[str] = Field(default_factory=list, description="Targets for reaction (e.g., overwatch)")
    weapon_id: str | None = Field(default=None, description="Weapon for overwatch attack")
    weapon_profile_id: str | None = Field(
        default=None,
        description="Weapon profile to use for weapons with multiple profiles",
    )


# =============================================================================
# Output Models
# =============================================================================


class ResourceChange(FrozenModel):
    """Change to a combatant's resources."""

    combatant_id: str
    hp_change: int = 0
    heat_change: int = 0
    structure_change: int = 0
    stress_change: int = 0
    repairs_change: int = 0


class AttackOutcome(FrozenModel):
    """Result of resolving a single attack.

    Used for both normal attacks and overwatch reactions to capture
    all relevant attack resolution details.
    """

    hit: bool = Field(..., description="Whether the attack hit")
    critical: bool = Field(default=False, description="Whether the attack was a critical hit")
    damage_dealt: int = Field(default=0, description="Total damage dealt to target")
    damage_breakdown: DamageBreakdown = Field(
        default_factory=DamageBreakdown, description="Net damage by type"
    )
    roll: int = Field(..., description="The d20 roll result")
    total: int = Field(..., description="Total attack value (roll + bonuses)")
    target_defense: int = Field(..., description="Target's defense value that was beaten")
    accuracy_bonus: int = Field(default=0, description="Net accuracy bonus applied")
    difficulty_bonus: int = Field(default=0, description="Net difficulty applied")
    effects: list[dict] = Field(
        default_factory=list, description="Effects applied (cover, status mods, etc.)"
    )
    resource_change: ResourceChange | None = Field(
        default=None, description="Resource change from damage"
    )
    structure_check: dict | None = Field(
        default=None, description="Structure check result if triggered"
    )


class OverwatchOpportunityInfo(FrozenModel):
    """Overwatch opportunity info in action result.

    Describes a detected overwatch opportunity when a combatant starts
    movement inside an enemy's weapon threat range. Per PR2 4395-4401,
    enemies can take overwatch reactions against targets starting movement
    in their threat range.
    """

    reactor_id: str = Field(..., description="ID of enemy who can react")
    weapon_id: str = Field(..., description="Weapon with threat covering position")
    weapon_threat: int = Field(..., ge=1, description="Threat range of the weapon")
    can_react: bool = Field(..., description="True if reactor has available reaction budget")
    prevention_reason: str | None = Field(
        default=None, description="Reason if reaction is prevented"
    )


class ActionExecutionResult(FrozenModel):
    """Result of executing a combat action."""

    success: bool = Field(..., description="Whether action executed successfully")
    error: str | None = Field(default=None, description="Error message if failed")
    action_use: ActionUse | None = Field(default=None, description="Recorded action for combat log")
    effects_applied: list[dict] = Field(default_factory=list, description="Effects that were applied")
    damage_dealt: int = Field(default=0, description="Total damage dealt")
    damage_breakdown: DamageBreakdown = Field(
        default_factory=DamageBreakdown, description="Net damage by type"
    )
    heat_generated: int = Field(default=0, description="Heat generated by this action")
    resource_changes: list[ResourceChange] = Field(
        default_factory=list, description="Resource changes to combatants"
    )
    statuses_applied: dict[str, list[StatusType]] = Field(
        default_factory=dict, description="Statuses applied to targets"
    )
    structure_checks: list[dict] = Field(
        default_factory=list,
        description="Structure check results triggered by damage"
    )
    overheat_checks: list[dict] = Field(
        default_factory=list,
        description="Overheat check results triggered by heat"
    )
    position_updates: dict[str, dict] = Field(
        default_factory=dict,
        description="Position changes keyed by combatant_id: {q, r}"
    )
    overwatch_opportunities: list[OverwatchOpportunityInfo] = Field(
        default_factory=list,
        description="Overwatch opportunities triggered at movement start"
    )


class TurnStartResult(FrozenModel):
    """Result of starting a combatant's turn."""

    actor_id: str = Field(..., description="ID of the actor whose turn started")
    actor_name: str = Field(..., description="Name of the actor")
    economy: ActionEconomyState = Field(..., description="Fresh action economy for this turn")
    available_actions: list[str] = Field(default_factory=list, description="Actions available to this actor")
    prepared_action_expired: bool = Field(default=False, description="Whether a prepared action expired")
    cooldowns_decremented: list[str] = Field(
        default_factory=list, description="Effect IDs whose cooldowns were decremented"
    )

    # Meltdown resolution
    meltdown_countdown_active: bool = Field(
        default=False,
        description="Whether combatant had active meltdown countdown"
    )
    meltdown_countdown_remaining: int | None = Field(
        default=None,
        description="Turns remaining after decrement (None if no countdown)"
    )
    meltdown_triggered: bool = Field(
        default=False,
        description="Whether meltdown explosion triggered this turn"
    )
    meltdown_explosion_damage: int = Field(
        default=0,
        description="Total damage from meltdown explosion (4d6)"
    )
    meltdown_affected_targets: list[str] = Field(
        default_factory=list,
        description="IDs of combatants damaged by meltdown explosion"
    )

    # Deployable/Drone turn processing (PR2 5070-5088)
    mines_armed: list[str] = Field(
        default_factory=list,
        description="IDs of mines that armed at start of this turn"
    )
    drone_heat_to_owner: int = Field(
        default=0,
        description="Heat from active latch drones applied to owner"
    )
    drones_ready_to_act: list[str] = Field(
        default_factory=list,
        description="IDs of drones that can act this turn"
    )


class BurnTickResult(FrozenModel):
    """Result of burn damage tick at end of turn."""

    target_id: str = Field(..., description="Combatant who took the burn tick")
    burn_amount: int = Field(..., description="Total burn marked before resolution")
    engineering_roll: int = Field(..., description="1d20 roll")
    engineering_bonus: int = Field(..., description="Engineering skill bonus")
    total: int = Field(..., description="Roll + bonus")
    dc: int = Field(default=10, description="Difficulty class (always 10)")
    success: bool = Field(..., description="Whether check succeeded")
    damage_taken: int = Field(default=0, description="Damage taken (0 if success)")
    burn_cleared: bool = Field(..., description="Whether burn was cleared")


class TurnEndResult(FrozenModel):
    """Result of ending a combatant's turn."""

    actor_id: str = Field(..., description="ID of the actor whose turn ended")
    next_actor_id: str | None = Field(default=None, description="ID of next actor, None if round ends")
    next_actor_name: str | None = Field(default=None, description="Name of next actor")
    round_advanced: bool = Field(default=False, description="Whether we advanced to a new round")
    new_round_number: int | None = Field(default=None, description="New round number if advanced")
    end_of_turn_effects: list[dict] = Field(default_factory=list, description="Effects applied at turn end")
    cooldowns_decremented: list[str] = Field(
        default_factory=list, description="Effect IDs whose cooldowns were decremented"
    )
    burn_tick_result: BurnTickResult | None = Field(
        default=None, description="Burn tick resolution if actor had burn"
    )

    # Deployable/Drone turn processing (PR2 5070-5088)
    drones_primed: list[str] = Field(
        default_factory=list,
        description="IDs of drones that primed at end of this turn"
    )


class ReactionResult(FrozenModel):
    """Result of declaring a reaction."""

    success: bool = Field(..., description="Whether reaction was valid")
    error: str | None = Field(default=None, description="Error message if failed")
    reaction_used: str | None = Field(default=None, description="Reaction type that was used")
    effects_applied: list[dict] = Field(default_factory=list, description="Effects from the reaction")
    damage_dealt: int = Field(default=0, description="Damage dealt by overwatch")
    damage_breakdown: DamageBreakdown = Field(
        default_factory=DamageBreakdown, description="Net damage by type"
    )
    # Extended fields for overwatch attack resolution
    attack_hit: bool | None = Field(default=None, description="Whether overwatch attack hit")
    attack_critical: bool | None = Field(default=None, description="Whether attack was critical")
    attack_roll: int | None = Field(default=None, description="The d20 roll for overwatch attack")
    resource_changes: list[ResourceChange] = Field(
        default_factory=list, description="Resource changes from overwatch attack"
    )
    structure_checks: list[dict] = Field(
        default_factory=list, description="Structure check results from overwatch damage"
    )


class AvailableAction(FrozenModel):
    """An action available to the current actor."""

    action_id: str = Field(..., description="Action identifier")
    action_name: str = Field(..., description="Display name")
    action_type: ActionType = Field(..., description="Action type (full/quick/free/reaction)")
    is_available: bool = Field(..., description="Whether action can be taken now")
    unavailable_reason: str | None = Field(default=None, description="Why action is unavailable")
    requires_target: bool = Field(default=False, description="Whether action needs a target")
    requires_weapon: bool = Field(default=False, description="Whether action needs a weapon")
    requires_system: bool = Field(default=False, description="Whether action needs a system")
    requires_path: bool = Field(default=False, description="Whether action needs a movement path")
    max_targets: int = Field(default=1, description="Maximum number of targets (e.g., 2 for barrage)")


class AvailableActionsResult(FrozenModel):
    """Available actions for the current actor."""

    actor_id: str = Field(..., description="Actor these actions are for")
    economy: ActionEconomyState = Field(..., description="Current action economy state")
    full_actions: list[AvailableAction] = Field(default_factory=list)
    quick_actions: list[AvailableAction] = Field(default_factory=list)
    free_actions: list[AvailableAction] = Field(default_factory=list)
    reactions: list[AvailableAction] = Field(default_factory=list)
    protocols: list[AvailableAction] = Field(default_factory=list)
    can_overcharge: bool = Field(default=True, description="Whether overcharge is available")


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Type aliases
    "StabilizePrimary",
    "StabilizeSecondary",
    # Input models
    "ActionExecutionInput",
    "ReactionInput",
    # Output models
    "ResourceChange",
    "AttackOutcome",
    "OverwatchOpportunityInfo",
    "ActionExecutionResult",
    "TurnStartResult",
    "BurnTickResult",
    "TurnEndResult",
    "ReactionResult",
    "AvailableAction",
    "AvailableActionsResult",
]
