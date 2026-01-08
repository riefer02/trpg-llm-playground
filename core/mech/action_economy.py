"""Action economy primitives for mech combat.

Provides:
- Overcharge orchestration with heat cost escalation
- Action economy state tracking per turn
- Reaction eligibility determination
- Phase-integrated action validation
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel

from core.shared.enums import ActionType
from core.shared.dice import DiceExpression, roll_dice

if TYPE_CHECKING:
    from core.mech.combat_state import CombatantState, OverchargeState
    from core.shared.combat.phased_tracker import PhasedTacticalTracker


OverchargeLevel = Literal[0, 1, 2, 3]


class OverchargeCostResult(FrozenModel):
    """Result of computing overcharge heat cost."""

    level: OverchargeLevel = Field(..., description="Escalation level")
    base_cost: int | DiceExpression = Field(..., description="Cost before modifiers")
    modified_cost: int = Field(..., ge=0, description="Final heat cost after modifiers")
    roll_result: int | None = Field(
        default=None, description="Dice roll result if applicable"
    )


class ActionEconomyState(FrozenModel):
    """Tracks action usage for a single turn.

    Per PR2 3726-3728:
    - 1 move + (2 quick OR 1 full) + any free + any reactions
    - Overcharge: 1/turn, gains 1 quick action
    - Reactions: 1/turn, any number/round, reset at round boundary
    """

    full_actions_used: int = Field(default=0, ge=0)
    quick_actions_used: int = Field(default=0, ge=0)
    overcharge_used: bool = False
    reactions_used_this_turn: int = Field(default=0, ge=0)

    @property
    def full_actions_remaining(self) -> int:
        """Full actions remaining this turn (max 1)."""
        return 1 - self.full_actions_used

    @property
    def quick_actions_remaining(self) -> int:
        """Quick actions remaining this turn (max 2, +1 if overcharged)."""
        base = 2 - self.quick_actions_used
        return max(0, base)

    @property
    def can_overcharge(self) -> bool:
        """Check if overcharge is available this turn."""
        return not self.overcharge_used

    @property
    def reactions_remaining_this_turn(self) -> int:
        """Reactions remaining this turn (max 1)."""
        return max(0, 1 - self.reactions_used_this_turn)


class ActionEconomyResult(FrozenModel):
    """Result of action economy validation."""

    can_take_action: bool = Field(..., description="Whether action can be taken")
    can_take_full_action: bool = Field(
        ..., description="Whether full action is available"
    )
    can_take_quick_action: bool = Field(
        ..., description="Whether quick action is available"
    )
    can_overcharge: bool = Field(..., description="Whether overcharge is available")
    can_take_reaction: bool = Field(..., description="Whether reaction can be taken")
    errors: list[str] = Field(
        default_factory=list, description="Blocking errors if any"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Non-blocking warnings"
    )


class OverchargeInput(FrozenModel):
    """Input for overcharge resolution."""

    combatant_id: str = Field(..., description="Actor overcharging")
    combatant_state: "CombatantState" = Field(
        ..., description="Current combatant state"
    )
    overcharge_state: "OverchargeState | None" = Field(
        default=None, description="Overcharge escalation state"
    )
    rules_override: dict | None = Field(
        default=None, description="Optional rules overrides"
    )


class OverchargeResult(FrozenModel):
    """Result of overcharge resolution."""

    success: bool = Field(..., description="Whether overcharge succeeded")
    heat_cost: int = Field(..., ge=0, description="Heat incurred")
    escalation_level: OverchargeLevel = Field(
        ..., description="Escalation level after use"
    )
    granted_action_type: Literal["quick"] = Field(
        default="quick", description="Action type granted by overcharge"
    )
    new_overcharge_state: "OverchargeState | None" = Field(
        ..., description="Updated overcharge state"
    )
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ReactionEligibilityInput(FrozenModel):
    """Input for reaction eligibility determination."""

    tracker: "PhasedTacticalTracker" = Field(..., description="Current combat tracker")
    actor_id: str = Field(..., description="Actor checking reactions")
    trigger_event: str | None = Field(
        default=None, description="Specific trigger event (e.g., 'enemy_attack')"
    )


class AvailableReaction(FrozenModel):
    """A reaction that is available to be taken."""

    reaction_id: str = Field(..., description="Reaction identifier")
    reaction_name: str = Field(..., description="Display name")
    per_round_remaining: int = Field(..., ge=0, description="Uses remaining this round")
    per_turn_remaining: int = Field(..., ge=0, description="Uses remaining this turn")
    is_trigger_conditional: bool = Field(
        default=False, description="Requires specific trigger to activate"
    )
    trigger_condition: str | None = Field(
        default=None, description="Trigger requirement"
    )


class ReactionEligibilityResult(FrozenModel):
    """Result of reaction eligibility determination."""

    actor_id: str = Field(..., description="Actor checked")
    current_phase: str = Field(..., description="Current turn phase")
    total_reactions_remaining: int = Field(
        ..., ge=0, description="Total reactions available now"
    )
    available_reactions: list[AvailableReaction] = Field(
        default_factory=list, description="List of available reactions"
    )
    blocked_reason: str | None = Field(
        default=None, description="Why no reactions available"
    )


def compute_overcharge_cost(
    level: OverchargeLevel,
    cost_cap: int | None = None,
) -> OverchargeCostResult:
    """Compute the heat cost for overcharge at a given escalation level.

    Per PR2 4372-4376:
    - 1st use: 1 heat
    - 2nd use: 1d3 heat
    - 3rd use: 1d6 heat
    - 4th use: 1d6+4 heat

    Args:
        level: Current escalation level (0-3)
        cost_cap: Optional maximum cost cap

    Returns:
        OverchargeCostResult with cost details
    """
    from core.mech.rules import DEFAULT_OVERCHARGE_RULES

    rules = DEFAULT_OVERCHARGE_RULES
    base_cost = rules.costs[level]

    roll_result = None
    modified_cost = 0

    if isinstance(base_cost, int):
        modified_cost = base_cost
    elif isinstance(base_cost, DiceExpression):
        roll_result = roll_dice(str(base_cost))
        modified_cost = roll_result

    if cost_cap is not None:
        modified_cost = min(modified_cost, cost_cap)

    return OverchargeCostResult(
        level=level,
        base_cost=base_cost,
        modified_cost=modified_cost,
        roll_result=roll_result,
    )


def resolve_overcharge(
    input_data: OverchargeInput,
) -> OverchargeResult:
    """Resolve an overcharge action.

    Per PR2 4372-4376:
    - Overcharge is a free action
    - Can only overcharge once per turn
    - Gains one additional quick action
    - Heat cost escalates with repeated use

    Args:
        input_data: Overcharge input with combatant state

    Returns:
        OverchargeResult with outcome details
    """
    from core.mech.combat_state import OverchargeState
    from core.mech.rules import DEFAULT_OVERCHARGE_RULES

    rules = DEFAULT_OVERCHARGE_RULES
    errors = []
    warnings = []

    combatant = input_data.combatant_state
    overcharge_state = input_data.overcharge_state

    if overcharge_state is None:
        overcharge_state = OverchargeState()

    if not overcharge_state.can_overcharge:
        errors.append("Overcharge already used this turn")
        return OverchargeResult(
            success=False,
            heat_cost=0,
            escalation_level=overcharge_state.current_level,
            granted_action_type="quick",
            new_overcharge_state=overcharge_state,
            errors=errors,
            warnings=warnings,
        )

    cost_result = compute_overcharge_cost(overcharge_state.current_level)
    new_heat = combatant.resources.heat_current + cost_result.modified_cost
    new_level = min(overcharge_state.current_level + 1, 3)  # Cap at level 3

    new_overcharge_state = OverchargeState(
        current_level=new_level,
        uses_this_turn=overcharge_state.uses_this_turn + 1,
    )

    if new_heat > combatant.resources.heat_cap:
        warnings.append(
            f"Overcharge will exceed heat cap ({new_heat}/{combatant.resources.heat_cap})"
        )

    return OverchargeResult(
        success=True,
        heat_cost=cost_result.modified_cost,
        escalation_level=new_level,
        granted_action_type="quick",
        new_overcharge_state=new_overcharge_state,
        errors=errors,
        warnings=warnings,
    )


def validate_action_economy(
    economy: ActionEconomyState,
    action_type: ActionType,
    is_overcharge: bool = False,
) -> ActionEconomyResult:
    """Validate if an action can be taken given current economy state.

    Args:
        economy: Current action economy state
        action_type: Type of action being taken
        is_overcharge: Whether this is an overcharge action

    Returns:
        ActionEconomyResult with validation outcome
    """
    errors = []
    warnings = []

    can_take_full = economy.full_actions_remaining > 0
    can_take_quick = economy.quick_actions_remaining > 0
    can_overcharge = economy.can_overcharge
    can_react = economy.reactions_remaining_this_turn > 0

    if action_type == "full":
        can_take = can_take_full
        if not can_take_full:
            errors.append("Full action already used this turn")
    elif action_type == "quick":
        can_take = can_take_quick
        if not can_take_quick:
            errors.append("Quick actions exhausted (max 2)")
    elif action_type == "free":
        can_take = True
    elif action_type == "reaction":
        can_take = can_react
        if not can_react:
            errors.append("Reaction already used this turn")
    else:
        can_take = True

    if is_overcharge:
        if not can_overcharge:
            errors.append("Overcharge already used this turn")
        can_take = can_overcharge

    return ActionEconomyResult(
        can_take_action=len(errors) == 0,
        can_take_full_action=can_take_full,
        can_take_quick_action=can_take_quick,
        can_overcharge=can_overcharge,
        can_take_reaction=can_react,
        errors=errors,
        warnings=warnings,
    )


def use_full_action(economy: ActionEconomyState) -> ActionEconomyState:
    """Use a full action, incrementing the counter."""
    return economy.model_copy(
        update={"full_actions_used": economy.full_actions_used + 1}
    )


def use_quick_action(economy: ActionEconomyState) -> ActionEconomyState:
    """Use a quick action, incrementing the counter."""
    return economy.model_copy(
        update={"quick_actions_used": economy.quick_actions_used + 1}
    )


def use_overcharge(economy: ActionEconomyState) -> ActionEconomyState:
    """Mark overcharge as used."""
    return economy.model_copy(update={"overcharge_used": True})


def use_reaction(economy: ActionEconomyState) -> ActionEconomyState:
    """Use a reaction, incrementing the turn counter."""
    return economy.model_copy(
        update={"reactions_used_this_turn": economy.reactions_used_this_turn + 1}
    )


def reset_economy_for_new_turn(economy: ActionEconomyState) -> ActionEconomyState:
    """Reset action economy for a new turn (keep per-round reaction limits)."""
    return ActionEconomyState(
        full_actions_used=0,
        quick_actions_used=0,
        overcharge_used=False,
        reactions_used_this_turn=0,
    )


def get_action_economy_summary(economy: ActionEconomyState) -> dict:
    """Get a summary of current action economy for display/debugging."""
    return {
        "full_actions": f"{economy.full_actions_used}/1",
        "quick_actions": f"{economy.quick_actions_used}/2",
        "overcharge": "used" if economy.overcharge_used else "available",
        "reactions_this_turn": f"{economy.reactions_used_this_turn}/1",
        "can_take_full": economy.full_actions_remaining > 0,
        "can_take_quick": economy.quick_actions_remaining > 0,
        "can_overcharge": economy.can_overcharge,
        "can_react": economy.reactions_remaining_this_turn > 0,
    }
