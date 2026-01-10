"""Status and condition effects.

Effects for granting, clearing, and managing status conditions.

Effects:
    - StatusGrant: Grants a status or condition
    - StatusClear: Removes a status or condition
    - StatusToggleEffect: Toggleable status at specific timing windows
    - StatusBreakCondition: Breaks a status at specific conditions
    - StatusStackLimit: Limits how many times a status can stack
    - MovementScopedStatus: Status that affects specific movement types
    - StatusRestriction: Restricts specific statuses
    - StatusActionOverrideEffect: Overrides action when statuses are present
    - StatusTrigger: Triggers effects when statuses change
    - AllegianceShiftEffect: Temporarily flips a target's allegiance

See Also:
    - PR2 3985-4012: Conditions and statuses
"""

from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from pydantic import Field

from core.shared.models import FrozenModel
from core.shared.enums import ActionType, AttackType, StatusType
from core.shared.effects.types import (
    ActionCategoryType,
    BreakTriggerType,
    EffectDuration,
    EffectTarget,
    EffectTargetNoAll,
    MovementMode,
    TriggerType,
    UsesPer,
)
from core.shared.effects.conditions import EffectCondition

if TYPE_CHECKING:
    from core.shared.effects.core import MechanicalEffect

__all__ = [
    "StatusGrant",
    "StatusClear",
    "StatusToggleEffect",
    "StatusBreakCondition",
    "StatusStackLimit",
    "MovementScopedStatus",
    "StatusRestriction",
    "StatusActionOverrideEffect",
    "StatusTrigger",
    "AllegianceShiftEffect",
]


class StatusToggleEffect(FrozenModel):
    """
    Toggleable status at specific timing windows.

    Examples:
        StatusToggleEffect(status="invisible", activation_action="protocol",
                           activation_timing="on_turn_start",
                           deactivation_action="protocol",
                           deactivation_timing="on_turn_start",
                           duration="end_of_next_turn")
    """

    status: StatusType
    target: EffectTargetNoAll = "self"
    activation_action: ActionType = "protocol"
    activation_timing: TriggerType = "on_turn_start"
    deactivation_action: ActionType | None = None
    deactivation_timing: TriggerType | None = None
    duration: EffectDuration | None = None
    condition: EffectCondition | None = None


class StatusGrant(FrozenModel):
    """
    Grants or inflicts a status condition.

    Examples:
        StatusGrant(status="invisible", target="self", trigger="after_move_8_plus")
        StatusGrant(status="prone", target="enemy", trigger="on_hit")
    """

    status: StatusType | Literal["invisible", "flying"]
    target: EffectTarget
    trigger: TriggerType | str | None = Field(default=None)
    condition: EffectCondition | None = None
    duration: Literal[
        "end_of_turn",
        "start_of_next_turn",
        "end_of_next_turn",
        "until_cleared",
        "until_attack",
        "match_trigger",
        "scene",
    ] = "end_of_turn"


class StatusClear(FrozenModel):
    """
    Clears a status or condition from a target.

    Examples:
        StatusClear(status="burn", target="ally")
        StatusClear(status="any", target="self")
    """

    status: StatusType | Literal["burn", "any", "tech"]
    target: EffectTarget
    count: int = Field(default=1, ge=1)


class StatusBreakCondition(FrozenModel):
    """
    Defines triggers that end a status early.

    Examples:
        StatusBreakCondition(status="invisible", break_triggers=["move", "reaction"])
    """

    status: StatusType
    target: EffectTarget = "self"
    break_triggers: list[BreakTriggerType]
    condition: EffectCondition | None = None


class StatusStackLimit(FrozenModel):
    """
    Limits stacking or re-application of a status.

    Examples:
        StatusStackLimit(status="invisible", max_stacks=1)
    """

    status: StatusType
    max_stacks: int = Field(default=1, ge=1)
    target: EffectTarget = "self"
    condition: EffectCondition | None = None


class MovementScopedStatus(FrozenModel):
    """
    Status that applies only during movement.

    Examples:
        MovementScopedStatus(status="invisible", movement_modes=["any"], ends_on="movement_end")
    """

    status: StatusType
    target: EffectTargetNoAll
    movement_modes: list[MovementMode] = Field(default_factory=list)
    ends_on: Literal["movement_end", "turn_end"] = "movement_end"
    condition: EffectCondition | None = None


class StatusRestriction(FrozenModel):
    """
    Restricts gaining or benefiting from statuses.

    Examples:
        StatusRestriction(statuses=["invisible"], restriction="cannot_benefit", target="enemy")
    """

    statuses: list[StatusType]
    restriction: Literal["cannot_gain", "cannot_benefit"] = "cannot_gain"
    target: EffectTarget = "enemy"
    duration: EffectDuration = "end_of_turn"
    condition: EffectCondition | None = None


class AllegianceShiftEffect(FrozenModel):
    """
    Temporarily flips a target's allegiance.

    Examples:
        AllegianceShiftEffect(duration="end_of_next_turn", ends_on_hostile_action=True)
    """

    target: EffectTarget = "enemy"
    duration: EffectDuration = "end_of_next_turn"
    ends_on_hostile_action: bool = True
    condition: EffectCondition | None = None


class StatusActionOverrideEffect(FrozenModel):
    """
    Allows actions that are normally restricted by a status.

    Examples:
        StatusActionOverrideEffect(status="jammed", allow_attack_types=["ranged"])
    """

    status: StatusType
    allow_attack_types: list[AttackType] = Field(default_factory=list)
    allow_action_categories: list[ActionCategoryType] = Field(default_factory=list)
    target: EffectTarget = "self"
    condition: EffectCondition | None = None


class StatusTrigger(FrozenModel):
    """
    Effect triggered by inflicting or clearing a status.

    Examples:
        StatusTrigger(trigger="on_inflict", status="immobilized",
                      effect=MechanicalEffect(status_grants=[StatusGrant(status="shredded", target="enemy",
                      duration="match_trigger")]))
    """

    trigger: Literal["on_inflict", "on_clear"]
    status: StatusType
    target: EffectTargetNoAll = "enemy"
    effect: "MechanicalEffect"
    condition: EffectCondition | None = None
    uses_per: UsesPer = "unlimited"
