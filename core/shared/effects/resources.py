"""Resource and capacity effects.

Effects for actions, reactions, resources, and capacity management.

Effects:
    - ActionGrant: Grants a new action or ability
    - ActionRestriction: Restricts actions
    - ReactionLimitEffect: Adjusts maximum reactions per turn
    - ReactionTriggerEffect: Grants or extends reaction triggers
    - NonCombatCapacityEffect: Non-combat capabilities
    - ResourceChange: Changes resources (HP, heat, etc.)
    - ScaledResourceChange: Scales resource changes by conditions
    - OverchargeCostCapEffect: Caps overcharge costs
    - LimitedUseBonusEffect: Bonus to limited use items
    - LimitedUseRechargeEffect: Recharges limited use items

See Also:
    - PR2 3726-4406: Actions and activation
    - PR2 4402-4410: Overcharge rules
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from core.shared.models import FrozenModel
from core.shared.dice import DiceExpression
from core.shared.enums import ActionType, SizeClass, StatusType
from core.shared.id_helpers import ReactionIdField
from core.shared.effects.types import (
    ActionCategoryType,
    EffectDuration,
    EffectTarget,
    EffectTargetNoAll,
    NonCombatInteractionScope,
    PassengerLocation,
    ReactionTriggerEvent,
    ResourceAmount,
    ResourceDirection,
    ResourceType,
    TriggerType,
    UsesPer,
)
from core.shared.effects.conditions import EffectCondition

__all__ = [
    "ActionGrant",
    "ActionRestriction",
    "ReactionLimitEffect",
    "ReactionTriggerEffect",
    "NonCombatCapacityEffect",
    "ResourceChange",
    "ScaledResourceChange",
    "OverchargeCostCapEffect",
    "LimitedUseBonusEffect",
    "LimitedUseRechargeEffect",
]


class ActionGrant(FrozenModel):
    """
    Grants a new action or ability.

    Examples:
        ActionGrant(action_type="quick", name="Afterburner", trigger="after_boost")
        ActionGrant(action_type="reaction", name="Juke", trigger="on_successful_agility_save")
    """

    action_type: ActionType
    name: str
    trigger: TriggerType | str | None = Field(
        default=None, description="When this action can be used"
    )
    uses_per: UsesPer = "unlimited"


class ReactionLimitEffect(FrozenModel):
    """
    Adjusts the maximum number of reactions per turn.

    Examples:
        ReactionLimitEffect(max_reactions_per_turn=2)
    """

    max_reactions_per_turn: int | None = Field(default=None, ge=1)
    bonus_reactions_per_turn: int | None = Field(default=None, ge=1)
    target: EffectTargetNoAll = "self"
    condition: EffectCondition | None = None


class ReactionTriggerEffect(FrozenModel):
    """
    Grants or extends reaction triggers for a specific reaction action.

    Examples:
        ReactionTriggerEffect(reaction_id="overwatch", trigger_events=["enemy_enters_threat"])
    """

    reaction_id: ReactionIdField
    trigger_events: list[ReactionTriggerEvent] = Field(default_factory=list)
    uses_per: UsesPer = "unlimited"
    condition: EffectCondition | None = None


class NonCombatCapacityEffect(FrozenModel):
    """
    Non-combat capabilities such as extra limbs or passenger space.

    Examples:
        NonCombatCapacityEffect(extra_limb_pairs=1, interaction_scope="pilot_scale")
        NonCombatCapacityEffect(extra_passenger_slots=1, max_passenger_size="size_half")
    """

    non_combat_only: bool = True
    interaction_scope: NonCombatInteractionScope | None = None
    extra_limb_pairs: int | None = Field(default=None, ge=1)
    extra_passenger_slots: int | None = Field(default=None, ge=1)
    max_passenger_size: SizeClass | None = None
    passenger_location: PassengerLocation = "cockpit"
    passenger_protected_from_external_effects: bool = True
    statuses_if_unlicensed_pilot: list[StatusType] = Field(default_factory=list)
    condition: EffectCondition | None = None


class ActionRestriction(FrozenModel):
    """
    Restrictions for combat action usage.

    Examples:
        ActionRestriction(disallow_attack_rolls=True)
        ActionRestriction(action_ids=["hide"], target="enemy")
    """

    disallow_attack_rolls: bool = False
    disallow_heat_generation: bool = False
    action_ids: list[str] = Field(default_factory=list)
    action_categories: list[ActionCategoryType] = Field(default_factory=list)
    target: EffectTarget = "self"
    duration: EffectDuration = "end_of_turn"
    condition: EffectCondition | None = None


class ResourceChange(FrozenModel):
    """
    Change a resource value such as HP or heat.

    Examples:
        ResourceChange(resource="hp", amount="half_max", target="ally", cost_repairs=1)
        ResourceChange(resource="heat", amount=DiceExpression.parse("1d6"), direction="lose", target="self")
    """

    resource: ResourceType
    amount: ResourceAmount
    direction: ResourceDirection = "gain"
    target: EffectTargetNoAll
    cost_repairs: int = Field(default=0, ge=0)
    cost_source: Literal["self", "target", "either"] = "self"


class ScaledResourceChange(FrozenModel):
    """
    Resource change based on a rolled value with a multiplier.

    Examples:
        ScaledResourceChange(resource="heat", roll=DiceExpression.parse("1d6"), multiplier=0.5)
    """

    resource: ResourceType
    roll: DiceExpression
    multiplier: float = Field(default=1.0, ge=0)
    direction: ResourceDirection = "gain"
    target: EffectTargetNoAll = "self"
    rounding: Literal["floor", "ceil", "round"] = "floor"
    condition: EffectCondition | None = None


class OverchargeCostCapEffect(FrozenModel):
    """
    Caps the heat cost for overcharge actions.

    Examples:
        OverchargeCostCapEffect(max_cost=DiceExpression.parse("1d6"))
    """

    max_cost: int | DiceExpression
    target: EffectTargetNoAll = "self"
    condition: EffectCondition | None = None


class LimitedUseBonusEffect(FrozenModel):
    """
    Adds limited-use charges to tagged systems, deployables, or weapons.

    Examples:
        LimitedUseBonusEffect(bonus_uses=2, applies_to=["system", "deployable"])
    """

    bonus_uses: int = Field(..., ge=1)
    applies_to: list[Literal["system", "deployable", "weapon"]] = Field(
        default_factory=list
    )
    requires_tag: str = "limited"
    condition: EffectCondition | None = None


class LimitedUseRechargeEffect(FrozenModel):
    """
    Replenishes limited-use charges as part of a rest or downtime action.

    Examples:
        LimitedUseRechargeEffect(bonus_uses=1, applies_to=["weapon", "deployable"], cost_repairs=2, uses_per="rest")
    """

    bonus_uses: int = Field(..., ge=1)
    applies_to: list[Literal["system", "deployable", "weapon"]] = Field(
        default_factory=list
    )
    requires_tag: str = "limited"
    cost_repairs: int = Field(default=0, ge=0)
    uses_per: UsesPer = "rest"
    target: EffectTarget = "self"
    condition: EffectCondition | None = None
