"""Tech action effects.

Effects for tech actions, targeting, and restrictions.

Effects:
    - TechRange: Range descriptor for tech actions
    - TechActionOverrideEffect: Overrides tech action targeting
    - TechAction: Defines a tech action granted by a system
    - TechAttackModifier: Accuracy/difficulty modifiers for tech attacks
    - TechActionRestriction: Restrictions or immunity affecting tech actions

See Also:
    - PR2 4060-4095: Tech actions
"""

from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from pydantic import Field

from core.shared.models import FrozenModel
from core.shared.enums import ActionType
from core.shared.effects.types import (
    EffectTarget,
    EffectTargetNoAll,
    EffectTargetWithObjectNoAll,
    TechActionScope,
    TechRangeType,
    UsesPer,
)
from core.shared.effects.conditions import EffectCondition

if TYPE_CHECKING:
    from core.shared.effects.core import MechanicalEffect

__all__ = [
    "TechRange",
    "TechActionOverrideEffect",
    "TechAction",
    "TechAttackModifier",
    "TechActionRestriction",
]


class TechRange(FrozenModel):
    """
    Range descriptor for tech actions.

    Examples:
        TechRange(range_type="sensors")  # Within sensors
        TechRange(range_type="range", value=10)  # Range 10
    """

    range_type: TechRangeType = "sensors"
    value: int | None = Field(default=None, ge=0)


class TechActionOverrideEffect(FrozenModel):
    """
    Overrides tech action targeting or requirements.

    Examples:
        TechActionOverrideEffect(requires_line_of_sight=False, range_override=TechRange(range_type="range", value=50))
    """

    applies_to: TechActionScope = "all"
    requires_line_of_sight: bool | None = None
    range_override: TechRange | None = None
    target: EffectTargetNoAll = "self"
    condition: EffectCondition | None = None


class TechAction(FrozenModel):
    """Defines a tech action granted by a system.

    Per PR2 4060-4095: Tech actions target E-Defense and can apply status
    effects, debuffs, or direct damage. Some tech actions are attacks.

    Examples:
        TechAction(name="Track", action_type="quick", is_attack=True, range=TechRange(range_type="sensors"))
    """

    name: str
    action_type: ActionType
    target: EffectTargetWithObjectNoAll = "enemy"
    range: TechRange | None = None
    is_attack: bool = False
    attack_vs: Literal["e_defense", "evasion"] = "e_defense"
    effect: "MechanicalEffect | None" = None
    on_hit: "MechanicalEffect | None" = None
    on_miss: "MechanicalEffect | None" = None
    on_success: "MechanicalEffect | None" = None
    on_failure: "MechanicalEffect | None" = None
    uses_per: UsesPer = "unlimited"
    special: str | None = None


class TechAttackModifier(FrozenModel):
    """
    Accuracy/difficulty modifiers for tech attacks.

    Examples:
        TechAttackModifier(value=-1, target="ally", condition="adjacent", max_stacks=3)
    """

    value: int = Field(..., description="Positive = accuracy, negative = difficulty")
    target: EffectTarget = "self"
    condition: EffectCondition | None = None
    max_stacks: int | None = Field(default=None, ge=1)
    reset_trigger: (
        Literal["turn_start", "turn_end", "round_end", "scene_end"] | None
    ) = None


class TechActionRestriction(FrozenModel):
    """
    Restrictions or immunity affecting tech actions.

    Examples:
        TechActionRestriction(disallow_tech_actions=True, end_tech_effects=True)
    """

    disallow_tech_actions: bool = False
    immune_to_tech: bool = False
    end_tech_effects: bool = False
    target: EffectTarget = "self"
    condition: EffectCondition | None = None
