"""Resource and capacity effects.

Re-exports resource-related effect classes from core.py for cleaner imports
while maintaining backward compatibility.

Effects:
    - ActionGrant: Grants a new action or ability
    - ActionRestriction: Restricts actions
    - ReactionLimitEffect: Adjusts maximum reactions per turn
    - ReactionTriggerEffect: Grants or extends reaction triggers
    - NonCombatCapacityEffect: Non-combat capabilities
    - ResourceChange: Changes resources (ammo, charges, etc.)
    - ScaledResourceChange: Scales resource changes by conditions
    - OverchargeCostCapEffect: Caps overcharge costs
    - LimitedUseBonusEffect: Bonus to limited use items
    - LimitedUseRechargeEffect: Recharges limited use items

See Also:
    - PR2 3726-4406: Actions and activation
"""

from __future__ import annotations

from core.shared.effects.core import (
    ActionGrant,
    ActionRestriction,
    ReactionLimitEffect,
    ReactionTriggerEffect,
    NonCombatCapacityEffect,
    ResourceChange,
    ScaledResourceChange,
    OverchargeCostCapEffect,
    LimitedUseBonusEffect,
    LimitedUseRechargeEffect,
)

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
