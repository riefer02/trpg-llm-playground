"""Tech action effects.

Re-exports tech action-related effect classes from core.py for cleaner imports
while maintaining backward compatibility.

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

from core.shared.effects.core import (
    TechRange,
    TechActionOverrideEffect,
    TechAction,
    TechAttackModifier,
    TechActionRestriction,
)

__all__ = [
    "TechRange",
    "TechActionOverrideEffect",
    "TechAction",
    "TechAttackModifier",
    "TechActionRestriction",
]
