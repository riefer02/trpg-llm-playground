"""Miscellaneous effect classes.

Re-exports miscellaneous effect classes from core.py for cleaner imports
while maintaining backward compatibility.

Effects:
    - EffectChoice: Select one of multiple effects
    - TriggeredEffect: Effect triggered by conditions
    - BondmateEffect: Bondmate relationship effects
    - AllegianceShiftEffect: Changes faction allegiance
    - AISystemLimitEffect: Limits AI systems
    - AIControlTransferEffect: Transfers AI control
    - EffectRemoval: Removes effects from targets
    - AccuracyModifier: Modifies accuracy/difficulty

See Also:
    - PR2 4407-4430: Various action and effect rules
"""

from __future__ import annotations

from core.shared.effects.core import (
    EffectChoice,
    TriggeredEffect,
    BondmateEffect,
    AllegianceShiftEffect,
    AISystemLimitEffect,
    AIControlTransferEffect,
    EffectRemoval,
    AccuracyModifier,
)

__all__ = [
    "EffectChoice",
    "TriggeredEffect",
    "BondmateEffect",
    "AllegianceShiftEffect",
    "AISystemLimitEffect",
    "AIControlTransferEffect",
    "EffectRemoval",
    "AccuracyModifier",
]
