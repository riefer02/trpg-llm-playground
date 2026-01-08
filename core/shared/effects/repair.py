"""Repair and damage mitigation effects.

Re-exports repair effect classes from core.py for cleaner imports
while maintaining backward compatibility.

Effects:
    - RepairCostModifier: Modifies repair costs
    - RepairShareEffect: Shares repair with allies
    - RepairActionEffect: Modifies repair actions
    - StructureDamageAvoidanceEffect: Avoids structure damage
    - ZeroHpSurvivalEffect: Survival at 0 HP

See Also:
    - PR2 4729-4785: Rest and repair
"""

from __future__ import annotations

from core.shared.effects.core import (
    RepairCostModifier,
    RepairShareEffect,
    RepairActionEffect,
    StructureDamageAvoidanceEffect,
    ZeroHpSurvivalEffect,
)

__all__ = [
    "RepairCostModifier",
    "RepairShareEffect",
    "RepairActionEffect",
    "StructureDamageAvoidanceEffect",
    "ZeroHpSurvivalEffect",
]
