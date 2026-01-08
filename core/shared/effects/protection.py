"""Damage resistance and immunity effects.

Re-exports protection effect classes from core.py for cleaner imports
while maintaining backward compatibility.

Effects:
    - Immunity: Complete immunity to damage types
    - TagImmunityEffect: Immunity based on tags
    - Resistance: Damage resistance with reduction
    - HeatResistanceEffect: Heat and burn resistance

See Also:
    - PR2 4005-4012: Resistance and immunity
"""

from __future__ import annotations

from core.shared.effects.core import (
    Immunity,
    TagImmunityEffect,
    Resistance,
    HeatResistanceEffect,
)

__all__ = [
    "Immunity",
    "TagImmunityEffect",
    "Resistance",
    "HeatResistanceEffect",
]
