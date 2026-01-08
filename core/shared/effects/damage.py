"""Direct damage, reduction, and sharing effects.

Re-exports damage-related effect classes from core.py for cleaner imports
while maintaining backward compatibility.

Effects:
    - DamageModifier: Bonus or penalty to damage
    - DamageMultiplierEffect: Multiplies damage, heat, or burn
    - RangeModifier: Modifies range or threat
    - DirectDamage: Direct damage not tied to weapon attacks
    - DamageReduction: Reduces incoming damage by a value
    - DamageReductionRollEffect: Reduces damage based on roll
    - DamageShareEffect: Shares damage with another target
    - DamageNegationEffect: Negates damage under conditions
    - DamageAbsorption: Absorbs damage as heat or other resources

See Also:
    - PR2 3960-3964: Damage resolution
"""

from __future__ import annotations

from core.shared.effects.core import (
    DamageModifier,
    DamageMultiplierEffect,
    RangeModifier,
    DirectDamage,
    DamageReduction,
    DamageReductionRollEffect,
    DamageShareEffect,
    DamageNegationEffect,
    DamageAbsorption,
)

__all__ = [
    "DamageModifier",
    "DamageMultiplierEffect",
    "RangeModifier",
    "DirectDamage",
    "DamageReduction",
    "DamageReductionRollEffect",
    "DamageShareEffect",
    "DamageNegationEffect",
    "DamageAbsorption",
]
