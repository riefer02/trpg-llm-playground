"""Weapon modification and granting effects.

Re-exports weapon-related effect classes from core.py for cleaner imports
while maintaining backward compatibility.

Effects:
    - WeaponTagGrant: Adds tags to weapons
    - WeaponRangeSpec: Overrides weapon range
    - WeaponSizeBonus: Bonus weapon size slots
    - WeaponGrantEffect: Grants a weapon
    - WeaponModEffect: Modifies weapon properties
    - WeaponSpinUpEffect: Spin up weapon for bonuses
    - WeaponAIControlEffect: AI control over weapons

See Also:
    - PR2 4444-4490: Weapon systems
"""

from __future__ import annotations

from core.shared.effects.core import (
    WeaponTagGrant,
    WeaponRangeSpec,
    WeaponSizeBonus,
    WeaponGrantEffect,
    WeaponModEffect,
    WeaponSpinUpEffect,
    WeaponAIControlEffect,
)

__all__ = [
    "WeaponTagGrant",
    "WeaponRangeSpec",
    "WeaponSizeBonus",
    "WeaponGrantEffect",
    "WeaponModEffect",
    "WeaponSpinUpEffect",
    "WeaponAIControlEffect",
]
