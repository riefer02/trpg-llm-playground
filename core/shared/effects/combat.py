"""Reload and out-of-play effects.

Re-exports combat effect classes from core.py for cleaner imports
while maintaining backward compatibility.

Effects:
    - ReloadEffect: Reloads weapons
    - ReloadRestrictionEffect: Restricts reloading
    - OutOfPlayEffect: Effects when out of play

See Also:
    - PR2 4332-4334: Reload action
"""

from __future__ import annotations

from core.shared.effects.core import (
    ReloadEffect,
    ReloadRestrictionEffect,
    OutOfPlayEffect,
)

__all__ = [
    "ReloadEffect",
    "ReloadRestrictionEffect",
    "OutOfPlayEffect",
]
