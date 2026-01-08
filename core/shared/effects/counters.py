"""Cooldown and counter tracking effects.

Re-exports counter/cooldown effect classes from core.py for cleaner imports
while maintaining backward compatibility.

Effects:
    - PerTargetCounter: Tracks effect applications per target
    - PerTargetCounterEffect: Applies per-target counters
    - CooldownState: State tracking for cooldowns
    - CooldownEffect: Applies cooldown to actions

See Also:
    - PR2 4407-4430: Cooldown actions
"""

from __future__ import annotations

from core.shared.effects.core import (
    PerTargetCounter,
    PerTargetCounterEffect,
    CooldownState,
    CooldownEffect,
)

__all__ = [
    "PerTargetCounter",
    "PerTargetCounterEffect",
    "CooldownState",
    "CooldownEffect",
]
