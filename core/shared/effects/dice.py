"""Dice pool and random check effects.

Re-exports dice-related effect classes from core.py for cleaner imports
while maintaining backward compatibility.

Effects:
    - DicePoolGain: Gain dice in a named pool when triggers occur
    - DicePoolSpendOption: Spend dice from a pool for effects
    - DicePoolEffect: Named pool of dice that can be gained and spent
    - CountdownDieTrigger: Triggers when countdown die reaches value
    - CountdownDieEffect: Manages countdown die mechanics
    - LeadershipDicePoolEffect: Leadership-inspired dice pool mechanics

See Also:
    - PR2 1350-1364: Dice and checks
"""

from __future__ import annotations

from core.shared.effects.core import (
    DicePoolGain,
    DicePoolSpendOption,
    DicePoolEffect,
    CountdownDieTrigger,
    CountdownDieEffect,
    LeadershipDicePoolEffect,
)

__all__ = [
    "DicePoolGain",
    "DicePoolSpendOption",
    "DicePoolEffect",
    "CountdownDieTrigger",
    "CountdownDieEffect",
    "LeadershipDicePoolEffect",
]
