"""Attack and damage resolution effects.

Re-exports attack-related effect classes from core.py for cleaner imports
while maintaining backward compatibility.

Effects:
    - AttackRollOverrideEffect: Overrides attack roll mechanics
    - AttackTargetingEffect: Multi-target attack selection
    - AreaAttackPattern: Blast, burst, line, cone attack patterns
    - LineAttackEffect: Custom line attack behavior
    - AttackSequenceModifierEffect: Accuracy/difficulty across attack sequences
    - AttackRerollEffect: Allows rerolling attack rolls
    - AttackOutcomeEffect: Modifies attack outcomes (hit/miss/crit)
    - CriticalDamageOverrideEffect: Overrides critical damage rules
    - DamageRollOverrideEffect: Overrides damage calculation
    - AccuracyTradeEffect: Trades accuracy for difficulty or vice versa
    - DelayedImpactEffect: Delays damage application

See Also:
    - PR2 3965-3984: Attack and damage resolution
"""

from __future__ import annotations

from core.shared.effects.core import (
    AttackRollOverrideEffect,
    AttackTargetingEffect,
    AreaAttackPattern,
    LineAttackEffect,
    AttackSequenceModifierEffect,
    AttackRerollEffect,
    AttackOutcomeEffect,
    CriticalDamageOverrideEffect,
    DamageRollOverrideEffect,
    AccuracyTradeEffect,
    DelayedImpactEffect,
)

__all__ = [
    "AttackRollOverrideEffect",
    "AttackTargetingEffect",
    "AreaAttackPattern",
    "LineAttackEffect",
    "AttackSequenceModifierEffect",
    "AttackRerollEffect",
    "AttackOutcomeEffect",
    "CriticalDamageOverrideEffect",
    "DamageRollOverrideEffect",
    "AccuracyTradeEffect",
    "DelayedImpactEffect",
]
