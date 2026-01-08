"""Condition primitives for effect targeting.

This module re-exports condition classes from core.py
for cleaner imports while maintaining backward compatibility.
"""

from __future__ import annotations

from core.shared.effects.core import (
    SpatialCondition,
    AttackContextCondition,
    SizeCondition,
    CheckContextCondition,
    ReactionCondition,
    ConditionGroup,
    EffectCondition,
)

__all__ = [
    "SpatialCondition",
    "AttackContextCondition",
    "SizeCondition",
    "CheckContextCondition",
    "ReactionCondition",
    "ConditionGroup",
    "EffectCondition",
]
