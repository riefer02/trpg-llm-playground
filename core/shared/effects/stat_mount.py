"""Stat and mount slot modification effects.

This module re-exports stat/mount effect classes from core.py
for cleaner imports while maintaining backward compatibility.
"""

from __future__ import annotations

from core.shared.effects.core import (
    StatModifier,
    CompanionStatModifierEffect,
    StatOverrideEffect,
    MountSlotGrant,
    MountSlotReplacement,
    MountSizeUpgradeEffect,
    IntegratedWeaponEffect,
)

__all__ = [
    "StatModifier",
    "CompanionStatModifierEffect",
    "StatOverrideEffect",
    "MountSlotGrant",
    "MountSlotReplacement",
    "MountSizeUpgradeEffect",
    "IntegratedWeaponEffect",
]
