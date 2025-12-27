"""Shared types and enums for Lancer TTRPG."""

from core.shared.enums import (
    ActionType,
    DamageType,
    RangeType,
    SizeClass,
    ManufacturerType,
    MountType,
    SystemType,
    StatusType,
)
from core.shared.dice import DiceExpression, DieSize, roll_dice, roll_with_advantage
from core.shared.effects import (
    StatType,
    ConditionType,
    StatModifier,
    DamageModifier,
    RangeModifier,
    ActionGrant,
    Immunity,
    Resistance,
    AccuracyModifier,
    MovementGrant,
    StatusGrant,
    MechanicalEffect,
    stat_bonus,
    damage_bonus,
    immunity_to,
)

__all__ = [
    # Enums
    "ActionType",
    "DamageType", 
    "RangeType",
    "SizeClass",
    "ManufacturerType",
    "MountType",
    "SystemType",
    "StatusType",
    # Dice
    "DiceExpression",
    "DieSize",
    "roll_dice",
    "roll_with_advantage",
    # Effects
    "StatType",
    "ConditionType",
    "StatModifier",
    "DamageModifier",
    "RangeModifier",
    "ActionGrant",
    "Immunity",
    "Resistance",
    "AccuracyModifier",
    "MovementGrant",
    "StatusGrant",
    "MechanicalEffect",
    "stat_bonus",
    "damage_bonus",
    "immunity_to",
]

