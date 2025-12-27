"""
Lancer TTRPG Core Type System

Type-driven schemas for game data, rules, and entities.
Uses Pydantic v2 for validation and JSON Schema export.

Legal Note: This type system encodes game mechanics (allowed under
the Lancer Third Party License), not copyrighted expression/flavor text.
"""

# Re-export pilot domain
from core.pilot import (
    Pilot,
    create_ll0_pilot,
    Skill,
    SkillSet,
    SkillType,
    Background,
    Talent,
    TalentDefinition,
    TalentRank,
    License,
    LicenseDefinition,
    Manufacturer,
    CoreBonus,
    CoreBonusDefinition,
)

# Re-export shared types
from core.shared import (
    ActionType,
    DamageType,
    RangeType,
    SizeClass,
    DiceExpression,
    roll_dice,
    # Effect primitives
    MechanicalEffect,
    StatModifier,
    DamageModifier,
    AccuracyModifier,
    ActionGrant,
    Immunity,
    Resistance,
)

__all__ = [
    # Pilot domain
    "Pilot",
    "create_ll0_pilot",
    "Skill",
    "SkillSet",
    "SkillType",
    "Background",
    "Talent",
    "TalentDefinition",
    "TalentRank",
    "License",
    "LicenseDefinition",
    "Manufacturer",
    "CoreBonus",
    "CoreBonusDefinition",
    # Shared types
    "ActionType",
    "DamageType",
    "RangeType",
    "SizeClass",
    "DiceExpression",
    "roll_dice",
    # Effect primitives
    "MechanicalEffect",
    "StatModifier",
    "DamageModifier",
    "AccuracyModifier",
    "ActionGrant",
    "Immunity",
    "Resistance",
]

