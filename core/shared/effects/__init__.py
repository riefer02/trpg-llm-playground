"""
Structured mechanical effect primitives for Lancer TTRPG.

This module defines composable effect building blocks that encode
game mechanics as structured data rather than description strings.

The effects system has been organized into submodules:
- types.py: Type aliases (Literal types)
- (effects_old.py: Effect classes - to be split further)

All exports remain available from this package for backward compatibility.
"""

# Re-export types from the new types module
from core.shared.effects.types import (
    StatType,
    ConditionType,
    TriggerType,
    ReactionTriggerEvent,
    ActionCategoryType,
    EffectDuration,
    EffectTarget,
    EffectTargetNoAll,
    EffectTargetWithObject,
    EffectTargetWithObjectNoAll,
    SpatialRelation,
    AttackAreaShape,
    MovementDistanceType,
    ForcedMovementDistanceType,
    MovementMode,
    IntelAudience,
    IntelType,
    CheckKind,
    WeaponSizeType,
    WeaponTypeType,
    AreaSelectionScope,
    ZoneEndTriggerType,
    ZoneEndScope,
    ResourceType,
    ResourceAmount,
    ResourceDirection,
    TechRangeType,
    TechActionScope,
    UsesPer,
    BreakTriggerType,
    NonCombatInteractionScope,
    PassengerLocation,
    RollPatternType,
    OutOfPlayDuration,
    DeploymentActivationCondition,
    DelayedImpactTiming,
    PhaseState,
    HologramTrailTrigger,
    HologramDetonationTrigger,
    DamageTypeScope,
    DirectDamageType,
)

# Re-export everything from core module for backward compatibility
# This ensures all effect classes are available
from core.shared.effects.core import *  # noqa: F401, F403

# Explicitly re-export the __all__ from core to maintain compatibility
from core.shared.effects.core import __all__ as _core_all

# Combine all exports
__all__ = [
    # Type aliases from types.py
    "StatType",
    "ConditionType",
    "SpatialRelation",
    "AttackAreaShape",
    "TriggerType",
    "ReactionTriggerEvent",
    "ActionCategoryType",
    "EffectDuration",
    "MovementDistanceType",
    "ForcedMovementDistanceType",
    "IntelAudience",
    "IntelType",
    "MovementMode",
    "CheckKind",
    "NonCombatInteractionScope",
    "PassengerLocation",
    "BreakTriggerType",
    "WeaponSizeType",
    "WeaponTypeType",
    "AreaSelectionScope",
    "ZoneEndTriggerType",
    "ZoneEndScope",
    "RollPatternType",
    "OutOfPlayDuration",
    "ResourceType",
    "ResourceAmount",
    "ResourceDirection",
    "TechRangeType",
    "TechActionScope",
    "DeploymentActivationCondition",
    "DelayedImpactTiming",
    "PhaseState",
    "HologramTrailTrigger",
    "HologramDetonationTrigger",
    "EffectTarget",
    "EffectTargetNoAll",
    "EffectTargetWithObject",
    "EffectTargetWithObjectNoAll",
    "UsesPer",
    "DamageTypeScope",
    "DirectDamageType",
] + _core_all
