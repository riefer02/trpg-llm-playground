"""
Structured mechanical effect primitives for Lancer TTRPG.

This module defines composable effect building blocks that encode
game mechanics as structured data rather than description strings.

The effects system has been organized into submodules:
- types.py: Type aliases (Literal types)
- conditions.py: Condition primitives
- stat_mount.py: Stat and mount slot effects
- damage.py: Damage and damage reduction effects
- movement.py: Movement and positioning effects
- status.py: Status and condition effects
- attack.py: Attack and damage resolution effects
- tech.py: Tech action effects
- resources.py: Resource and capacity effects
- dice.py: Dice pool and random check effects
- weapon.py: Weapon modification and granting effects
- deployment.py: Deployment and attachment effects
- zone.py: Zone and area effect definitions
- special.py: Special combat and interaction effects
- covers.py: Cover and line of sight effects
- intel.py: Intel and targeting effects
- checks.py: Save and check effects
- protection.py: Damage resistance and immunity effects
- repair.py: Repair and damage mitigation effects
- combat.py: Reload and out-of-play effects
- mode.py: Mode, protocol, and progression effects
- counters.py: Cooldown and counter tracking effects
- misc.py: Miscellaneous effect classes
- core.py: Main MechanicalEffect class and remaining classes

All exports remain available from this package for backward compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

from core.shared.effects.conditions import (
    SpatialCondition,
    AttackContextCondition,
    SizeCondition,
    CheckContextCondition,
    ReactionCondition,
    ConditionGroup,
    EffectCondition,
)

from core.shared.effects.stat_mount import (
    StatModifier,
    CompanionStatModifierEffect,
    StatOverrideEffect,
    MountSlotGrant,
    MountSlotReplacement,
    MountSizeUpgradeEffect,
    IntegratedWeaponEffect,
)

from core.shared.effects.damage import (
    DamageModifier,
    DamageMultiplierEffect,
    RangeModifier,
    DirectDamage,
    DamageReduction,
    DamageReductionRollEffect,
    DamageShareEffect,
    DamageNegationEffect,
    DamageAbsorption,
)

from core.shared.effects.movement import (
    MovementGrant,
    MoveAdjacentEffect,
    PositionSwapEffect,
    ForcedMovement,
    MovementRestrictionEffect,
    MovementSurfaceEffect,
    MovementModeAccessEffect,
    JumpDistanceEffect,
    MovementOverrideEffect,
)

from core.shared.effects.status import (
    StatusGrant,
    StatusClear,
    StatusToggleEffect,
    StatusBreakCondition,
    StatusStackLimit,
    MovementScopedStatus,
    StatusRestriction,
    StatusActionOverrideEffect,
    StatusTrigger,
)

from core.shared.effects.attack import (
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

from core.shared.effects.tech import (
    TechRange,
    TechAction,
    TechAttackModifier,
    TechActionOverrideEffect,
    TechActionRestriction,
)

from core.shared.effects.resources import (
    ActionGrant,
    ActionRestriction,
    ReactionLimitEffect,
    ReactionTriggerEffect,
    NonCombatCapacityEffect,
    ResourceChange,
    ScaledResourceChange,
    OverchargeCostCapEffect,
    LimitedUseBonusEffect,
    LimitedUseRechargeEffect,
)

from core.shared.effects.dice import (
    DicePoolGain,
    DicePoolSpendOption,
    DicePoolEffect,
    CountdownDieTrigger,
    CountdownDieEffect,
    LeadershipDicePoolEffect,
)

from core.shared.effects.weapon import (
    WeaponTagGrant,
    WeaponRangeSpec,
    WeaponSizeBonus,
    WeaponGrantEffect,
    WeaponModEffect,
    WeaponSpinUpEffect,
    WeaponAIControlEffect,
)

from core.shared.effects.deployment import (
    DeploymentEffect,
    AttachmentEffect,
    SystemLinkEffect,
)

from core.shared.effects.zone import (
    ZoneEndCondition,
    AttackCaptureEffect,
    ZoneEffect,
    AreaSelectionEffect,
)

from core.shared.effects.special import (
    TetherEffect,
    GrappleEffect,
    SizeInteractionEffect,
    MountedAllyEffect,
    EffectRemoval,
    HolographicDuplicateEffect,
    MovementTrailEffect,
    HologramTrailEffect,
)

from core.shared.effects.covers import (
    CoverGrant,
    CoverRestriction,
    LineOfSightRestriction,
)

from core.shared.effects.intel import (
    IntelEffect,
    TargetMarkEffect,
)

from core.shared.effects.checks import (
    SaveOverrideEffect,
    SaveCheck,
    RandomCheckEffect,
    RollPatternEffect,
    CheckModifierEffect,
    CheckValueModifierEffect,
)

from core.shared.effects.protection import (
    Immunity,
    TagImmunityEffect,
    Resistance,
    HeatResistanceEffect,
)

from core.shared.effects.repair import (
    RepairCostModifier,
    RepairShareEffect,
    RepairActionEffect,
    StructureDamageAvoidanceEffect,
    ZeroHpSurvivalEffect,
)

from core.shared.effects.combat import (
    ReloadEffect,
    ReloadRestrictionEffect,
    OutOfPlayEffect,
)

from core.shared.effects.mode import (
    ModeEffect,
    ProtocolEffect,
    CorePowerEffect,
    ProgressionState,
    GateProgressionEffect,
    ProgressionEffect,
)

from core.shared.effects.counters import (
    PerTargetCounter,
    PerTargetCounterEffect,
    CooldownState,
    CooldownEffect,
)

from core.shared.effects.misc import (
    EffectChoice,
    TriggeredEffect,
    BondmateEffect,
    AllegianceShiftEffect,
    AISystemLimitEffect,
    AIControlTransferEffect,
    EffectRemoval,
    AccuracyModifier,
)

from core.shared.effects.core import (
    ZoneShape,
    ProgressionResetTrigger,
    CooldownResetTrigger,
    PhaseShiftEffect,
    MechanicalEffect,
    stat_bonus,
    damage_bonus,
    immunity_to,
)

__all__ = [
    "StatType",
    "ConditionType",
    "TriggerType",
    "ReactionTriggerEvent",
    "ActionCategoryType",
    "EffectDuration",
    "EffectTarget",
    "EffectTargetNoAll",
    "EffectTargetWithObject",
    "EffectTargetWithObjectNoAll",
    "SpatialRelation",
    "AttackAreaShape",
    "MovementDistanceType",
    "ForcedMovementDistanceType",
    "MovementMode",
    "IntelAudience",
    "IntelType",
    "CheckKind",
    "WeaponSizeType",
    "WeaponTypeType",
    "AreaSelectionScope",
    "ZoneEndTriggerType",
    "ZoneEndScope",
    "ResourceType",
    "ResourceAmount",
    "ResourceDirection",
    "TechRangeType",
    "TechActionScope",
    "UsesPer",
    "BreakTriggerType",
    "NonCombatInteractionScope",
    "PassengerLocation",
    "RollPatternType",
    "OutOfPlayDuration",
    "DeploymentActivationCondition",
    "DelayedImpactTiming",
    "PhaseState",
    "HologramTrailTrigger",
    "HologramDetonationTrigger",
    "DamageTypeScope",
    "DirectDamageType",
    "SpatialCondition",
    "AttackContextCondition",
    "SizeCondition",
    "CheckContextCondition",
    "ReactionCondition",
    "ConditionGroup",
    "EffectCondition",
    "StatModifier",
    "CompanionStatModifierEffect",
    "StatOverrideEffect",
    "MountSlotGrant",
    "MountSlotReplacement",
    "MountSizeUpgradeEffect",
    "IntegratedWeaponEffect",
    "DamageModifier",
    "DamageMultiplierEffect",
    "RangeModifier",
    "DirectDamage",
    "DamageReduction",
    "DamageReductionRollEffect",
    "DamageShareEffect",
    "DamageNegationEffect",
    "DamageAbsorption",
    "MovementGrant",
    "MoveAdjacentEffect",
    "PositionSwapEffect",
    "ForcedMovement",
    "MovementRestrictionEffect",
    "MovementSurfaceEffect",
    "MovementModeAccessEffect",
    "JumpDistanceEffect",
    "MovementOverrideEffect",
    "StatusGrant",
    "StatusClear",
    "StatusToggleEffect",
    "StatusBreakCondition",
    "StatusStackLimit",
    "MovementScopedStatus",
    "StatusRestriction",
    "StatusActionOverrideEffect",
    "StatusTrigger",
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
    "TechRange",
    "TechAction",
    "TechAttackModifier",
    "TechActionOverrideEffect",
    "TechActionRestriction",
    "ActionGrant",
    "ActionRestriction",
    "ReactionLimitEffect",
    "ReactionTriggerEffect",
    "NonCombatCapacityEffect",
    "ResourceChange",
    "ScaledResourceChange",
    "OverchargeCostCapEffect",
    "LimitedUseBonusEffect",
    "LimitedUseRechargeEffect",
    "DicePoolGain",
    "DicePoolSpendOption",
    "DicePoolEffect",
    "CountdownDieTrigger",
    "CountdownDieEffect",
    "LeadershipDicePoolEffect",
    "WeaponTagGrant",
    "WeaponRangeSpec",
    "WeaponSizeBonus",
    "WeaponGrantEffect",
    "WeaponModEffect",
    "WeaponSpinUpEffect",
    "WeaponAIControlEffect",
    "DeploymentEffect",
    "AttachmentEffect",
    "SystemLinkEffect",
    "ZoneEndCondition",
    "AttackCaptureEffect",
    "ZoneEffect",
    "AreaSelectionEffect",
    "TetherEffect",
    "GrappleEffect",
    "SizeInteractionEffect",
    "MountedAllyEffect",
    "EffectRemoval",
    "HolographicDuplicateEffect",
    "MovementTrailEffect",
    "HologramTrailEffect",
    "CoverGrant",
    "CoverRestriction",
    "LineOfSightRestriction",
    "IntelEffect",
    "TargetMarkEffect",
    "SaveOverrideEffect",
    "SaveCheck",
    "RandomCheckEffect",
    "RollPatternEffect",
    "CheckModifierEffect",
    "CheckValueModifierEffect",
    "Immunity",
    "TagImmunityEffect",
    "Resistance",
    "HeatResistanceEffect",
    "RepairCostModifier",
    "RepairShareEffect",
    "RepairActionEffect",
    "StructureDamageAvoidanceEffect",
    "ZeroHpSurvivalEffect",
    "ReloadEffect",
    "ReloadRestrictionEffect",
    "OutOfPlayEffect",
    "ModeEffect",
    "ProtocolEffect",
    "CorePowerEffect",
    "ProgressionState",
    "GateProgressionEffect",
    "ProgressionEffect",
    "PerTargetCounter",
    "PerTargetCounterEffect",
    "CooldownState",
    "CooldownEffect",
    "EffectChoice",
    "TriggeredEffect",
    "BondmateEffect",
    "AllegianceShiftEffect",
    "AISystemLimitEffect",
    "AIControlTransferEffect",
    "ZoneShape",
    "ProgressionResetTrigger",
    "CooldownResetTrigger",
    "PhaseShiftEffect",
    "AccuracyModifier",
    "MechanicalEffect",
    "stat_bonus",
    "damage_bonus",
    "immunity_to",
]
