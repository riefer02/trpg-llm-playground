"""
Structured mechanical effect primitives for Lancer TTRPG.

This module defines composable effect building blocks that encode
game mechanics as structured data rather than description strings.

Legal Note: These represent pure game mechanics (allowed under the
Lancer Third Party License), not copyrighted expression/flavor text.
"""

from __future__ import annotations

__all__ = [
    # Type aliases
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
    "EffectCondition",
    "ZoneShape",
    "ProgressionResetTrigger",
    "CooldownResetTrigger",
    # Condition classes
    "SpatialCondition",
    "AttackContextCondition",
    "SizeCondition",
    "CheckContextCondition",
    "ReactionCondition",
    "ConditionGroup",
    # Stat modification effects
    "StatModifier",
    "CompanionStatModifierEffect",
    "StatOverrideEffect",
    # Mount slot effects
    "MountSlotGrant",
    "MountSlotReplacement",
    "MountSizeUpgradeEffect",
    "IntegratedWeaponEffect",
    # Damage effects
    "DamageModifier",
    "DamageMultiplierEffect",
    "DirectDamage",
    "DamageReduction",
    "DamageReductionRollEffect",
    # Attack effects
    "RangeModifier",
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
    # Tech effects
    "TechRange",
    "TechActionOverrideEffect",
    "TechAction",
    "TechAttackModifier",
    "TechActionRestriction",
    # Resource & capacity effects
    "ActionGrant",
    "ActionRestriction",
    "ReactionLimitEffect",
    "ReactionTriggerEffect",
    "NonCombatCapacityEffect",
    # Status effects
    "StatusToggleEffect",
    "StatusGrant",
    "StatusClear",
    "StatusBreakCondition",
    "StatusStackLimit",
    "MovementScopedStatus",
    "StatusRestriction",
    "StatusActionOverrideEffect",
    "StatusTrigger",
    # Movement effects
    "MovementGrant",
    "MoveAdjacentEffect",
    "PositionSwapEffect",
    "ForcedMovement",
    "MovementRestrictionEffect",
    "MovementSurfaceEffect",
    "MovementModeAccessEffect",
    "JumpDistanceEffect",
    "MovementOverrideEffect",
    # Cover & targeting effects
    "CoverRestriction",
    "CoverGrant",
    "LineOfSightRestriction",
    # Save & check effects
    "SaveOverrideEffect",
    "SaveCheck",
    "RandomCheckEffect",
    "RollPatternEffect",
    "CheckModifierEffect",
    "CheckValueModifierEffect",
    # Heat & structure effects
    "HeatResistanceEffect",
    "StructureDamageAvoidanceEffect",
    "ZeroHpSurvivalEffect",
    # Resistance & immunity effects
    "Immunity",
    "TagImmunityEffect",
    "Resistance",
    # Accuracy effects
    "AccuracyModifier",
    # Special targeting effects
    "TargetMarkEffect",
    "IntelEffect",
    # Dice pool effects
    "DicePoolGain",
    "DicePoolSpendOption",
    "DicePoolEffect",
    "LeadershipDicePoolEffect",
    # Countdown effects
    "CountdownDieTrigger",
    "CountdownDieEffect",
    # Area effects
    "AreaSelectionEffect",
    # Repair effects
    "RepairCostModifier",
    "RepairShareEffect",
    "RepairActionEffect",
    # Deployment effects
    "DeploymentEffect",
    "AttachmentEffect",
    # Weapon effects
    "WeaponTagGrant",
    "WeaponRangeSpec",
    "WeaponSizeBonus",
    "WeaponGrantEffect",
    "WeaponModEffect",
    "WeaponSpinUpEffect",
    "WeaponAIControlEffect",
    # AI effects
    "AISystemLimitEffect",
    "AIControlTransferEffect",
    # Phase & protocol effects
    "PhaseShiftEffect",
    "ProtocolEffect",
    "CorePowerEffect",
    # Zone effects
    "ZoneEndCondition",
    "AttackCaptureEffect",
    "ZoneEffect",
    # Special combat effects
    "TetherEffect",
    "GrappleEffect",
    "SizeInteractionEffect",
    "MountedAllyEffect",
    "EffectRemoval",
    # Hologram effects
    "HolographicDuplicateEffect",
    "MovementTrailEffect",
    "HologramTrailEffect",
    # Reload effects
    "ReloadRestrictionEffect",
    "ReloadEffect",
    # Damage sharing effects
    "DamageShareEffect",
    "DamageNegationEffect",
    "DamageAbsorption",
    # Out of play effects
    "OutOfPlayEffect",
    # Mode & progression effects
    "ModeEffect",
    "ProgressionState",
    "GateProgressionEffect",
    "ProgressionEffect",
    # Counter effects
    "PerTargetCounter",
    "PerTargetCounterEffect",
    # Cooldown effects
    "CooldownState",
    "CooldownEffect",
    # Resource change effects
    "ResourceChange",
    "ScaledResourceChange",
    "OverchargeCostCapEffect",
    "LimitedUseBonusEffect",
    "LimitedUseRechargeEffect",
    # Allegiance effects
    "AllegianceShiftEffect",
    # Bond effects
    "BondmateEffect",
    # Delayed impact effects
    "DelayedImpactEffect",
    # Triggered effect
    "TriggeredEffect",
    # Choice effect
    "EffectChoice",
    # System link effect
    "SystemLinkEffect",
    # Main effect class
    "MechanicalEffect",
    # Convenience functions
    "stat_bonus",
    "damage_bonus",
    "immunity_to",
]

from typing import Literal
from pydantic import Field, model_validator
from core.shared.models import FrozenModel

from core.shared.enums import (
    ActionType,
    AttackType,
    CoverType,
    DamageType,
    MountSlotType,
    RangeType,
    SaveType,
    SizeClass,
    StatusType,
)
from core.shared.dice import DiceExpression
from core.shared.payloads import DamageSpec

# Import type aliases from types module
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


# NOTE: The following type aliases have been moved to core.shared.effects.types
# They are imported above and re-exported for backward compatibility.
# The original definitions have been removed from this file.


# Structured condition primitives - kept here as they reference FrozenModel
# TODO: These should eventually move to effects/conditions.py


# Type aliases moved to core.shared.effects.types - imported above


class SpatialCondition(FrozenModel):
    """
    Spatial relationship predicates (adjacent, range, zone entry, cover sourcing).

    Examples:
        SpatialCondition(relation="adjacent", target="ally")
        SpatialCondition(relation="outside_range", range=5, range_type="range")
    """

    relation: SpatialRelation
    target: EffectTarget = "enemy"
    source: EffectTargetNoAll = "self"
    range: int | None = Field(default=None, ge=0)
    range_type: RangeType | None = None
    cover: CoverType | None = None
    requires_line_of_sight: bool | None = None
    requires_free_space: bool | None = None


class AttackContextCondition(FrozenModel):
    """
    Attack context predicates (attack types, ranges, targeting scope).

    Examples:
        AttackContextCondition(attack_types=["ranged"], applies_to="incoming")
        AttackContextCondition(range_comparator="gt", range=5, applies_to="outgoing")
    """

    attack_types: list[AttackType] = Field(default_factory=list)
    area_shapes: list[AttackAreaShape] = Field(default_factory=list)
    weapon_ids: list[str] = Field(default_factory=list)
    applies_to: Literal["incoming", "outgoing", "mutual"] = "incoming"
    target: EffectTarget = "enemy"
    range_comparator: Literal["gt", "lt", "within"] | None = None
    range: int | None = Field(default=None, ge=0)
    range_type: RangeType | None = None
    requires_line_of_sight: bool | None = None
    requires_line_of_effect_crossing_zone: bool | None = None


class SizeCondition(FrozenModel):
    """
    Size comparison predicates for source/target/self.

    Examples:
        SizeCondition(subject="source", comparator="lt", size="size_5")
    """

    subject: Literal["source", "target", "self"] = "source"
    comparator: Literal["lt", "lte", "gt", "gte", "eq"] = "lt"
    size: SizeClass


class CheckContextCondition(FrozenModel):
    """
    Non-attack check/save predicates and equipment gating.

    Examples:
        CheckContextCondition(check_kinds=["save"], saves=["hull"])
        CheckContextCondition(equipment_materials=["metal"])
    """

    check_kinds: list[Literal["check", "save", "contested", "search"]] = Field(
        default_factory=list
    )
    saves: list[SaveType] = Field(default_factory=list)
    target: EffectTarget = "self"
    requires_line_of_sight: bool | None = None
    equipment_materials: list[str] = Field(default_factory=list)
    equipment_tags: list[str] = Field(default_factory=list)


class ReactionCondition(FrozenModel):
    """
    Reaction-specific predicates (brace, overwatch).

    Examples:
        ReactionCondition(reaction_id="brace")
        ReactionCondition(reaction_id="overwatch", is_attack=True)
    """

    reaction_id: str | None = None
    is_attack: bool | None = None

    @model_validator(mode="after")
    def _validate_reaction_condition(self) -> "ReactionCondition":
        """Ensure at least one condition parameter is set.

        Raises:
            ValueError: If both reaction_id and is_attack are None.
        """
        if self.reaction_id is None and self.is_attack is None:
            raise ValueError(
                "ReactionCondition must define 'reaction_id' or 'is_attack'"
            )
        return self


class ConditionGroup(FrozenModel):
    """
    Combine multiple conditions with AND/OR/NOT semantics.

    Examples:
        ConditionGroup(all_of=[SpatialCondition(relation="adjacent", target="ally")])
        ConditionGroup(
            all_of=[SpatialCondition(relation="using_cover_from_source", cover="hard")],
            any_of=[AttackContextCondition(area_shapes=["blast", "line", "cone"])],
        )
    """

    all_of: list["EffectCondition"] = Field(default_factory=list)
    any_of: list["EffectCondition"] = Field(default_factory=list)
    none_of: list["EffectCondition"] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_group(self) -> "ConditionGroup":
        """Ensure at least one condition list is defined.

        Raises:
            ValueError: If all_of, any_of, and none_of are all empty.
        """
        if not self.all_of and not self.any_of and not self.none_of:
            raise ValueError("ConditionGroup must define at least one condition list")
        return self


EffectCondition = (
    ConditionType
    | str
    | SpatialCondition
    | AttackContextCondition
    | CheckContextCondition
    | ReactionCondition
    | SizeCondition
    | ConditionGroup
)

# Additional type aliases moved to core.shared.effects.types


class StatModifier(FrozenModel):
    """
    Numeric modifier to a stat.

    Examples:
        StatModifier(stat="hp", value=5)  # +5 HP
        StatModifier(stat="size", value=1)  # +1 Size
    """

    stat: StatType
    value: int


class CompanionStatModifierEffect(FrozenModel):
    """
    Stat modifiers that apply to deployables, drones, or other companion entities.

    Examples:
        CompanionStatModifierEffect(stat="hp", value=5, applies_to=["drone", "deployable"])
    """

    stat: StatType
    value: int
    applies_to: list[Literal["drone", "deployable", "object"]] = Field(
        default_factory=list
    )
    duration: EffectDuration | None = None
    condition: EffectCondition | None = None


class StatOverrideEffect(FrozenModel):
    """
    Sets a stat to a specific value while a condition is met.

    Examples:
        StatOverrideEffect(stat="evasion", value=5, condition="reserve_power_mode")
    """

    stat: StatType
    value: int
    target: EffectTargetNoAll = "self"
    duration: EffectDuration | None = None
    condition: EffectCondition | None = None


class MountSlotGrant(FrozenModel):
    """
    Adds one or more mount slots.

    Examples:
        MountSlotGrant(slot_type="flexible", count=1, requires_mount_count_lt=3)
    """

    slot_type: MountSlotType
    count: int = Field(default=1, ge=1)
    requires_mount_count_lt: int | None = Field(default=None, ge=0)
    target: EffectTargetNoAll = "self"
    condition: EffectCondition | None = None


class MountSlotReplacement(FrozenModel):
    """
    Replaces existing mount slots with a new slot type.

    Examples:
        MountSlotReplacement(new_slot_type="main_aux", count=1)
    """

    new_slot_type: MountSlotType
    count: int = Field(default=1, ge=1)
    target: EffectTargetNoAll = "self"
    condition: EffectCondition | None = None


class MountSizeUpgradeEffect(FrozenModel):
    """
    Increases the weapon size that a mount can accept.

    Examples:
        MountSizeUpgradeEffect(increase_by=1, count=1)
    """

    increase_by: int = Field(default=1, ge=1)
    count: int = Field(default=1, ge=1)
    max_size: WeaponSizeType | None = None
    target: EffectTargetNoAll = "self"
    condition: EffectCondition | None = None


class IntegratedWeaponEffect(FrozenModel):
    """
    Grants an integrated weapon mount with linked free-fire rules.

    Examples:
        IntegratedWeaponEffect(weapon_size="aux", free_attack_uses_per="round")
    """

    weapon_size: WeaponSizeType = "aux"
    free_attack_action_type: ActionType = "free"
    free_attack_uses_per: UsesPer = "round"
    free_attack_trigger: TriggerType = "on_attack_roll"
    requires_other_weapon_attack: bool = True
    cannot_be_modified: bool = True
    target: EffectTargetNoAll = "self"
    condition: EffectCondition | None = None


class DamageModifier(FrozenModel):
    """
    Bonus damage under specific conditions.

    Examples:
        DamageModifier(flat=1, condition="melee_attack")  # +1 melee damage
        DamageModifier(dice=DiceExpression.parse("1d6"), damage_type="kinetic")  # +1d6 kinetic
    """

    dice: DiceExpression | None = Field(
        default=None, description="Bonus dice (e.g., DiceExpression.parse('1d6'))"
    )
    flat: int = Field(default=0, description="Flat bonus damage")
    damage_type: DamageType | None = Field(default=None)
    condition: EffectCondition | None = Field(default=None)


class DamageMultiplierEffect(FrozenModel):
    """
    Multiplies damage, heat, or burn dealt or taken.

    Examples:
        DamageMultiplierEffect(multiplier=0.5, applies_to="outgoing", condition="target_range_gt_5")
    """

    multiplier: float = Field(..., ge=0)
    damage_types: list[DamageTypeScope] = Field(default_factory=list)
    applies_to: Literal["outgoing", "incoming"] = "outgoing"
    target: EffectTarget = "enemy"
    condition: EffectCondition | None = None


class RangeModifier(FrozenModel):
    """
    Modifier to range or threat.

    Examples:
        RangeModifier(range_type="threat", value=1)  # +1 Threat
        RangeModifier(range_type="range", value=5)  # +5 Range
    """

    range_type: Literal["range", "threat", "sensors"]
    value: int
    condition: EffectCondition | None = None


class DirectDamage(FrozenModel):
    """Direct damage not tied to a standard weapon attack.

    Per PR2 3960-3964: Damage is calculated from dice and flat modifiers,
    then reduced by armor and resistance. AP ignores armor.

    Examples:
        DirectDamage(damage_type="explosive", dice=DiceExpression.parse("1d3"), ap=True)
    """

    damage_type: DirectDamageType
    dice: DiceExpression | None = None
    flat: int = 0
    multiplier: float = Field(default=1.0, ge=0)
    ap: bool = False
    target: EffectTargetWithObject = "enemy"
    condition: EffectCondition | None = None


class ActionGrant(FrozenModel):
    """
    Grants a new action or ability.

    Examples:
        ActionGrant(action_type="quick", name="Afterburner", trigger="after_boost")
        ActionGrant(action_type="reaction", name="Juke", trigger="on_successful_agility_save")
    """

    action_type: ActionType
    name: str
    trigger: TriggerType | str | None = Field(
        default=None, description="When this action can be used"
    )
    uses_per: UsesPer = "unlimited"


class ReactionLimitEffect(FrozenModel):
    """
    Adjusts the maximum number of reactions per turn.

    Examples:
        ReactionLimitEffect(max_reactions_per_turn=2)
    """

    max_reactions_per_turn: int | None = Field(default=None, ge=1)
    bonus_reactions_per_turn: int | None = Field(default=None, ge=1)
    target: EffectTargetNoAll = "self"
    condition: EffectCondition | None = None


class ReactionTriggerEffect(FrozenModel):
    """
    Grants or extends reaction triggers for a specific reaction action.

    Examples:
        ReactionTriggerEffect(reaction_id="overwatch", trigger_events=["enemy_enters_threat"])
    """

    reaction_id: str
    trigger_events: list[ReactionTriggerEvent] = Field(default_factory=list)
    uses_per: UsesPer = "unlimited"
    condition: EffectCondition | None = None


class NonCombatCapacityEffect(FrozenModel):
    """
    Non-combat capabilities such as extra limbs or passenger space.

    Examples:
        NonCombatCapacityEffect(extra_limb_pairs=1, interaction_scope="pilot_scale")
        NonCombatCapacityEffect(extra_passenger_slots=1, max_passenger_size="size_half")
    """

    non_combat_only: bool = True
    interaction_scope: NonCombatInteractionScope | None = None
    extra_limb_pairs: int | None = Field(default=None, ge=1)
    extra_passenger_slots: int | None = Field(default=None, ge=1)
    max_passenger_size: SizeClass | None = None
    passenger_location: PassengerLocation = "cockpit"
    passenger_protected_from_external_effects: bool = True
    statuses_if_unlicensed_pilot: list[StatusType] = Field(default_factory=list)
    condition: EffectCondition | None = None


class BondmateEffect(FrozenModel):
    """
    Defines a bondmate relationship for pilot talents or systems.

    Examples:
        BondmateEffect(allowed_target_types=["pilot", "npc"], can_change_between_missions=True)
    """

    allowed_target_types: list[Literal["pilot", "npc"]] = Field(default_factory=list)
    can_change_between_missions: bool = True
    target: EffectTargetNoAll = "self"
    condition: EffectCondition | None = None


class TargetMarkEffect(FrozenModel):
    """
    Marks a target and applies ongoing concentration-style effects.

    Examples:
        TargetMarkEffect(
            name="Mark for Death",
            action_type="full",
            range=30,
            min_range=5,
            self_effects=MechanicalEffect(...),
            attack_effects=MechanicalEffect(...),
        )
    """

    name: str
    action_type: ActionType
    target: Literal["enemy", "ally", "any"] = "enemy"
    range: int | None = Field(default=None, ge=0)
    min_range: int | None = Field(default=None, ge=0)
    requires_line_of_sight: bool = True
    duration: EffectDuration = "until_cleared"
    exclusive: bool = True
    end_action: ActionType | None = None
    end_timing: TriggerType | None = None
    self_effects: MechanicalEffect | None = None
    target_effects: MechanicalEffect | None = None
    attack_effects: MechanicalEffect | None = None
    condition: EffectCondition | None = None


class TechRange(FrozenModel):
    """
    Range descriptor for tech actions.

    Examples:
        TechRange(range_type="sensors")  # Within sensors
        TechRange(range_type="range", value=10)  # Range 10
    """

    range_type: TechRangeType = "sensors"
    value: int | None = Field(default=None, ge=0)


class TechActionOverrideEffect(FrozenModel):
    """
    Overrides tech action targeting or requirements.

    Examples:
        TechActionOverrideEffect(requires_line_of_sight=False, range_override=TechRange(range_type="range", value=50))
    """

    applies_to: TechActionScope = "all"
    requires_line_of_sight: bool | None = None
    range_override: TechRange | None = None
    target: EffectTargetNoAll = "self"
    condition: EffectCondition | None = None


class TechAction(FrozenModel):
    """Defines a tech action granted by a system.

    Per PR2 4060-4095: Tech actions target E-Defense and can apply status
    effects, debuffs, or direct damage. Some tech actions are attacks.

    Examples:
        TechAction(name="Track", action_type="quick", is_attack=True, range=TechRange(range_type="sensors"))
    """

    name: str
    action_type: ActionType
    target: EffectTargetWithObjectNoAll = "enemy"
    range: TechRange | None = None
    is_attack: bool = False
    attack_vs: Literal["e_defense", "evasion"] = "e_defense"
    effect: MechanicalEffect | None = None
    on_hit: MechanicalEffect | None = None
    on_miss: MechanicalEffect | None = None
    on_success: MechanicalEffect | None = None
    on_failure: MechanicalEffect | None = None
    uses_per: UsesPer = "unlimited"
    special: str | None = None


class TechAttackModifier(FrozenModel):
    """
    Accuracy/difficulty modifiers for tech attacks.

    Examples:
        TechAttackModifier(value=-1, target="ally", condition="adjacent", max_stacks=3)
    """

    value: int = Field(..., description="Positive = accuracy, negative = difficulty")
    target: EffectTarget = "self"
    condition: EffectCondition | None = None
    max_stacks: int | None = Field(default=None, ge=1)
    reset_trigger: (
        Literal["turn_start", "turn_end", "round_end", "scene_end"] | None
    ) = None


class TechActionRestriction(FrozenModel):
    """
    Restrictions or immunity affecting tech actions.

    Examples:
        TechActionRestriction(disallow_tech_actions=True, end_tech_effects=True)
    """

    disallow_tech_actions: bool = False
    immune_to_tech: bool = False
    end_tech_effects: bool = False
    target: EffectTarget = "self"
    condition: EffectCondition | None = None


class EffectChoice(FrozenModel):
    """
    Select one of multiple effects.

    Examples:
        EffectChoice(name="Option A", effect=MechanicalEffect(...))
    """

    name: str
    effect: MechanicalEffect
    target: EffectTarget = "enemy"
    range: TechRange | None = None
    condition: EffectCondition | None = None


class DicePoolGain(FrozenModel):
    """
    Gain dice in a named dice pool when a trigger occurs.

    Examples:
        DicePoolGain(trigger="on_hit", amount=1, uses_per="round")
    """

    trigger: TriggerType
    amount: int = Field(default=1, ge=1)
    uses_per: UsesPer = "unlimited"
    condition: EffectCondition | None = None
    requires_no_spend_this_turn: bool = False


class DicePoolSpendOption(FrozenModel):
    """
    Spend dice from a pool for a specific effect.

    Examples:
        DicePoolSpendOption(name="Parry", action_type="reaction", dice_cost=1,
                            effect=MechanicalEffect(damage_multipliers=[...]))
    """

    name: str
    effect: MechanicalEffect
    action_type: ActionType | None = None
    trigger: TriggerType | None = None
    dice_cost: int | None = Field(default=1)
    spend_any_number: bool = False
    spend_all: bool = False
    roll: DiceExpression | None = None
    roll_threshold: int | None = Field(default=None, ge=1)
    condition: EffectCondition | None = None
    effect_per_die: MechanicalEffect | None = None
    bonus_effect: MechanicalEffect | None = None
    bonus_requires_spend_at_least: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_spend_option(self) -> "DicePoolSpendOption":
        """Validate spend option configuration.

        Ensures dice_cost, spend_any_number, spend_all, and roll/roll_threshold
        are consistent.

        Raises:
            ValueError: If configuration is invalid.
        """
        if self.spend_all and not self.spend_any_number:
            raise ValueError("spend_any_number must be True when spend_all is True")
        if self.spend_any_number:
            if self.dice_cost is not None:
                raise ValueError("dice_cost must be None when spend_any_number is True")
        else:
            if self.dice_cost is None:
                raise ValueError("dice_cost is required when spend_any_number is False")
            if self.dice_cost < 1:
                raise ValueError("dice_cost must be >= 1")
        if (self.roll_threshold is None) ^ (self.roll is None):
            raise ValueError("roll and roll_threshold must be provided together")
        if self.spend_all and self.dice_cost is not None:
            raise ValueError("dice_cost must be None when spend_all is True")
        if self.effect_per_die and not (self.spend_any_number or self.spend_all):
            raise ValueError("effect_per_die requires spend_any_number or spend_all")
        if self.bonus_requires_spend_at_least is not None and self.bonus_effect is None:
            raise ValueError(
                "bonus_effect is required when bonus_requires_spend_at_least is set"
            )
        if self.bonus_effect is not None and not (
            self.spend_any_number or self.spend_all
        ):
            raise ValueError("bonus_effect requires spend_any_number or spend_all")
        return self


class DicePoolEffect(FrozenModel):
    """
    Named pool of dice that can be gained and spent for effects.

    Examples:
        DicePoolEffect(
            pool_name="blademaster",
            die_size=6,
            max_dice=3,
            gain_triggers=[DicePoolGain(trigger="on_hit", uses_per="round")],
            spend_options=[DicePoolSpendOption(name="Parry", action_type="reaction", dice_cost=1)],
        )
    """

    pool_name: str
    die_size: int = Field(default=6, ge=2)
    max_dice: int | None = Field(default=None, ge=1)
    starting_dice: int = Field(default=0, ge=0)
    gain_triggers: list[DicePoolGain] = Field(default_factory=list)
    spend_options: list[DicePoolSpendOption] = Field(default_factory=list)
    weapon_id: str | None = None
    expires_on_scene_end: bool = False
    lost_on_rest: bool = False
    lost_on_full_repair: bool = False
    condition: EffectCondition | None = None

    @model_validator(mode="after")
    def _validate_pool(self) -> "DicePoolEffect":
        """Validate dice pool configuration.

        Ensures starting_dice does not exceed max_dice.

        Raises:
            ValueError: If starting_dice exceeds max_dice.
        """
        if self.max_dice is not None and self.starting_dice > self.max_dice:
            raise ValueError("starting_dice cannot exceed max_dice")
        return self


class CountdownDieTrigger(FrozenModel):
    """
    Trigger that decrements a countdown die.

    Examples:
        CountdownDieTrigger(trigger="on_hit", condition="aux_ranged_attack", optional=True)
    """

    trigger: TriggerType
    decrement: int = Field(default=1, ge=1)
    optional: bool = False
    uses_per: UsesPer = "unlimited"
    condition: EffectCondition | None = None


class CountdownDieEffect(FrozenModel):
    """
    Countdown die that changes value based on triggers and can be spent at a threshold.

    Examples:
        CountdownDieEffect(
            die_name="gunslinger",
            die_size=6,
            starting_value=6,
            decrement_triggers=[CountdownDieTrigger(trigger="on_hit", condition="aux_ranged_attack", optional=True)],
            spend_requires_value=1,
            reset_value=6,
        )
    """

    die_name: str
    die_size: int = Field(default=6, ge=2)
    starting_value: int = Field(default=6, ge=1)
    minimum_value: int = Field(default=1, ge=1)
    decrement_triggers: list[CountdownDieTrigger] = Field(default_factory=list)
    spend_options: list[DicePoolSpendOption] = Field(default_factory=list)
    spend_requires_value: int = Field(default=1, ge=1)
    reset_value: int | None = Field(default=None, ge=1)
    expires_on_scene_end: bool = False
    lost_on_rest: bool = False
    lost_on_full_repair: bool = False
    condition: EffectCondition | None = None

    @model_validator(mode="after")
    def _validate_countdown(self) -> "CountdownDieEffect":
        """Validate countdown die configuration.

        Ensures starting_value, spend_requires_value, and reset_value are
        within valid ranges relative to minimum_value and die_size.

        Raises:
            ValueError: If any value is out of valid range.
        """
        if self.starting_value < self.minimum_value:
            raise ValueError("starting_value cannot be below minimum_value")
        if self.starting_value > self.die_size:
            raise ValueError("starting_value cannot exceed die_size")
        if self.spend_requires_value < self.minimum_value:
            raise ValueError("spend_requires_value cannot be below minimum_value")
        if self.spend_requires_value > self.die_size:
            raise ValueError("spend_requires_value cannot exceed die_size")
        if self.reset_value is not None and (
            self.reset_value < self.minimum_value or self.reset_value > self.die_size
        ):
            raise ValueError("reset_value must be within die range")
        return self
        return self


class LeadershipDicePoolEffect(FrozenModel):
    """
    Shared dice pool that can be granted to allies for specific effects.

    Examples:
        LeadershipDicePoolEffect(dice_count=3, grant_action_type="free")
    """

    pool_name: str = "leadership"
    die_size: int = Field(default=6, ge=2)
    dice_count: int = Field(..., ge=1)
    max_per_ally: int = Field(default=1, ge=1)
    grant_action_type: ActionType = "free"
    grant_trigger: TriggerType | None = None
    grant_requires_communication: bool = True
    recover_on_rest: int = Field(default=1, ge=0)
    recover_all_on_full_repair: bool = True
    replenish_requires_empty_pool: bool = True
    spend_options: list[EffectChoice] = Field(default_factory=list)
    condition: EffectCondition | None = None


class ActionRestriction(FrozenModel):
    """
    Restrictions for combat action usage.

    Examples:
        ActionRestriction(disallow_attack_rolls=True)
        ActionRestriction(action_ids=["hide"], target="enemy")
    """

    disallow_attack_rolls: bool = False
    disallow_heat_generation: bool = False
    action_ids: list[str] = Field(default_factory=list)
    action_categories: list[ActionCategoryType] = Field(default_factory=list)
    target: EffectTarget = "self"
    duration: EffectDuration = "end_of_turn"
    condition: EffectCondition | None = None


class StatusActionOverrideEffect(FrozenModel):
    """
    Allows actions that are normally restricted by a status.

    Examples:
        StatusActionOverrideEffect(status="jammed", allow_attack_types=["ranged"])
    """

    status: StatusType
    allow_attack_types: list[AttackType] = Field(default_factory=list)
    allow_action_categories: list[ActionCategoryType] = Field(default_factory=list)
    target: EffectTarget = "self"
    condition: EffectCondition | None = None


class LineOfSightRestriction(FrozenModel):
    """
    Restricts line of sight tracing for a target.

    Examples:
        LineOfSightRestriction(cannot_trace_outside_zone=True, target="all")
    """

    target: EffectTarget = "all"
    cannot_trace_outside_zone: bool = False
    only_adjacent: bool = False
    duration: EffectDuration = "end_of_turn"
    excludes_source: bool = False
    condition: EffectCondition | None = None


class Immunity(FrozenModel):
    """
    Immunity to a condition, damage type, or effect.

    Examples:
        Immunity(target="burn")  # Immune to Burn
        Immunity(target="knockback", condition="from_smaller")  # Immune to knockback from smaller
    """

    target: str = Field(
        ..., description="What you're immune to (condition, damage type, or effect)"
    )
    condition: EffectCondition | None = Field(
        default=None, description="Conditional immunity"
    )


class TagImmunityEffect(FrozenModel):
    """
    Immunity to effects or damage originating from tagged items.

    Examples:
        TagImmunityEffect(tags=["smart", "seeking"], immune_to_damage=True)
    """

    tags: list[str] = Field(default_factory=list)
    immune_to_damage: bool = True
    immune_to_effects: bool = True
    target: EffectTargetNoAll = "self"
    condition: EffectCondition | None = None


class Resistance(FrozenModel):
    """
    Resistance (half damage) to a damage type.

    Examples:
        Resistance(damage_type="energy")
        Resistance(damage_type="all", condition="can_see_bondmate")
    """

    damage_type: DamageType | Literal["all"]
    target: EffectTarget = "self"
    condition: EffectCondition | None = Field(default=None)
    duration: EffectDuration | None = None


class HeatResistanceEffect(FrozenModel):
    """
    Resistance (usually half) to heat gained from effects.

    Examples:
        HeatResistanceEffect(condition="blast_line_cone")
    """

    multiplier: float = Field(default=0.5, ge=0)
    target: EffectTarget = "self"
    condition: EffectCondition | None = None


class AccuracyModifier(FrozenModel):
    """
    Modifier to accuracy/difficulty on rolls.

    Examples:
        AccuracyModifier(value=1, condition="target_has_lock_on")  # +1 Accuracy vs Lock On
        AccuracyModifier(value=-1)  # -1 Accuracy (difficulty)
    """

    value: int = Field(..., description="Positive = accuracy, negative = difficulty")
    condition: EffectCondition | None = Field(default=None)
    applies_to: Literal["all", "melee", "ranged", "tech"] = "all"
    target: EffectTarget = "self"


class CheckModifierEffect(FrozenModel):
    """
    Modifier to accuracy/difficulty on non-attack checks or saves.

    Examples:
        CheckModifierEffect(value=-1, check_types=["engineering"], check_kinds=["check", "save"])
        CheckModifierEffect(value=1, condition="gm_approved_non_combat_check")
    """

    value: int = Field(..., description="Positive = accuracy, negative = difficulty")
    check_types: list[SaveType] = Field(default_factory=list)
    check_kinds: list[CheckKind] = Field(default_factory=list)
    target: EffectTargetNoAll = "self"
    condition: EffectCondition | None = Field(default=None)


class CheckValueModifierEffect(FrozenModel):
    """
    Flat modifier to non-attack checks or saves.

    Examples:
        CheckValueModifierEffect(value=2, check_kinds=["save"])
    """

    value: int
    check_types: list[SaveType] = Field(default_factory=list)
    check_kinds: list[CheckKind] = Field(default_factory=list)
    target: EffectTargetNoAll = "self"
    condition: EffectCondition | None = Field(default=None)


class DamageReduction(FrozenModel):
    """
    Flat damage reduction that applies before resistance.

    Examples:
        DamageReduction(amount=2, damage_type="all")
    """

    amount: int = Field(..., ge=0)
    damage_type: DamageType | Literal["all", "heat", "burn"] = "all"
    target: EffectTargetWithObject = "self"
    minimum_damage: int | None = Field(default=None, ge=0)
    condition: EffectCondition | None = None


class DamageReductionRollEffect(FrozenModel):
    """
    Damage reduction based on a roll.

    Examples:
        DamageReductionRollEffect(roll=DiceExpression.parse("1d6"))
    """

    roll: DiceExpression
    damage_type: DamageType | Literal["all", "heat", "burn"] = "all"
    target: EffectTargetWithObject = "self"
    minimum_damage: int | None = Field(default=None, ge=0)
    condition: EffectCondition | None = None


class MovementGrant(FrozenModel):
    """Grants movement or teleportation.

    Per PR2 4132-4145: Movement includes walking, flying, and teleporting.
    Movement grants can be triggered by actions, reactions, or conditions.

    Examples:
        MovementGrant(spaces=2, movement_type="fly", trigger="on_successful_save")
        MovementGrant(spaces=3, movement_type="teleport", trigger="after_boost")
        MovementGrant(spaces="speed", movement_type="walk", trigger="brace")
    """

    spaces: MovementDistanceType
    movement_type: Literal["walk", "fly", "teleport", "boost"] = "walk"
    trigger: TriggerType | str | None = Field(default=None)
    target: EffectTargetNoAll = "self"
    distance_is_maximum: bool = False
    requires_line_of_sight: bool = True
    requires_free_space: bool = False
    fails_if_occupied: bool = False
    ignores_engagement: bool = False
    provokes_reactions: bool = True
    condition: EffectCondition | None = None


class MoveAdjacentEffect(FrozenModel):
    """
    Moves the source adjacent to a target under specific conditions.

    Examples:
        MoveAdjacentEffect(target="enemy", movement_type="fly", trigger="on_turn_end")
    """

    target: Literal["enemy", "ally", "any"] = "enemy"
    movement_type: Literal["walk", "fly", "teleport"] = "walk"
    trigger: TriggerType | None = None
    action_type: ActionType | None = None
    requires_line_of_sight: bool = True
    requires_free_space: bool = True
    fails_if_occupied: bool = True
    ignores_engagement: bool = False
    provokes_reactions: bool = True
    uses_per: UsesPer = "unlimited"
    condition: EffectCondition | None = None


class PositionSwapEffect(FrozenModel):
    """
    Swaps positions between the source and a target.

    Examples:
        PositionSwapEffect(action_type="quick", range=10, range_type="sensors", uses_per="scene")
    """

    action_type: ActionType
    target: EffectTargetNoAll = "ally"
    range: int | None = Field(default=None, ge=0)
    range_type: TechRangeType = "sensors"
    requires_line_of_sight: bool = True
    uses_per: UsesPer = "scene"
    condition: EffectCondition | None = None


class ForcedMovement(FrozenModel):
    """Forced push/pull movement applied to a target.

    Per PR2 4146-4151: Forced movement moves targets against their will.
    Push moves away from source, pull moves toward source or zone.

    Examples:
        ForcedMovement(direction="pull", distance=5, ignores_engagement=True)
    """

    direction: Literal["pull", "push"]
    distance: ForcedMovementDistanceType
    target: EffectTarget = "enemy"
    toward: Literal["source", "zone_center"] = "source"
    ignores_engagement: bool = False
    provokes_reactions: bool = True
    must_obey_obstructions: bool = True
    on_collision: MechanicalEffect | None = None
    condition: EffectCondition | None = None


class StatusToggleEffect(FrozenModel):
    """
    Toggleable status at specific timing windows.

    Examples:
        StatusToggleEffect(status="invisible", activation_action="protocol",
                           activation_timing="on_turn_start",
                           deactivation_action="protocol",
                           deactivation_timing="on_turn_start",
                           duration="end_of_next_turn")
    """

    status: StatusType
    target: EffectTargetNoAll = "self"
    activation_action: ActionType = "protocol"
    activation_timing: TriggerType = "on_turn_start"
    deactivation_action: ActionType | None = None
    deactivation_timing: TriggerType | None = None
    duration: EffectDuration | None = None
    condition: EffectCondition | None = None


class StatusGrant(FrozenModel):
    """
    Grants or inflicts a status condition.

    Examples:
        StatusGrant(status="invisible", target="self", trigger="after_move_8_plus")
        StatusGrant(status="prone", target="enemy", trigger="on_hit")
    """

    status: StatusType | Literal["invisible", "flying"]
    target: EffectTarget
    trigger: TriggerType | str | None = Field(default=None)
    condition: EffectCondition | None = None
    duration: Literal[
        "end_of_turn",
        "start_of_next_turn",
        "end_of_next_turn",
        "until_cleared",
        "until_attack",
        "match_trigger",
        "scene",
    ] = "end_of_turn"


class StatusClear(FrozenModel):
    """
    Clears a status or condition from a target.

    Examples:
        StatusClear(status="burn", target="ally")
        StatusClear(status="any", target="self")
    """

    status: StatusType | Literal["burn", "any", "tech"]
    target: EffectTarget
    count: int = Field(default=1, ge=1)


class StatusBreakCondition(FrozenModel):
    """
    Defines triggers that end a status early.

    Examples:
        StatusBreakCondition(status="invisible", break_triggers=["move", "reaction"])
    """

    status: StatusType
    target: EffectTarget = "self"
    break_triggers: list[BreakTriggerType]
    condition: EffectCondition | None = None


class StatusStackLimit(FrozenModel):
    """
    Limits stacking or re-application of a status.

    Examples:
        StatusStackLimit(status="invisible", max_stacks=1)
    """

    status: StatusType
    max_stacks: int = Field(default=1, ge=1)
    target: EffectTarget = "self"
    condition: EffectCondition | None = None


class MovementScopedStatus(FrozenModel):
    """
    Status that applies only during movement.

    Examples:
        MovementScopedStatus(status="invisible", movement_modes=["any"], ends_on="movement_end")
    """

    status: StatusType
    target: EffectTargetNoAll
    movement_modes: list[MovementMode] = Field(default_factory=list)
    ends_on: Literal["movement_end", "turn_end"] = "movement_end"
    condition: EffectCondition | None = None


class StatusRestriction(FrozenModel):
    """
    Restricts gaining or benefiting from statuses.

    Examples:
        StatusRestriction(statuses=["invisible"], restriction="cannot_benefit", target="enemy")
    """

    statuses: list[StatusType]
    restriction: Literal["cannot_gain", "cannot_benefit"] = "cannot_gain"
    target: EffectTarget = "enemy"
    duration: EffectDuration = "end_of_turn"
    condition: EffectCondition | None = None


class AllegianceShiftEffect(FrozenModel):
    """
    Temporarily flips a target's allegiance.

    Examples:
        AllegianceShiftEffect(duration="end_of_next_turn", ends_on_hostile_action=True)
    """

    target: EffectTarget = "enemy"
    duration: EffectDuration = "end_of_next_turn"
    ends_on_hostile_action: bool = True
    condition: EffectCondition | None = None


class CoverRestriction(FrozenModel):
    """
    Restricts cover benefits for targets.

    Examples:
        CoverRestriction(max_cover="none", target="enemy")
    """

    max_cover: CoverType = "none"
    target: EffectTarget = "enemy"
    duration: EffectDuration = "end_of_turn"
    condition: EffectCondition | None = None


class CoverGrant(FrozenModel):
    """
    Grants cover to targets for a duration.

    Examples:
        CoverGrant(cover="soft", target="ally", duration="start_of_next_turn")
    """

    cover: CoverType
    target: EffectTarget = "ally"
    duration: EffectDuration = "end_of_turn"
    condition: EffectCondition | None = None


class IntelEffect(FrozenModel):
    """
    Reveals information or grants enhanced vision.

    Examples:
        IntelEffect(reveal=["location", "hp"], audience="self", duration="until_cleared")
    """

    reveal: list[IntelType] = Field(default_factory=list)
    audience: IntelAudience = "self"
    target: EffectTarget = "enemy"
    perfect_vision: bool = False
    grants_line_of_sight: bool = True
    duration: EffectDuration = "end_of_turn"
    condition: EffectCondition | None = None


class MovementRestrictionEffect(FrozenModel):
    """
    Movement restriction applied to a target.

    Examples:
        MovementRestrictionEffect(target="enemy", cannot_move_closer_to_source=True)
    """

    target: EffectTarget = "enemy"
    movement_modes: list[MovementMode] = Field(default_factory=list)
    max_voluntary_speed: int | None = Field(default=None, ge=0)
    cannot_move_closer_to_source: bool = False
    cannot_move_further_from_source: bool = False
    must_move_straight_line: bool = False
    duration: EffectDuration = "end_of_turn"
    condition: EffectCondition | None = None


class MovementSurfaceEffect(FrozenModel):
    """
    Modifies terrain/surface rules for movement.

    Examples:
        MovementSurfaceEffect(ignore_difficult_terrain=True)
        MovementSurfaceEffect(treat_vertical_as_ground=True, fall_if_prone=True)
    """

    target: EffectTarget = "self"
    ignore_difficult_terrain: bool = False
    treat_vertical_as_ground: bool = False
    fall_if_prone: bool = False
    duration: EffectDuration = "scene"
    condition: EffectCondition | None = None


class MovementModeAccessEffect(FrozenModel):
    """
    Grants access to movement modes or improves traversal.

    Examples:
        MovementModeAccessEffect(climb_at_full_speed=True, swim_at_full_speed=True)
    """

    target: EffectTargetNoAll = "self"
    climb_at_full_speed: bool = False
    swim_at_full_speed: bool = False
    condition: EffectCondition | None = None


class JumpDistanceEffect(FrozenModel):
    """
    Modifies jump distances relative to speed.

    Examples:
        JumpDistanceEffect(horizontal_multiplier=1.0, vertical_multiplier=0.5)
    """

    target: EffectTargetNoAll = "self"
    horizontal_multiplier: float | None = Field(default=None, ge=0)
    vertical_multiplier: float | None = Field(default=None, ge=0)
    condition: EffectCondition | None = None


class MovementOverrideEffect(FrozenModel):
    """
    Replaces movement modes with another type.

    Examples:
        MovementOverrideEffect(movement_modes=["move", "boost"], override_type="teleport")
    """

    target: EffectTargetNoAll = "self"
    movement_modes: list[MovementMode] = Field(default_factory=list)
    override_type: Literal["walk", "fly", "teleport"] = "teleport"
    same_distance: bool = True
    must_end_on_surface: bool = False
    duration: EffectDuration = "end_of_turn"
    condition: EffectCondition | None = None


class ResourceChange(FrozenModel):
    """
    Change a resource value such as HP or heat.

    Examples:
        ResourceChange(resource="hp", amount="half_max", target="ally", cost_repairs=1)
        ResourceChange(resource="heat", amount=DiceExpression.parse("1d6"), direction="lose", target="self")
    """

    resource: ResourceType
    amount: ResourceAmount
    direction: ResourceDirection = "gain"
    target: EffectTargetNoAll
    cost_repairs: int = Field(default=0, ge=0)
    cost_source: Literal["self", "target", "either"] = "self"


class ScaledResourceChange(FrozenModel):
    """
    Resource change based on a rolled value with a multiplier.

    Examples:
        ScaledResourceChange(resource="heat", roll=DiceExpression.parse("1d6"), multiplier=0.5)
    """

    resource: ResourceType
    roll: DiceExpression
    multiplier: float = Field(default=1.0, ge=0)
    direction: ResourceDirection = "gain"
    target: EffectTargetNoAll = "self"
    rounding: Literal["floor", "ceil", "round"] = "floor"
    condition: EffectCondition | None = None


class OverchargeCostCapEffect(FrozenModel):
    """
    Caps the heat cost for overcharge actions.

    Examples:
        OverchargeCostCapEffect(max_cost=DiceExpression.parse("1d6"))
    """

    max_cost: int | DiceExpression
    target: EffectTargetNoAll = "self"
    condition: EffectCondition | None = None


class LimitedUseBonusEffect(FrozenModel):
    """
    Adds limited-use charges to tagged systems, deployables, or weapons.

    Examples:
        LimitedUseBonusEffect(bonus_uses=2, applies_to=["system", "deployable"])
    """

    bonus_uses: int = Field(..., ge=1)
    applies_to: list[Literal["system", "deployable", "weapon"]] = Field(
        default_factory=list
    )
    requires_tag: str = "limited"
    condition: EffectCondition | None = None


class LimitedUseRechargeEffect(FrozenModel):
    """
    Replenishes limited-use charges as part of a rest or downtime action.

    Examples:
        LimitedUseRechargeEffect(bonus_uses=1, applies_to=["weapon", "deployable"], cost_repairs=2, uses_per="rest")
    """

    bonus_uses: int = Field(..., ge=1)
    applies_to: list[Literal["system", "deployable", "weapon"]] = Field(
        default_factory=list
    )
    requires_tag: str = "limited"
    cost_repairs: int = Field(default=0, ge=0)
    uses_per: UsesPer = "rest"
    target: EffectTarget = "self"
    condition: EffectCondition | None = None


class RepairCostModifier(FrozenModel):
    """
    Overrides repair costs for specific repair actions.

    Examples:
        RepairCostModifier(structure_repair_cost=1)
    """

    structure_repair_cost: int | None = Field(default=None, ge=0)
    target: EffectTargetNoAll = "self"
    condition: EffectCondition | None = None


class RepairShareEffect(FrozenModel):
    """
    Allows others to spend your repairs as their own.

    Examples:
        RepairShareEffect(target="ally", range=1, requires_adjacent=True, requires_choice=True)
    """

    target: EffectTargetNoAll = "ally"
    range: int | None = Field(default=None, ge=0)
    requires_adjacent: bool = False
    requires_choice: bool = False
    spend_repairs_as_own: bool = True
    condition: EffectCondition | None = None


class RepairActionEffect(FrozenModel):
    """
    Action-based repair options that spend repairs for a specific outcome.

    Examples:
        RepairActionEffect(
            name="combat_repair",
            action_type="full",
            repairs_cost=4,
            range=1,
            requires_adjacent=True,
            restore_structure=1,
            restore_hp=1,
            condition="target_destroyed",
        )
    """

    name: str
    action_type: ActionType
    repairs_cost: int = Field(..., ge=0)
    target: EffectTargetNoAll = "ally"
    range: int | None = Field(default=None, ge=0)
    requires_adjacent: bool = False
    requires_line_of_sight: bool = False
    restore_structure: int | None = Field(default=None, ge=0)
    restore_hp: int | None = Field(default=None, ge=0)
    condition: EffectCondition | None = None


class StructureDamageAvoidanceEffect(FrozenModel):
    """
    Chance to ignore structure damage, optionally healing after prevention.

    Examples:
        StructureDamageAvoidanceEffect(
            roll=DiceExpression.parse("1d6"),
            success_threshold=6,
            heal_hp_to=1,
            uses_per="full_repair",
            condition="structure_damage",
        )
    """

    trigger: TriggerType = "on_take_damage"
    roll: DiceExpression
    success_threshold: int = Field(..., ge=1)
    ignore_structure_damage: bool = True
    heal_hp_to: int | None = Field(default=None, ge=0)
    uses_per: UsesPer = "full_repair"
    target: EffectTargetNoAll = "self"
    condition: EffectCondition | None = None


class ZeroHpSurvivalEffect(FrozenModel):
    """
    Prevents destruction at 0 HP and enforces structure checks on damage.

    Examples:
        ZeroHpSurvivalEffect(block_hp_regain=True, duration="until_rest")
    """

    trigger: TriggerType = "on_take_damage"
    prevent_destruction_at_zero_hp: bool = True
    structure_check_on_damage: bool = True
    block_hp_regain: bool = True
    duration: OutOfPlayDuration = "until_rest"
    target: EffectTargetNoAll = "self"
    condition: EffectCondition | None = None


class AreaSelectionEffect(FrozenModel):
    """
    Specifies how areas or spaces are selected for an effect.

    Examples:
        AreaSelectionEffect(scope_options=["area"], shape="blast", size=1, count_options=[1, 2],
                            non_overlapping=True, range=20)
    """

    scope_options: list[AreaSelectionScope] = Field(default_factory=list)
    shape: ZoneShape | None = None
    size: int | None = Field(default=None, ge=0)
    count: int | None = Field(default=None, ge=1)
    count_options: list[int] = Field(default_factory=list)
    range: int | None = Field(default=None, ge=0)
    requires_line_of_sight: bool = True
    requires_free_space: bool = False
    non_adjacent: bool = False
    non_overlapping: bool = False
    visible_to_all: bool = False
    vertical_range: int | None = Field(default=None, ge=0)
    origin: Literal["self", "point"] = "self"


class AttackRollOverrideEffect(FrozenModel):
    """Overrides attack roll resolution details for a weapon or effect.

    Per PR2 3965-3969: Attack rolls use d20 + relevant skill + any bonuses.
    This effect allows overriding the target defense, attack type restrictions,
    and cover/line-of-sight handling.

    Examples:
        AttackRollOverrideEffect(attack_vs="evasion", fixed_target_defense=8, target="ally",
                                 allowed_attack_types=["ranged"])
    """

    attack_vs: Literal["evasion", "e_defense"] = "e_defense"
    fixed_target_defense: int | None = Field(default=None, ge=0)
    target: EffectTarget = "enemy"
    allowed_attack_types: list[AttackType] = Field(default_factory=list)
    respects_cover: bool = True
    respects_line_of_sight: bool = True
    condition: EffectCondition | None = None


class AttackTargetingEffect(FrozenModel):
    """
    Describes multi-target attack selection.

    Examples:
        AttackTargetingEffect(target_count_options=[1, 2], separate_attack_rolls=True)
    """

    target_count_options: list[int] = Field(default_factory=list)
    separate_attack_rolls: bool = True
    require_distinct_targets: bool = True
    condition: EffectCondition | None = None


class AreaAttackPattern(FrozenModel):
    """Defines multi-area attack patterns for weapons.

    Per PR2 3970-3974: Area attacks use geometric patterns (blast, burst, line, cone)
    to affect multiple targets simultaneously.

    Examples:
        AreaAttackPattern(area_shape="blast", area_size=1, area_count_options=[1, 2], non_overlapping=True)
    """

    area_shape: Literal["blast", "burst", "line", "cone"]
    area_size: int = Field(..., ge=0)
    area_count_options: list[int] = Field(default_factory=list)
    non_overlapping: bool = False


class LineAttackEffect(FrozenModel):
    """Line-based attack behavior that augments a weapon or action.

    Per PR2 3970-3974: Line attacks affect all targets in a straight line.
    This effect allows customizing line length, origin point, and obstacle handling.

    Examples:
        LineAttackEffect(between_self_and_target=True, heat_per_additional_target=1)
        LineAttackEffect(
            length=3,
            attack_origin_at_line_end=True,
            measure_range_from_line_end=True,
            measure_line_of_sight_from_line_end=True,
            line_passes_through_obstacles=True,
            line_damage=DirectDamage(damage_type="kinetic", dice=DiceExpression.parse("1d3"), ap=True),
            line_hits_objects=True,
        )
    """

    length: int | None = Field(
        default=None,
        ge=0,
        description="Line length in spaces; None = between self and target",
    )
    between_self_and_target: bool = False
    attack_origin_at_line_end: bool = False
    measure_range_from_line_end: bool = False
    measure_line_of_sight_from_line_end: bool = False
    measure_cover_from_line_end: bool = False
    line_passes_through_obstacles: bool = False
    fixed_direction_after_firing: bool = False
    line_hits_characters: bool = True
    line_hits_objects: bool = False
    line_damage: DirectDamage | None = None
    heat_per_additional_target: int | None = Field(default=None, ge=0)
    condition: EffectCondition | None = None


class AttackSequenceModifierEffect(FrozenModel):
    """
    Modifies accuracy/difficulty across a sequence of attacks in a turn.

    Examples:
        AttackSequenceModifierEffect(
            first_attack_accuracy=1,
            first_attack_optional=True,
            subsequent_attack_difficulty=1,
            duration="end_of_turn",
        )
    """

    trigger: TriggerType = "on_turn_start"
    first_attack_accuracy: int = 0
    first_attack_optional: bool = False
    subsequent_attack_difficulty: int = 0
    duration: EffectDuration = "end_of_turn"
    condition: EffectCondition | None = None


class AttackRerollEffect(FrozenModel):
    """
    Allows rerolling and retargeting attacks under constraints.

    Examples:
        AttackRerollEffect(trigger="on_miss", allowed_attack_types=["melee", "ranged"],
                           require_new_target=True, max_rerolls_per_attack=1)
    """

    trigger: TriggerType = "on_miss"
    allowed_attack_types: list[AttackType] = Field(default_factory=list)
    require_new_target: bool = False
    allow_same_area_for_aoe: bool = False
    aoe_types: list[RangeType] = Field(default_factory=list)
    max_rerolls_per_attack: int = Field(default=1, ge=1)
    disallow_already_hit_targets: bool = False
    keep_second: bool = False
    uses_reaction: bool = False
    duration: EffectDuration = "end_of_turn"
    uses_per: UsesPer = "unlimited"
    condition: EffectCondition | None = None


class AttackOutcomeEffect(FrozenModel):
    """
    Modifies the outcome of an attack (e.g., upgrade to critical).

    Examples:
        AttackOutcomeEffect(trigger="on_ally_hit", upgrade_to_crit=True, uses_reaction=True)
    """

    trigger: TriggerType
    upgrade_to_crit: bool = False
    force_miss: bool = False
    target: EffectTarget = "enemy"
    attacker: EffectTarget = "ally"
    uses_reaction: bool = False
    duration: EffectDuration = "end_of_turn"
    uses_per: UsesPer = "unlimited"
    condition: EffectCondition | None = None


class CriticalDamageOverrideEffect(FrozenModel):
    """
    Overrides critical damage resolution.

    Examples:
        CriticalDamageOverrideEffect(mode="max", requires_natural_20=True)
    """

    trigger: TriggerType = "on_crit"
    applies_to: list[AttackType] = Field(default_factory=list)
    mode: Literal["max"] = "max"
    requires_natural_20: bool = False
    target: EffectTarget = "enemy"
    attacker: EffectTarget = "self"
    condition: EffectCondition | None = None


class DamageRollOverrideEffect(FrozenModel):
    """
    Overrides how damage rolls are resolved.

    Examples:
        DamageRollOverrideEffect(mode="average", optional=True)
    """

    mode: Literal["average"] = "average"
    optional: bool = True
    target: EffectTargetNoAll = "self"
    condition: EffectCondition | None = None


class AccuracyTradeEffect(FrozenModel):
    """
    Trade accuracy dice for bonus effects.

    Examples:
        AccuracyTradeEffect(accuracy_cost=1, bonus_damage=DiceExpression.parse("1d6"), requires_crit=True)
    """

    accuracy_cost: int = Field(..., ge=1)
    bonus_damage: DiceExpression
    bonus_damage_type: DamageType | DirectDamageType | None = None
    requires_crit: bool = False
    trigger: TriggerType = "on_attack_roll"
    applies_to_attack_types: list[AttackType] = Field(default_factory=list)
    allowed_weapon_types: list[WeaponTypeType] = Field(default_factory=list)
    condition: EffectCondition | None = None


class DelayedImpactEffect(FrozenModel):
    """
    Optional delayed impact behavior for weapon attacks.

    Examples:
        DelayedImpactEffect(
            delay_timing="end_of_next_round",
            delayed_damage=DiceExpression.parse("3d6"),
            delayed_damage_type="explosive",
            self_slow_duration="end_of_next_turn",
            reveal_area=True,
        )
    """

    delay_optional: bool = True
    delay_timing: DelayedImpactTiming = "end_of_next_round"
    delayed_damage: DiceExpression | None = None
    delayed_damage_type: DamageType | None = None
    self_slow_duration: EffectDuration | None = None
    reveal_area: bool = False
    reveal_audience: IntelAudience = "all"


class WeaponTagGrant(FrozenModel):
    """
    Tag granted to a weapon.

    Examples:
        WeaponTagGrant(tag="ordnance")
    """

    tag: str
    value: int | None = None


class WeaponRangeSpec(FrozenModel):
    """
    Range entry for a granted weapon profile.

    Examples:
        WeaponRangeSpec(range_type="range", value=8)
    """

    range_type: RangeType
    value: int = Field(..., ge=0)


class WeaponSizeBonus(FrozenModel):
    """
    Size-based bonus for a weapon mod.

    Examples:
        WeaponSizeBonus(size="main", burn=2)
    """

    size: WeaponSizeType
    burn: int = Field(..., ge=0)


class WeaponGrantEffect(FrozenModel):
    """
    Grants a specific weapon profile as part of an effect.

    Examples:
        WeaponGrantEffect(
            weapon_id="nuclear_cavalier_fuel_rod_gun",
            name="Fuel Rod Gun",
            size="main",
            weapon_type="cqb",
            ranges=[WeaponRangeSpec(range_type="range", value=8)],
            damage=[DamageSpec(damage_type="energy", dice=DiceExpression.parse("1d3+2"))],
            limited_uses=3,
            integrated_mount=True,
        )
    """

    weapon_id: str | None = None
    name: str
    size: WeaponSizeType
    weapon_type: WeaponTypeType
    ranges: list[WeaponRangeSpec] = Field(default_factory=list)
    damage: list[DamageSpec] = Field(default_factory=list)
    tags: list[WeaponTagGrant] = Field(default_factory=list)
    limited_uses: int | None = Field(default=None, ge=0)
    unique: bool = False
    integrated_mount: bool = False
    target: EffectTargetNoAll = "self"
    condition: EffectCondition | None = None


class WeaponModEffect(FrozenModel):
    """
    Modifies a selected weapon with tags or bonuses.

    Examples:
        WeaponModEffect(
            allowed_weapon_types=["launcher", "cannon"],
            range_bonus=5,
            add_tags=[WeaponTagGrant(tag="ordnance")],
        )
    """

    allowed_weapon_types: list[WeaponTypeType] = Field(default_factory=list)
    allowed_weapon_sizes: list[WeaponSizeType] = Field(default_factory=list)
    range_bonus: int = 0
    add_tags: list[WeaponTagGrant] = Field(default_factory=list)
    burn_by_size: list[WeaponSizeBonus] = Field(default_factory=list)
    increase_existing_burn: bool = False
    damage_unreducible: bool = False
    condition: EffectCondition | None = None


class WeaponSpinUpEffect(FrozenModel):
    """
    Spin-up mode for a weapon with alternate behavior while active.

    Examples:
        WeaponSpinUpEffect(
            spin_up_action_type="quick",
            spin_down_action_type="free",
            spin_down_timing="start_of_turn",
            effects_while_spun_up=MechanicalEffect(
                status_grants=[StatusGrant(status="slowed", target="self", duration="until_cleared")],
                weapon_mods=[WeaponModEffect(add_tags=[WeaponTagGrant(tag="reliable", value=3)])],
            ),
        )
    """

    spin_up_action_type: ActionType
    spin_down_action_type: ActionType | None = None
    spin_down_timing: Literal["start_of_turn", "end_of_turn", "anytime"] | None = None
    allow_skirmish_with_base_profile: bool = False
    requires_barrage_while_spun_up: bool = False
    effects_while_spun_up: MechanicalEffect = Field(
        default_factory=lambda: MechanicalEffect()
    )
    condition: EffectCondition | None = None


class WeaponAIControlEffect(FrozenModel):
    """
    AI-controlled weapon handling rules.

    Examples:
        WeaponAIControlEffect(
            allowed_weapon_sizes=["aux", "main", "heavy"],
            free_attack_difficulty=2,
            cannot_fire_if_used_this_turn=True,
            weapon_locked_until_next_turn_after_ai_attack=True,
            ai_selects_target_if_unshackled=True,
        )
    """

    allowed_weapon_sizes: list[WeaponSizeType] = Field(default_factory=list)
    free_attack_action_type: ActionType = "free"
    free_attack_uses_per: UsesPer = "round"
    free_attack_difficulty: int = 0
    cannot_fire_if_used_this_turn: bool = False
    weapon_locked_until_next_turn_after_ai_attack: bool = False
    ai_selects_target_if_unshackled: bool = False
    ai_control_scope: Literal["weapon_only", "mech"] = "weapon_only"
    condition: EffectCondition | None = None


class AISystemLimitEffect(FrozenModel):
    """
    Adjusts the maximum number of AI-tagged systems that can be installed.

    Examples:
        AISystemLimitEffect(bonus_systems=1, max_ai_systems=2)
    """

    bonus_systems: int = 0
    max_ai_systems: int | None = Field(default=None, ge=0)
    condition: EffectCondition | None = None


class AIControlTransferEffect(FrozenModel):
    """
    Handles AI control transfer when another AI becomes unshackled.

    Examples:
        AIControlTransferEffect(transfer_on_unshackle=True)
    """

    transfer_on_unshackle: bool = True
    source_tag: str = "ai"
    target_tag: str = "ai"
    condition: EffectCondition | None = None


class DeploymentEffect(FrozenModel):
    """
    Represents deploying a device that primes and can be activated later.

    Examples:
        DeploymentEffect(
            action_type="quick",
            placement_range=1,
            placement_relation="adjacent",
            primes_after="turn_end",
            activation_condition="adjacent_start_or_move",
            activation_action="quick",
            activation_target="ally",
            consumes_on_activation=True,
        )
    """

    action_type: ActionType
    placement_range: int = Field(..., ge=0)
    placement_relation: Literal["adjacent", "range"] = "adjacent"
    placement_requires_free_space: bool = False
    primes_after: Literal["turn_end", "immediate"] = "immediate"
    activation_condition: DeploymentActivationCondition
    activation_action: ActionType | None = None
    activation_target: EffectTargetNoAll = "self"
    activation_effect: MechanicalEffect | None = None
    consumes_on_activation: bool = True
    becomes_object_on_activation: bool = False
    open_topped: bool = False
    immobile: bool = False
    can_deactivate: bool = True


class AttachmentEffect(FrozenModel):
    """
    Attaches a device or effect to a target.

    Examples:
        AttachmentEffect(action_type="quick", range=5, target="ally")
    """

    action_type: ActionType
    range: int = Field(..., ge=0)
    target: EffectTargetNoAll = "ally"
    requires_line_of_sight: bool = True
    max_instances_per_target: int | None = Field(default=None, ge=1)
    duration: EffectDuration | None = None
    condition: EffectCondition | None = None


class SystemLinkEffect(FrozenModel):
    """
    Persistent link between two characters for shared systems/conditions.

    Examples:
        SystemLinkEffect(
            action_type="quick",
            range=10,
            range_type="sensors",
            target="ally",
            duration="scene",
            tech_action_uses_target_sensors=True,
            tech_action_uses_target_los=True,
            check_stat_proxy_to_target=["systems"],
            check_kinds=["check", "save"],
            share_conditions=True,
            share_heat_from_tech=True,
        )
    """

    action_type: ActionType
    range: int | None = Field(default=None, ge=0)
    range_type: TechRangeType = "sensors"
    target: EffectTargetNoAll = "ally"
    requires_line_of_sight: bool = True
    duration: EffectDuration | None = None
    max_links_per_source: int | None = Field(default=None, ge=1)
    breaks_on_out_of_range: bool = False
    break_if_statuses: list[StatusType] = Field(default_factory=list)
    shares_space: bool = False
    moves_with_target: bool = False
    cover_from_target: CoverType | None = None
    share_conditions: bool = False
    share_heat_from_tech: bool = False
    share_heat_all: bool = False
    tech_action_uses_target_sensors: bool = False
    tech_action_uses_target_los: bool = False
    stat_proxy_to_target: list[StatType] = Field(default_factory=list)
    stat_proxy_to_source: list[StatType] = Field(default_factory=list)
    check_stat_proxy_to_target: list[SaveType] = Field(default_factory=list)
    check_stat_proxy_to_source: list[SaveType] = Field(default_factory=list)
    check_kinds: list[CheckKind] = Field(default_factory=list)
    condition: EffectCondition | None = None


class PhaseShiftEffect(FrozenModel):
    """
    Phase/intangible state changes with periodic checks.

    Examples:
        PhaseShiftEffect(
            activation_action="quick",
            roll=DiceExpression.parse("1d6"),
            success_threshold=4,
            out_of_phase_duration="start_of_next_turn",
            duration="scene",
        )
    """

    activation_action: ActionType
    starts_out_of_phase: bool = True
    roll_trigger: TriggerType = "on_turn_start"
    roll: DiceExpression
    success_threshold: int = Field(..., ge=1)
    out_of_phase_duration: Literal["start_of_next_turn", "end_of_turn"] = (
        "start_of_next_turn"
    )
    duration: EffectDuration = "scene"
    deactivation_action: ActionType | None = None
    intangible: bool = True
    ignore_obstructions: bool = True
    cannot_end_in_obstruction: bool = True
    cannot_interact: bool = True
    immune_to_damage: bool = True


ZoneShape = Literal["burst", "blast", "line", "cone", "square"]


class ZoneEndCondition(FrozenModel):
    """
    Describes when a zone ends early.

    Examples:
        ZoneEndCondition(trigger="enter", end_scope="triggered_space")
    """

    trigger: ZoneEndTriggerType
    target: EffectTarget = "all"
    end_scope: ZoneEndScope = "zone"
    condition: EffectCondition | None = None


class AttackCaptureEffect(FrozenModel):
    """
    Captures attacks that cross a zone boundary and resolves them later.

    Examples:
        AttackCaptureEffect(
            attack_types=["ranged"],
            damage_types=["kinetic", "explosive"],
            capture_max=6,
            resolve_trigger="turn_end",
            attack_bonus_per_capture=1,
            attack_bonus_max=6,
            damage_per_capture=DiceExpression.parse("1d6"),
            damage_type="kinetic",
        )
    """

    attack_types: list[AttackType] = Field(default_factory=list)
    damage_types: list[DamageType] = Field(default_factory=list)
    block_crossing_boundary: bool = True
    allow_internal_attacks: bool = True
    capture_attacks: bool = True
    capture_max: int | None = Field(default=None, ge=0)
    resolve_trigger: ZoneEndTriggerType = "turn_end"
    resolve_scope: Literal["targets_in_zone"] = "targets_in_zone"
    attack_bonus_per_capture: int = Field(default=0, ge=0)
    attack_bonus_max: int | None = Field(default=None, ge=0)
    damage_per_capture: DiceExpression | None = None
    damage_type: DamageType | None = None
    negate_damage_on_capture: bool = True


class ZoneEffect(FrozenModel):
    """
    Persistent area effects such as hazard zones or shield fields.

    Examples:
        ZoneEffect(shape="burst", size=1, duration="scene", difficult_terrain=True)
    """

    shape: ZoneShape
    size: int | None = Field(default=None, ge=0)
    width: int | None = Field(default=None, ge=0)
    height: int | None = Field(default=None, ge=0)
    placement: Literal["self", "target_area", "deployable"] = "target_area"
    placement_range: int | None = Field(default=None, ge=0)
    placement_requires_line_of_sight: bool = False
    placement_requires_free_space: bool = False
    placement_requires_adjacent: bool = False
    placement_non_adjacent: bool = False
    placement_visible_to_all: bool = False
    vertical_range: int | None = Field(default=None, ge=0)
    retarget_action: ActionType | None = None
    retarget_range: int | None = Field(default=None, ge=0)
    retarget_requires_line_of_sight: bool = False
    retarget_replaces_existing: bool = True
    duration: Literal[
        "end_of_turn",
        "start_of_next_turn",
        "end_of_next_turn",
        "scene",
    ] = "scene"
    difficult_terrain: bool = False
    extinguishes_fires: bool = False
    blocks_movement: bool = False
    blocks_movement_condition: EffectCondition | None = None
    counts_as_obstruction: bool = False
    blocks_line_of_sight: bool = False
    cover: CoverType | None = None
    cover_all_directions: bool = False
    applies_to: Literal["all", "ally", "enemy", "object"] = "all"
    effects_on_enter: MechanicalEffect | None = None
    effects_on_start_turn: MechanicalEffect | None = None
    effects_on_end_turn: MechanicalEffect | None = None
    continuous_effects: MechanicalEffect | None = None
    attack_capture: AttackCaptureEffect | None = None
    total_effect_cap: int | None = Field(default=None, ge=0)
    deactivate_on_effect_cap: bool = False
    ends_on_source_destroyed: bool = False
    max_instances_per_source: int | None = Field(default=None, ge=1)
    end_conditions: list[ZoneEndCondition] = Field(default_factory=list)
    condition: EffectCondition | None = None


class TetherEffect(FrozenModel):
    """
    Represents a tether/drag connection between two entities.

    Examples:
        TetherEffect(action_type="quick", range=1, max_distance=5, tow_slowed=True)
    """

    action_type: ActionType
    range: int = Field(..., ge=0)
    max_distance: int = Field(..., ge=0)
    tow_slowed: bool = False
    auto_attach_if_willing: bool = False
    auto_attach_if_stunned: bool = False
    detach_on_hit: bool = True
    detach_attack_evasion: int | None = Field(default=None, ge=0)
    can_attach_to_objects: bool = False
    object_attach_range: int | None = Field(default=None, ge=0)
    object_strain_capacity: int | None = Field(default=None, ge=0)
    climb_no_speed_penalty: bool = False


class GrappleEffect(FrozenModel):
    """
    Modifies grapple behavior or grants grapple-related options.

    Examples:
        GrappleEffect(range=5, pull_grappler_adjacent=True, break_if_no_adjacent_path=True)
        GrappleEffect(allow_boost_while_grappling=True, allow_reactions_while_grappling=True)
        GrappleEffect(
            movement_trigger="move",
            movement_type="fly",
            movement_distance="speed",
            movement_must_be_straight_line=True,
            movement_requires_surface_end=True,
            movement_can_hold_surface_if_immobile=True,
            fall_if_prone_or_knockback=True,
        )
    """

    range: int | None = Field(default=None, ge=0)
    requires_line_of_sight: bool = True
    pull_grappler_adjacent: bool = False
    pull_target_adjacent: bool = False
    break_if_no_adjacent_path: bool = False
    allow_boost_while_grappling: bool | None = None
    allow_reactions_while_grappling: bool | None = None
    movement_trigger: MovementMode | None = None
    movement_uses_per: UsesPer | None = None
    movement_type: Literal["walk", "fly", "teleport"] | None = None
    movement_distance: MovementDistanceType | None = None
    movement_requires_clear_path: bool = True
    movement_must_be_straight_line: bool = False
    movement_requires_surface_end: bool = False
    movement_can_hold_surface_if_immobile: bool = False
    fall_if_prone_or_knockback: bool = False
    drag_down_action_type: ActionType | None = None
    drag_down_range: int | None = Field(default=None, ge=0)
    drag_down_requires_line_of_sight: bool = True
    drag_down_contested_stat: SaveType | None = None
    drag_down_knocks_prone: bool = False
    break_all_grapples: bool = False
    condition: EffectCondition | None = None


class SizeInteractionEffect(FrozenModel):
    """
    Adjusts size comparisons and carry capacity for specific actions.

    Examples:
        SizeInteractionEffect(
            applies_to_actions=["ram", "grapple"],
            treat_as_same_size_if_target_larger=True,
            treat_as_larger_if_target_same_or_smaller=True,
            lift_capacity_multiplier=2,
            drag_capacity_multiplier=2,
        )
    """

    applies_to_actions: list[Literal["ram", "grapple"]] = Field(default_factory=list)
    treat_as_same_size_if_target_larger: bool = False
    treat_as_larger_if_target_same_or_smaller: bool = False
    lift_capacity_multiplier: int | None = Field(default=None, ge=1)
    drag_capacity_multiplier: int | None = Field(default=None, ge=1)
    condition: EffectCondition | None = None


class MountedAllyEffect(FrozenModel):
    """
    Allows allies to mount and ride on a mech.

    Examples:
        MountedAllyEffect(max_total_size_relative="self_minus_half", mount_action_type="quick")
    """

    max_total_size_relative: Literal["self_minus_half"] | None = None
    max_total_size: SizeClass | None = None
    mount_action_type: ActionType = "quick"
    requires_adjacent: bool = True
    disallow_target_statuses: list[StatusType] = Field(default_factory=list)
    provides_soft_cover: bool = True
    shares_space: bool = True
    moves_with_carrier: bool = True
    dismount_on_carrier_statuses: list[StatusType] = Field(default_factory=list)
    dismount_on_carrier_destroyed: bool = False
    dismount_on_rider_statuses: list[StatusType] = Field(default_factory=list)
    knocks_prone_on_dismount: bool = False
    condition: EffectCondition | None = None


class EffectRemoval(FrozenModel):
    """
    Describes how an ongoing effect can be removed.

    Examples:
        EffectRemoval(action_type="quick", check_type="engineering", check_kind="check")
    """

    action_type: ActionType
    check_type: SaveType | None = None
    check_kind: Literal["check", "save"] = "check"
    target: EffectTargetNoAll = "self"
    condition: EffectCondition | None = None


class HolographicDuplicateEffect(FrozenModel):
    """
    Creates a holographic duplicate decoy.

    Examples:
        HolographicDuplicateEffect(duration="scene", breaks_on_hit=True)
    """

    duration: EffectDuration = "scene"
    breaks_on_hit: bool = True
    target: EffectTargetNoAll = "self"
    condition: EffectCondition | None = None


class MovementTrailEffect(FrozenModel):
    """
    Creates a persistent trail as you move or boost.

    Examples:
        MovementTrailEffect(trigger="move_or_boost", cover="hard", trail_height=1)
    """

    trigger: HologramTrailTrigger
    trail_height: int = Field(default=1, ge=0)
    length_matches_movement: bool = True
    cover: CoverType | None = None
    cover_all_directions: bool = False
    applies_to_adjacent: bool = True
    resistances: list[Resistance] = Field(default_factory=list)
    blocks_movement: bool = False
    counts_as_obstruction: bool = False
    creation_duration: EffectDuration = "end_of_turn"
    trail_duration: EffectDuration = "scene"
    ends_on_reactivation: bool = False
    condition: EffectCondition | None = None


class HologramTrailEffect(FrozenModel):
    """
    Trail of holograms that detonate and allow teleportation.

    Examples:
        HologramTrailEffect(
            trigger="move_or_boost",
            detonation_damage=DiceExpression.parse("1d6"),
            detonation_damage_type="energy",
            detonation_save="agility",
            teleport_action="quick",
            teleport_range=50,
            detonate_all_burst=1,
            suppress_new_until="start_of_next_turn",
        )
    """

    trigger: HologramTrailTrigger
    hologram_size: Literal["match_self"] = "match_self"
    detonation_triggers: list[HologramDetonationTrigger]
    detonation_damage: DiceExpression
    detonation_damage_type: DamageType
    detonation_save: SaveType
    detonation_half_on_success: bool = True
    detonation_targets_hostile_only: bool = True
    teleport_action: ActionType
    teleport_range: int = Field(..., ge=0)
    detonate_all_on_teleport: bool = True
    detonate_all_burst: int = Field(default=1, ge=0)
    suppress_new_until: Literal["start_of_next_turn", "end_of_next_turn"] | None = None
    duration: EffectDuration = "scene"


class ReloadRestrictionEffect(FrozenModel):
    """
    Restricts reload actions for specific weapon types.

    Examples:
        ReloadRestrictionEffect(applies_to="limited", disallow_reload=True)
    """

    applies_to: Literal["limited", "all"] = "limited"
    disallow_reload: bool = True
    target: EffectTargetNoAll = "self"
    condition: EffectCondition | None = None


class ReloadEffect(FrozenModel):
    """Reloads one or more weapons, optionally filtered by tag.

    Per PR2 4332-4334: The Reload quick action reloads all loading weapons
    on a mount or reloads a specific weapon.

    Examples:
        ReloadEffect(target="ally", count=1, requires_tag="loading")
        ReloadEffect(target="self", count="all", requires_tag="loading")
    """

    target: EffectTargetNoAll
    count: int | Literal["all"] = 1
    requires_tag: str | None = "loading"
    consumes_source: bool = False


class DamageShareEffect(FrozenModel):
    """
    Shares a fraction of damage from a target to a source.

    Examples:
        DamageShareEffect(share_fraction=0.5, source="self", target="ally", requires_adjacent=True)
    """

    share_fraction: float = Field(..., ge=0, le=1)
    source: EffectTargetNoAll = "self"
    target: EffectTargetNoAll = "ally"
    timing: Literal["before_armor_and_reduction", "after_armor_and_reduction"] = (
        "before_armor_and_reduction"
    )
    requires_adjacent: bool = False
    breaks_on_separation: bool = False
    condition: EffectCondition | None = None


class DamageNegationEffect(FrozenModel):
    """
    Negates an incoming damage instance.

    Examples:
        DamageNegationEffect(target="self")
    """

    target: EffectTargetNoAll = "self"
    negate_damage: bool = True
    negate_heat: bool = False
    negate_burn: bool = False
    condition: EffectCondition | None = None


class DamageAbsorption(FrozenModel):
    """
    Absorbs damage before it affects the target.

    Examples:
        DamageAbsorption(target="ally", base_hp=4, bonus_hp_per_grit=1)
    """

    target: EffectTargetNoAll
    base_hp: int = Field(..., ge=0)
    bonus_hp_per_grit: int = Field(default=0, ge=0)
    max_instances_per_target: int = Field(default=1, ge=1)
    spillover: bool = True
    ends_on_zero: bool = True
    duration: Literal["scene", "until_destroyed"] = "until_destroyed"


class OutOfPlayEffect(FrozenModel):
    """
    Removes a target from play for a duration.

    Examples:
        OutOfPlayEffect(target="self", duration="until_rest", gm_may_override=True)
    """

    target: EffectTargetNoAll = "self"
    duration: OutOfPlayDuration = "until_cleared"
    return_to_previous_space: bool = True
    fallback_to_nearest_free: bool = True
    gm_may_override: bool = False
    condition: EffectCondition | None = None


class SaveOverrideEffect(FrozenModel):
    """
    Overrides or guarantees outcomes for saves or contested checks.

    Examples:
        SaveOverrideEffect(saves=["hull", "agility"], auto_pass=True, include_contested_checks=True)
    """

    saves: list[SaveType] = Field(default_factory=list)
    auto_pass: bool = False
    auto_fail: bool = False
    include_contested_checks: bool = False
    duration: EffectDuration = "end_of_turn"
    condition: EffectCondition | None = None


class SaveCheck(FrozenModel):
    """
    Save-based conditional effect.

    Example:
        SaveCheck(
            trigger="on_hit",
            save="hull",
            on_failure=MechanicalEffect(status_grants=[StatusGrant(status="prone", target="enemy")]),
        )
    """

    trigger: TriggerType = "on_hit"
    condition: EffectCondition | None = None
    save: SaveType
    target: EffectTarget = "enemy"
    on_success: MechanicalEffect | None = None
    on_failure: MechanicalEffect | None = None


class RandomCheckEffect(FrozenModel):
    """
    Random check with success/failure effects.

    Examples:
        RandomCheckEffect(
            trigger="on_ally_damaged",
            roll=DiceExpression.parse("1d6"),
            success_threshold=4,
            on_success=MechanicalEffect(...),
        )
    """

    trigger: TriggerType
    roll: DiceExpression
    success_threshold: int = Field(..., ge=1)
    target: EffectTargetNoAll = "ally"
    on_success: MechanicalEffect | None = None
    on_failure: MechanicalEffect | None = None
    uses_per: UsesPer = "unlimited"
    condition: EffectCondition | None = None


class RollPatternEffect(FrozenModel):
    """
    Triggers an effect based on a roll pattern (e.g., triples).

    Examples:
        RollPatternEffect(trigger="on_activation", roll=DiceExpression.parse("3d6"),
                          pattern="triples", effect=MechanicalEffect(...))
    """

    trigger: TriggerType
    roll: DiceExpression
    pattern: RollPatternType
    target: EffectTargetNoAll = "self"
    effect: MechanicalEffect
    condition: EffectCondition | None = None


class StatusTrigger(FrozenModel):
    """
    Effect triggered by inflicting or clearing a status.

    Examples:
        StatusTrigger(trigger="on_inflict", status="immobilized",
                      effect=MechanicalEffect(status_grants=[StatusGrant(status="shredded", target="enemy",
                      duration="match_trigger")]))
    """

    trigger: Literal["on_inflict", "on_clear"]
    status: StatusType
    target: EffectTargetNoAll = "enemy"
    effect: MechanicalEffect
    condition: EffectCondition | None = None
    uses_per: UsesPer = "unlimited"


class TriggeredEffect(FrozenModel):
    """
    Effect that only applies when a trigger occurs.

    Examples:
        TriggeredEffect(
            trigger="on_reload",
            effect=MechanicalEffect(action_grants=[ActionGrant(action_type="free", name="reload_fire")]),
        )
    """

    trigger: TriggerType
    condition: EffectCondition | None = None
    effect: MechanicalEffect
    uses_per: UsesPer = "unlimited"


class ModeEffect(FrozenModel):
    """
    Toggleable mode that bundles multiple effects.

    Examples:
        ModeEffect(
            name="Reserve Power Mode",
            activation_action_id="shutdown",
            activation_action_type="quick",
            deactivation_action_id="boot_up",
            deactivation_action_type="full",
            effects=MechanicalEffect(...),
        )
    """

    name: str
    activation_action_id: str
    activation_action_type: ActionType
    deactivation_action_id: str | None = None
    deactivation_action_type: ActionType | None = None
    duration: EffectDuration | None = None
    effects: MechanicalEffect = Field(default_factory=lambda: MechanicalEffect())
    condition: EffectCondition | None = None


class ProtocolEffect(FrozenModel):
    """
    Protocol action that grants ongoing effects.

    Examples:
        ProtocolEffect(
            name="Skirmisher Protocol",
            duration="scene",
            effects=MechanicalEffect(movement_grants=[MovementGrant(spaces=4, movement_type="walk")]),
        )
    """

    name: str
    action_type: ActionType = "protocol"
    trigger: TriggerType = "on_activation"
    duration: EffectDuration | None = None
    effects: MechanicalEffect = Field(default_factory=lambda: MechanicalEffect())


class CorePowerEffect(FrozenModel):
    """
    Core power activation and effects.

    Examples:
        CorePowerEffect(
            name="Skirmisher Protocol",
            action_type="protocol",
            duration="scene",
            effects=MechanicalEffect(movement_grants=[MovementGrant(spaces=4, movement_type="walk")]),
        )
    """

    name: str
    action_type: ActionType
    trigger: TriggerType = "on_activation"
    duration: EffectDuration | None = None
    effects: MechanicalEffect = Field(default_factory=lambda: MechanicalEffect())


ProgressionResetTrigger = Literal["scene_end", "rest", "full_repair", "never"]


class ProgressionState(FrozenModel):
    """
    Tracks sequential progression through numbered states (gates).

    Used for effects like OSIRIS NHP gates where effects unlock sequentially.

    Examples:
        ProgressionState(current_gate=1, max_gate=4, reset_on="rest", per_target=True)
    """

    current_gate: int = Field(default=1, ge=1, le=4)
    max_gate: int = Field(default=4, ge=1)
    reset_on: ProgressionResetTrigger = "scene_end"
    per_target: bool = True
    target_id: str | None = None


class GateProgressionEffect(FrozenModel):
    """
    Effect that unlocks based on previous gate completion.

    Used for sequential effects like OSIRIS NHP where each gate unlocks the next.

    Examples:
        GateProgressionEffect(
            gate_number=2,
            prerequisite_gate=1,
            effect=MechanicalEffect(status_grants=[...])
        )
    """

    gate_number: int = Field(..., ge=1, le=4)
    prerequisite_gate: int | None = Field(default=None, ge=1, le=4)
    effect: "MechanicalEffect"
    condition: EffectCondition | None = None


class ProgressionEffect(FrozenModel):
    """
    Tracks sequential gate progression and applies effects at each gate.

    Examples:
        ProgressionEffect(
            progression_name="OSIRIS_Gates",
            reset_on="rest",
            max_gate=4,
            gates=[
                GateProgressionEffect(gate_number=1, effect=MechanicalEffect(...)),
                GateProgressionEffect(gate_number=2, prerequisite_gate=1, effect=MechanicalEffect(...)),
            ]
        )
    """

    progression_name: str
    reset_on: ProgressionResetTrigger = "scene_end"
    max_gate: int = Field(default=4, ge=1)
    per_target: bool = True
    gates: list[GateProgressionEffect] = Field(default_factory=list)
    condition: EffectCondition | None = None


class PerTargetCounter(FrozenModel):
    """
    Tracks effect usage per specific target.

    Used for effects like Basilisk stun or H0r_OS invasions where each target
    can only be affected a limited number of times per combat/scene.

    Examples:
        PerTargetCounter(effect_id="basilisk_stun", max_count=1, reset_on="scene_end")
    """

    effect_id: str
    current_count: int = 0
    max_count: int = Field(default=1, ge=1)
    reset_on: ProgressionResetTrigger = "scene_end"
    target_id: str | None = None


class PerTargetCounterEffect(FrozenModel):
    """
    Applies an effect with per-target limits.

    Used for effects that limit how many times a specific target can be affected.

    Examples:
        PerTargetCounterEffect(
            effect_id="basilisk_stun",
            max_count=1,
            reset_on="scene_end",
            effect=MechanicalEffect(status_grants=[...])
        )
    """

    effect_id: str
    max_count: int = Field(default=1, ge=1)
    reset_on: ProgressionResetTrigger = "scene_end"
    effect: "MechanicalEffect"
    condition: EffectCondition | None = None


CooldownResetTrigger = Literal[
    "scene_end", "rest", "full_repair", "turn_start", "turn_end", "round_end", "never"
]


class CooldownState(FrozenModel):
    """
    Tracks cooldowns preventing repeated effects.

    Used for effects that require a cooldown period between uses.

    Examples:
        CooldownState(effect_id="ability_id", duration=1, trigger_on="on_hit")
    """

    effect_id: str
    turns_remaining: int = 0
    duration: int = Field(default=1, ge=1)
    trigger_on: TriggerType | None = None
    reset_on: CooldownResetTrigger = "scene_end"
    per_target: bool = False
    target_id: str | None = None


class CooldownEffect(FrozenModel):
    """
    Applies an effect with a cooldown preventing immediate re-use.

    Examples:
        CooldownEffect(
            effect_id="ability_id",
            duration=1,
            effect=MechanicalEffect(damage_mods=[...])
        )
    """

    effect_id: str
    duration: int = Field(default=1, ge=1)
    trigger_on: TriggerType | None = None
    reset_on: CooldownResetTrigger = "scene_end"
    effect: "MechanicalEffect"
    condition: EffectCondition | None = None


class MechanicalEffect(FrozenModel):
    """
    Composable mechanical effect for talents, core bonuses, systems, etc.

    This is the primary building block for encoding game mechanics
    without relying on natural language descriptions.

    Example:
        MechanicalEffect(
            stat_mods=[StatModifier(stat="hp", value=5)],
            immunities=[Immunity(target="burn")],
        )
    """

    # Stat modifications
    stat_mods: list[StatModifier] = Field(default_factory=list)
    companion_stat_mods: list[CompanionStatModifierEffect] = Field(default_factory=list)
    stat_overrides: list[StatOverrideEffect] = Field(default_factory=list)
    mount_slot_grants: list[MountSlotGrant] = Field(default_factory=list)
    mount_slot_replacements: list[MountSlotReplacement] = Field(default_factory=list)
    mount_size_upgrades: list[MountSizeUpgradeEffect] = Field(default_factory=list)
    integrated_weapons: list[IntegratedWeaponEffect] = Field(default_factory=list)
    size_interactions: list[SizeInteractionEffect] = Field(default_factory=list)
    mount_carries: list[MountedAllyEffect] = Field(default_factory=list)
    accuracy_mods: list[AccuracyModifier] = Field(default_factory=list)
    check_mods: list[CheckModifierEffect] = Field(default_factory=list)
    check_value_mods: list[CheckValueModifierEffect] = Field(default_factory=list)
    damage_mods: list[DamageModifier] = Field(default_factory=list)
    damage_multipliers: list[DamageMultiplierEffect] = Field(default_factory=list)
    direct_damages: list[DirectDamage] = Field(default_factory=list)
    range_mods: list[RangeModifier] = Field(default_factory=list)
    limited_use_bonuses: list[LimitedUseBonusEffect] = Field(default_factory=list)
    limited_use_recharges: list[LimitedUseRechargeEffect] = Field(default_factory=list)
    non_combat_capabilities: list[NonCombatCapacityEffect] = Field(default_factory=list)

    # Grants and immunities
    action_grants: list[ActionGrant] = Field(default_factory=list)
    reaction_limits: list[ReactionLimitEffect] = Field(default_factory=list)
    reaction_triggers: list[ReactionTriggerEffect] = Field(default_factory=list)
    bondmates: list[BondmateEffect] = Field(default_factory=list)
    target_marks: list[TargetMarkEffect] = Field(default_factory=list)
    leadership_dice_pools: list[LeadershipDicePoolEffect] = Field(default_factory=list)
    dice_pools: list[DicePoolEffect] = Field(default_factory=list)
    countdown_dice: list[CountdownDieEffect] = Field(default_factory=list)
    tech_actions: list[TechAction] = Field(default_factory=list)
    tech_attack_mods: list[TechAttackModifier] = Field(default_factory=list)
    tech_restrictions: list[TechActionRestriction] = Field(default_factory=list)
    tech_action_overrides: list[TechActionOverrideEffect] = Field(default_factory=list)
    movement_grants: list[MovementGrant] = Field(default_factory=list)
    move_adjacent_effects: list[MoveAdjacentEffect] = Field(default_factory=list)
    position_swaps: list[PositionSwapEffect] = Field(default_factory=list)
    forced_movements: list[ForcedMovement] = Field(default_factory=list)
    status_toggles: list[StatusToggleEffect] = Field(default_factory=list)
    status_grants: list[StatusGrant] = Field(default_factory=list)
    status_clears: list[StatusClear] = Field(default_factory=list)
    status_breaks: list[StatusBreakCondition] = Field(default_factory=list)
    status_stack_limits: list[StatusStackLimit] = Field(default_factory=list)
    movement_scoped_statuses: list[MovementScopedStatus] = Field(default_factory=list)
    action_restrictions: list[ActionRestriction] = Field(default_factory=list)
    status_action_overrides: list[StatusActionOverrideEffect] = Field(
        default_factory=list
    )
    line_of_sight_restrictions: list[LineOfSightRestriction] = Field(
        default_factory=list
    )
    status_restrictions: list[StatusRestriction] = Field(default_factory=list)
    allegiance_shifts: list[AllegianceShiftEffect] = Field(default_factory=list)
    cover_restrictions: list[CoverRestriction] = Field(default_factory=list)
    cover_grants: list[CoverGrant] = Field(default_factory=list)
    intel_effects: list[IntelEffect] = Field(default_factory=list)
    movement_restrictions: list[MovementRestrictionEffect] = Field(default_factory=list)
    movement_surface_effects: list[MovementSurfaceEffect] = Field(default_factory=list)
    movement_mode_accesses: list[MovementModeAccessEffect] = Field(default_factory=list)
    jump_distance_effects: list[JumpDistanceEffect] = Field(default_factory=list)
    movement_overrides: list[MovementOverrideEffect] = Field(default_factory=list)
    immunities: list[Immunity] = Field(default_factory=list)
    tag_immunities: list[TagImmunityEffect] = Field(default_factory=list)
    resistances: list[Resistance] = Field(default_factory=list)
    heat_resistances: list[HeatResistanceEffect] = Field(default_factory=list)
    damage_reductions: list[DamageReduction] = Field(default_factory=list)
    damage_reduction_rolls: list[DamageReductionRollEffect] = Field(
        default_factory=list
    )
    damage_negations: list[DamageNegationEffect] = Field(default_factory=list)
    resource_changes: list[ResourceChange] = Field(default_factory=list)
    scaled_resource_changes: list[ScaledResourceChange] = Field(default_factory=list)
    overcharge_cost_caps: list[OverchargeCostCapEffect] = Field(default_factory=list)
    repair_cost_mods: list[RepairCostModifier] = Field(default_factory=list)
    repair_share_effects: list[RepairShareEffect] = Field(default_factory=list)
    repair_actions: list[RepairActionEffect] = Field(default_factory=list)
    structure_damage_avoidances: list[StructureDamageAvoidanceEffect] = Field(
        default_factory=list
    )
    zero_hp_survival_effects: list[ZeroHpSurvivalEffect] = Field(default_factory=list)
    attack_sequence_mods: list[AttackSequenceModifierEffect] = Field(
        default_factory=list
    )
    attack_roll_overrides: list[AttackRollOverrideEffect] = Field(default_factory=list)
    targetings: list[AttackTargetingEffect] = Field(default_factory=list)
    area_selections: list[AreaSelectionEffect] = Field(default_factory=list)
    area_attack_patterns: list[AreaAttackPattern] = Field(default_factory=list)
    line_attacks: list[LineAttackEffect] = Field(default_factory=list)
    attack_rerolls: list[AttackRerollEffect] = Field(default_factory=list)
    attack_outcomes: list[AttackOutcomeEffect] = Field(default_factory=list)
    critical_damage_overrides: list[CriticalDamageOverrideEffect] = Field(
        default_factory=list
    )
    damage_roll_overrides: list[DamageRollOverrideEffect] = Field(default_factory=list)
    accuracy_trade_effects: list[AccuracyTradeEffect] = Field(default_factory=list)
    delayed_impacts: list[DelayedImpactEffect] = Field(default_factory=list)
    weapon_grants: list[WeaponGrantEffect] = Field(default_factory=list)
    weapon_mods: list[WeaponModEffect] = Field(default_factory=list)
    weapon_spin_ups: list[WeaponSpinUpEffect] = Field(default_factory=list)
    weapon_ai_controls: list[WeaponAIControlEffect] = Field(default_factory=list)
    ai_system_limits: list[AISystemLimitEffect] = Field(default_factory=list)
    ai_control_transfers: list[AIControlTransferEffect] = Field(default_factory=list)
    deployments: list[DeploymentEffect] = Field(default_factory=list)
    attachments: list[AttachmentEffect] = Field(default_factory=list)
    system_links: list[SystemLinkEffect] = Field(default_factory=list)
    zones: list[ZoneEffect] = Field(default_factory=list)
    reloads: list[ReloadEffect] = Field(default_factory=list)
    reload_restrictions: list[ReloadRestrictionEffect] = Field(default_factory=list)
    damage_absorptions: list[DamageAbsorption] = Field(default_factory=list)
    damage_shares: list[DamageShareEffect] = Field(default_factory=list)
    out_of_play_effects: list[OutOfPlayEffect] = Field(default_factory=list)
    tethers: list[TetherEffect] = Field(default_factory=list)
    grapple_effects: list[GrappleEffect] = Field(default_factory=list)
    save_overrides: list[SaveOverrideEffect] = Field(default_factory=list)
    save_checks: list[SaveCheck] = Field(default_factory=list)
    random_checks: list[RandomCheckEffect] = Field(default_factory=list)
    roll_patterns: list[RollPatternEffect] = Field(default_factory=list)
    triggered_effects: list[TriggeredEffect] = Field(default_factory=list)
    status_triggers: list[StatusTrigger] = Field(default_factory=list)
    mode_effects: list[ModeEffect] = Field(default_factory=list)
    protocols: list[ProtocolEffect] = Field(default_factory=list)
    core_powers: list[CorePowerEffect] = Field(default_factory=list)
    choices: list[EffectChoice] = Field(default_factory=list)
    phase_shifts: list[PhaseShiftEffect] = Field(default_factory=list)
    effect_removals: list[EffectRemoval] = Field(default_factory=list)
    movement_trails: list[MovementTrailEffect] = Field(default_factory=list)
    hologram_trails: list[HologramTrailEffect] = Field(default_factory=list)
    holographic_duplicates: list[HolographicDuplicateEffect] = Field(
        default_factory=list
    )
    progression_effects: list[ProgressionEffect] = Field(default_factory=list)
    per_target_counter_effects: list[PerTargetCounterEffect] = Field(
        default_factory=list
    )
    cooldown_effects: list[CooldownEffect] = Field(default_factory=list)

    def is_empty(self) -> bool:
        """Check if this effect has no components."""
        return (
            not self.stat_mods
            and not self.companion_stat_mods
            and not self.stat_overrides
            and not self.mount_slot_grants
            and not self.mount_slot_replacements
            and not self.mount_size_upgrades
            and not self.integrated_weapons
            and not self.size_interactions
            and not self.mount_carries
            and not self.accuracy_mods
            and not self.check_mods
            and not self.check_value_mods
            and not self.damage_mods
            and not self.damage_multipliers
            and not self.direct_damages
            and not self.range_mods
            and not self.limited_use_bonuses
            and not self.limited_use_recharges
            and not self.non_combat_capabilities
            and not self.action_grants
            and not self.reaction_limits
            and not self.reaction_triggers
            and not self.bondmates
            and not self.target_marks
            and not self.leadership_dice_pools
            and not self.dice_pools
            and not self.countdown_dice
            and not self.tech_actions
            and not self.tech_attack_mods
            and not self.tech_restrictions
            and not self.tech_action_overrides
            and not self.movement_grants
            and not self.move_adjacent_effects
            and not self.position_swaps
            and not self.forced_movements
            and not self.status_toggles
            and not self.status_grants
            and not self.status_clears
            and not self.status_breaks
            and not self.status_stack_limits
            and not self.movement_scoped_statuses
            and not self.action_restrictions
            and not self.status_action_overrides
            and not self.line_of_sight_restrictions
            and not self.status_restrictions
            and not self.allegiance_shifts
            and not self.cover_restrictions
            and not self.cover_grants
            and not self.intel_effects
            and not self.movement_restrictions
            and not self.movement_surface_effects
            and not self.movement_mode_accesses
            and not self.jump_distance_effects
            and not self.movement_overrides
            and not self.immunities
            and not self.tag_immunities
            and not self.resistances
            and not self.heat_resistances
            and not self.damage_reductions
            and not self.damage_reduction_rolls
            and not self.damage_negations
            and not self.resource_changes
            and not self.scaled_resource_changes
            and not self.overcharge_cost_caps
            and not self.repair_cost_mods
            and not self.repair_share_effects
            and not self.repair_actions
            and not self.structure_damage_avoidances
            and not self.zero_hp_survival_effects
            and not self.attack_sequence_mods
            and not self.attack_roll_overrides
            and not self.targetings
            and not self.area_selections
            and not self.area_attack_patterns
            and not self.line_attacks
            and not self.attack_rerolls
            and not self.attack_outcomes
            and not self.critical_damage_overrides
            and not self.damage_roll_overrides
            and not self.accuracy_trade_effects
            and not self.delayed_impacts
            and not self.weapon_grants
            and not self.weapon_mods
            and not self.weapon_spin_ups
            and not self.weapon_ai_controls
            and not self.ai_system_limits
            and not self.ai_control_transfers
            and not self.deployments
            and not self.attachments
            and not self.system_links
            and not self.zones
            and not self.reloads
            and not self.reload_restrictions
            and not self.damage_absorptions
            and not self.damage_shares
            and not self.out_of_play_effects
            and not self.tethers
            and not self.grapple_effects
            and not self.save_overrides
            and not self.save_checks
            and not self.random_checks
            and not self.roll_patterns
            and not self.triggered_effects
            and not self.status_triggers
            and not self.mode_effects
            and not self.protocols
            and not self.core_powers
            and not self.choices
            and not self.phase_shifts
            and not self.effect_removals
            and not self.movement_trails
            and not self.hologram_trails
            and not self.holographic_duplicates
            and not self.progression_effects
            and not self.per_target_counter_effects
            and not self.cooldown_effects
        )


# Convenience constructors for common patterns


def stat_bonus(stat: StatType, value: int) -> MechanicalEffect:
    """Create a simple stat bonus effect."""
    return MechanicalEffect(stat_mods=[StatModifier(stat=stat, value=value)])


def damage_bonus(
    flat: int = 0,
    dice: DiceExpression | str | None = None,
    condition: EffectCondition | None = None,
) -> MechanicalEffect:
    """Create a damage bonus effect."""
    bonus_dice = DiceExpression.parse(dice) if isinstance(dice, str) else dice
    return MechanicalEffect(
        damage_mods=[DamageModifier(flat=flat, dice=bonus_dice, condition=condition)]
    )


def immunity_to(
    target: str, condition: EffectCondition | None = None
) -> MechanicalEffect:
    """Create an immunity effect."""
    return MechanicalEffect(immunities=[Immunity(target=target, condition=condition)])
