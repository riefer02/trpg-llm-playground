"""
Structured mechanical effect primitives for Lancer TTRPG.

This module defines composable effect building blocks that encode
game mechanics as structured data rather than description strings.

Legal Note: These represent pure game mechanics (allowed under the
Lancer Third Party License), not copyrighted expression/flavor text.
"""

from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

from core.shared.enums import ActionType, CoverType, DamageType, SaveType, StatusType
from core.shared.dice import DiceExpression


# Stats that can be modified
StatType = Literal[
    # Pilot stats
    "hp",
    "armor", 
    "evasion",
    "e_defense",
    "speed",
    # Mech stats
    "heat_cap",
    "repair_cap",
    "tech_attack",
    "save_target",
    "sensor_range",
    "size",
    # Mounts
    "limited_bonus",  # +N to Limited weapon uses
    # System budget
    "system_points",
]

# Conditions for conditional effects
ConditionType = Literal[
    # Target conditions
    "target_prone",
    "target_immobilized",
    "target_below_half_hp",
    "target_has_lock_on",
    "target_larger",
    "target_smaller",
    "target_same_size",
    "target_hidden",
    "target_jammed",
    "target_shredded",
    "target_impaired",
    "target_slowed",
    "target_stunned",
    "target_exposed",
    "target_invisible",
    "target_engaged",
    # Attacker conditions
    "after_boost",
    "after_move_8_plus",
    "in_danger_zone",
    "hidden",
    "engaged",
    "exposed",
    "overheated",
    # Attack type conditions
    "melee_attack",
    "ranged_attack",
    "tech_attack",
    "ram_attack",
    "melee_attack_no_other_adjacent",
    # Save conditions
    "save_vs_knockback_or_prone",
]

# Common trigger points for conditional effects
TriggerType = Literal[
    "on_hit",
    "on_miss",
    "on_reload",
    "on_first_attack",
    "on_crit",
    "on_kill",
    "on_activation",
    "on_attack_roll",
    "on_move",
    "on_reaction",
    "on_take_damage",
    "on_ally_damaged",
    "on_turn_start",
    "on_turn_end",
]
ActionCategoryType = Literal[
    "attack",
    "movement",
    "tech",
    "utility",
    "defense",
    "reaction",
]
EffectDuration = Literal[
    "end_of_turn",
    "start_of_next_turn",
    "end_of_next_turn",
    "until_cleared",
    "scene",
]
IntelAudience = Literal["self", "allies", "all"]
IntelType = Literal[
    "location",
    "hp",
    "structure",
    "heat",
    "speed",
    "evasion",
    "e_defense",
    "weapons",
    "systems",
    "area",
]
MovementMode = Literal["move", "boost", "other", "any"]
BreakTriggerType = Literal[
    "move",
    "reaction",
    "turn_start",
    "take_damage",
    "stunned",
    "manual_deactivate",
]
WeaponSizeType = Literal["aux", "main", "heavy", "superheavy"]
WeaponTypeType = Literal["cqb", "rifle", "launcher", "cannon", "melee", "nexus"]

ResourceType = Literal["hp", "heat", "repairs", "structure", "stress"]
ResourceAmount = int | DiceExpression | Literal["half_max", "full"]
ResourceDirection = Literal["gain", "lose", "set"]
TechRangeType = Literal["sensors", "range"]
DeploymentActivationCondition = Literal[
    "adjacent_start_or_move",
    "adjacent_start",
    "adjacent_move",
    "manual",
    "pass_over",
]
DelayedImpactTiming = Literal["end_of_next_round", "end_of_next_turn", "start_of_next_turn"]
PhaseState = Literal["in_phase", "out_of_phase"]
HologramTrailTrigger = Literal["move", "boost", "move_or_boost"]
HologramDetonationTrigger = Literal["start_turn", "move_through", "move_adjacent"]


class StatModifier(BaseModel):
    """
    Numeric modifier to a stat.
    
    Examples:
        StatModifier(stat="hp", value=5)  # +5 HP
        StatModifier(stat="size", value=1)  # +1 Size
    """
    stat: StatType
    value: int
    
    model_config = {"frozen": True}


class DamageModifier(BaseModel):
    """
    Bonus damage under specific conditions.
    
    Examples:
        DamageModifier(flat=1, condition="melee_attack")  # +1 melee damage
        DamageModifier(dice=DiceExpression.parse("1d6"), damage_type="kinetic")  # +1d6 kinetic
    """
    dice: DiceExpression | None = Field(default=None, description="Bonus dice (e.g., DiceExpression.parse('1d6'))")
    flat: int = Field(default=0, description="Flat bonus damage")
    damage_type: DamageType | None = Field(default=None)
    condition: ConditionType | None = Field(default=None)
    
    model_config = {"frozen": True}


class RangeModifier(BaseModel):
    """
    Modifier to range or threat.
    
    Examples:
        RangeModifier(range_type="threat", value=1)  # +1 Threat
        RangeModifier(range_type="range", value=5)  # +5 Range
    """
    range_type: Literal["range", "threat", "sensors"]
    value: int
    
    model_config = {"frozen": True}


DirectDamageType = DamageType | Literal["heat"]


class DirectDamage(BaseModel):
    """
    Direct damage not tied to a standard weapon attack.

    Examples:
        DirectDamage(damage_type="explosive", dice=DiceExpression.parse("1d3"), ap=True)
    """
    damage_type: DirectDamageType
    dice: DiceExpression | None = None
    flat: int = 0
    ap: bool = False
    target: Literal["self", "enemy", "ally", "adjacent", "all", "object"] = "enemy"
    condition: str | None = None

    model_config = {"frozen": True}


class ActionGrant(BaseModel):
    """
    Grants a new action or ability.
    
    Examples:
        ActionGrant(action_type="quick", name="Afterburner", trigger="after_boost")
        ActionGrant(action_type="reaction", name="Juke", trigger="on_successful_agility_save")
    """
    action_type: ActionType
    name: str
    trigger: str | None = Field(default=None, description="When this action can be used")
    uses_per: Literal["unlimited", "round", "scene", "mission"] = "unlimited"
    
    model_config = {"frozen": True}


class TechRange(BaseModel):
    """
    Range descriptor for tech actions.

    Examples:
        TechRange(range_type="sensors")  # Within sensors
        TechRange(range_type="range", value=10)  # Range 10
    """
    range_type: TechRangeType = "sensors"
    value: int | None = Field(default=None, ge=0)

    model_config = {"frozen": True}


class TechAction(BaseModel):
    """
    Defines a tech action granted by a system.

    Examples:
        TechAction(name="Track", action_type="quick", is_attack=True, range=TechRange(range_type="sensors"))
    """
    name: str
    action_type: ActionType
    target: Literal["self", "enemy", "ally", "adjacent", "object"] = "enemy"
    range: TechRange | None = None
    is_attack: bool = False
    attack_vs: Literal["e_defense", "evasion"] = "e_defense"
    effect: MechanicalEffect | None = None
    on_hit: MechanicalEffect | None = None
    on_miss: MechanicalEffect | None = None
    on_success: MechanicalEffect | None = None
    on_failure: MechanicalEffect | None = None
    uses_per: Literal["unlimited", "round", "scene", "mission"] = "unlimited"
    special: str | None = None

    model_config = {"frozen": True}


class TechAttackModifier(BaseModel):
    """
    Accuracy/difficulty modifiers for tech attacks.

    Examples:
        TechAttackModifier(value=-1, target="ally", condition="adjacent", max_stacks=3)
    """
    value: int = Field(..., description="Positive = accuracy, negative = difficulty")
    target: Literal["self", "enemy", "ally", "adjacent", "all"] = "self"
    condition: str | None = None
    max_stacks: int | None = Field(default=None, ge=1)
    reset_trigger: Literal["turn_start", "turn_end", "round_end", "scene_end"] | None = None

    model_config = {"frozen": True}


class TechActionRestriction(BaseModel):
    """
    Restrictions or immunity affecting tech actions.

    Examples:
        TechActionRestriction(disallow_tech_actions=True, end_tech_effects=True)
    """
    disallow_tech_actions: bool = False
    immune_to_tech: bool = False
    end_tech_effects: bool = False
    target: Literal["self", "enemy", "ally", "adjacent", "all"] = "self"
    condition: str | None = None

    model_config = {"frozen": True}


class EffectChoice(BaseModel):
    """
    Select one of multiple effects.

    Examples:
        EffectChoice(name="Option A", effect=MechanicalEffect(...))
    """
    name: str
    effect: MechanicalEffect
    target: Literal["self", "enemy", "ally", "adjacent"] = "enemy"
    range: TechRange | None = None
    condition: str | None = None

    model_config = {"frozen": True}


class ActionRestriction(BaseModel):
    """
    Restrictions for combat action usage.

    Examples:
        ActionRestriction(disallow_attack_rolls=True)
        ActionRestriction(action_ids=["hide"], target="enemy")
    """
    disallow_attack_rolls: bool = False
    action_ids: list[str] = Field(default_factory=list)
    action_categories: list[ActionCategoryType] = Field(default_factory=list)
    target: Literal["self", "enemy", "ally", "adjacent", "all"] = "self"
    duration: EffectDuration = "end_of_turn"
    condition: str | None = None

    model_config = {"frozen": True}


class Immunity(BaseModel):
    """
    Immunity to a condition, damage type, or effect.
    
    Examples:
        Immunity(target="burn")  # Immune to Burn
        Immunity(target="knockback", condition="from_smaller")  # Immune to knockback from smaller
    """
    target: str = Field(..., description="What you're immune to (condition, damage type, or effect)")
    condition: str | None = Field(default=None, description="Conditional immunity")
    
    model_config = {"frozen": True}


class Resistance(BaseModel):
    """
    Resistance (half damage) to a damage type.
    
    Examples:
        Resistance(damage_type="energy")
        Resistance(damage_type="all", condition="can_see_bondmate")
    """
    damage_type: DamageType | Literal["all"]
    target: Literal["self", "enemy", "ally", "adjacent", "all"] = "self"
    condition: str | None = Field(default=None)
    
    model_config = {"frozen": True}


class AccuracyModifier(BaseModel):
    """
    Modifier to accuracy/difficulty on rolls.
    
    Examples:
        AccuracyModifier(value=1, condition="target_has_lock_on")  # +1 Accuracy vs Lock On
        AccuracyModifier(value=-1)  # -1 Accuracy (difficulty)
    """
    value: int = Field(..., description="Positive = accuracy, negative = difficulty")
    condition: ConditionType | str | None = Field(default=None)
    applies_to: Literal["all", "melee", "ranged", "tech"] = "all"
    
    model_config = {"frozen": True}


class DamageReduction(BaseModel):
    """
    Flat damage reduction that applies before resistance.

    Examples:
        DamageReduction(amount=2, damage_type="all")
    """
    amount: int = Field(..., ge=0)
    damage_type: DamageType | Literal["all", "heat", "burn"] = "all"
    target: Literal["self", "enemy", "ally", "adjacent", "all", "object"] = "self"
    condition: str | None = None

    model_config = {"frozen": True}


class MovementGrant(BaseModel):
    """
    Grants movement or teleportation.
    
    Examples:
        MovementGrant(spaces=2, movement_type="fly", trigger="on_successful_save")
        MovementGrant(spaces=3, movement_type="teleport", trigger="after_boost")
    """
    spaces: int
    movement_type: Literal["walk", "fly", "teleport"] = "walk"
    trigger: str | None = Field(default=None)
    target: Literal["self", "enemy", "ally", "adjacent"] = "self"
    
    model_config = {"frozen": True}


class ForcedMovement(BaseModel):
    """
    Forced push/pull movement applied to a target.

    Examples:
        ForcedMovement(direction="pull", distance=5, ignores_engagement=True)
    """
    direction: Literal["pull", "push"]
    distance: int | DiceExpression
    target: Literal["self", "enemy", "ally", "adjacent", "all"] = "enemy"
    ignores_engagement: bool = False
    provokes_reactions: bool = True
    must_obey_obstructions: bool = True
    on_collision: MechanicalEffect | None = None

    model_config = {"frozen": True}


class StatusGrant(BaseModel):
    """
    Grants or inflicts a status condition.
    
    Examples:
        StatusGrant(status="invisible", target="self", trigger="after_move_8_plus")
        StatusGrant(status="prone", target="enemy", trigger="on_hit")
    """
    status: StatusType | Literal["invisible", "flying"]
    target: Literal["self", "enemy", "ally", "adjacent"]
    trigger: str | None = Field(default=None)
    condition: str | None = None
    duration: Literal[
        "end_of_turn",
        "start_of_next_turn",
        "end_of_next_turn",
        "until_cleared",
        "until_attack",
        "match_trigger",
        "scene",
    ] = "end_of_turn"
    
    model_config = {"frozen": True}


class StatusClear(BaseModel):
    """
    Clears a status or condition from a target.

    Examples:
        StatusClear(status="burn", target="ally")
        StatusClear(status="any", target="self")
    """
    status: StatusType | Literal["burn", "any", "tech"]
    target: Literal["self", "enemy", "ally", "adjacent", "all"]
    count: int = Field(default=1, ge=1)

    model_config = {"frozen": True}


class StatusBreakCondition(BaseModel):
    """
    Defines triggers that end a status early.

    Examples:
        StatusBreakCondition(status="invisible", break_triggers=["move", "reaction"])
    """
    status: StatusType
    target: Literal["self", "enemy", "ally", "adjacent", "all"] = "self"
    break_triggers: list[BreakTriggerType]
    condition: str | None = None

    model_config = {"frozen": True}


class StatusStackLimit(BaseModel):
    """
    Limits stacking or re-application of a status.

    Examples:
        StatusStackLimit(status="invisible", max_stacks=1)
    """
    status: StatusType
    max_stacks: int = Field(default=1, ge=1)
    target: Literal["self", "enemy", "ally", "adjacent", "all"] = "self"
    condition: str | None = None

    model_config = {"frozen": True}


class MovementScopedStatus(BaseModel):
    """
    Status that applies only during movement.

    Examples:
        MovementScopedStatus(status="invisible", movement_modes=["any"], ends_on="movement_end")
    """
    status: StatusType
    target: Literal["self", "enemy", "ally", "adjacent"]
    movement_modes: list[MovementMode] = Field(default_factory=list)
    ends_on: Literal["movement_end", "turn_end"] = "movement_end"
    condition: str | None = None

    model_config = {"frozen": True}


class StatusRestriction(BaseModel):
    """
    Restricts gaining or benefiting from statuses.

    Examples:
        StatusRestriction(statuses=["invisible"], restriction="cannot_benefit", target="enemy")
    """
    statuses: list[StatusType]
    restriction: Literal["cannot_gain", "cannot_benefit"] = "cannot_gain"
    target: Literal["self", "enemy", "ally", "adjacent", "all"] = "enemy"
    duration: EffectDuration = "end_of_turn"
    condition: str | None = None

    model_config = {"frozen": True}


class CoverRestriction(BaseModel):
    """
    Restricts cover benefits for targets.

    Examples:
        CoverRestriction(max_cover="none", target="enemy")
    """
    max_cover: CoverType = "none"
    target: Literal["self", "enemy", "ally", "adjacent", "all"] = "enemy"
    duration: EffectDuration = "end_of_turn"
    condition: str | None = None

    model_config = {"frozen": True}


class CoverGrant(BaseModel):
    """
    Grants cover to targets for a duration.

    Examples:
        CoverGrant(cover="soft", target="ally", duration="start_of_next_turn")
    """
    cover: CoverType
    target: Literal["self", "enemy", "ally", "adjacent", "all"] = "ally"
    duration: EffectDuration = "end_of_turn"
    condition: str | None = None

    model_config = {"frozen": True}


class IntelEffect(BaseModel):
    """
    Reveals information or grants enhanced vision.

    Examples:
        IntelEffect(reveal=["location", "hp"], audience="self", duration="until_cleared")
    """
    reveal: list[IntelType] = Field(default_factory=list)
    audience: IntelAudience = "self"
    target: Literal["self", "enemy", "ally", "adjacent", "all"] = "enemy"
    perfect_vision: bool = False
    grants_line_of_sight: bool = True
    duration: EffectDuration = "end_of_turn"
    condition: str | None = None

    model_config = {"frozen": True}


class MovementRestrictionEffect(BaseModel):
    """
    Movement restriction applied to a target.

    Examples:
        MovementRestrictionEffect(target="enemy", cannot_move_closer_to_source=True)
    """
    target: Literal["self", "enemy", "ally", "adjacent"] = "enemy"
    max_voluntary_speed: int | None = Field(default=None, ge=0)
    cannot_move_closer_to_source: bool = False
    duration: EffectDuration = "end_of_turn"
    condition: str | None = None

    model_config = {"frozen": True}


class ResourceChange(BaseModel):
    """
    Change a resource value such as HP or heat.

    Examples:
        ResourceChange(resource="hp", amount="half_max", target="ally", cost_repairs=1)
        ResourceChange(resource="heat", amount=DiceExpression.parse("1d6"), direction="lose", target="self")
    """
    resource: ResourceType
    amount: ResourceAmount
    direction: ResourceDirection = "gain"
    target: Literal["self", "enemy", "ally", "adjacent"]
    cost_repairs: int = Field(default=0, ge=0)
    cost_source: Literal["self", "target", "either"] = "self"

    model_config = {"frozen": True}


class AttackTargetingEffect(BaseModel):
    """
    Describes multi-target attack selection.

    Examples:
        AttackTargetingEffect(target_count_options=[1, 2], separate_attack_rolls=True)
    """
    target_count_options: list[int] = Field(default_factory=list)
    separate_attack_rolls: bool = True
    require_distinct_targets: bool = True
    condition: str | None = None

    model_config = {"frozen": True}


class AreaAttackPattern(BaseModel):
    """
    Defines multi-area attack patterns for weapons.

    Examples:
        AreaAttackPattern(area_shape="blast", area_size=1, area_count_options=[1, 2], non_overlapping=True)
    """
    area_shape: Literal["blast", "burst", "line", "cone"]
    area_size: int = Field(..., ge=0)
    area_count_options: list[int] = Field(default_factory=list)
    non_overlapping: bool = False

    model_config = {"frozen": True}


class DelayedImpactEffect(BaseModel):
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

    model_config = {"frozen": True}


class WeaponTagGrant(BaseModel):
    """
    Tag granted to a weapon.

    Examples:
        WeaponTagGrant(tag="ordnance")
    """
    tag: str
    value: int | None = None

    model_config = {"frozen": True}


class WeaponSizeBonus(BaseModel):
    """
    Size-based bonus for a weapon mod.

    Examples:
        WeaponSizeBonus(size="main", burn=2)
    """
    size: WeaponSizeType
    burn: int = Field(..., ge=0)

    model_config = {"frozen": True}


class WeaponModEffect(BaseModel):
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
    condition: str | None = None

    model_config = {"frozen": True}


class DeploymentEffect(BaseModel):
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
    primes_after: Literal["turn_end", "immediate"] = "immediate"
    activation_condition: DeploymentActivationCondition
    activation_action: ActionType | None = None
    activation_target: Literal["self", "enemy", "ally", "adjacent"] = "self"
    activation_effect: MechanicalEffect | None = None
    consumes_on_activation: bool = True

    model_config = {"frozen": True}


class PhaseShiftEffect(BaseModel):
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
    out_of_phase_duration: Literal["start_of_next_turn", "end_of_turn"] = "start_of_next_turn"
    duration: EffectDuration = "scene"
    deactivation_action: ActionType | None = None
    intangible: bool = True
    ignore_obstructions: bool = True
    cannot_end_in_obstruction: bool = True
    cannot_interact: bool = True
    immune_to_damage: bool = True

    model_config = {"frozen": True}


ZoneShape = Literal["burst", "blast", "line", "cone", "square"]


class ZoneEffect(BaseModel):
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
    retarget_action: ActionType | None = None
    retarget_range: int | None = Field(default=None, ge=0)
    retarget_requires_line_of_sight: bool = False
    retarget_replaces_existing: bool = True
    duration: Literal["end_of_turn", "end_of_next_turn", "scene"] = "scene"
    difficult_terrain: bool = False
    cover: CoverType | None = None
    cover_all_directions: bool = False
    applies_to: Literal["all", "ally", "enemy", "object"] = "all"
    effects_on_enter: MechanicalEffect | None = None
    effects_on_start_turn: MechanicalEffect | None = None
    effects_on_end_turn: MechanicalEffect | None = None
    continuous_effects: MechanicalEffect | None = None
    total_effect_cap: int | None = Field(default=None, ge=0)

    model_config = {"frozen": True}


class TetherEffect(BaseModel):
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

    model_config = {"frozen": True}


class EffectRemoval(BaseModel):
    """
    Describes how an ongoing effect can be removed.

    Examples:
        EffectRemoval(action_type="quick", check_type="engineering", check_kind="check")
    """
    action_type: ActionType
    check_type: SaveType | None = None
    check_kind: Literal["check", "save"] = "check"
    target: Literal["self", "enemy", "ally", "adjacent"] = "self"
    condition: str | None = None

    model_config = {"frozen": True}


class HologramTrailEffect(BaseModel):
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

    model_config = {"frozen": True}


class ReloadEffect(BaseModel):
    """
    Reloads one or more weapons, optionally filtered by tag.

    Examples:
        ReloadEffect(target="ally", count=1, requires_tag="loading")
    """
    target: Literal["self", "enemy", "ally", "adjacent"]
    count: int = Field(default=1, ge=1)
    requires_tag: str | None = "loading"
    consumes_source: bool = False

    model_config = {"frozen": True}


class DamageAbsorption(BaseModel):
    """
    Absorbs damage before it affects the target.

    Examples:
        DamageAbsorption(target="ally", base_hp=4, bonus_hp_per_grit=1)
    """
    target: Literal["self", "enemy", "ally", "adjacent"]
    base_hp: int = Field(..., ge=0)
    bonus_hp_per_grit: int = Field(default=0, ge=0)
    max_instances_per_target: int = Field(default=1, ge=1)
    spillover: bool = True
    ends_on_zero: bool = True
    duration: Literal["scene", "until_destroyed"] = "until_destroyed"

    model_config = {"frozen": True}


class SaveCheck(BaseModel):
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
    condition: str | None = None
    save: SaveType
    target: Literal["self", "enemy", "ally", "adjacent"] = "enemy"
    on_success: MechanicalEffect | None = None
    on_failure: MechanicalEffect | None = None

    model_config = {"frozen": True}


class RandomCheckEffect(BaseModel):
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
    target: Literal["self", "enemy", "ally", "adjacent"] = "ally"
    on_success: MechanicalEffect | None = None
    on_failure: MechanicalEffect | None = None
    uses_per: Literal["unlimited", "round", "scene", "mission"] = "unlimited"
    condition: str | None = None

    model_config = {"frozen": True}


class StatusTrigger(BaseModel):
    """
    Effect triggered by inflicting or clearing a status.

    Examples:
        StatusTrigger(trigger="on_inflict", status="immobilized",
                      effect=MechanicalEffect(status_grants=[StatusGrant(status="shredded", target="enemy",
                      duration="match_trigger")]))
    """
    trigger: Literal["on_inflict", "on_clear"]
    status: StatusType
    target: Literal["self", "enemy", "ally", "adjacent"] = "enemy"
    effect: MechanicalEffect
    condition: str | None = None
    uses_per: Literal["unlimited", "round", "scene", "mission"] = "unlimited"

    model_config = {"frozen": True}


class TriggeredEffect(BaseModel):
    """
    Effect that only applies when a trigger occurs.

    Examples:
        TriggeredEffect(
            trigger="on_reload",
            effect=MechanicalEffect(action_grants=[ActionGrant(action_type="free", name="reload_fire")]),
        )
    """
    trigger: TriggerType
    condition: str | None = None
    effect: MechanicalEffect
    uses_per: Literal["unlimited", "round", "scene", "mission"] = "unlimited"

    model_config = {"frozen": True}


class MechanicalEffect(BaseModel):
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
    accuracy_mods: list[AccuracyModifier] = Field(default_factory=list)
    damage_mods: list[DamageModifier] = Field(default_factory=list)
    direct_damages: list[DirectDamage] = Field(default_factory=list)
    range_mods: list[RangeModifier] = Field(default_factory=list)
    
    # Grants and immunities
    action_grants: list[ActionGrant] = Field(default_factory=list)
    tech_actions: list[TechAction] = Field(default_factory=list)
    tech_attack_mods: list[TechAttackModifier] = Field(default_factory=list)
    tech_restrictions: list[TechActionRestriction] = Field(default_factory=list)
    movement_grants: list[MovementGrant] = Field(default_factory=list)
    forced_movements: list[ForcedMovement] = Field(default_factory=list)
    status_grants: list[StatusGrant] = Field(default_factory=list)
    status_clears: list[StatusClear] = Field(default_factory=list)
    status_breaks: list[StatusBreakCondition] = Field(default_factory=list)
    status_stack_limits: list[StatusStackLimit] = Field(default_factory=list)
    movement_scoped_statuses: list[MovementScopedStatus] = Field(default_factory=list)
    action_restrictions: list[ActionRestriction] = Field(default_factory=list)
    status_restrictions: list[StatusRestriction] = Field(default_factory=list)
    cover_restrictions: list[CoverRestriction] = Field(default_factory=list)
    cover_grants: list[CoverGrant] = Field(default_factory=list)
    intel_effects: list[IntelEffect] = Field(default_factory=list)
    movement_restrictions: list[MovementRestrictionEffect] = Field(default_factory=list)
    immunities: list[Immunity] = Field(default_factory=list)
    resistances: list[Resistance] = Field(default_factory=list)
    damage_reductions: list[DamageReduction] = Field(default_factory=list)
    resource_changes: list[ResourceChange] = Field(default_factory=list)
    targetings: list[AttackTargetingEffect] = Field(default_factory=list)
    area_attack_patterns: list[AreaAttackPattern] = Field(default_factory=list)
    delayed_impacts: list[DelayedImpactEffect] = Field(default_factory=list)
    weapon_mods: list[WeaponModEffect] = Field(default_factory=list)
    deployments: list[DeploymentEffect] = Field(default_factory=list)
    zones: list[ZoneEffect] = Field(default_factory=list)
    reloads: list[ReloadEffect] = Field(default_factory=list)
    damage_absorptions: list[DamageAbsorption] = Field(default_factory=list)
    tethers: list[TetherEffect] = Field(default_factory=list)
    save_checks: list[SaveCheck] = Field(default_factory=list)
    random_checks: list[RandomCheckEffect] = Field(default_factory=list)
    triggered_effects: list[TriggeredEffect] = Field(default_factory=list)
    status_triggers: list[StatusTrigger] = Field(default_factory=list)
    choices: list[EffectChoice] = Field(default_factory=list)
    phase_shifts: list[PhaseShiftEffect] = Field(default_factory=list)
    effect_removals: list[EffectRemoval] = Field(default_factory=list)
    hologram_trails: list[HologramTrailEffect] = Field(default_factory=list)
    
    # For complex effects not yet fully modeled
    # This should be mechanical shorthand, not flavor text
    special: str | None = Field(
        default=None, 
        description="Brief mechanical note for effects not yet modeled (e.g., 'grapple_pull_5')"
    )
    
    model_config = {"frozen": True}
    
    def is_empty(self) -> bool:
        """Check if this effect has no components."""
        return (
            not self.stat_mods
            and not self.accuracy_mods
            and not self.damage_mods
            and not self.direct_damages
            and not self.range_mods
            and not self.action_grants
            and not self.tech_actions
            and not self.tech_attack_mods
            and not self.tech_restrictions
            and not self.movement_grants
            and not self.forced_movements
            and not self.status_grants
            and not self.status_clears
            and not self.status_breaks
            and not self.status_stack_limits
            and not self.movement_scoped_statuses
            and not self.action_restrictions
            and not self.status_restrictions
            and not self.cover_restrictions
            and not self.cover_grants
            and not self.intel_effects
            and not self.movement_restrictions
            and not self.immunities
            and not self.resistances
            and not self.damage_reductions
            and not self.resource_changes
            and not self.targetings
            and not self.area_attack_patterns
            and not self.delayed_impacts
            and not self.weapon_mods
            and not self.deployments
            and not self.zones
            and not self.reloads
            and not self.damage_absorptions
            and not self.tethers
            and not self.save_checks
            and not self.random_checks
            and not self.triggered_effects
            and not self.status_triggers
            and not self.choices
            and not self.phase_shifts
            and not self.effect_removals
            and not self.hologram_trails
            and not self.special
        )


# Convenience constructors for common patterns

def stat_bonus(stat: StatType, value: int) -> MechanicalEffect:
    """Create a simple stat bonus effect."""
    return MechanicalEffect(stat_mods=[StatModifier(stat=stat, value=value)])


def damage_bonus(
    flat: int = 0,
    dice: DiceExpression | str | None = None,
    condition: ConditionType | None = None,
) -> MechanicalEffect:
    """Create a damage bonus effect."""
    bonus_dice = DiceExpression.parse(dice) if isinstance(dice, str) else dice
    return MechanicalEffect(damage_mods=[DamageModifier(flat=flat, dice=bonus_dice, condition=condition)])


def immunity_to(target: str, condition: str | None = None) -> MechanicalEffect:
    """Create an immunity effect."""
    return MechanicalEffect(immunities=[Immunity(target=target, condition=condition)])
