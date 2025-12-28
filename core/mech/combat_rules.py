"""Comprehensive mech combat ruleset models."""

from typing import Literal
from pydantic import BaseModel, Field

from core.shared.enums import DamageType
from core.shared.dice import DiceExpression
from core.mech.rules import (
    ActionEconomyRules,
    AttackRules,
    BonusDamageRules,
    CoverRules,
    CriticalHitRules,
    HeatRules,
    BurnRules,
    OverchargeRules,
    InvisibilityRules,
    ObjectRules,
    ReactionRules,
)


class TurnOrderRules(BaseModel):
    """Turn order and round cadence rules."""

    players_act_first: bool = True
    nomination_required: bool = True
    gm_chooses_if_no_nomination: bool = True
    alternate_sides: bool = True
    remaining_side_any_order: bool = True
    next_round_start_other_side: bool = True

    model_config = {"frozen": True}


class TurnActionRules(BaseModel):
    """Per-turn action economy rules."""

    move_per_turn: int = Field(default=1, ge=0)
    action_economy: ActionEconomyRules = Field(default_factory=ActionEconomyRules)
    free_actions_on_turn_only: bool = True
    overcharge_limit_per_turn: int = Field(default=1, ge=0)

    model_config = {"frozen": True}


class EngagementRules(BaseModel):
    """Engagement rules and penalties."""

    ranged_attack_difficulty: int = Field(default=1, ge=0)
    stop_on_engage_same_size_or_larger: bool = True

    model_config = {"frozen": True}


class ObstructionRules(BaseModel):
    """Obstruction and pass-through rules."""

    allies_obstruct: bool = False
    can_end_in_allied_space: bool = False
    smaller_obstruct_larger: bool = False
    can_move_through_smaller: bool = True
    can_end_in_smaller_space: bool = False

    model_config = {"frozen": True}


class TerrainRules(BaseModel):
    """Terrain interaction rules."""

    difficult_terrain_cost: int = Field(default=2, ge=1)
    dangerous_terrain_check_skill: Literal["engineering"] = "engineering"
    dangerous_terrain_damage: int = Field(default=5, ge=0)
    dangerous_terrain_check_once_per_round: bool = True
    climb_cost: int = Field(default=2, ge=1)

    model_config = {"frozen": True}


class FallingRules(BaseModel):
    """Falling damage rules."""

    min_distance_for_damage: int = Field(default=3, ge=0)
    damage_per_3_spaces: int = Field(default=3, ge=0)
    damage_type: DamageType = "kinetic"
    armor_piercing: bool = True
    max_damage: int = Field(default=9, ge=0)
    resolves_end_of_turn: bool = True

    model_config = {"frozen": True}


class ZeroGRules(BaseModel):
    """Zero-g or underwater movement rules."""

    slowed_without_propulsion: bool = True
    can_fly_while_moving: bool = True
    no_falling: bool = True

    model_config = {"frozen": True}


class TeleportRules(BaseModel):
    """Teleport rules."""

    counts_as_movement_spaces: int = Field(default=1, ge=0)
    requires_surface_start: bool = True
    requires_surface_end: bool = True
    ignores_engagement: bool = True
    ignores_reactions: bool = True
    ignores_obstructions: bool = True
    ignores_line_of_sight: bool = True
    fails_if_destination_occupied: bool = True

    model_config = {"frozen": True}


class FlightRules(BaseModel):
    """Flight rules and restrictions."""

    must_move_min_spaces: int = Field(default=1, ge=0)
    straight_line_per_movement: bool = True
    falls_if_immobilized_or_stunned: bool = True
    agility_save_on_structure_or_overheat: bool = True
    combat_altitude_limit: int = Field(default=10, ge=0)
    beyond_altitude_only_move_or_boost: bool = True
    ignore_obstructions: bool = True
    requires_physical_space: bool = True
    engage_only_on_adjacent: bool = True
    cannot_be_prone: bool = True
    hover_allows_stationary: bool = True
    hover_ignores_straight_line: bool = True
    carry_max_total_size: float = Field(default=0.5, ge=0.0)
    carry_limit_ignored_in_zero_g: bool = True

    model_config = {"frozen": True}


class AttackPatternDefinition(BaseModel):
    """Definition for area attack patterns."""

    pattern: Literal["line", "cone", "blast", "burst"]
    size: int = Field(..., ge=0)
    separate_attack_per_target: bool = True
    single_damage_roll: bool = True

    model_config = {"frozen": True}


class LineOfSightRules(BaseModel):
    """Line of sight interaction rules."""

    requires_line_of_sight: bool = True
    arcing_allows_no_los: bool = True
    seeking_ignores_cover: bool = True
    seeking_ignores_los: bool = True
    arcing_requires_path_clear: bool = True
    seeking_requires_path_clear: bool = True
    adjacent_cover_does_not_block_los: bool = True

    model_config = {"frozen": True}


class ValidTargetRules(BaseModel):
    """Target eligibility rules."""

    allow_characters: bool = True
    allow_objects: bool = True
    allow_points: bool = True
    allow_self: bool = False

    model_config = {"frozen": True}


class DamageResolutionRules(BaseModel):
    """Damage resolution order and constraints."""

    armor_applies_to: list[DamageType] = Field(
        default_factory=lambda: ["kinetic", "explosive", "energy"]
    )
    apply_armor_before_resistance: bool = True
    resistance_stacks: bool = False
    ap_ignores_armor: bool = True
    apply_multipliers_before_reduction: bool = True

    model_config = {"frozen": True}


class D6Range(BaseModel):
    """Inclusive d6 roll range."""

    roll_min: int = Field(..., ge=1, le=6)
    roll_max: int = Field(..., ge=1, le=6)

    model_config = {"frozen": True}


class SystemTraumaRules(BaseModel):
    """System trauma selection and fallback rules."""

    roll: DiceExpression = Field(default_factory=lambda: DiceExpression.parse("1d6"))
    mount_on: D6Range = Field(default_factory=lambda: D6Range(roll_min=1, roll_max=3))
    system_on: D6Range = Field(default_factory=lambda: D6Range(roll_min=4, roll_max=6))
    choose_destroyed_by: Literal["target", "attacker"] = "target"
    exclude_limited_no_charges: bool = True
    fallback_to_other_if_none: bool = True
    fallback_to_direct_hit_if_none: bool = True

    model_config = {"frozen": True}


class StructureOutcomeType(BaseModel):
    """Outcome detail for structure damage."""

    name: Literal["glancing_blow", "system_trauma", "direct_hit", "crushing_hit"]
    impaired_until_end_next_turn: bool = False
    destroy_mount: bool = False
    destroy_system: bool = False
    stunned_until_end_next_turn: bool = False
    hull_check_required: bool = False
    destroyed: bool = False

    model_config = {"frozen": True}


class StructureTableEntry(BaseModel):
    """Table entry for structure damage results."""

    roll_min: int = Field(..., ge=1, le=6)
    roll_max: int = Field(..., ge=1, le=6)
    outcome: StructureOutcomeType

    model_config = {"frozen": True}


class DirectHitOutcome(BaseModel):
    """Direct hit outcome by remaining structure."""

    remaining_structure_min: int = Field(..., ge=0)
    remaining_structure_max: int | None = Field(default=None, ge=0)
    outcome: StructureOutcomeType

    model_config = {"frozen": True}


class StructureDamageRules(BaseModel):
    """Structure damage and check rules."""

    pc_structure: int = Field(default=4, ge=0)
    npc_structure: int = Field(default=1, ge=0)
    check_on_zero_hp: bool = True
    reset_hp_on_structure: bool = True
    spillover_damage_applies: bool = True
    dice_per_structure_marked: int = Field(default=1, ge=1)
    choose_lowest: bool = True
    multiple_ones_crushing: bool = True
    system_trauma_rules: SystemTraumaRules = Field(default_factory=SystemTraumaRules)
    crushing_hit_outcome: StructureOutcomeType = Field(
        default_factory=lambda: StructureOutcomeType(name="crushing_hit", destroyed=True)
    )
    table: list[StructureTableEntry] = Field(default_factory=list)
    direct_hit_outcomes: list[DirectHitOutcome] = Field(default_factory=list)

    model_config = {"frozen": True}


class OverheatOutcomeType(BaseModel):
    """Outcome detail for overheat checks."""

    name: Literal["emergency_shunt", "power_plant_destabilize", "meltdown", "irreversible_meltdown"]
    impaired_until_end_next_turn: bool = False
    exposed_until_cleared: bool = False
    meltdown_immediate: bool = False
    meltdown_countdown: bool = False
    engineering_check_to_delay: bool = False

    model_config = {"frozen": True}


class OverheatTableEntry(BaseModel):
    """Table entry for overheat results."""

    roll_min: int = Field(..., ge=1, le=6)
    roll_max: int = Field(..., ge=1, le=6)
    outcome: OverheatOutcomeType

    model_config = {"frozen": True}


class MeltdownOutcome(BaseModel):
    """Meltdown outcome by remaining stress."""

    remaining_stress_min: int = Field(..., ge=0)
    remaining_stress_max: int | None = Field(default=None, ge=0)
    outcome: OverheatOutcomeType

    model_config = {"frozen": True}


class OverheatRules(BaseModel):
    """Heat and overheat table rules."""

    stress_per_overheat: int = Field(default=1, ge=0)
    roll_dice_per_stress: bool = True
    choose_lowest: bool = True
    reset_heat_after_overheat: bool = True
    meltdown_at_zero_stress: bool = True
    danger_zone_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    danger_zone_rounding: Literal["up", "down"] = "up"
    table: list[OverheatTableEntry] = Field(default_factory=list)
    meltdown_outcomes: list[MeltdownOutcome] = Field(default_factory=list)
    irreversible_meltdown_on_multiple_ones: bool = True
    irreversible_meltdown_outcome: OverheatOutcomeType = Field(
        default_factory=lambda: OverheatOutcomeType(name="irreversible_meltdown", meltdown_countdown=True)
    )

    model_config = {"frozen": True}


class ReactorMeltdownRules(BaseModel):
    """Reactor meltdown resolution rules."""

    burst_radius: int = Field(default=2, ge=0)
    damage: DiceExpression = Field(default_factory=lambda: DiceExpression.parse("4d6"))
    damage_type: DamageType = "explosive"
    save_skill: Literal["agility"] = "agility"
    save_halves_damage: bool = True
    pilot_survival: bool = False

    model_config = {"frozen": True}


class RepairSpendOption(BaseModel):
    """Repairs spend option during rest or stabilize."""

    repairs_spent: int = Field(..., ge=0)
    effect: Literal[
        "full_hp",
        "repair_weapon_or_system",
        "repair_structure",
        "repair_stress",
        "repair_destroyed_mech",
    ]

    model_config = {"frozen": True}


class RestRepairRules(BaseModel):
    """Rest and repair rules."""

    rest_hours: int = Field(default=1, ge=0)
    heal_pilot_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    cool_heat_on_rest: bool = True
    end_mech_conditions_on_rest: bool = True
    repair_options: list[RepairSpendOption] = Field(default_factory=list)
    destroyed_mech_becomes_cover: bool = True
    destroyed_mech_difficult_terrain: bool = True

    model_config = {"frozen": True}


DEFAULT_STRUCTURE_DAMAGE_RULES = StructureDamageRules(
    table=[
        StructureTableEntry(
            roll_min=5,
            roll_max=6,
            outcome=StructureOutcomeType(name="glancing_blow", impaired_until_end_next_turn=True),
        ),
        StructureTableEntry(
            roll_min=2,
            roll_max=4,
            outcome=StructureOutcomeType(name="system_trauma", destroy_mount=True, destroy_system=True),
        ),
        StructureTableEntry(
            roll_min=1,
            roll_max=1,
            outcome=StructureOutcomeType(name="direct_hit"),
        ),
    ],
    direct_hit_outcomes=[
        DirectHitOutcome(
            remaining_structure_min=3,
            remaining_structure_max=None,
            outcome=StructureOutcomeType(name="direct_hit", stunned_until_end_next_turn=True),
        ),
        DirectHitOutcome(
            remaining_structure_min=2,
            remaining_structure_max=2,
            outcome=StructureOutcomeType(
                name="direct_hit", hull_check_required=True, stunned_until_end_next_turn=True
            ),
        ),
        DirectHitOutcome(
            remaining_structure_min=1,
            remaining_structure_max=1,
            outcome=StructureOutcomeType(name="direct_hit", destroyed=True),
        ),
    ],
)


DEFAULT_OVERHEAT_RULES = OverheatRules(
    table=[
        OverheatTableEntry(
            roll_min=5,
            roll_max=6,
            outcome=OverheatOutcomeType(name="emergency_shunt", impaired_until_end_next_turn=True),
        ),
        OverheatTableEntry(
            roll_min=2,
            roll_max=4,
            outcome=OverheatOutcomeType(name="power_plant_destabilize", exposed_until_cleared=True),
        ),
        OverheatTableEntry(
            roll_min=1,
            roll_max=1,
            outcome=OverheatOutcomeType(name="meltdown"),
        ),
    ],
    meltdown_outcomes=[
        MeltdownOutcome(
            remaining_stress_min=3,
            remaining_stress_max=None,
            outcome=OverheatOutcomeType(name="meltdown", exposed_until_cleared=True),
        ),
        MeltdownOutcome(
            remaining_stress_min=2,
            remaining_stress_max=2,
            outcome=OverheatOutcomeType(
                name="meltdown", meltdown_countdown=True, engineering_check_to_delay=True
            ),
        ),
        MeltdownOutcome(
            remaining_stress_min=1,
            remaining_stress_max=1,
            outcome=OverheatOutcomeType(name="meltdown", meltdown_immediate=True),
        ),
    ],
)


DEFAULT_REACTOR_MELTDOWN_RULES = ReactorMeltdownRules()


DEFAULT_REST_REPAIR_RULES = RestRepairRules(
    repair_options=[
        RepairSpendOption(repairs_spent=1, effect="full_hp"),
        RepairSpendOption(repairs_spent=1, effect="repair_weapon_or_system"),
        RepairSpendOption(repairs_spent=2, effect="repair_structure"),
        RepairSpendOption(repairs_spent=2, effect="repair_stress"),
        RepairSpendOption(repairs_spent=4, effect="repair_destroyed_mech"),
    ],
)


class MechCombatRules(BaseModel):
    """Top-level combat rules bundle."""

    turn_order: TurnOrderRules = Field(default_factory=TurnOrderRules)
    turn_actions: TurnActionRules = Field(default_factory=TurnActionRules)
    engagement: EngagementRules = Field(default_factory=EngagementRules)
    obstructions: ObstructionRules = Field(default_factory=ObstructionRules)
    terrain: TerrainRules = Field(default_factory=TerrainRules)
    falling: FallingRules = Field(default_factory=FallingRules)
    zero_g: ZeroGRules = Field(default_factory=ZeroGRules)
    teleport: TeleportRules = Field(default_factory=TeleportRules)
    flight: FlightRules = Field(default_factory=FlightRules)
    attack_rules: AttackRules = Field(default_factory=AttackRules)
    cover_rules: CoverRules = Field(default_factory=CoverRules)
    line_of_sight_rules: LineOfSightRules = Field(default_factory=LineOfSightRules)
    valid_target_rules: ValidTargetRules = Field(default_factory=ValidTargetRules)
    critical_hit_rules: CriticalHitRules = Field(default_factory=CriticalHitRules)
    bonus_damage_rules: BonusDamageRules = Field(default_factory=BonusDamageRules)
    damage_resolution: DamageResolutionRules = Field(default_factory=DamageResolutionRules)
    heat_rules: HeatRules = Field(default_factory=HeatRules)
    burn_rules: BurnRules = Field(default_factory=BurnRules)
    overcharge_rules: OverchargeRules = Field(default_factory=OverchargeRules)
    overheat_rules: OverheatRules = Field(default_factory=lambda: DEFAULT_OVERHEAT_RULES)
    structure_rules: StructureDamageRules = Field(default_factory=lambda: DEFAULT_STRUCTURE_DAMAGE_RULES)
    invisibility_rules: InvisibilityRules = Field(default_factory=InvisibilityRules)
    object_rules: ObjectRules = Field(default_factory=ObjectRules)
    reaction_rules: ReactionRules = Field(default_factory=ReactionRules)
    reactor_meltdown: ReactorMeltdownRules = Field(default_factory=ReactorMeltdownRules)
    rest_repair_rules: RestRepairRules = Field(default_factory=lambda: DEFAULT_REST_REPAIR_RULES)

    model_config = {"frozen": True}


DEFAULT_MECH_COMBAT_RULES = MechCombatRules()
