"""Type aliases for Lancer effect system.

This module contains all Literal type definitions used by the effects system.
Separating type aliases from effect classes improves readability and allows
importing just the types without loading all effect models.
"""

from typing import Literal

from core.shared.enums import DamageType
from core.shared.dice import DiceExpression

__all__ = [
    # Core types
    "StatType",
    "ConditionType",
    "TriggerType",
    "ReactionTriggerEvent",
    "ActionCategoryType",
    "EffectDuration",
    # Targeting
    "EffectTarget",
    "EffectTargetNoAll",
    "EffectTargetWithObject",
    "EffectTargetWithObjectNoAll",
    # Spatial
    "SpatialRelation",
    "AttackAreaShape",
    # Movement
    "MovementDistanceType",
    "ForcedMovementDistanceType",
    "MovementMode",
    # Intel
    "IntelAudience",
    "IntelType",
    # Checks
    "CheckKind",
    # Weapons
    "WeaponSizeType",
    "WeaponTypeType",
    # Areas/Zones
    "AreaSelectionScope",
    "ZoneEndTriggerType",
    "ZoneEndScope",
    # Resources
    "ResourceType",
    "ResourceAmount",
    "ResourceDirection",
    # Tech
    "TechRangeType",
    "TechActionScope",
    # Miscellaneous
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
    # Damage
    "DamageTypeScope",
    "DirectDamageType",
]

# -----------------------------------------------------------------------------
# Core Types
# -----------------------------------------------------------------------------

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
    "limited_bonus",
    # System budget
    "system_points",
]
"""Stats that can be modified by effects."""

ConditionType = Literal[
    # Target conditions
    "target_prone",
    "target_immobilized",
    "target_below_half_hp",
    "target_below_max_hp",
    "target_has_lock_on",
    "target_larger",
    "target_smaller",
    "target_same_size",
    "from_smaller",
    "target_hidden",
    "target_jammed",
    "target_shredded",
    "target_impaired",
    "target_slowed",
    "target_stunned",
    "target_grappled",
    "target_biological",
    "target_exposed",
    "target_invisible",
    "target_engaged",
    "target_marked",
    "target_within_range_3",
    "covering_fire_target",
    "covering_fire_target_moves",
    "covering_fire_bracketing",
    "covering_fire_overwatch",
    "jackhammer_target_object",
    "jackhammer_adjacent_to_object",
    "cannon_collateral_burst",
    "different_target_in_threat",
    "threat_except_original_target",
    "exemplar_marked_target",
    "ally_attacks_exemplar_marked_target",
    "spotter_adjacent_ally_consumes_lock_on",
    "exemplar_marked_target_attacks_other_within_3",
    "exemplar_challenge_active",
    "exemplar_challenge_other_targets",
    "during_rest",
    # Attacker conditions
    "while_flying",
    "while_hidden",
    "hidden_at_turn_start",
    "hidden_end_turn",
    "after_boost",
    "juggernaut_supercharge",
    "juggernaut_supercharge_through_character",
    "juggernaut_supercharge_through_obstacle",
    "after_move_8_plus",
    "in_danger_zone",
    "hidden",
    "engaged",
    "exposed",
    "overheated",
    "stunned",
    "braced",
    "brace",
    "reserve_power_mode",
    "steady_aim",
    "steady_aim_rifle_attack",
    "steady_aim_rifle_attack_crit",
    "rifle_attack",
    "rifle_attack_crit",
    "bondmate_adjacent",
    "bondmate_adjacent_and_damaged",
    "bondmate_targeted",
    "cover_me_overwatch",
    "leadership_die",
    "steel_assassin_charge",
    "selected_mount",
    "selected_weapon",
    "exclude_integrated_mounts",
    "structure_1_or_less",
    "zero_hp",
    "ram_or_grapple",
    "melee_knockback",
    "flying_this_turn",
    "natural_20",
    "after_miss",
    "turn_only_move_hide_boost",
    "caused_by_self",
    "gain_stress",
    "stabilize_cool",
    "tech_attack_hit",
    "invade_action",
    "performed_tech_action",
    "lock_on_action",
    "hacker_jam_cockpit",
    "hacker_disable_life_support",
    "hacker_hack_slash",
    "shutdown_action",
    "burn_equals_current_heat",
    "on_turn",
    "next_melee_vs_tech_hit_target",
    "ally_engaged_with_target",
    "higher_elevation_than_target",
    "target_not_in_cover",
    "cqb_target_within_3",
    "cqb_target_within_3_with_los",
    "next_ranged_vs_target_after_melee_hit",
    "next_melee_vs_target_after_ranged_hit",
    "adjacent_to_engaged_self",
    "before_or_after_skirmish",
    "first_reaction_attack_against_self",
    "flank_overwatch",
    "cqb_overwatch",
    "nexus_weapon_crit",
    "lock_on_consumed_drone_or_nexus_attack",
    "lock_on_consumed_tech_attack",
    "chaff_soft_cover_break_on_attack_or_save",
    "stormbringer_concussive_action",
    "stormbringer_torrent_attack",
    "stormbringer_thunder_blast",
    "thrown_aux_melee_attack",
    "adjacent_to_target",
    "adjacent_to_self",
    "restricting_swarm_active",
    "target_grappled_by_self",
    # Attack type conditions
    "heavy_or_superheavy_melee_attack",
    "backswing_cut_followup",
    "melee_attack",
    "ranged_attack",
    "tech_attack",
    "ram_attack",
    "ram_attack_after_boost",
    "ram_knockback_into_character",
    "ram_knockback_into_obstacle",
    "improvised_attack",
    "main_melee_attack",
    "aux_ranged_attack",
    "aux_melee_attack",
    "launcher_attack",
    "lock_on_consumed_launcher_attack",
    "cannon_attack",
    "fuel_rod_gun_attack",
    "melee_attack_no_other_adjacent",
    "melee_crit",
    "attack_roll_1_or_2",
    # Save conditions
    "save_vs_knockback_or_prone",
    # Spatial/targeting conditions
    "adjacent",
    "adjacent_recover_charged_stake",
    "adjacent_start_or_move",
    "ally_target",
    "ally_target_range_5",
    "hostile_target_range_5",
    "target_empty_space",
    "target_destroyed",
    "target_is_object",
    "target_is_drone",
    "target_is_deployable",
    "target_prone_or_immobilized_or_stunned",
    # Attack context conditions
    "after_boost_melee_attack",
    "first_ranged_attack_each_round",
    "attacks_from_owner",
    "attacks_within_range_3",
    "attacker_within_range_3_before_attack",
    "kinetic_attack_against_self_or_adjacent_ally",
    "launcher_attack_consumes_lock_on",
    "next_loading_attack",
    "next_ranged_attack",
    "ranged_attack_vs_marked_target_no_cover_outside_range_5",
    "ram_attack_vs_object",
    # State/mode conditions
    "ai_controlled_until_protocol_resume",
    "core_power_active",
    "core_power_full_action_mag_field",
    "protocol_ends",
    "stabilizer_attached",
    "structure_damage",
    # Movement conditions
    "after_move_or_boost",
    "boost_speed_plus_2",
    "hover_no_landing_required",
    "must_move_max_speed_each_move",
    "no_move",
    "slipstream_jump",
    # Miscellaneous conditions
    "exclusive_target",
    "from_triggering_attack",
    "half_damage",
    "melee_knockback_plus_2",
    "melee_attack_knockback_bonus",
    "melee_weapon_spend_charge",
    # Complex conditions
    "first_turn",
    "gm_approved_non_combat_check",
    "no_attacks_or_forced_saves",
    "toward_self_stop_adjacent_if_possible",
    "deactivate_start_turn",
    "ranged_threat_minimum",
    "sekhmet_auto_pilot_melee_only",
    "snare_trap_immobilize",
    "charged_stake_immobilized",
    "charged_stake_immobilize",
    "willing_or_stunned_adjacent_target",
    "unwilling_adjacent_target",
    "enter_zone_metal_target",
    "in_zone_metal_target",
    "reaction_parry_kinetic_attack_against_self_or_adjacent_ally",
    "hull_checks_and_saves",
    "stabilize_singularity",
    "agility_checks_and_saves",
    "until_end_of_next_turn",
    "invisible_to_chosen_target_only",
    "benefiting_from_trail_cover",
    "source_is_chosen_target",
    "ally_other_than_self_hits_markerlight_target",
    # HORUS conditions
    "mimic_mesh_copy_active",
    "monitor_charge_accumulated",
    "cascade_status_spread",
    "damage_type_selectable",
    "weapon_profile_copied",
    "chain_to_secondary_target",
    # HA conditions
    "overwatch_reaction",
    "overwatch_attack",
    "limit_break_active",
    "apocalypse_rail_attack",
    "warp_shield_redirect",
    "teleport_on_hit",
    "gravity_well_pull",
    "plasma_gauntlet_active",
    # Phase conditions
    "out_of_phase",
]
"""Conditions for conditional effects."""

TriggerType = Literal[
    "on_hit",
    "on_miss",
    "on_reload",
    "on_first_attack",
    "on_overkill",
    "on_crit",
    "on_kill",
    "on_activation",
    "on_attack_roll",
    "on_detonate",
    "on_move",
    "on_reaction",
    "on_action",
    "on_take_damage",
    "on_hide",
    "on_overheat",
    "on_ally_hit",
    "on_ally_miss",
    "on_ally_damaged",
    "on_ally_targeted",
    "on_turn_start",
    "on_turn_end",
    "on_overcharge",
    "on_core_power_spent",
    "on_structure_loss",
    "on_heat_gain",
    "on_target_failed_save",
    "on_tech_attack_hit",
    "on_ally_turn_start",
    "on_stabilize",
    "on_skirmish",
    "on_enter",
    "on_inflict",
    "on_brace",
    "on_move_or_boost",
    "on_slipstream_jump",
    "on_extra_overwatch",
    "on_ally_hit_target_within_range",
    "on_lock_on_consumed",
    "on_deploy",
    "on_destroyed",
    "on_adjacent",
    "on_attacked",
    "on_ally_killed",
    "on_hp_below_half",
    "on_damage_dealt",
    "on_first_adjacent_turn",
    "on_any_damage",
]
"""Common trigger points for conditional effects."""

ReactionTriggerEvent = Literal[
    "enemy_starts_movement_in_threat",
    "enemy_enters_threat",
    "enemy_leaves_threat",
    "enemy_exits_threat",
    "ally_hits_target",
    "enemy_attacks_ally",
    "covering_fire_target_moves",
]
"""Events that can trigger reactions."""

ActionCategoryType = Literal[
    "attack",
    "movement",
    "tech",
    "utility",
    "defense",
    "reaction",
]
"""Categories of actions."""

EffectDuration = Literal[
    "end_of_turn",
    "start_of_next_turn",
    "end_of_next_turn",
    "until_cleared",
    "scene",
]
"""Duration types for effects."""

# -----------------------------------------------------------------------------
# Targeting Types
# -----------------------------------------------------------------------------

EffectTarget = Literal["self", "enemy", "ally", "adjacent", "all"]
"""Who an effect can target."""

EffectTargetNoAll = Literal["self", "enemy", "ally", "adjacent"]
"""Effect targets excluding 'all'."""

EffectTargetWithObject = Literal["self", "enemy", "ally", "adjacent", "all", "object"]
"""Effect targets including objects."""

EffectTargetWithObjectNoAll = Literal["self", "enemy", "ally", "adjacent", "object"]
"""Effect targets with objects, excluding 'all'."""

# -----------------------------------------------------------------------------
# Spatial Types
# -----------------------------------------------------------------------------

SpatialRelation = Literal[
    "adjacent",
    "within_range",
    "outside_range",
    "within_threat",
    "enter_zone",
    "line_of_attack",
    "line_of_sight",
    "using_cover_from_source",
]
"""Spatial relationship types."""

AttackAreaShape = Literal["blast", "burst", "line", "cone"]
"""Shapes for area attacks."""

# -----------------------------------------------------------------------------
# Movement Types
# -----------------------------------------------------------------------------

MovementDistanceType = int | DiceExpression | Literal["speed"]
"""How to specify movement distance."""

ForcedMovementDistanceType = int | DiceExpression | Literal["as_far_as_possible"]
"""How to specify forced movement distance."""

MovementMode = Literal["move", "boost", "other", "any"]
"""Types of movement."""

# -----------------------------------------------------------------------------
# Intel Types
# -----------------------------------------------------------------------------

IntelAudience = Literal["self", "allies", "all"]
"""Who receives intel from scanning/searching."""

IntelType = Literal[
    "location",
    "hp",
    "armor",
    "structure",
    "heat",
    "hase",
    "speed",
    "evasion",
    "e_defense",
    "weapons",
    "systems",
    "area",
]
"""Types of information that can be revealed."""

# -----------------------------------------------------------------------------
# Check Types
# -----------------------------------------------------------------------------

CheckKind = Literal["check", "save", "contested_check", "search"]
"""Types of checks/rolls."""

# -----------------------------------------------------------------------------
# Weapon Types
# -----------------------------------------------------------------------------

WeaponSizeType = Literal["aux", "main", "heavy", "superheavy"]
"""Weapon mount sizes."""

WeaponTypeType = Literal["cqb", "rifle", "launcher", "cannon", "melee", "nexus"]
"""Weapon type categories."""

# -----------------------------------------------------------------------------
# Area/Zone Types
# -----------------------------------------------------------------------------

AreaSelectionScope = Literal["area", "spaces", "battlefield"]
"""Scope for area selection effects."""

ZoneEndTriggerType = Literal[
    "enter",
    "start_turn",
    "end_turn",
    "turn_start",
    "turn_end",
]
"""Events that can end a zone effect."""

ZoneEndScope = Literal["zone", "triggered_space"]
"""What ends when a zone ends."""

# -----------------------------------------------------------------------------
# Resource Types
# -----------------------------------------------------------------------------

ResourceType = Literal["hp", "heat", "repairs", "structure", "stress", "core_power"]
"""Types of resources that can be modified."""

ResourceAmount = int | DiceExpression | Literal["quarter_max", "half_max", "full"]
"""How to specify resource amounts."""

ResourceDirection = Literal["gain", "lose", "set"]
"""Direction of resource change."""

# -----------------------------------------------------------------------------
# Tech Types
# -----------------------------------------------------------------------------

TechRangeType = Literal["sensors", "range"]
"""How tech action range is specified."""

TechActionScope = Literal["all", "hack", "invade", "quick_tech", "full_tech"]
"""Which tech actions an effect applies to."""

# -----------------------------------------------------------------------------
# Miscellaneous Types
# -----------------------------------------------------------------------------

UsesPer = Literal["unlimited", "round", "scene", "mission", "rest", "full_repair"]
"""How often an ability can be used."""

BreakTriggerType = Literal[
    "attack",
    "force_save",
    "move",
    "reaction",
    "turn_start",
    "take_damage",
    "stunned",
    "manual_deactivate",
    "source_destroyed",
]
"""Events that can break concentration/sustained effects."""

NonCombatInteractionScope = Literal["pilot_scale", "environment", "any"]
"""Scope for non-combat interactions."""

PassengerLocation = Literal["cockpit", "compartment"]
"""Where passengers are located in a mech."""

RollPatternType = Literal["triples", "doubles"]
"""Special roll patterns to check for."""

OutOfPlayDuration = Literal[
    "until_cleared", "until_rest", "scene", "mission", "start_of_next_turn"
]
"""How long something stays out of play."""

DeploymentActivationCondition = Literal[
    "adjacent_start_or_move",
    "adjacent_start",
    "adjacent_move",
    "manual",
    "on_prime",
    "pass_over",
]
"""Conditions for activating deployables."""

DelayedImpactTiming = Literal[
    "end_of_next_round", "end_of_next_turn", "start_of_next_turn"
]
"""When delayed effects trigger."""

PhaseState = Literal["in_phase", "out_of_phase"]
"""Phase shift states."""

HologramTrailTrigger = Literal["move", "boost", "move_or_boost"]
"""What triggers hologram trail creation."""

HologramDetonationTrigger = Literal["start_turn", "move_through", "move_adjacent"]
"""What triggers hologram detonation."""

# -----------------------------------------------------------------------------
# Damage Types (compound)
# -----------------------------------------------------------------------------

DamageTypeScope = DamageType | Literal["heat", "burn", "all"]
"""Damage types including heat and burn."""

DirectDamageType = DamageType | Literal["heat"]
"""Damage types for direct damage effects."""
