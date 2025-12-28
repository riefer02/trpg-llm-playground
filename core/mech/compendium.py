"""Mechanics-only compendium definitions for mech frames, weapons, and systems."""

from core.mech.frame import (
    CoreSystemDefinition,
    FrameTrait,
    MechFrameBaseStats,
    MechFrameDefinition,
)
from core.mech.mounts import MountSlot
from core.mech.system import (
    AreaEffect,
    DamageSpec,
    DeployableEffect,
    DeployableObject,
    DronePayload,
    DroneReaction,
    FlightEffect,
    GrenadePayload,
    MechSystemDefinition,
    MinePayload,
    SystemTag,
)
from core.mech.weapon import (
    MechWeaponDefinition,
    WeaponDamage,
    WeaponRange,
    WeaponTag,
)
from core.shared.dice import DiceExpression
from core.shared.effects import (
    AccuracyModifier,
    ActionGrant,
    ActionRestriction,
    AreaAttackPattern,
    CoverRestriction,
    CoverGrant,
    DelayedImpactEffect,
    DamageModifier,
    DamageReduction,
    DirectDamage,
    EffectRemoval,
    EffectChoice,
    ForcedMovement,
    HologramTrailEffect,
    IntelEffect,
    Immunity,
    Resistance,
    MechanicalEffect,
    MovementGrant,
    MovementRestrictionEffect,
    MovementScopedStatus,
    PhaseShiftEffect,
    RandomCheckEffect,
    ResourceChange,
    DeploymentEffect,
    ReloadEffect,
    DamageAbsorption,
    TetherEffect,
    SaveCheck,
    StatModifier,
    StatusBreakCondition,
    StatusGrant,
    StatusClear,
    StatusStackLimit,
    StatusRestriction,
    StatusTrigger,
    AttackTargetingEffect,
    WeaponModEffect,
    WeaponSizeBonus,
    WeaponTagGrant,
    TriggeredEffect,
    TechAction,
    TechAttackModifier,
    TechActionRestriction,
    TechRange,
    ZoneEffect,
)


GMS_FRAMES: list[MechFrameDefinition] = [
    MechFrameDefinition(
        id="gms_everest",
        name="GMS Everest",
        manufacturer="GMS",
        license_id=None,
        license_rank=None,
        base_stats=MechFrameBaseStats(
            size="size_1",
            armor=0,
            hp=10,
            evasion=8,
            e_defense=8,
            speed=4,
            sensor_range=10,
            tech_attack=0,
            heat_cap=6,
            repair_cap=5,
            save_target=10,
        ),
        mounts=[
            MountSlot(slot_type="heavy"),
            MountSlot(slot_type="main"),
            MountSlot(slot_type="flexible"),
        ],
        system_points=6,
        core_system=CoreSystemDefinition(
            id="gms_hyperspec_fuel_injector",
            name="GMS Hyperspec Fuel Injector",
            effects=MechanicalEffect(
                special="core_power_extra_full_or_two_quick_actions_free",
            ),
        ),
        traits=[
            FrameTrait(
                name="Initiative",
                effects=MechanicalEffect(
                    triggered_effects=[
                        TriggeredEffect(
                            trigger="on_turn_start",
                            condition="first_turn",
                            effect=MechanicalEffect(
                                action_grants=[
                                    ActionGrant(
                                        action_type="quick",
                                        name="bonus_quick_action",
                                        uses_per="round",
                                    )
                                ],
                            ),
                        )
                    ],
                ),
            ),
            FrameTrait(
                name="Replaceable Parts",
                effects=MechanicalEffect(
                    special="repair_structure_cost_1_instead_of_2",
                ),
            ),
        ],
    ),
]


GMS_WEAPONS: list[MechWeaponDefinition] = [
    MechWeaponDefinition(
        id="anti_material_rifle",
        name="Anti-Material Rifle",
        size="heavy",
        weapon_type="rifle",
        damage_type="kinetic",
        ranges=[WeaponRange(range_type="range", value=20)],
        damage=[WeaponDamage(damage_type="kinetic", dice=DiceExpression.parse("2d6"))],
        tags=[
            WeaponTag(tag="ap"),
            WeaponTag(tag="loading"),
            WeaponTag(tag="ordnance"),
            WeaponTag(tag="accurate"),
        ],
    ),
    MechWeaponDefinition(
        id="assault_rifle",
        name="Assault Rifle",
        size="main",
        weapon_type="rifle",
        damage_type="kinetic",
        ranges=[WeaponRange(range_type="range", value=10)],
        damage=[WeaponDamage(damage_type="kinetic", dice=DiceExpression.parse("1d6"))],
        tags=[WeaponTag(tag="reliable", value=2)],
    ),
    MechWeaponDefinition(
        id="charged_blade",
        name="Charged Blade",
        size="main",
        weapon_type="melee",
        damage_type="energy",
        ranges=[WeaponRange(range_type="threat", value=1)],
        damage=[WeaponDamage(damage_type="energy", dice=DiceExpression.parse("1d3+3"))],
        tags=[WeaponTag(tag="ap")],
    ),
    MechWeaponDefinition(
        id="nexus_light",
        name="Nexus (Light)",
        size="aux",
        weapon_type="nexus",
        damage_type="kinetic",
        ranges=[WeaponRange(range_type="range", value=10)],
        damage=[WeaponDamage(damage_type="kinetic", dice=DiceExpression.parse("1d3"))],
        tags=[WeaponTag(tag="smart")],
    ),
    MechWeaponDefinition(
        id="nexus_hunter_killer",
        name="Nexus (Hunter-Killer)",
        size="main",
        weapon_type="nexus",
        damage_type="kinetic",
        ranges=[WeaponRange(range_type="range", value=10)],
        damage=[WeaponDamage(damage_type="kinetic", dice=DiceExpression.parse("1d6"))],
        tags=[WeaponTag(tag="smart")],
    ),
    MechWeaponDefinition(
        id="heavy_machine_gun",
        name="Heavy Machine Gun",
        size="heavy",
        weapon_type="cannon",
        damage_type="kinetic",
        ranges=[WeaponRange(range_type="range", value=8)],
        damage=[WeaponDamage(damage_type="kinetic", dice=DiceExpression.parse("2d6+4"))],
        tags=[WeaponTag(tag="inaccurate")],
    ),
    MechWeaponDefinition(
        id="heavy_melee_weapon",
        name="Heavy Melee Weapon",
        size="heavy",
        weapon_type="melee",
        damage_type="kinetic",
        ranges=[WeaponRange(range_type="threat", value=1)],
        damage=[WeaponDamage(damage_type="kinetic", dice=DiceExpression.parse("2d6+1"))],
    ),
    MechWeaponDefinition(
        id="heavy_charged_blade",
        name="Heavy Charged Blade",
        size="heavy",
        weapon_type="melee",
        damage_type="energy",
        ranges=[WeaponRange(range_type="threat", value=1)],
        damage=[WeaponDamage(damage_type="energy", dice=DiceExpression.parse("1d6+3"))],
        tags=[WeaponTag(tag="ap")],
    ),
    MechWeaponDefinition(
        id="howitzer",
        name="Howitzer",
        size="heavy",
        weapon_type="cannon",
        damage_type="explosive",
        ranges=[
            WeaponRange(range_type="range", value=20),
            WeaponRange(range_type="blast", value=2),
        ],
        damage=[WeaponDamage(damage_type="explosive", dice=DiceExpression.parse("2d6"))],
        tags=[
            WeaponTag(tag="arcing"),
            WeaponTag(tag="inaccurate"),
            WeaponTag(tag="loading"),
            WeaponTag(tag="ordnance"),
        ],
    ),
    MechWeaponDefinition(
        id="missile_rack",
        name="Missile Rack",
        size="aux",
        weapon_type="launcher",
        damage_type="explosive",
        ranges=[
            WeaponRange(range_type="range", value=10),
            WeaponRange(range_type="blast", value=1),
        ],
        damage=[WeaponDamage(damage_type="explosive", dice=DiceExpression.parse("1d3+1"))],
        tags=[WeaponTag(tag="loading")],
    ),
    MechWeaponDefinition(
        id="mortar",
        name="Mortar",
        size="main",
        weapon_type="cannon",
        damage_type="explosive",
        ranges=[
            WeaponRange(range_type="range", value=15),
            WeaponRange(range_type="blast", value=1),
        ],
        damage=[WeaponDamage(damage_type="explosive", dice=DiceExpression.parse("1d6+1"))],
        tags=[WeaponTag(tag="arcing"), WeaponTag(tag="inaccurate")],
    ),
    MechWeaponDefinition(
        id="pistol",
        name="Pistol",
        size="aux",
        weapon_type="cqb",
        damage_type="kinetic",
        ranges=[
            WeaponRange(range_type="range", value=5),
            WeaponRange(range_type="threat", value=3),
        ],
        damage=[WeaponDamage(damage_type="kinetic", dice=DiceExpression.parse("1d3"))],
    ),
    MechWeaponDefinition(
        id="progressive_knife",
        name="Progressive Knife",
        size="aux",
        weapon_type="melee",
        damage_type="energy",
        ranges=[WeaponRange(range_type="threat", value=1)],
        damage=[WeaponDamage(damage_type="energy", dice=DiceExpression.parse("1d3+1"))],
        tags=[WeaponTag(tag="overkill")],
    ),
    MechWeaponDefinition(
        id="cyclone_pulse_rifle",
        name="Cyclone Pulse Rifle",
        size="superheavy",
        weapon_type="rifle",
        damage_type="kinetic",
        ranges=[WeaponRange(range_type="range", value=15)],
        damage=[WeaponDamage(damage_type="kinetic", dice=DiceExpression.parse("3d6+3"))],
        tags=[
            WeaponTag(tag="reliable", value=5),
            WeaponTag(tag="accurate"),
            WeaponTag(tag="loading"),
        ],
    ),
    MechWeaponDefinition(
        id="rpg",
        name="RPG",
        size="main",
        weapon_type="launcher",
        damage_type="explosive",
        ranges=[
            WeaponRange(range_type="range", value=10),
            WeaponRange(range_type="blast", value=2),
        ],
        damage=[WeaponDamage(damage_type="explosive", dice=DiceExpression.parse("1d6+1"))],
        tags=[WeaponTag(tag="loading"), WeaponTag(tag="ordnance")],
    ),
    MechWeaponDefinition(
        id="shotgun",
        name="Shotgun",
        size="main",
        weapon_type="cqb",
        damage_type="kinetic",
        ranges=[
            WeaponRange(range_type="range", value=5),
            WeaponRange(range_type="threat", value=3),
        ],
        damage=[WeaponDamage(damage_type="kinetic", dice=DiceExpression.parse("1d6"))],
    ),
    MechWeaponDefinition(
        id="tactical_melee",
        name="Tactical Melee",
        size="main",
        weapon_type="melee",
        damage_type="kinetic",
        ranges=[WeaponRange(range_type="threat", value=1)],
        damage=[WeaponDamage(damage_type="kinetic", dice=DiceExpression.parse("1d6+2"))],
    ),
    MechWeaponDefinition(
        id="tactical_knife",
        name="Tactical Knife",
        size="aux",
        weapon_type="melee",
        damage_type="kinetic",
        ranges=[
            WeaponRange(range_type="threat", value=1),
            WeaponRange(range_type="thrown", value=3),
        ],
        damage=[WeaponDamage(damage_type="kinetic", dice=DiceExpression.parse("1d3+1"))],
    ),
    MechWeaponDefinition(
        id="thermal_pistol",
        name="Thermal Pistol",
        size="aux",
        weapon_type="cqb",
        damage_type="energy",
        ranges=[WeaponRange(range_type="line", value=5)],
        damage=[WeaponDamage(damage_type="energy", flat=2)],
    ),
    MechWeaponDefinition(
        id="thermal_rifle",
        name="Thermal Rifle",
        size="main",
        weapon_type="rifle",
        damage_type="energy",
        ranges=[WeaponRange(range_type="range", value=5)],
        damage=[WeaponDamage(damage_type="energy", dice=DiceExpression.parse("1d3+2"))],
        tags=[WeaponTag(tag="ap")],
    ),
    MechWeaponDefinition(
        id="thermal_lance",
        name="Thermal Lance",
        size="heavy",
        weapon_type="cannon",
        damage_type="energy",
        ranges=[WeaponRange(range_type="line", value=10)],
        damage=[WeaponDamage(damage_type="energy", dice=DiceExpression.parse("1d6+3"))],
        tags=[WeaponTag(tag="heat_self", value=2)],
    ),
]


GMS_SYSTEMS: list[MechSystemDefinition] = [
    MechSystemDefinition(
        id="gms_manipulators",
        name="Manipulators",
        sp_cost=1,
        unique=True,
        tags=[SystemTag(tag="mod")],
        effects=MechanicalEffect(
            special="extra_limbs_non_combat_interaction",
        ),
    ),
    MechSystemDefinition(
        id="gms_expanded_compartment",
        name="Expanded Compartment",
        sp_cost=1,
        unique=True,
        tags=[SystemTag(tag="mod")],
        effects=MechanicalEffect(
            special="extra_size_half_passenger_in_cockpit",
        ),
    ),
    MechSystemDefinition(
        id="gms_custom_paint_job",
        name="Custom Paint Job",
        sp_cost=1,
        unique=True,
        tags=[SystemTag(tag="mod")],
        effects=MechanicalEffect(
            special="structure_damage_ignore_on_6_once_per_full_repair",
        ),
    ),
    MechSystemDefinition(
        id="personalizations",
        name="Personalizations",
        sp_cost=1,
        unique=True,
        tags=[SystemTag(tag="mod")],
        effects=MechanicalEffect(
            stat_mods=[StatModifier(stat="hp", value=2)],
            special="minor_noncombat_mod_plus_1_accuracy_if_relevant",
        ),
    ),
    MechSystemDefinition(
        id="gms_stable_structure",
        name="Stable Structure",
        sp_cost=2,
        unique=True,
        tags=[SystemTag(tag="mod")],
        effects=MechanicalEffect(
            accuracy_mods=[
                AccuracyModifier(value=1, applies_to="all", condition="save_vs_knockback_or_prone")
            ],
        ),
    ),
    MechSystemDefinition(
        id="gms_dummy_plug",
        name="Companion/Concierge-Class Dummy Plug",
        sp_cost=2,
        unique=True,
        system_type="ai",
        tags=[SystemTag(tag="ai"), SystemTag(tag="protocol")],
        effects=MechanicalEffect(
            special="protocol_hand_over_control_ai_no_independent_initiative",
        ),
    ),
    MechSystemDefinition(
        id="gms_shield_type_1",
        name="GMS Shield Type-I",
        sp_cost=2,
        unique=True,
        system_type="shield",
        tags=[SystemTag(tag="shield"), SystemTag(tag="protocol")],
        effects=MechanicalEffect(
            special="protocol_target_mutual_attacks_difficulty_2_until_next_turn",
        ),
    ),
    MechSystemDefinition(
        id="gms_filter_smoke_charges",
        name="GMS Pattern-A Filter Smoke Charges",
        sp_cost=2,
        unique=True,
        limited_uses=3,
        tags=[SystemTag(tag="grenade"), SystemTag(tag="mine")],
        grenades=[
            GrenadePayload(
                name="Smoke Grenade",
                range=5,
                area=AreaEffect(
                    pattern="blast",
                    size=2,
                    duration="end_of_next_turn",
                    cover="soft",
                ),
            ),
        ],
        mines=[
            MinePayload(
                name="Shroud Mine",
                detonation="ally_adjacent_movement",
                area=AreaEffect(
                    pattern="burst",
                    size=3,
                    duration="end_of_turn",
                    cover="soft",
                ),
            ),
        ],
    ),
    MechSystemDefinition(
        id="gms_hex_charges",
        name="GMS Pattern-B Hex Charges",
        sp_cost=2,
        unique=True,
        limited_uses=3,
        tags=[SystemTag(tag="grenade"), SystemTag(tag="mine")],
        grenades=[
            GrenadePayload(
                name="Frag Grenade",
                range=5,
                area=AreaEffect(
                    pattern="blast",
                    size=1,
                    damage=DamageSpec(
                        damage_type="explosive",
                        dice=DiceExpression.parse("1d6"),
                    ),
                    save="agility",
                    half_on_success=True,
                ),
            ),
        ],
        mines=[
            MinePayload(
                name="Explosive Mine",
                area=AreaEffect(
                    pattern="burst",
                    size=1,
                    damage=DamageSpec(
                        damage_type="explosive",
                        dice=DiceExpression.parse("2d6"),
                    ),
                    save="agility",
                    half_on_success=True,
                ),
            ),
        ],
    ),
    MechSystemDefinition(
        id="gms_jericho_cover",
        name="GMS Pattern-A Jericho Deployable Cover",
        sp_cost=2,
        unique=True,
        system_type="deployable",
        tags=[SystemTag(tag="deployable"), SystemTag(tag="quick_action")],
        deployable=DeployableEffect(
            count=2,
            obj=DeployableObject(
                size=1,
                cover="hard",
                evasion=5,
                hp=10,
            ),
            pickup_action="full",
        ),
    ),
    MechSystemDefinition(
        id="gms_turret_drones",
        name="GMS Turret Drones",
        sp_cost=2,
        unique=True,
        limited_uses=3,
        system_type="drone",
        tags=[SystemTag(tag="drone"), SystemTag(tag="reaction"), SystemTag(tag="quick_action")],
        drone=DronePayload(
            name="Turret Drone",
            size="size_half",
            hp=10,
            evasion=10,
            e_defense=10,
            reactions=[
                DroneReaction(
                    name="Turret Attack",
                    trigger="ally_hit_target_within_range",
                    range=10,
                    damage=DamageSpec(
                        damage_type="kinetic",
                        flat=3,
                    ),
                ),
            ],
        ),
    ),
    MechSystemDefinition(
        id="gms_eva_module",
        name="GMS EVA Module",
        sp_cost=1,
        unique=True,
        flight=FlightEffect(
            mode="environmental",
            environment=["low_g", "zero_g", "submarine"],
            ignores_slowed_in_environment=True,
        ),
    ),
    MechSystemDefinition(
        id="gms_jump_jet_burst",
        name="GMS Burst Jump Jet System",
        sp_cost=2,
        unique=True,
        flight=FlightEffect(
            mode="boost",
            must_end_on_surface=True,
        ),
    ),
    MechSystemDefinition(
        id="gms_flight_system_type_1",
        name="GMS Type I Flight System",
        sp_cost=3,
        unique=True,
        flight=FlightEffect(
            mode="move_or_boost",
            heat_on_turn_end="size_plus_1",
        ),
    ),
]


IPSN_FRAMES: list[MechFrameDefinition] = [
    MechFrameDefinition(
        id="ipsn_raleigh",
        name="IPS-N Raleigh",
        manufacturer="IPS-N",
        license_id="raleigh",
        license_rank=2,
        base_stats=MechFrameBaseStats(
            size="size_1",
            armor=1,
            hp=10,
            evasion=8,
            e_defense=7,
            speed=4,
            sensor_range=10,
            tech_attack=-1,
            heat_cap=5,
            repair_cap=5,
            save_target=10,
        ),
        mounts=[
            MountSlot(slot_type="heavy"),
            MountSlot(slot_type="flexible"),
            MountSlot(slot_type="aux_aux"),
            MountSlot(slot_type="integrated", integrated_weapon_id="ipsn_m35_mjolnir"),
        ],
        system_points=5,
        core_system=CoreSystemDefinition(
            id="ipsn_thunder_god_protocol",
            name="Thunder God Protocol",
            effects=MechanicalEffect(
                special=(
                    "protocol_core_power_thunder_god: "
                    "if_mjolnir_not_fired_gain_2_chambers_end_turn "
                    "starts_0 "
                    "fire_all_chambers_4_damage_each_max_6 "
                    "if_chambers_4_plus_ap_and_target_shredded_until_end_next_turn"
                ),
            ),
        ),
        traits=[
            FrameTrait(
                name="Full Metal Jacket",
                effects=MechanicalEffect(
                    triggered_effects=[
                        TriggeredEffect(
                            trigger="on_turn_end",
                            condition="no_attacks_or_forced_saves",
                            effect=MechanicalEffect(
                                special="reload_all_loading_weapons",
                            ),
                        )
                    ],
                ),
            ),
            FrameTrait(
                name="Shielded Magazines",
                effects=MechanicalEffect(
                    special="can_make_ranged_attacks_while_jammed",
                ),
            ),
        ],
    ),
    MechFrameDefinition(
        id="ipsn_blackbeard",
        name="IPS-N Blackbeard",
        manufacturer="IPS-N",
        license_id="blackbeard",
        license_rank=2,
        base_stats=MechFrameBaseStats(
            size="size_1",
            armor=1,
            hp=12,
            evasion=8,
            e_defense=6,
            speed=5,
            sensor_range=5,
            tech_attack=-2,
            heat_cap=4,
            repair_cap=5,
            save_target=10,
        ),
        mounts=[
            MountSlot(slot_type="flexible"),
            MountSlot(slot_type="main"),
            MountSlot(slot_type="heavy"),
        ],
        system_points=5,
        core_system=CoreSystemDefinition(
            id="ipsn_assault_grapples",
            name="Assault Grapples",
            effects=MechanicalEffect(
                special=(
                    "core_power_quick_action_range_5_multi_target_hull_save_2d6_kinetic "
                    "half_on_success_fail_prone_pull_adjacent_immobilized_end_next_turn"
                ),
            ),
        ),
        traits=[
            FrameTrait(
                name="Cable Grapple",
                effects=MechanicalEffect(
                    special="grapple_range_5_pull_adjacent_on_success_break_if_no_adjacent_path",
                ),
            ),
            FrameTrait(
                name="Lock/Kill Subsystem",
                effects=MechanicalEffect(
                    special="can_boost_and_react_while_grappling",
                ),
            ),
            FrameTrait(
                name="Exposed Reactor",
                effects=MechanicalEffect(
                    special="plus_1_difficulty_engineering_checks_and_saves",
                ),
            ),
        ],
    ),
    MechFrameDefinition(
        id="ipsn_drake",
        name="IPS-N Drake",
        manufacturer="IPS-N",
        license_id="drake",
        license_rank=2,
        base_stats=MechFrameBaseStats(
            size="size_2",
            armor=3,
            hp=8,
            evasion=6,
            e_defense=6,
            speed=3,
            sensor_range=10,
            tech_attack=0,
            heat_cap=5,
            repair_cap=5,
            save_target=10,
        ),
        mounts=[
            MountSlot(slot_type="main"),
            MountSlot(slot_type="main"),
            MountSlot(slot_type="heavy"),
        ],
        system_points=5,
        core_system=CoreSystemDefinition(
            id="ipsn_fortress",
            name="Fortress",
            effects=MechanicalEffect(
                special=(
                    "core_power_fortress_protocol_immobilized "
                    "brace_full_action_next_turn "
                    "deploy_two_line2_size1_hard_cover_immune_damage "
                    "self_hard_cover "
                    "allies_using_cover_gain_immunity_knockback_prone_involuntary_movement "
                    "allies_and_self_resist_damage_heat_burn_from_blast_line_cone "
                    "deactivate_start_turn_protocol"
                ),
            ),
        ),
        traits=[
            FrameTrait(
                name="Heavy Frame",
                effects=MechanicalEffect(
                    immunities=[
                        Immunity(target="knockback", condition="from_smaller"),
                        Immunity(target="prone", condition="from_smaller"),
                    ],
                ),
            ),
            FrameTrait(
                name="Blast Plating",
                effects=MechanicalEffect(
                    resistances=[
                        Resistance(damage_type="all", condition="blast_line_cone")
                    ],
                    special="resist_heat_from_blast_line_cone",
                ),
            ),
            FrameTrait(
                name="Guardian",
                effects=MechanicalEffect(
                    cover_grants=[
                        CoverGrant(
                            cover="hard",
                            target="ally",
                            duration="scene",
                            condition="adjacent_to_self",
                        )
                    ],
                ),
            ),
            FrameTrait(
                name="Slow",
                effects=MechanicalEffect(
                    special="plus_1_difficulty_agility_checks_and_saves",
                ),
            ),
        ],
    ),
    MechFrameDefinition(
        id="ipsn_lancaster",
        name="IPS-N Lancaster",
        manufacturer="IPS-N",
        license_id="lancaster",
        license_rank=2,
        base_stats=MechFrameBaseStats(
            size="size_2",
            armor=1,
            hp=6,
            evasion=8,
            e_defense=8,
            speed=6,
            sensor_range=8,
            tech_attack=1,
            heat_cap=6,
            repair_cap=10,
            save_target=10,
        ),
        mounts=[
            MountSlot(slot_type="main_aux"),
            MountSlot(slot_type="integrated", integrated_weapon_id="ipsn_latch_drone"),
        ],
        system_points=8,
        core_system=CoreSystemDefinition(
            id="ipsn_supercharger",
            name="Supercharger",
            effects=MechanicalEffect(
                special=(
                    "core_power_supercharger_quick_action_target_ally_range8 "
                    "self_heat_each_turn_target_accuracy_all_attacks_checks_saves "
                    "target_immunity_impaired_jammed_slowed_shredded_immobilized_from_others "
                    "ends_if_self_or_target_stunned "
                    "cannot_fire_latch_drone_while_active"
                ),
            ),
        ),
        traits=[
            FrameTrait(
                name="Redundant Systems",
                effects=MechanicalEffect(
                    special="adjacent_allies_can_spend_lancaster_repairs",
                ),
            ),
            FrameTrait(
                name="Combat Repair",
                effects=MechanicalEffect(
                    special="full_action_spend_4_repairs_restore_destroyed_mech_1_structure_1_hp",
                ),
            ),
            FrameTrait(
                name="Insulated",
                effects=MechanicalEffect(
                    immunities=[Immunity(target="burn")],
                ),
            ),
        ],
    ),
    MechFrameDefinition(
        id="ipsn_nelson",
        name="IPS-N Nelson",
        manufacturer="IPS-N",
        license_id="nelson",
        license_rank=2,
        base_stats=MechFrameBaseStats(
            size="size_1",
            armor=0,
            hp=8,
            evasion=11,
            e_defense=7,
            speed=5,
            sensor_range=5,
            tech_attack=0,
            heat_cap=6,
            repair_cap=5,
            save_target=10,
        ),
        mounts=[
            MountSlot(slot_type="flexible"),
            MountSlot(slot_type="main_aux"),
        ],
        system_points=6,
        core_system=CoreSystemDefinition(
            id="ipsn_perpetual_momentum_drive",
            name="Perpetual Momentum Drive",
            effects=MechanicalEffect(
                special="core_power_protocol_skirmisher_move_4_scene",
            ),
        ),
        traits=[
            FrameTrait(
                name="Momentum",
                effects=MechanicalEffect(
                    triggered_effects=[
                        TriggeredEffect(
                            trigger="on_hit",
                            condition="after_boost_melee_attack",
                            uses_per="round",
                            effect=MechanicalEffect(
                                damage_mods=[
                                    DamageModifier(
                                        dice=DiceExpression.parse("1d6"),
                                        condition="melee_attack",
                                    )
                                ],
                            ),
                        )
                    ],
                ),
            ),
            FrameTrait(
                name="Skirmisher",
                effects=MechanicalEffect(
                    movement_grants=[
                        MovementGrant(
                            spaces=1,
                            movement_type="walk",
                            trigger="after_attack_ignore_reactions_ignore_engagement_not_immobilized_or_slowed",
                        )
                    ],
                ),
            ),
        ],
    ),
    MechFrameDefinition(
        id="ipsn_tortuga",
        name="IPS-N Tortuga",
        manufacturer="IPS-N",
        license_id="tortuga",
        license_rank=2,
        base_stats=MechFrameBaseStats(
            size="size_2",
            armor=2,
            hp=8,
            evasion=6,
            e_defense=10,
            speed=3,
            sensor_range=15,
            tech_attack=1,
            heat_cap=6,
            repair_cap=6,
            save_target=10,
        ),
        mounts=[
            MountSlot(slot_type="main"),
            MountSlot(slot_type="heavy"),
        ],
        system_points=6,
        core_system=CoreSystemDefinition(
            id="ipsn_sentinel",
            name="Sentinel",
            effects=MechanicalEffect(
                triggered_effects=[
                    TriggeredEffect(
                        trigger="on_hit",
                        condition="overwatch_attack",
                        effect=MechanicalEffect(
                            status_grants=[
                                StatusGrant(
                                    status="immobilized",
                                    target="enemy",
                                    duration="start_of_next_turn",
                                )
                            ],
                        ),
                    )
                ],
                special="core_power_protocol_ranged_threat_min_5_extra_overwatch_per_round",
            ),
        ),
        traits=[
            FrameTrait(
                name="Sentinel",
                effects=MechanicalEffect(
                    accuracy_mods=[
                        AccuracyModifier(value=1, condition="reaction_attack")
                    ],
                ),
            ),
            FrameTrait(
                name="Guardian",
                effects=MechanicalEffect(
                    cover_grants=[
                        CoverGrant(
                            cover="hard",
                            target="ally",
                            duration="scene",
                            condition="adjacent_to_self",
                        )
                    ],
                ),
            ),
        ],
    ),
    MechFrameDefinition(
        id="ipsn_vlad",
        name="IPS-N Vlad",
        manufacturer="IPS-N",
        license_id="vlad",
        license_rank=2,
        base_stats=MechFrameBaseStats(
            size="size_1",
            armor=2,
            hp=8,
            evasion=8,
            e_defense=8,
            speed=4,
            sensor_range=5,
            tech_attack=-1,
            heat_cap=6,
            repair_cap=4,
            save_target=10,
        ),
        mounts=[
            MountSlot(slot_type="flexible"),
            MountSlot(slot_type="main"),
            MountSlot(slot_type="heavy"),
        ],
        system_points=5,
        core_system=CoreSystemDefinition(
            id="ipsn_shrike_armor",
            name="Shrike Armor",
            effects=MechanicalEffect(
                resistances=[
                    Resistance(damage_type="all", condition="attacks_within_range_3")
                ],
                special="core_power_protocol_shrike_damage_to_attackers_3_ap_kinetic",
            ),
        ),
        traits=[
            FrameTrait(
                name="Dismemberment",
                effects=MechanicalEffect(
                    status_triggers=[
                        StatusTrigger(
                            trigger="on_inflict",
                            status="immobilized",
                            target="enemy",
                            effect=MechanicalEffect(
                                status_grants=[
                                    StatusGrant(
                                        status="shredded",
                                        target="enemy",
                                        duration="match_trigger",
                                    )
                                ],
                            ),
                        )
                    ],
                ),
            ),
            FrameTrait(
                name="Shrike Armor",
                effects=MechanicalEffect(
                    direct_damages=[
                        DirectDamage(
                            damage_type="kinetic",
                            flat=1,
                            ap=True,
                            target="enemy",
                            condition="attacker_within_range_3_before_attack",
                        )
                    ],
                ),
            ),
        ],
    ),
]


IPSN_WEAPONS: list[MechWeaponDefinition] = [
    MechWeaponDefinition(
        id="hand_cannon",
        name="Hand Cannon",
        size="aux",
        weapon_type="cqb",
        damage_type="kinetic",
        license_id="raleigh",
        license_rank=1,
        ranges=[
            WeaponRange(range_type="range", value=5),
            WeaponRange(range_type="threat", value=3),
        ],
        damage=[WeaponDamage(damage_type="kinetic", dice=DiceExpression.parse("1d6"))],
        tags=[WeaponTag(tag="loading"), WeaponTag(tag="reliable", value=1)],
    ),
    MechWeaponDefinition(
        id="bolt_thrower",
        name="Bolt Thrower",
        size="heavy",
        weapon_type="cannon",
        damage_type="kinetic",
        license_id="raleigh",
        license_rank=2,
        ranges=[WeaponRange(range_type="range", value=8)],
        damage=[
            WeaponDamage(damage_type="kinetic", dice=DiceExpression.parse("2d6")),
            WeaponDamage(damage_type="explosive", dice=DiceExpression.parse("1d6")),
        ],
        tags=[WeaponTag(tag="loading"), WeaponTag(tag="reliable", value=2)],
    ),
    MechWeaponDefinition(
        id="kinetic_hammer",
        name="Kinetic Hammer",
        size="heavy",
        weapon_type="melee",
        damage_type="kinetic",
        license_id="raleigh",
        license_rank=3,
        ranges=[WeaponRange(range_type="threat", value=1)],
        damage=[WeaponDamage(damage_type="kinetic", dice=DiceExpression.parse("2d6+2"))],
        tags=[WeaponTag(tag="reliable", value=4)],
    ),
    MechWeaponDefinition(
        id="ipsn_m35_mjolnir",
        name="IPS-N M35 Mjolnir Cannon",
        size="main",
        weapon_type="cqb",
        damage_type="kinetic",
        license_id="raleigh",
        license_rank=2,
        integrated_only=True,
        integrated_frame_id="ipsn_raleigh",
        unique=True,
        ranges=[
            WeaponRange(range_type="range", value=5),
            WeaponRange(range_type="threat", value=3),
        ],
        damage=[WeaponDamage(damage_type="kinetic", flat=4)],
        effects=MechanicalEffect(
            triggered_effects=[
                TriggeredEffect(
                    trigger="on_reload",
                    effect=MechanicalEffect(
                        action_grants=[
                            ActionGrant(
                                action_type="free",
                                name="mjolnir_free_attack",
                                uses_per="round",
                            )
                        ],
                    ),
                )
            ],
        ),
    ),
    MechWeaponDefinition(
        id="ipsn_chain_axe",
        name="Chain Axe",
        size="main",
        weapon_type="melee",
        damage_type="kinetic",
        license_id="blackbeard",
        license_rank=1,
        ranges=[WeaponRange(range_type="threat", value=1)],
        damage=[WeaponDamage(damage_type="kinetic", dice=DiceExpression.parse("1d6"))],
        tags=[WeaponTag(tag="reliable", value=2)],
        effects=MechanicalEffect(
            triggered_effects=[
                TriggeredEffect(
                    trigger="on_crit",
                    effect=MechanicalEffect(
                        status_grants=[
                            StatusGrant(
                                status="shredded",
                                target="enemy",
                                duration="end_of_turn",
                            )
                        ],
                    ),
                )
            ],
        ),
    ),
    MechWeaponDefinition(
        id="ipsn_flechette_launcher",
        name="Flechette Launcher",
        size="aux",
        weapon_type="cqb",
        damage_type="kinetic",
        license_id="blackbeard",
        license_rank=2,
        ranges=[WeaponRange(range_type="burst", value=1)],
        damage=[WeaponDamage(damage_type="kinetic", flat=1)],
        effects=MechanicalEffect(
            triggered_effects=[
                TriggeredEffect(
                    trigger="on_hit",
                    condition="target_grappled_or_biological",
                    effect=MechanicalEffect(
                        damage_mods=[DamageModifier(flat=2)],
                    ),
                )
            ],
            special="ignore_melee_engagement_penalty",
        ),
    ),
    MechWeaponDefinition(
        id="ipsn_nanocarbon_sword",
        name="Nanocarbon Sword",
        size="heavy",
        weapon_type="melee",
        damage_type="kinetic",
        license_id="blackbeard",
        license_rank=2,
        ranges=[WeaponRange(range_type="threat", value=2)],
        damage=[WeaponDamage(damage_type="kinetic", dice=DiceExpression.parse("1d6+4"))],
        tags=[WeaponTag(tag="reliable", value=3)],
    ),
    MechWeaponDefinition(
        id="ipsn_assault_cannon",
        name="Assault Cannon",
        size="main",
        weapon_type="cannon",
        damage_type="kinetic",
        license_id="drake",
        license_rank=1,
        ranges=[WeaponRange(range_type="range", value=8)],
        damage=[WeaponDamage(damage_type="kinetic", dice=DiceExpression.parse("1d6+2"))],
        tags=[WeaponTag(tag="heat_self", value=1), WeaponTag(tag="overkill")],
        effects=MechanicalEffect(
            special="quick_action_spin_up_gain_reliable_3_self_slowed_until_stop_free_action",
        ),
    ),
    MechWeaponDefinition(
        id="ipsn_concussion_missiles",
        name="Concussion Missiles",
        size="main",
        weapon_type="launcher",
        damage_type="explosive",
        license_id="drake",
        license_rank=2,
        ranges=[WeaponRange(range_type="range", value=5)],
        damage=[WeaponDamage(damage_type="explosive", dice=DiceExpression.parse("1d3"))],
        tags=[WeaponTag(tag="knockback", value=2)],
        effects=MechanicalEffect(
            save_checks=[
                SaveCheck(
                    trigger="on_hit",
                    save="hull",
                    target="enemy",
                    on_failure=MechanicalEffect(
                        status_grants=[
                            StatusGrant(
                                status="impaired",
                                target="enemy",
                                duration="start_of_next_turn",
                            )
                        ],
                    ),
                )
            ],
        ),
    ),
    MechWeaponDefinition(
        id="ipsn_leviathan_hac",
        name="Leviathan Heavy Assault Cannon",
        size="superheavy",
        weapon_type="cannon",
        damage_type="kinetic",
        license_id="drake",
        license_rank=3,
        ranges=[WeaponRange(range_type="range", value=8)],
        damage=[WeaponDamage(damage_type="kinetic", dice=DiceExpression.parse("1d6"))],
        effects=MechanicalEffect(
            special=(
                "can_skirmish_with_base_profile "
                "quick_action_spin_up_self_slowed_damage_4d6_plus_4 "
                "requires_barrage_when_spun_up "
                "gain_reliable_5_and_heat_self_2 "
                "stop_spin_up_free_action_start_turn"
            ),
        ),
    ),
    MechWeaponDefinition(
        id="ipsn_latch_drone",
        name="Latch Drone",
        size="main",
        weapon_type="launcher",
        damage_type=None,
        license_id="lancaster",
        license_rank=2,
        integrated_only=True,
        integrated_frame_id="ipsn_lancaster",
        unique=True,
        ranges=[WeaponRange(range_type="range", value=8)],
        effects=MechanicalEffect(
            triggered_effects=[
                TriggeredEffect(
                    trigger="on_hit",
                    effect=MechanicalEffect(
                        resource_changes=[
                            ResourceChange(
                                resource="hp",
                                amount="half_max",
                                target="ally",
                                cost_repairs=1,
                                cost_source="either",
                            )
                        ],
                    ),
                )
            ],
            special="ranged_attack_vs_evasion_8_target_friendly",
        ),
    ),
    MechWeaponDefinition(
        id="ipsn_plasma_cutter",
        name="Plasma Cutter",
        size="aux",
        weapon_type="melee",
        damage_type="energy",
        license_id="lancaster",
        license_rank=3,
        ranges=[WeaponRange(range_type="threat", value=1)],
        damage=[
            WeaponDamage(damage_type="energy", flat=1),
            WeaponDamage(damage_type="heat", flat=1),
        ],
        tags=[
            WeaponTag(tag="heat_self", value=1),
            WeaponTag(tag="burn", value=1),
        ],
        effects=MechanicalEffect(
            triggered_effects=[
                TriggeredEffect(
                    trigger="on_hit",
                    condition="target_is_object",
                    effect=MechanicalEffect(
                        special="object_damage_10_ap_energy",
                    ),
                )
            ],
        ),
    ),
    MechWeaponDefinition(
        id="ipsn_war_pike",
        name="War Pike",
        size="main",
        weapon_type="melee",
        damage_type="kinetic",
        license_id="nelson",
        license_rank=1,
        ranges=[
            WeaponRange(range_type="threat", value=3),
            WeaponRange(range_type="thrown", value=5),
        ],
        damage=[WeaponDamage(damage_type="kinetic", dice=DiceExpression.parse("1d6"))],
        tags=[WeaponTag(tag="knockback", value=1)],
    ),
    MechWeaponDefinition(
        id="ipsn_power_knuckles",
        name="Power Knuckles",
        size="aux",
        weapon_type="melee",
        damage_type="explosive",
        license_id="nelson",
        license_rank=3,
        ranges=[WeaponRange(range_type="threat", value=1)],
        damage=[WeaponDamage(damage_type="explosive", dice=DiceExpression.parse("1d3+1"))],
        effects=MechanicalEffect(
            save_checks=[
                SaveCheck(
                    trigger="on_crit",
                    save="hull",
                    target="enemy",
                    on_failure=MechanicalEffect(
                        status_grants=[
                            StatusGrant(
                                status="prone",
                                target="enemy",
                                duration="until_cleared",
                            )
                        ],
                    ),
                )
            ],
        ),
    ),
    MechWeaponDefinition(
        id="ipsn_automatic_shotgun",
        name="Automatic Shotgun",
        size="main",
        weapon_type="cqb",
        damage_type="kinetic",
        license_id="tortuga",
        license_rank=1,
        ranges=[
            WeaponRange(range_type="range", value=3),
            WeaponRange(range_type="threat", value=3),
        ],
        damage=[WeaponDamage(damage_type="kinetic", dice=DiceExpression.parse("2d6"))],
        tags=[WeaponTag(tag="inaccurate")],
    ),
    MechWeaponDefinition(
        id="ipsn_daisy_cutter",
        name="Daisy Cutter",
        size="heavy",
        weapon_type="cqb",
        damage_type="kinetic",
        license_id="tortuga",
        license_rank=2,
        ranges=[WeaponRange(range_type="cone", value=7)],
        damage=[WeaponDamage(damage_type="kinetic", dice=DiceExpression.parse("3d6"))],
        limited_uses=2,
        effects=MechanicalEffect(
            zones=[
                ZoneEffect(
                    shape="cone",
                    size=7,
                    duration="end_of_next_turn",
                    cover="soft",
                )
            ],
        ),
    ),
    MechWeaponDefinition(
        id="ipsn_catalytic_hammer",
        name="Catalytic Hammer",
        size="main",
        weapon_type="melee",
        damage_type="kinetic",
        license_id="tortuga",
        license_rank=2,
        ranges=[WeaponRange(range_type="threat", value=1)],
        damage=[WeaponDamage(damage_type="kinetic", dice=DiceExpression.parse("1d3+5"))],
        tags=[WeaponTag(tag="loading")],
        effects=MechanicalEffect(
            save_checks=[
                SaveCheck(
                    trigger="on_crit",
                    save="hull",
                    target="enemy",
                    on_failure=MechanicalEffect(
                        status_grants=[
                            StatusGrant(
                                status="stunned",
                                target="enemy",
                                duration="end_of_next_turn",
                            )
                        ],
                    ),
                )
            ],
        ),
    ),
    MechWeaponDefinition(
        id="ipsn_impact_lance",
        name="Impact Lance",
        size="main",
        weapon_type="melee",
        damage_type="energy",
        license_id="vlad",
        license_rank=1,
        ranges=[WeaponRange(range_type="threat", value=3)],
        damage=[WeaponDamage(damage_type="energy", dice=DiceExpression.parse("1d6"))],
        effects=MechanicalEffect(
            special="line_attack_between_target_and_self_heat_self_per_extra_target",
        ),
    ),
    MechWeaponDefinition(
        id="ipsn_nail_gun",
        name="Nail Gun",
        size="main",
        weapon_type="cqb",
        damage_type="kinetic",
        license_id="vlad",
        license_rank=2,
        ranges=[
            WeaponRange(range_type="range", value=5),
            WeaponRange(range_type="threat", value=3),
        ],
        damage=[WeaponDamage(damage_type="kinetic", dice=DiceExpression.parse("1d6+1"))],
        tags=[WeaponTag(tag="heat_self", value=1)],
        effects=MechanicalEffect(
            save_checks=[
                SaveCheck(
                    trigger="on_hit",
                    save="hull",
                    target="enemy",
                    on_failure=MechanicalEffect(
                        status_grants=[
                            StatusGrant(
                                status="immobilized",
                                target="enemy",
                                duration="end_of_next_turn",
                            )
                        ],
                    ),
                )
            ],
        ),
    ),
    MechWeaponDefinition(
        id="ipsn_combat_drill",
        name="Combat Drill",
        size="superheavy",
        weapon_type="melee",
        damage_type="kinetic",
        license_id="vlad",
        license_rank=3,
        ranges=[WeaponRange(range_type="threat", value=1)],
        damage=[
            WeaponDamage(damage_type="kinetic", dice=DiceExpression.parse("3d6")),
            WeaponDamage(damage_type="energy", dice=DiceExpression.parse("1d6")),
        ],
        tags=[WeaponTag(tag="overkill"), WeaponTag(tag="ap")],
        effects=MechanicalEffect(
            special="overkill_bonus_vs_prone_immobilized_or_stunned_targets",
        ),
    ),
]


IPSN_SYSTEMS: list[MechSystemDefinition] = [
    MechSystemDefinition(
        id="ipsn_breaching_charges",
        name="Breaching Charges",
        sp_cost=2,
        license_id="raleigh",
        license_rank=1,
        unique=True,
        limited_uses=3,
        tags=[SystemTag(tag="grenade"), SystemTag(tag="mine")],
        grenades=[
            GrenadePayload(
                name="Thermal Grenade",
                range=5,
                area=AreaEffect(
                    pattern="blast",
                    size=1,
                    damage=DamageSpec(
                        damage_type="energy",
                        dice=DiceExpression.parse("1d6"),
                    ),
                    save="agility",
                    half_on_success=True,
                    object_damage=DamageSpec(
                        damage_type="energy",
                        flat=10,
                        ap=True,
                    ),
                    objects_auto_hit=True,
                ),
            ),
        ],
        mines=[
            MinePayload(
                name="Breaching Charge",
                detonation="manual",
                detonation_action="quick",
                can_attach_to_terrain=True,
                area=AreaEffect(
                    pattern="burst",
                    size=1,
                    damage=DamageSpec(
                        damage_type="energy",
                        dice=DiceExpression.parse("2d6"),
                        ap=True,
                    ),
                    save="agility",
                    half_on_success=True,
                    object_damage=DamageSpec(
                        damage_type="explosive",
                        flat=30,
                        ap=True,
                    ),
                    objects_auto_hit=True,
                ),
            ),
        ],
    ),
    MechSystemDefinition(
        id="ipsn_roland_chamber",
        name="ROLAND Chamber",
        sp_cost=2,
        license_id="raleigh",
        license_rank=2,
        unique=True,
        tags=[SystemTag(tag="mod")],
        effects=MechanicalEffect(
            triggered_effects=[
                TriggeredEffect(
                    trigger="on_reload",
                    condition="next_loading_attack",
                    effect=MechanicalEffect(
                        damage_mods=[
                            DamageModifier(
                                dice=DiceExpression.parse("1d6"),
                                damage_type="explosive",
                            )
                        ],
                        save_checks=[
                            SaveCheck(
                                trigger="on_hit",
                                save="hull",
                                target="enemy",
                                on_failure=MechanicalEffect(
                                    status_grants=[
                                        StatusGrant(
                                            status="prone",
                                            target="enemy",
                                        )
                                    ],
                                ),
                            )
                        ],
                    ),
                )
            ],
        ),
    ),
    MechSystemDefinition(
        id="ipsn_uncle_cc",
        name="UNCLE-class C/C",
        sp_cost=3,
        license_id="raleigh",
        license_rank=3,
        unique=True,
        system_type="ai",
        tags=[SystemTag(tag="ai"), SystemTag(tag="mod")],
        effects=MechanicalEffect(
            special=(
                "choose_weapon_heavy_or_smaller_grant_ai_control "
                "1_per_turn_free_attack_plus_2_difficulty "
                "cannot_fire_if_used_this_turn "
                "after_uncle_attack_weapon_locked_until_next_turn "
                "if_unshackled_uncle_selects_target_or_no_attack"
            ),
        ),
    ),
    MechSystemDefinition(
        id="ipsn_synthetic_muscle_netting",
        name="Synthetic Muscle Netting",
        sp_cost=2,
        license_id="blackbeard",
        license_rank=1,
        unique=True,
        tags=[SystemTag(tag="mod")],
        effects=MechanicalEffect(
            special="ram_grapple_size_equal_or_larger_and_lift_drag_double",
        ),
    ),
    MechSystemDefinition(
        id="ipsn_reinforced_grapples",
        name="Reinforced Grapples",
        sp_cost=2,
        license_id="blackbeard",
        license_rank=3,
        effects=MechanicalEffect(
            special=(
                "grapple_movement_once_per_turn_move_fly_straight_line "
                "must_end_on_surface_or_fall_can_hold_surface_if_immobile "
                "falls_if_prone_or_knockback "
                "quick_action_drag_down_contested_hull_knock_prone"
            ),
        ),
    ),
    MechSystemDefinition(
        id="ipsn_sekhmet_nhp",
        name="SEKHMET-class NHP",
        sp_cost=3,
        license_id="blackbeard",
        license_rank=3,
        unique=True,
        system_type="ai",
        tags=[SystemTag(tag="ai"), SystemTag(tag="protocol")],
        effects=MechanicalEffect(
            special=(
                "sekhmet_protocol_melee_crits_bonus_1d6 "
                "free_melee_skirmish_each_turn "
                "lose_direct_control_chase_nearest_target_melee "
                "end_protocol_start_turn_stunned_until_next_turn"
            ),
        ),
    ),
    MechSystemDefinition(
        id="ipsn_argonaut_shield",
        name="IPS-N Argonaut Shield",
        sp_cost=2,
        license_id="drake",
        license_rank=1,
        tags=[SystemTag(tag="quick_action"), SystemTag(tag="shield")],
        effects=MechanicalEffect(
            special=(
                "quick_action_adjacent_ally_resistance_all_damage_share_half_damage "
                "breaks_on_separation_repeat_to_renew"
            ),
        ),
    ),
    MechSystemDefinition(
        id="ipsn_aegis_shield_generator",
        name="Aegis Shield Generator",
        sp_cost=2,
        license_id="drake",
        license_rank=2,
        unique=True,
        limited_uses=1,
        system_type="shield",
        tags=[SystemTag(tag="shield"), SystemTag(tag="deployable"), SystemTag(tag="quick_action")],
        deployable=DeployableEffect(
            obj=DeployableObject(
                size=1,
                evasion=5,
                hp=10,
            ),
        ),
        effects=MechanicalEffect(
            zones=[
                ZoneEffect(
                    shape="burst",
                    size=1,
                    placement="deployable",
                    duration="scene",
                    continuous_effects=MechanicalEffect(
                        damage_reductions=[
                            DamageReduction(amount=2, damage_type="all", target="all")
                        ],
                    ),
                    total_effect_cap=20,
                )
            ],
            special=(
                "generator_reduces_total_20_hp_then_deactivates "
                "ends_scene_or_destroyed"
            ),
        ),
    ),
    MechSystemDefinition(
        id="ipsn_portable_bunker",
        name="Portable Bunker",
        sp_cost=2,
        license_id="drake",
        license_rank=3,
        unique=True,
        limited_uses=1,
        system_type="deployable",
        tags=[SystemTag(tag="deployable"), SystemTag(tag="quick_action")],
        deployable=DeployableEffect(
            obj=DeployableObject(
                size=4,
                cover="hard",
                evasion=5,
                hp=40,
            ),
        ),
        effects=MechanicalEffect(
            zones=[
                ZoneEffect(
                    shape="square",
                    width=4,
                    height=4,
                    placement="deployable",
                    duration="scene",
                    cover="hard",
                    cover_all_directions=True,
                    continuous_effects=MechanicalEffect(
                        resistances=[
                            Resistance(damage_type="all", condition="blast_line_cone_from_outside")
                        ],
                    ),
                )
            ],
            special=(
                "deploy_adjacent_free_4x4_area_unfolds_start_next_turn "
                "open_topped_immobile_cannot_deactivate"
            ),
        ),
    ),
    MechSystemDefinition(
        id="ipsn_bulwark_mods",
        name="Bulwark Mods",
        sp_cost=1,
        license_id="nelson",
        license_rank=1,
        tags=[SystemTag(tag="mod")],
        effects=MechanicalEffect(
            special="ignore_difficult_terrain",
        ),
    ),
    MechSystemDefinition(
        id="ipsn_thermal_charge",
        name="Thermal Charge",
        sp_cost=2,
        license_id="nelson",
        license_rank=2,
        limited_uses=3,
        tags=[SystemTag(tag="mod")],
        effects=MechanicalEffect(
            triggered_effects=[
                TriggeredEffect(
                    trigger="on_hit",
                    condition="melee_weapon_spend_charge",
                    effect=MechanicalEffect(
                        damage_mods=[
                            DamageModifier(
                                dice=DiceExpression.parse("1d6"),
                                damage_type="explosive",
                            )
                        ],
                    ),
                )
            ],
        ),
    ),
    MechSystemDefinition(
        id="ipsn_armor_lock_system",
        name="Armor Lock System",
        sp_cost=1,
        license_id="nelson",
        license_rank=2,
        unique=True,
        tags=[SystemTag(tag="mod")],
        effects=MechanicalEffect(
            special=(
                "brace_reaction_cost_heat_2 "
                "attacks_against_you_plus_1_difficulty_until_end_next_turn "
                "immune_failed_agility_hull_saves_and_contested "
                "immune_knockback_grapple_prone_or_forced_move_by_smaller_than_size_5 "
                "end_all_grapples"
            ),
        ),
    ),
    MechSystemDefinition(
        id="ipsn_ramjet",
        name="RAMJET",
        sp_cost=3,
        license_id="nelson",
        license_rank=3,
        unique=True,
        tags=[SystemTag(tag="protocol")],
        effects=MechanicalEffect(
            special=(
                "protocol_heat_self_2_until_start_next_turn "
                "boost_speed_plus_2_melee_knockback_plus_2 "
                "must_move_max_speed_each_move_straight_line_only"
            ),
        ),
    ),
    MechSystemDefinition(
        id="ipsn_siege_ram",
        name="Siege Ram",
        sp_cost=2,
        license_id="tortuga",
        license_rank=1,
        unique=True,
        tags=[SystemTag(tag="mod")],
        effects=MechanicalEffect(
            damage_mods=[
                DamageModifier(
                    dice=DiceExpression.parse("1d3"),
                    damage_type="kinetic",
                    condition="ram_attack",
                )
            ],
            special="ram_attacks_vs_objects_deal_10_ap_kinetic",
        ),
    ),
    MechSystemDefinition(
        id="ipsn_throughbolt_rounds",
        name="Throughbolt Rounds",
        sp_cost=2,
        license_id="tortuga",
        license_rank=3,
        unique=True,
        tags=[SystemTag(tag="mod")],
        effects=MechanicalEffect(
            special=(
                "weapon_mod_cqb_cannon_rifle "
                "project_line_3_from_self_attack_origin_at_line_end "
                "line_deals_1d3_ap_kinetic_to_objects_or_characters"
            ),
        ),
    ),
    MechSystemDefinition(
        id="ipsn_hyper_dense_armor",
        name="Hyper Dense Armor",
        sp_cost=3,
        license_id="tortuga",
        license_rank=3,
        unique=True,
        system_type="shield",
        tags=[SystemTag(tag="shield"), SystemTag(tag="quick_action")],
        effects=MechanicalEffect(
            special=(
                "quick_action_resistance_all_damage_heat_burn_from_range_gt_5 "
                "self_slowed_and_deal_half_damage_heat_burn_to_targets_range_gt_5 "
                "deactivate_quick_action"
            ),
        ),
    ),
    MechSystemDefinition(
        id="ipsn_snare_trap",
        name="Snare Trap",
        sp_cost=1,
        license_id="vlad",
        license_rank=1,
        unique=True,
        limited_uses=2,
        system_type="deployable",
        tags=[SystemTag(tag="deployable"), SystemTag(tag="quick_action")],
        deployable=DeployableEffect(
            obj=DeployableObject(
                size=1,
                evasion=5,
                hp=10,
            ),
        ),
        effects=MechanicalEffect(
            deployments=[
                DeploymentEffect(
                    action_type="quick",
                    placement_range=1,
                    placement_relation="adjacent",
                    primes_after="turn_end",
                    activation_condition="pass_over",
                    activation_action=None,
                    activation_target="enemy",
                    activation_effect=MechanicalEffect(
                        save_checks=[
                            SaveCheck(
                                trigger="on_activation",
                                save="hull",
                                target="enemy",
                                on_failure=MechanicalEffect(
                                    direct_damages=[
                                        DirectDamage(
                                            damage_type="kinetic",
                                            dice=DiceExpression.parse("1d6"),
                                            ap=True,
                                            target="enemy",
                                        )
                                    ],
                                    status_grants=[
                                        StatusGrant(
                                            status="immobilized",
                                            target="enemy",
                                            duration="until_cleared",
                                        )
                                    ],
                                ),
                            )
                        ],
                    ),
                    consumes_on_activation=True,
                )
            ],
            special=(
                "trap_becomes_object_on_trigger "
                "immobilize_persists_while_trap_intact"
            ),
        ),
    ),
    MechSystemDefinition(
        id="ipsn_caltrop_launcher",
        name="Caltrop Launcher",
        sp_cost=1,
        license_id="vlad",
        license_rank=2,
        unique=True,
        tags=[SystemTag(tag="quick_action")],
        effects=MechanicalEffect(
            zones=[
                ZoneEffect(
                    shape="blast",
                    size=1,
                    placement_range=5,
                    duration="scene",
                    difficult_terrain=True,
                    effects_on_enter=MechanicalEffect(
                        direct_damages=[
                            DirectDamage(
                                damage_type="explosive",
                                dice=DiceExpression.parse("1d3"),
                                ap=True,
                                target="enemy",
                            )
                        ],
                    ),
                    effects_on_start_turn=MechanicalEffect(
                        direct_damages=[
                            DirectDamage(
                                damage_type="explosive",
                                dice=DiceExpression.parse("1d3"),
                                ap=True,
                                target="enemy",
                            )
                        ],
                    ),
                )
            ],
        ),
    ),
    MechSystemDefinition(
        id="ipsn_charged_stake",
        name="Charged Stake",
        sp_cost=2,
        license_id="vlad",
        license_rank=3,
        tags=[SystemTag(tag="full_action")],
        effects=MechanicalEffect(
            save_checks=[
                SaveCheck(
                    trigger="on_activation",
                    save="hull",
                    target="enemy",
                    on_failure=MechanicalEffect(
                        direct_damages=[
                            DirectDamage(
                                damage_type="energy",
                                dice=DiceExpression.parse("1d6"),
                                ap=True,
                                target="enemy",
                            )
                        ],
                        status_grants=[
                            StatusGrant(
                                status="immobilized",
                                target="enemy",
                                duration="until_cleared",
                            )
                        ],
                    ),
                )
            ],
            special=(
                "adjacent_target_repeat_hull_save_end_turn_or_take_5_ap_energy "
                "only_one_target_at_time_quick_action_recover_from_adjacent"
            ),
        ),
    ),
    MechSystemDefinition(
        id="ipsn_restock_drone",
        name="Restock Drone",
        sp_cost=2,
        license_id="lancaster",
        license_rank=1,
        unique=True,
        limited_uses=2,
        system_type="drone",
        tags=[SystemTag(tag="drone"), SystemTag(tag="quick_action")],
        drone=DronePayload(
            name="Restock Drone",
            size="size_half",
            hp=10,
            evasion=10,
            e_defense=10,
        ),
        effects=MechanicalEffect(
            deployments=[
                DeploymentEffect(
                    action_type="quick",
                    placement_range=1,
                    placement_relation="adjacent",
                    primes_after="turn_end",
                    activation_condition="adjacent_start_or_move",
                    activation_action="quick",
                    activation_target="ally",
                    activation_effect=MechanicalEffect(
                        resource_changes=[
                            ResourceChange(
                                resource="heat",
                                amount=DiceExpression.parse("1d6"),
                                direction="lose",
                                target="ally",
                            )
                        ],
                        status_clears=[
                            StatusClear(
                                status="any",
                                target="ally",
                            )
                        ],
                        reloads=[
                            ReloadEffect(
                                target="ally",
                                count=1,
                                requires_tag="loading",
                                consumes_source=True,
                            )
                        ],
                    ),
                    consumes_on_activation=True,
                )
            ],
        ),
    ),
    MechSystemDefinition(
        id="ipsn_cable_winch",
        name="Cable Winch System",
        sp_cost=1,
        license_id="lancaster",
        license_rank=1,
        tags=[SystemTag(tag="quick_action")],
        effects=MechanicalEffect(
            triggered_effects=[
                TriggeredEffect(
                    trigger="on_activation",
                    condition="willing_or_stunned_adjacent_target",
                    effect=MechanicalEffect(
                        tethers=[
                            TetherEffect(
                                action_type="quick",
                                range=1,
                                max_distance=5,
                                tow_slowed=True,
                                auto_attach_if_willing=True,
                                auto_attach_if_stunned=True,
                                detach_on_hit=True,
                                detach_attack_evasion=10,
                                can_attach_to_objects=True,
                                object_attach_range=5,
                                object_strain_capacity=6,
                                climb_no_speed_penalty=True,
                            )
                        ],
                    ),
                )
            ],
            save_checks=[
                SaveCheck(
                    trigger="on_activation",
                    condition="unwilling_adjacent_target",
                    save="hull",
                    target="enemy",
                    on_failure=MechanicalEffect(
                        tethers=[
                            TetherEffect(
                                action_type="quick",
                                range=1,
                                max_distance=5,
                                tow_slowed=True,
                                auto_attach_if_willing=True,
                                auto_attach_if_stunned=True,
                                detach_on_hit=True,
                                detach_attack_evasion=10,
                                can_attach_to_objects=True,
                                object_attach_range=5,
                                object_strain_capacity=6,
                                climb_no_speed_penalty=True,
                            )
                        ],
                    ),
                )
            ],
        ),
    ),
    MechSystemDefinition(
        id="ipsn_mule_harness",
        name="MULE Harness",
        sp_cost=2,
        license_id="lancaster",
        license_rank=2,
        unique=True,
        effects=MechanicalEffect(
            special=(
                "carry_allies_total_size_half_less_than_self "
                "adjacent_allies_quick_action_mount_soft_cover "
                "dismount_on_prone_stunned_destroyed_or_immobilized"
            ),
        ),
    ),
    MechSystemDefinition(
        id="ipsn_sealant_spray",
        name="Sealant Spray",
        sp_cost=2,
        license_id="lancaster",
        license_rank=2,
        tags=[SystemTag(tag="quick_action")],
        effects=MechanicalEffect(
            save_checks=[
                SaveCheck(
                    trigger="on_activation",
                    condition="hostile_target_range_5",
                    save="agility",
                    target="enemy",
                    on_failure=MechanicalEffect(
                        status_grants=[
                            StatusGrant(
                                status="slowed",
                                target="enemy",
                                duration="start_of_next_turn",
                            )
                        ],
                        status_clears=[
                            StatusClear(status="burn", target="enemy"),
                        ],
                    ),
                )
            ],
            triggered_effects=[
                TriggeredEffect(
                    trigger="on_activation",
                    condition="ally_target_range_5",
                    effect=MechanicalEffect(
                        status_grants=[
                            StatusGrant(
                                status="slowed",
                                target="ally",
                                duration="start_of_next_turn",
                            )
                        ],
                        status_clears=[
                            StatusClear(status="burn", target="ally"),
                        ],
                    ),
                )
            ],
            special="target_empty_space_blast_1_difficult_terrain_scene_put_out_fires",
        ),
    ),
    MechSystemDefinition(
        id="ipsn_aceso_stabilizer",
        name="Aceso Stabilizer",
        sp_cost=3,
        license_id="lancaster",
        license_rank=3,
        unique=True,
        limited_uses=3,
        system_type="shield",
        tags=[SystemTag(tag="shield"), SystemTag(tag="quick_action")],
        effects=MechanicalEffect(
            immunities=[
                Immunity(target="impaired", condition="stabilizer_attached"),
                Immunity(target="jammed", condition="stabilizer_attached"),
            ],
            damage_absorptions=[
                DamageAbsorption(
                    target="ally",
                    base_hp=4,
                    bonus_hp_per_grit=1,
                    max_instances_per_target=1,
                    spillover=True,
                    ends_on_zero=True,
                )
            ],
            special="attach_ally_range_5",
        ),
    ),
]


SSC_FRAMES: list[MechFrameDefinition] = [
    MechFrameDefinition(
        id="ssc_black_witch",
        name="SSC Black Witch",
        manufacturer="SSC",
        license_id="black_witch",
        license_rank=2,
        base_stats=MechFrameBaseStats(
            size="size_1",
            armor=1,
            hp=6,
            evasion=10,
            e_defense=12,
            speed=5,
            sensor_range=15,
            tech_attack=1,
            heat_cap=6,
            repair_cap=3,
            save_target=10,
        ),
        mounts=[
            MountSlot(slot_type="main_aux"),
        ],
        system_points=8,
        core_system=CoreSystemDefinition(
            id="ssc_mag_projector",
            name="Mag Projector",
            effects=MechanicalEffect(
                special=(
                    "core_power_full_action_blast_4_mag_field "
                    "blocks_ranged_kinetic_explosive_through_field "
                    "difficult_terrain "
                    "metal_targets_hull_save_or_pull_to_center_and_immobilized "
                    "detonation_end_next_turn_resolve_saved_attacks"
                ),
            ),
        ),
        traits=[
            FrameTrait(
                name="Repulsor Field",
                effects=MechanicalEffect(
                    resistances=[Resistance(damage_type="kinetic")],
                ),
            ),
            FrameTrait(
                name="Mag Parry",
                effects=MechanicalEffect(
                    special="reaction_parry_kinetic_attack_1_per_round_roll_5_plus_miss",
                ),
            ),
        ],
    ),
    MechFrameDefinition(
        id="ssc_deaths_head",
        name="SSC Death's Head",
        manufacturer="SSC",
        license_id="deaths_head",
        license_rank=2,
        base_stats=MechFrameBaseStats(
            size="size_1",
            armor=0,
            hp=8,
            evasion=10,
            e_defense=8,
            speed=5,
            sensor_range=20,
            tech_attack=0,
            heat_cap=6,
            repair_cap=2,
            save_target=10,
        ),
        mounts=[
            MountSlot(slot_type="main_aux"),
            MountSlot(slot_type="heavy"),
        ],
        system_points=6,
        core_system=CoreSystemDefinition(
            id="ssc_precognitive_targeting",
            name="Precognitive Targeting",
            effects=MechanicalEffect(
                special=(
                    "core_power_protocol_mark_for_death_full_action_range_30_min_range_5 "
                    "immobilized_no_reactions_while_concentrating "
                    "ranged_crits_vs_marked_target_deal_bonus_3d6_if_no_cover"
                ),
            ),
        ),
        traits=[
            FrameTrait(
                name="Neuro-Linked",
                effects=MechanicalEffect(
                    special="reroll_first_ranged_attack_per_round_keep_second",
                ),
            ),
            FrameTrait(
                name="Perfected Targeting",
                effects=MechanicalEffect(
                    accuracy_mods=[
                        AccuracyModifier(value=1, condition="ranged_attack")
                    ],
                ),
            ),
        ],
    ),
    MechFrameDefinition(
        id="ssc_dusk_wing",
        name="SSC Dusk Wing",
        manufacturer="SSC",
        license_id="dusk_wing",
        license_rank=2,
        base_stats=MechFrameBaseStats(
            size="size_half",
            armor=0,
            hp=6,
            evasion=12,
            e_defense=8,
            speed=6,
            sensor_range=10,
            tech_attack=1,
            heat_cap=4,
            repair_cap=3,
            save_target=10,
        ),
        mounts=[
            MountSlot(slot_type="aux_aux"),
            MountSlot(slot_type="flexible"),
        ],
        system_points=6,
        core_system=CoreSystemDefinition(
            id="ssc_hall_of_mirrors",
            name="Hall of Mirrors",
            effects=MechanicalEffect(
                hologram_trails=[
                    HologramTrailEffect(
                        trigger="move_or_boost",
                        detonation_triggers=[
                            "start_turn",
                            "move_through",
                            "move_adjacent",
                        ],
                        detonation_damage=DiceExpression.parse("1d6"),
                        detonation_damage_type="energy",
                        detonation_save="agility",
                        teleport_action="quick",
                        teleport_range=50,
                        detonate_all_burst=1,
                        suppress_new_until="start_of_next_turn",
                    )
                ],
            ),
        ),
        traits=[
            FrameTrait(
                name="Integrated Hover Flight",
                effects=MechanicalEffect(
                    special="hover_flight_on_move_or_boost",
                ),
            ),
            FrameTrait(
                name="Harlequin Cloak",
                effects=MechanicalEffect(
                    triggered_effects=[
                        TriggeredEffect(
                            trigger="on_turn_start",
                            effect=MechanicalEffect(
                                status_grants=[
                                    StatusGrant(
                                        status="invisible",
                                        target="self",
                                        duration="end_of_turn",
                                    )
                                ],
                            ),
                        )
                    ],
                ),
            ),
            FrameTrait(
                name="Fragile",
                effects=MechanicalEffect(
                    accuracy_mods=[
                        AccuracyModifier(value=-1, condition="hull_checks_and_saves")
                    ],
                ),
            ),
        ],
    ),
    MechFrameDefinition(
        id="ssc_metalmark",
        name="SSC Metalmark",
        manufacturer="SSC",
        license_id="metalmark",
        license_rank=2,
        base_stats=MechFrameBaseStats(
            size="size_1",
            armor=1,
            hp=8,
            evasion=10,
            e_defense=6,
            speed=5,
            sensor_range=10,
            tech_attack=0,
            heat_cap=5,
            repair_cap=4,
            save_target=10,
        ),
        mounts=[
            MountSlot(slot_type="aux_aux"),
            MountSlot(slot_type="main"),
            MountSlot(slot_type="heavy"),
        ],
        system_points=5,
        core_system=CoreSystemDefinition(
            id="ssc_tactical_cloak",
            name="Tactical Cloak",
            effects=MechanicalEffect(
                triggered_effects=[
                    TriggeredEffect(
                        trigger="on_activation",
                        effect=MechanicalEffect(
                            status_grants=[
                                StatusGrant(
                                    status="invisible",
                                    target="self",
                                    duration="scene",
                                )
                            ],
                        ),
                    )
                ],
            ),
        ),
        traits=[
            FrameTrait(
                name="Flash Cloak",
                effects=MechanicalEffect(
                    movement_scoped_statuses=[
                        MovementScopedStatus(
                            status="invisible",
                            target="self",
                            movement_modes=["any"],
                        )
                    ],
                ),
            ),
            FrameTrait(
                name="Carapace Adaptation",
                effects=MechanicalEffect(
                    accuracy_mods=[
                        AccuracyModifier(
                            value=-1,
                            condition="ranged_attacks_against_self_in_soft_cover",
                        )
                    ],
                ),
            ),
        ],
    ),
    MechFrameDefinition(
        id="ssc_monarch",
        name="SSC Monarch",
        manufacturer="SSC",
        license_id="monarch",
        license_rank=2,
        base_stats=MechFrameBaseStats(
            size="size_2",
            armor=1,
            hp=8,
            evasion=8,
            e_defense=8,
            speed=5,
            sensor_range=15,
            tech_attack=1,
            heat_cap=6,
            repair_cap=3,
            save_target=10,
        ),
        mounts=[
            MountSlot(slot_type="flexible"),
            MountSlot(slot_type="main"),
            MountSlot(slot_type="heavy"),
        ],
        system_points=5,
        core_system=CoreSystemDefinition(
            id="ssc_avenger_silos",
            name="Avenger Silos",
            effects=MechanicalEffect(
                special="core_power_full_action_divine_punishment_burst_50_agility_half_1d6_plus_4_explosive_no_los",
            ),
        ),
        traits=[
            FrameTrait(
                name="Avenger Silos",
                effects=MechanicalEffect(
                    triggered_effects=[
                        TriggeredEffect(
                            trigger="on_crit",
                            condition="ranged_weapon_other_target_range_15_los",
                            uses_per="round",
                            effect=MechanicalEffect(
                                direct_damages=[
                                    DirectDamage(
                                        damage_type="explosive",
                                        flat=3,
                                        target="enemy",
                                    )
                                ],
                            ),
                        )
                    ],
                ),
            ),
            FrameTrait(
                name="Seeking Payload",
                effects=MechanicalEffect(
                    triggered_effects=[
                        TriggeredEffect(
                            trigger="on_activation",
                            condition="launcher_attack_consumes_lock_on",
                            effect=MechanicalEffect(
                                special="gain_seeking_and_damage_unreducible",
                            ),
                        )
                    ],
                ),
            ),
        ],
    ),
    MechFrameDefinition(
        id="ssc_mourning_cloak",
        name="SSC Mourning Cloak",
        manufacturer="SSC",
        license_id="mourning_cloak",
        license_rank=2,
        base_stats=MechFrameBaseStats(
            size="size_1",
            armor=0,
            hp=8,
            evasion=12,
            e_defense=6,
            speed=5,
            sensor_range=15,
            tech_attack=0,
            heat_cap=4,
            repair_cap=3,
            save_target=10,
        ),
        mounts=[
            MountSlot(slot_type="main_aux"),
            MountSlot(slot_type="flexible"),
        ],
        system_points=6,
        core_system=CoreSystemDefinition(
            id="ssc_ex_slipstream_module",
            name="EX Slipstream Module",
            effects=MechanicalEffect(
                special=(
                    "full_action_teleport_3d6_fail_if_occupied_triples_vanish "
                    "core_power_protocol_move_or_boost_teleport_same_distance_scene"
                ),
            ),
        ),
        traits=[
            FrameTrait(
                name="Hunter",
                effects=MechanicalEffect(
                    damage_mods=[
                        DamageModifier(
                            dice=DiceExpression.parse("1d6"),
                            condition="melee_attack_no_other_adjacent",
                        )
                    ],
                ),
            ),
            FrameTrait(
                name="Biotic Components",
                effects=MechanicalEffect(
                    accuracy_mods=[
                        AccuracyModifier(value=1, condition="agility_checks_and_saves")
                    ],
                ),
            ),
        ],
    ),
    MechFrameDefinition(
        id="ssc_swallowtail",
        name="SSC Swallowtail",
        manufacturer="SSC",
        license_id="swallowtail",
        license_rank=2,
        base_stats=MechFrameBaseStats(
            size="size_1",
            armor=0,
            hp=6,
            evasion=10,
            e_defense=10,
            speed=6,
            sensor_range=20,
            tech_attack=1,
            heat_cap=4,
            repair_cap=5,
            save_target=10,
        ),
        mounts=[
            MountSlot(slot_type="aux_aux"),
            MountSlot(slot_type="flexible"),
        ],
        system_points=6,
        core_system=CoreSystemDefinition(
            id="ssc_cloudscout_tacsim_swarms",
            name="Cloudscout TACSIM Swarms",
            effects=MechanicalEffect(
                random_checks=[
                    RandomCheckEffect(
                        trigger="on_ally_damaged",
                        roll=DiceExpression.parse("1d6"),
                        success_threshold=4,
                        target="ally",
                        uses_per="round",
                        condition="reaction_visible_source_and_target",
                        on_success=MechanicalEffect(
                            resistances=[
                                Resistance(
                                    damage_type="all",
                                    target="ally",
                                    condition="from_triggering_attack",
                                )
                            ],
                            movement_grants=[
                                MovementGrant(
                                    spaces=3,
                                    movement_type="teleport",
                                    target="ally",
                                )
                            ],
                        ),
                        on_failure=MechanicalEffect(
                            movement_grants=[
                                MovementGrant(
                                    spaces=6,
                                    movement_type="teleport",
                                    target="ally",
                                )
                            ],
                        ),
                    )
                ],
            ),
        ),
        traits=[
            FrameTrait(
                name="Integrated Cloak",
                effects=MechanicalEffect(
                    triggered_effects=[
                        TriggeredEffect(
                            trigger="on_turn_end",
                            condition="no_move",
                            effect=MechanicalEffect(
                                status_grants=[
                                    StatusGrant(
                                        status="invisible",
                                        target="self",
                                        duration="start_of_next_turn",
                                    )
                                ],
                            ),
                        )
                    ],
                    status_breaks=[
                        StatusBreakCondition(
                            status="invisible",
                            target="self",
                            break_triggers=[
                                "move",
                                "reaction",
                                "turn_start",
                            ],
                        )
                    ],
                ),
            ),
            FrameTrait(
                name="Prophetic Scanners",
                effects=MechanicalEffect(
                    status_triggers=[
                        StatusTrigger(
                            trigger="on_inflict",
                            status="lock_on",
                            target="enemy",
                            uses_per="round",
                            effect=MechanicalEffect(
                                status_grants=[
                                    StatusGrant(
                                        status="shredded",
                                        target="enemy",
                                        duration="start_of_next_turn",
                                    )
                                ],
                            ),
                        )
                    ],
                ),
            ),
        ],
    ),
]


SSC_WEAPONS: list[MechWeaponDefinition] = [
    MechWeaponDefinition(
        id="ssc_mag_cannon",
        name="Mag Cannon",
        size="main",
        weapon_type="cannon",
        damage_type="energy",
        license_id="black_witch",
        license_rank=1,
        ranges=[WeaponRange(range_type="line", value=8)],
        damage=[WeaponDamage(damage_type="energy", dice=DiceExpression.parse("1d3+1"))],
        effects=MechanicalEffect(
            save_checks=[
                SaveCheck(
                    trigger="on_hit",
                    condition="line_targets",
                    save="hull",
                    target="enemy",
                    on_failure=MechanicalEffect(
                        forced_movements=[
                            ForcedMovement(
                                direction="pull",
                                distance=DiceExpression.parse("1d3+1"),
                                target="enemy",
                            )
                        ],
                    ),
                )
            ],
        ),
    ),
    MechWeaponDefinition(
        id="ssc_vulture_dmr",
        name="Vulture DMR",
        size="main",
        weapon_type="rifle",
        damage_type="kinetic",
        license_id="deaths_head",
        license_rank=2,
        ranges=[WeaponRange(range_type="range", value=15)],
        damage=[WeaponDamage(damage_type="kinetic", dice=DiceExpression.parse("1d6+1"))],
        tags=[
            WeaponTag(tag="heat_self", value=1),
            WeaponTag(tag="overkill"),
            WeaponTag(tag="accurate"),
        ],
    ),
    MechWeaponDefinition(
        id="ssc_railgun",
        name="Railgun",
        size="heavy",
        weapon_type="rifle",
        damage_type="kinetic",
        license_id="deaths_head",
        license_rank=3,
        ranges=[WeaponRange(range_type="line", value=20)],
        damage=[WeaponDamage(damage_type="kinetic", dice=DiceExpression.parse("1d6+4"))],
        tags=[
            WeaponTag(tag="heat_self", value=2),
            WeaponTag(tag="ap"),
            WeaponTag(tag="ordnance"),
        ],
    ),
    MechWeaponDefinition(
        id="ssc_veil_rifle",
        name="Veil Rifle",
        size="main",
        weapon_type="rifle",
        damage_type="energy",
        license_id="dusk_wing",
        license_rank=1,
        ranges=[WeaponRange(range_type="line", value=10)],
        damage=[WeaponDamage(damage_type="energy", dice=DiceExpression.parse("1d3+1"))],
        tags=[WeaponTag(tag="accurate")],
        effects=MechanicalEffect(
            cover_grants=[
                CoverGrant(
                    cover="soft",
                    target="ally",
                    duration="start_of_next_turn",
                    condition="in_line_of_attack",
                )
            ],
        ),
    ),
    MechWeaponDefinition(
        id="ssc_burst_launcher",
        name="Burst Launcher",
        size="main",
        weapon_type="launcher",
        damage_type="explosive",
        license_id="dusk_wing",
        license_rank=2,
        ranges=[WeaponRange(range_type="range", value=15)],
        damage=[
            WeaponDamage(damage_type="explosive", dice=DiceExpression.parse("1d3")),
            WeaponDamage(damage_type="heat", flat=1),
        ],
        tags=[WeaponTag(tag="arcing"), WeaponTag(tag="accurate")],
        effects=MechanicalEffect(
            triggered_effects=[
                TriggeredEffect(
                    trigger="on_crit",
                    effect=MechanicalEffect(
                        status_grants=[
                            StatusGrant(
                                status="impaired",
                                target="enemy",
                                duration="start_of_next_turn",
                            )
                        ],
                    ),
                )
            ],
        ),
    ),
    MechWeaponDefinition(
        id="ssc_rail_rifle",
        name="Rail Rifle",
        size="main",
        weapon_type="rifle",
        damage_type="kinetic",
        license_id="metalmark",
        license_rank=2,
        ranges=[WeaponRange(range_type="line", value=10)],
        damage=[WeaponDamage(damage_type="kinetic", dice=DiceExpression.parse("1d6+1"))],
        tags=[WeaponTag(tag="heat_self", value=1)],
    ),
    MechWeaponDefinition(
        id="ssc_shock_knife",
        name="Shock Knife",
        size="aux",
        weapon_type="melee",
        damage_type="energy",
        license_id="metalmark",
        license_rank=2,
        ranges=[
            WeaponRange(range_type="threat", value=1),
            WeaponRange(range_type="thrown", value=5),
        ],
        damage=[WeaponDamage(damage_type="energy", flat=1)],
        tags=[
            WeaponTag(tag="heat_self", value=1),
            WeaponTag(tag="burn", value=2),
        ],
    ),
    MechWeaponDefinition(
        id="ssc_sharanga_missiles",
        name="Sharanga Missiles",
        size="main",
        weapon_type="launcher",
        damage_type="explosive",
        license_id="monarch",
        license_rank=1,
        ranges=[WeaponRange(range_type="range", value=15)],
        damage=[WeaponDamage(damage_type="explosive", flat=3)],
        tags=[WeaponTag(tag="arcing")],
        effects=MechanicalEffect(
            targetings=[
                AttackTargetingEffect(
                    target_count_options=[1, 2],
                    separate_attack_rolls=True,
                    require_distinct_targets=True,
                )
            ],
        ),
    ),
    MechWeaponDefinition(
        id="ssc_gandiva_missiles",
        name="Gandiva Missiles",
        size="heavy",
        weapon_type="launcher",
        damage_type="energy",
        license_id="monarch",
        license_rank=2,
        ranges=[WeaponRange(range_type="range", value=15)],
        damage=[WeaponDamage(damage_type="energy", dice=DiceExpression.parse("1d6+3"))],
        tags=[
            WeaponTag(tag="smart"),
            WeaponTag(tag="seeking"),
            WeaponTag(tag="accurate"),
        ],
    ),
    MechWeaponDefinition(
        id="ssc_pinaka_missiles",
        name="Pinaka Missiles",
        size="superheavy",
        weapon_type="launcher",
        damage_type="explosive",
        license_id="monarch",
        license_rank=3,
        ranges=[
            WeaponRange(range_type="range", value=20),
            WeaponRange(range_type="blast", value=1),
        ],
        damage=[WeaponDamage(damage_type="explosive", dice=DiceExpression.parse("2d6"))],
        tags=[
            WeaponTag(tag="arcing"),
            WeaponTag(tag="heat_self", value=2),
        ],
        effects=MechanicalEffect(
            area_attack_patterns=[
                AreaAttackPattern(
                    area_shape="blast",
                    area_size=1,
                    area_count_options=[1, 2],
                    non_overlapping=True,
                )
            ],
            delayed_impacts=[
                DelayedImpactEffect(
                    delay_timing="end_of_next_round",
                    delayed_damage=DiceExpression.parse("3d6"),
                    delayed_damage_type="explosive",
                    self_slow_duration="end_of_next_turn",
                    reveal_area=True,
                    reveal_audience="all",
                )
            ],
        ),
    ),
    MechWeaponDefinition(
        id="ssc_vijaya_rockets",
        name="Vijaya Rockets",
        size="aux",
        weapon_type="launcher",
        damage_type="explosive",
        license_id="mourning_cloak",
        license_rank=1,
        ranges=[WeaponRange(range_type="range", value=5)],
        damage=[WeaponDamage(damage_type="explosive", dice=DiceExpression.parse("1d3"))],
        tags=[WeaponTag(tag="accurate")],
    ),
    MechWeaponDefinition(
        id="ssc_fold_knife",
        name="Fold Knife",
        size="aux",
        weapon_type="melee",
        damage_type="kinetic",
        license_id="mourning_cloak",
        license_rank=1,
        ranges=[WeaponRange(range_type="threat", value=1)],
        damage=[WeaponDamage(damage_type="kinetic", dice=DiceExpression.parse("1d3"))],
        tags=[WeaponTag(tag="accurate")],
        effects=MechanicalEffect(
            triggered_effects=[
                TriggeredEffect(
                    trigger="on_crit",
                    effect=MechanicalEffect(
                        movement_grants=[
                            MovementGrant(spaces=2, movement_type="teleport")
                        ],
                    ),
                )
            ],
        ),
    ),
    MechWeaponDefinition(
        id="ssc_variable_sword",
        name="Variable Sword",
        size="main",
        weapon_type="melee",
        damage_type="kinetic",
        license_id="mourning_cloak",
        license_rank=3,
        ranges=[WeaponRange(range_type="threat", value=2)],
        damage=[WeaponDamage(damage_type="kinetic", flat=3)],
        tags=[WeaponTag(tag="accurate")],
        effects=MechanicalEffect(
            triggered_effects=[
                TriggeredEffect(
                    trigger="on_crit",
                    effect=MechanicalEffect(
                        damage_mods=[
                            DamageModifier(
                                dice=DiceExpression.parse("1d6"),
                                damage_type="kinetic",
                            )
                        ],
                    ),
                )
            ],
        ),
    ),
    MechWeaponDefinition(
        id="ssc_oracle_lmg_i",
        name="Oracle LMG-I",
        size="aux",
        weapon_type="rifle",
        damage_type="kinetic",
        license_id="swallowtail",
        license_rank=2,
        ranges=[WeaponRange(range_type="range", value=15)],
        damage=[WeaponDamage(damage_type="kinetic", dice=DiceExpression.parse("1d3"))],
        tags=[
            WeaponTag(tag="arcing"),
            WeaponTag(tag="accurate"),
        ],
    ),
]


SSC_SYSTEMS: list[MechSystemDefinition] = [
    MechSystemDefinition(
        id="ssc_ferrous_lash",
        name="Ferrous Lash",
        sp_cost=2,
        license_id="black_witch",
        license_rank=1,
        tags=[SystemTag(tag="quick_action")],
        effects=MechanicalEffect(
            save_checks=[
                SaveCheck(
                    trigger="on_activation",
                    save="agility",
                    target="enemy",
                    on_failure=MechanicalEffect(
                        forced_movements=[
                            ForcedMovement(
                                direction="pull",
                                distance=5,
                                target="enemy",
                                ignores_engagement=True,
                                provokes_reactions=False,
                                on_collision=MechanicalEffect(
                                    status_grants=[
                                        StatusGrant(
                                            status="prone",
                                            target="enemy",
                                            duration="until_cleared",
                                        )
                                    ],
                                ),
                            )
                        ],
                    ),
                )
            ],
            triggered_effects=[
                TriggeredEffect(
                    trigger="on_activation",
                    condition="ally_target",
                    effect=MechanicalEffect(
                        forced_movements=[
                            ForcedMovement(
                                direction="pull",
                                distance=5,
                                target="ally",
                                ignores_engagement=True,
                                provokes_reactions=False,
                            )
                        ],
                    ),
                )
            ],
        ),
    ),
    MechSystemDefinition(
        id="ssc_iceout_drone",
        name="ICEOUT Drone",
        sp_cost=2,
        license_id="black_witch",
        license_rank=2,
        limited_uses=2,
        system_type="drone",
        tags=[SystemTag(tag="drone"), SystemTag(tag="quick_action")],
        drone=DronePayload(
            name="ICEOUT Drone",
            size="size_half",
            hp=10,
            evasion=10,
            e_defense=10,
        ),
        effects=MechanicalEffect(
            zones=[
                ZoneEffect(
                    shape="burst",
                    size=1,
                    placement="deployable",
                    duration="scene",
                    continuous_effects=MechanicalEffect(
                        tech_restrictions=[
                            TechActionRestriction(
                                disallow_tech_actions=True,
                                end_tech_effects=True,
                                target="all",
                            )
                        ],
                        status_clears=[StatusClear(status="tech", target="all")],
                    ),
                )
            ],
            action_grants=[
                ActionGrant(action_type="quick", name="move_drone"),
            ],
        ),
    ),
    MechSystemDefinition(
        id="ssc_mag_deployer",
        name="Mag Deployer",
        sp_cost=2,
        license_id="black_witch",
        license_rank=2,
        system_type="deployable",
        tags=[SystemTag(tag="deployable"), SystemTag(tag="quick_action")],
        deployable=DeployableEffect(
            obj=DeployableObject(
                size=2,
                evasion=5,
                hp=20,
            ),
        ),
        effects=MechanicalEffect(
            zones=[
                ZoneEffect(
                    shape="square",
                    width=2,
                    height=2,
                    placement="deployable",
                    placement_range=5,
                    duration="scene",
                    effects_on_enter=MechanicalEffect(
                        save_checks=[
                            SaveCheck(
                                trigger="on_activation",
                                condition="repulse_mode_hostile_enter",
                                save="hull",
                                target="enemy",
                                on_failure=MechanicalEffect(
                                    forced_movements=[
                                        ForcedMovement(
                                            direction="push",
                                            distance=3,
                                            target="enemy",
                                            on_collision=MechanicalEffect(
                                                status_grants=[
                                                    StatusGrant(
                                                        status="prone",
                                                        target="enemy",
                                                        duration="until_cleared",
                                                    )
                                                ],
                                            ),
                                        )
                                    ],
                                ),
                            ),
                            SaveCheck(
                                trigger="on_activation",
                                condition="attract_mode_any_enter",
                                save="hull",
                                target="enemy",
                                on_failure=MechanicalEffect(
                                    status_grants=[
                                        StatusGrant(
                                            status="immobilized",
                                            target="enemy",
                                            duration="until_cleared",
                                        )
                                    ],
                                ),
                            ),
                        ],
                        movement_grants=[
                            MovementGrant(
                                spaces=3,
                                movement_type="fly",
                                trigger="repulse_mode_ally_enter_free",
                            )
                        ],
                    ),
                )
            ],
            special="attract_mode_escape_quick_action_hull_save_only_one_deployer",
        ),
    ),
    MechSystemDefinition(
        id="ssc_black_ice_module",
        name="Black ICE Module",
        sp_cost=3,
        license_id="black_witch",
        license_rank=3,
        unique=True,
        effects=MechanicalEffect(
            tech_attack_mods=[
                TechAttackModifier(
                    value=-1,
                    target="ally",
                    condition="adjacent",
                    max_stacks=3,
                    reset_trigger="turn_end",
                )
            ],
        ),
    ),
    MechSystemDefinition(
        id="ssc_mag_shield",
        name="Mag Shield",
        sp_cost=2,
        license_id="black_witch",
        license_rank=3,
        unique=True,
        system_type="shield",
        tags=[SystemTag(tag="shield"), SystemTag(tag="quick_action")],
        effects=MechanicalEffect(
            zones=[
                ZoneEffect(
                    shape="line",
                    size=4,
                    placement_range=1,
                    duration="scene",
                    continuous_effects=MechanicalEffect(
                        resistances=[
                            Resistance(damage_type="kinetic", condition="attacks_through_field"),
                            Resistance(damage_type="explosive", condition="attacks_through_field"),
                        ],
                    ),
                )
            ],
            special="line_4_force_field_height_4_blocks_metal_movement_no_cover_or_los_only_one",
        ),
    ),
    MechSystemDefinition(
        id="ssc_high_stress_mag_clamps",
        name="High Stress Mag Clamps",
        sp_cost=1,
        license_id="deaths_head",
        license_rank=1,
        unique=True,
        tags=[SystemTag(tag="mod")],
        effects=MechanicalEffect(
            special="treat_vertical_surfaces_as_ground_fall_if_prone",
        ),
    ),
    MechSystemDefinition(
        id="ssc_tracking_drone",
        name="Tracking Drone",
        sp_cost=2,
        license_id="deaths_head",
        license_rank=1,
        tags=[SystemTag(tag="quick_action")],
        effects=MechanicalEffect(
            tech_actions=[
                TechAction(
                    name="Tracking Lock",
                    action_type="quick",
                    is_attack=True,
                    range=TechRange(range_type="sensors"),
                    on_hit=MechanicalEffect(
                        intel_effects=[
                            IntelEffect(
                                reveal=["location", "hp", "structure", "heat", "speed"],
                                audience="self",
                                target="enemy",
                                duration="until_cleared",
                            )
                        ],
                        action_restrictions=[
                            ActionRestriction(
                                action_ids=["hide"],
                                target="enemy",
                                duration="until_cleared",
                            )
                        ],
                        status_restrictions=[
                            StatusRestriction(
                                statuses=["hidden"],
                                restriction="cannot_gain",
                                target="enemy",
                                duration="until_cleared",
                            ),
                            StatusRestriction(
                                statuses=["invisible"],
                                restriction="cannot_benefit",
                                target="enemy",
                                duration="until_cleared",
                                condition="attacks_from_owner",
                            ),
                        ],
                        effect_removals=[
                            EffectRemoval(
                                action_type="quick",
                                check_type="engineering",
                                check_kind="check",
                                target="enemy",
                            )
                        ],
                    ),
                )
            ],
        ),
    ),
    MechSystemDefinition(
        id="ssc_core_siphon",
        name="Core Siphon",
        sp_cost=2,
        license_id="deaths_head",
        license_rank=2,
        unique=True,
        effects=MechanicalEffect(
            special=(
                "start_turn_optional_first_attack_plus_1_accuracy "
                "additional_attacks_plus_1_difficulty_until_turn_end"
            ),
        ),
    ),
    MechSystemDefinition(
        id="ssc_kinetic_compensator",
        name="Kinetic Compensator",
        sp_cost=2,
        license_id="deaths_head",
        license_rank=3,
        unique=True,
        effects=MechanicalEffect(
            triggered_effects=[
                TriggeredEffect(
                    trigger="on_miss",
                    condition="ranged_attack",
                    effect=MechanicalEffect(
                        accuracy_mods=[
                            AccuracyModifier(value=1, condition="next_ranged_attack")
                        ],
                    ),
                )
            ],
        ),
    ),
    MechSystemDefinition(
        id="ssc_neurospike_mk1",
        name="SSC Neurospike Mk1",
        sp_cost=2,
        license_id="dusk_wing",
        license_rank=1,
        unique=True,
        system_type="tech",
        tags=[SystemTag(tag="quick_action")],
        effects=MechanicalEffect(
            tech_actions=[
                TechAction(
                    name="Neurospike Invasion",
                    action_type="quick",
                    is_attack=True,
                    range=TechRange(range_type="sensors"),
                    on_hit=MechanicalEffect(
                        resource_changes=[
                            ResourceChange(
                                resource="heat",
                                amount=2,
                                direction="gain",
                                target="enemy",
                            )
                        ],
                        choices=[
                            EffectChoice(
                                name="Shrike Code",
                                target="enemy",
                                range=TechRange(range_type="sensors"),
                                effect=MechanicalEffect(
                                    triggered_effects=[
                                        TriggeredEffect(
                                            trigger="on_attack_roll",
                                            condition="until_end_of_next_turn",
                                            effect=MechanicalEffect(
                                                resource_changes=[
                                                    ResourceChange(
                                                        resource="heat",
                                                        amount=2,
                                                        direction="gain",
                                                        target="enemy",
                                                    )
                                                ],
                                            ),
                                        )
                                    ],
                                ),
                            ),
                            EffectChoice(
                                name="Mirage",
                                target="ally",
                                condition="choose_self_or_ally_in_los",
                                effect=MechanicalEffect(
                                    status_grants=[
                                        StatusGrant(
                                            status="invisible",
                                            target="ally",
                                            duration="end_of_next_turn",
                                            condition="invisible_to_chosen_target_only",
                                        )
                                    ],
                                ),
                            ),
                        ],
                    ),
                )
            ],
        ),
    ),
    MechSystemDefinition(
        id="ssc_flicker_field",
        name="Flicker Field",
        sp_cost=1,
        license_id="dusk_wing",
        license_rank=2,
        unique=True,
        effects=MechanicalEffect(
            triggered_effects=[
                TriggeredEffect(
                    trigger="on_activation",
                    condition="after_move_or_boost",
                    uses_per="round",
                    effect=MechanicalEffect(
                        status_grants=[
                            StatusGrant(
                                status="invisible",
                                target="self",
                                duration="until_attack",
                            )
                        ],
                    ),
                )
            ],
            status_stack_limits=[
                StatusStackLimit(status="invisible", target="self", max_stacks=1)
            ],
        ),
    ),
    MechSystemDefinition(
        id="ssc_stuncrown",
        name="StunCrown",
        sp_cost=2,
        license_id="dusk_wing",
        license_rank=3,
        limited_uses=2,
        tags=[SystemTag(tag="quick_action")],
        effects=MechanicalEffect(
            save_checks=[
                SaveCheck(
                    trigger="on_activation",
                    condition="burst_3_visible_no_cover",
                    save="agility",
                    target="enemy",
                    on_failure=MechanicalEffect(
                        status_grants=[
                            StatusGrant(
                                status="jammed",
                                target="enemy",
                                duration="end_of_next_turn",
                            )
                        ],
                    ),
                ),
                SaveCheck(
                    trigger="on_activation",
                    condition="burst_3_visible_no_cover",
                    save="systems",
                    target="enemy",
                    on_failure=MechanicalEffect(
                        status_grants=[
                            StatusGrant(
                                status="impaired",
                                target="enemy",
                                duration="end_of_next_turn",
                            )
                        ],
                    ),
                ),
            ],
        ),
    ),
    MechSystemDefinition(
        id="ssc_oasis",
        name="OASIS",
        sp_cost=3,
        license_id="dusk_wing",
        license_rank=3,
        unique=True,
        tags=[SystemTag(tag="protocol")],
        effects=MechanicalEffect(
            triggered_effects=[
                TriggeredEffect(
                    trigger="on_activation",
                    effect=MechanicalEffect(
                        resource_changes=[
                            ResourceChange(
                                resource="heat",
                                amount=2,
                                direction="gain",
                                target="self",
                            )
                        ],
                    ),
                )
            ],
            special=(
                "protocol_straight_line_movement_light_construct_hard_cover_adjacent_energy_resistance "
                "lingers_scene_or_until_reactivated"
            ),
        ),
    ),
    MechSystemDefinition(
        id="ssc_flash_charges",
        name="Flash Charges",
        sp_cost=2,
        license_id="metalmark",
        license_rank=1,
        unique=True,
        limited_uses=2,
        tags=[SystemTag(tag="grenade"), SystemTag(tag="mine")],
        grenades=[
            GrenadePayload(
                name="Flash Grenade",
                range=5,
                area=AreaEffect(
                    pattern="blast",
                    size=2,
                    duration="end_of_next_turn",
                ),
            )
        ],
        mines=[
            MinePayload(
                name="Flash Mine",
                area=AreaEffect(
                    pattern="burst",
                    size=1,
                    duration="end_of_next_turn",
                    save="agility",
                ),
            )
        ],
        effects=MechanicalEffect(
            special="grenade_blocks_los_out_mine_fail_los_adjacent_only",
        ),
    ),
    MechSystemDefinition(
        id="ssc_reactive_weave",
        name="Reactive Weave",
        sp_cost=1,
        license_id="metalmark",
        license_rank=1,
        unique=True,
        effects=MechanicalEffect(
            triggered_effects=[
                TriggeredEffect(
                    trigger="on_activation",
                    condition="brace",
                    effect=MechanicalEffect(
                        status_grants=[
                            StatusGrant(
                                status="invisible",
                                target="self",
                                duration="end_of_next_turn",
                            )
                        ],
                        special="brace_move_speed",
                    ),
                )
            ],
        ),
    ),
    MechSystemDefinition(
        id="ssc_active_camouflage",
        name="Active Camouflage",
        sp_cost=3,
        license_id="metalmark",
        license_rank=3,
        unique=True,
        tags=[SystemTag(tag="protocol")],
        effects=MechanicalEffect(
            triggered_effects=[
                TriggeredEffect(
                    trigger="on_activation",
                    effect=MechanicalEffect(
                        resource_changes=[
                            ResourceChange(
                                resource="heat",
                                amount=2,
                                direction="gain",
                                target="self",
                            )
                        ],
                        status_grants=[
                            StatusGrant(
                                status="invisible",
                                target="self",
                                duration="end_of_next_turn",
                            )
                        ],
                    ),
                )
            ],
            status_breaks=[
                StatusBreakCondition(
                    status="invisible",
                    target="self",
                    break_triggers=["take_damage", "stunned", "manual_deactivate"],
                )
            ],
            effect_removals=[
                EffectRemoval(action_type="protocol", target="self", condition="manual_deactivate")
            ],
        ),
    ),
    MechSystemDefinition(
        id="ssc_shock_wreath",
        name="Shock Wreath",
        sp_cost=2,
        license_id="metalmark",
        license_rank=3,
        tags=[SystemTag(tag="mod")],
        effects=MechanicalEffect(
            weapon_mods=[
                WeaponModEffect(
                    allowed_weapon_types=["melee"],
                    burn_by_size=[
                        WeaponSizeBonus(size="aux", burn=1),
                        WeaponSizeBonus(size="main", burn=2),
                        WeaponSizeBonus(size="heavy", burn=3),
                        WeaponSizeBonus(size="superheavy", burn=3),
                    ],
                    increase_existing_burn=True,
                )
            ],
        ),
    ),
    MechSystemDefinition(
        id="ssc_javelin_rockets",
        name="Javelin Rockets",
        sp_cost=2,
        license_id="monarch",
        license_rank=1,
        unique=True,
        tags=[SystemTag(tag="quick_action")],
        effects=MechanicalEffect(
            special="mark_three_spaces_range_15_los_non_adjacent_auto_damage_3_kinetic_on_pass_or_start_end_start_next_turn",
        ),
    ),
    MechSystemDefinition(
        id="ssc_stabilizer_mod",
        name="Stabilizer Mod",
        sp_cost=2,
        license_id="monarch",
        license_rank=2,
        tags=[SystemTag(tag="mod")],
        effects=MechanicalEffect(
            weapon_mods=[
                WeaponModEffect(
                    allowed_weapon_types=["launcher", "cannon"],
                    range_bonus=5,
                    add_tags=[WeaponTagGrant(tag="ordnance")],
                )
            ],
        ),
    ),
    MechSystemDefinition(
        id="ssc_tlaloc_nhp",
        name="TLALOC-Class NHP",
        sp_cost=3,
        license_id="monarch",
        license_rank=3,
        unique=True,
        system_type="ai",
        tags=[SystemTag(tag="protocol")],
        effects=MechanicalEffect(
            triggered_effects=[
                TriggeredEffect(
                    trigger="on_activation",
                    effect=MechanicalEffect(
                        resource_changes=[
                            ResourceChange(
                                resource="heat",
                                amount=2,
                                direction="gain",
                                target="self",
                            )
                        ],
                        status_grants=[
                            StatusGrant(
                                status="immobilized",
                                target="self",
                                duration="start_of_next_turn",
                            )
                        ],
                    ),
                ),
                TriggeredEffect(
                    trigger="on_miss",
                    condition="melee_or_ranged_attack_different_target",
                    effect=MechanicalEffect(
                        action_grants=[
                            ActionGrant(action_type="free", name="reroll_attack")
                        ],
                    ),
                ),
            ],
            special="reroll_once_per_attack_no_retarget_same_target",
        ),
    ),
    MechSystemDefinition(
        id="ssc_exposed_singularity",
        name="Exposed Singularity",
        sp_cost=2,
        license_id="mourning_cloak",
        license_rank=2,
        unique=True,
        tags=[SystemTag(tag="reaction")],
        effects=MechanicalEffect(
            triggered_effects=[
                TriggeredEffect(
                    trigger="on_activation",
                    condition="on_take_damage",
                    uses_per="round",
                    effect=MechanicalEffect(
                        special="teleport_1d6_spaces_on_damage",
                    ),
                )
            ],
        ),
    ),
    MechSystemDefinition(
        id="ssc_hunter_logic",
        name="Hunter Logic",
        sp_cost=2,
        license_id="mourning_cloak",
        license_rank=2,
        system_type="tech",
        tags=[SystemTag(tag="quick_action")],
        effects=MechanicalEffect(
            tech_actions=[
                TechAction(
                    name="Hunter Logic Invasion",
                    action_type="quick",
                    is_attack=True,
                    range=TechRange(range_type="sensors"),
                    on_hit=MechanicalEffect(
                        resource_changes=[
                            ResourceChange(
                                resource="heat",
                                amount=2,
                                direction="gain",
                                target="enemy",
                            )
                        ],
                        choices=[
                            EffectChoice(
                                name="Stalk Prey",
                                target="enemy",
                                range=TechRange(range_type="sensors"),
                                condition="exclusive_target",
                                effect=MechanicalEffect(
                                    status_grants=[
                                        StatusGrant(
                                            status="invisible",
                                            target="self",
                                            duration="until_cleared",
                                            condition="invisible_to_chosen_target_only",
                                        )
                                    ],
                                    status_breaks=[
                                        StatusBreakCondition(
                                            status="invisible",
                                            target="self",
                                            break_triggers=["take_damage"],
                                            condition="source_is_chosen_target",
                                        )
                                    ],
                                ),
                            ),
                            EffectChoice(
                                name="Terrify",
                                target="enemy",
                                range=TechRange(range_type="sensors"),
                                effect=MechanicalEffect(
                                    status_grants=[
                                        StatusGrant(
                                            status="impaired",
                                            target="enemy",
                                            duration="end_of_next_turn",
                                        )
                                    ],
                                    movement_restrictions=[
                                        MovementRestrictionEffect(
                                            target="enemy",
                                            cannot_move_closer_to_source=True,
                                            duration="end_of_next_turn",
                                        )
                                    ],
                                ),
                            ),
                        ],
                    ),
                )
            ],
        ),
    ),
    MechSystemDefinition(
        id="ssc_fade_cloak",
        name="FADE Cloak",
        sp_cost=2,
        license_id="mourning_cloak",
        license_rank=3,
        unique=True,
        tags=[SystemTag(tag="quick_action")],
        effects=MechanicalEffect(
            phase_shifts=[
                PhaseShiftEffect(
                    activation_action="quick",
                    roll=DiceExpression.parse("1d6"),
                    success_threshold=4,
                    out_of_phase_duration="start_of_next_turn",
                    duration="scene",
                    deactivation_action="quick",
                )
            ],
        ),
    ),
    MechSystemDefinition(
        id="ssc_markerlight",
        name="Markerlight",
        sp_cost=2,
        license_id="swallowtail",
        license_rank=1,
        system_type="tech",
        tags=[SystemTag(tag="full_action")],
        effects=MechanicalEffect(
            tech_actions=[
                TechAction(
                    name="Paint Target",
                    action_type="full",
                    is_attack=True,
                    range=TechRange(range_type="sensors"),
                    on_hit=MechanicalEffect(
                        resource_changes=[
                            ResourceChange(
                                resource="heat",
                                amount=4,
                                direction="gain",
                                target="enemy",
                            )
                        ],
                        status_grants=[
                            StatusGrant(
                                status="lock_on",
                                target="enemy",
                                duration="until_cleared",
                            )
                        ],
                        special="ally_hit_reaction_upgrade_to_crit_until_start_next_turn",
                    ),
                )
            ],
        ),
    ),
    MechSystemDefinition(
        id="ssc_scout_drone",
        name="Scout Drone",
        sp_cost=2,
        license_id="swallowtail",
        license_rank=1,
        system_type="drone",
        tags=[SystemTag(tag="drone"), SystemTag(tag="quick_action")],
        drone=DronePayload(
            name="Scout Drone",
            size="size_half",
            hp=10,
            evasion=10,
            e_defense=10,
            deploy_range_type="sensors",
            deploy_requires_line_of_sight=True,
            invisible=True,
            redeploy_action="quick",
            recall_action="quick",
        ),
        effects=MechanicalEffect(
            zones=[
                ZoneEffect(
                    shape="burst",
                    size=2,
                    placement="deployable",
                    duration="scene",
                    effects_on_end_turn=MechanicalEffect(
                        status_clears=[StatusClear(status="hidden", target="enemy")],
                    ),
                    continuous_effects=MechanicalEffect(
                        intel_effects=[
                            IntelEffect(
                                reveal=["hp", "evasion", "e_defense", "heat"],
                                audience="self",
                                target="enemy",
                                perfect_vision=True,
                                duration="scene",
                            )
                        ],
                        status_restrictions=[
                            StatusRestriction(
                                statuses=["invisible"],
                                restriction="cannot_benefit",
                                target="enemy",
                                duration="scene",
                            )
                        ],
                    ),
                )
            ],
        ),
    ),
    MechSystemDefinition(
        id="ssc_low_profile",
        name="Low Profile",
        sp_cost=1,
        license_id="swallowtail",
        license_rank=2,
        unique=True,
        tags=[SystemTag(tag="protocol")],
        effects=MechanicalEffect(
            accuracy_mods=[
                AccuracyModifier(value=-1, condition="search_hidden_against_self"),
                AccuracyModifier(value=-1, condition="ranged_or_tech_attacks_against_self"),
            ],
            status_grants=[
                StatusGrant(
                    status="slowed",
                    target="self",
                    duration="until_cleared",
                )
            ],
            action_restrictions=[
                ActionRestriction(
                    disallow_attack_rolls=True,
                    target="self",
                    duration="until_cleared",
                )
            ],
            effect_removals=[
                EffectRemoval(
                    action_type="quick",
                    target="self",
                )
            ],
        ),
    ),
    MechSystemDefinition(
        id="ssc_athena_nhp",
        name="ATHENA-Class NHP",
        sp_cost=3,
        license_id="swallowtail",
        license_rank=3,
        unique=True,
        system_type="ai",
        tags=[SystemTag(tag="quick_action")],
        effects=MechanicalEffect(
            zones=[
                ZoneEffect(
                    shape="blast",
                    size=3,
                    placement="target_area",
                    placement_range=50,
                    retarget_action="quick",
                    retarget_range=50,
                    retarget_replaces_existing=True,
                    duration="scene",
                    effects_on_end_turn=MechanicalEffect(
                        status_clears=[
                            StatusClear(status="hidden", target="enemy"),
                            StatusClear(status="invisible", target="enemy"),
                        ],
                        status_grants=[
                            StatusGrant(
                                status="lock_on",
                                target="enemy",
                                duration="until_cleared",
                            )
                        ],
                    ),
                    continuous_effects=MechanicalEffect(
                        intel_effects=[
                            IntelEffect(
                                reveal=[
                                    "hp",
                                    "evasion",
                                    "e_defense",
                                    "heat",
                                    "weapons",
                                    "systems",
                                ],
                                audience="self",
                                target="enemy",
                                perfect_vision=True,
                                grants_line_of_sight=False,
                                duration="scene",
                            )
                        ],
                        action_restrictions=[
                            ActionRestriction(
                                action_ids=["hide"],
                                target="enemy",
                                duration="scene",
                            )
                        ],
                        status_restrictions=[
                            StatusRestriction(
                                statuses=["hidden"],
                                restriction="cannot_gain",
                                target="enemy",
                                duration="scene",
                            ),
                            StatusRestriction(
                                statuses=["invisible"],
                                restriction="cannot_gain",
                                target="enemy",
                                duration="scene",
                            ),
                        ],
                        cover_restrictions=[
                            CoverRestriction(
                                max_cover="none",
                                target="enemy",
                                duration="scene",
                            )
                        ],
                    ),
                )
            ],
        ),
    ),
    MechSystemDefinition(
        id="ssc_cloaking_field",
        name="Cloaking Field",
        sp_cost=4,
        license_id="swallowtail",
        license_rank=3,
        tags=[SystemTag(tag="quick_action")],
        effects=MechanicalEffect(
            triggered_effects=[
                TriggeredEffect(
                    trigger="on_activation",
                    effect=MechanicalEffect(
                        resource_changes=[
                            ResourceChange(
                                resource="heat",
                                amount=2,
                                direction="gain",
                                target="self",
                            )
                        ],
                    ),
                ),
                TriggeredEffect(
                    trigger="on_take_damage",
                    effect=MechanicalEffect(
                        status_clears=[StatusClear(status="invisible", target="ally")]
                    ),
                ),
            ],
            zones=[
                ZoneEffect(
                    shape="burst",
                    size=2,
                    placement="self",
                    duration="end_of_next_turn",
                    applies_to="ally",
                    continuous_effects=MechanicalEffect(
                        status_grants=[
                            StatusGrant(
                                status="invisible",
                                target="ally",
                                duration="end_of_next_turn",
                            )
                        ],
                    ),
                )
            ],
            status_triggers=[
                StatusTrigger(
                    trigger="on_inflict",
                    status="stunned",
                    target="self",
                    effect=MechanicalEffect(
                        status_clears=[StatusClear(status="invisible", target="ally")]
                    ),
                )
            ],
            effect_removals=[EffectRemoval(action_type="quick", target="self")],
        ),
    ),
]


ALL_FRAMES: list[MechFrameDefinition] = GMS_FRAMES + IPSN_FRAMES + SSC_FRAMES
ALL_WEAPONS: list[MechWeaponDefinition] = GMS_WEAPONS + IPSN_WEAPONS + SSC_WEAPONS
ALL_SYSTEMS: list[MechSystemDefinition] = GMS_SYSTEMS + IPSN_SYSTEMS + SSC_SYSTEMS

FRAME_DEFINITIONS_BY_ID = {frame.id: frame for frame in ALL_FRAMES}
WEAPON_DEFINITIONS_BY_ID = {weapon.id: weapon for weapon in ALL_WEAPONS}
SYSTEM_DEFINITIONS_BY_ID = {system.id: system for system in ALL_SYSTEMS}


def get_frame_definition(frame_id: str) -> MechFrameDefinition | None:
    """Look up a frame definition by ID."""
    return FRAME_DEFINITIONS_BY_ID.get(frame_id)


def get_weapon_definition(weapon_id: str) -> MechWeaponDefinition | None:
    """Look up a weapon definition by ID."""
    return WEAPON_DEFINITIONS_BY_ID.get(weapon_id)


def get_system_definition(system_id: str) -> MechSystemDefinition | None:
    """Look up a system definition by ID."""
    return SYSTEM_DEFINITIONS_BY_ID.get(system_id)
