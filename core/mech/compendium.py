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
from core.shared.effects import AccuracyModifier, MechanicalEffect, StatModifier


GMS_FRAMES: list[MechFrameDefinition] = [
    MechFrameDefinition(
        id="gms_everest",
        name="GMS Everest",
        manufacturer="GMS",
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
                    special="first_turn_extra_quick_action_free",
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
        ],
        system_points=5,
        core_system=CoreSystemDefinition(
            id="ipsn_m35_mjolnir",
            name="IPS-N M35 Mjolnir Cannon",
            effects=MechanicalEffect(
                special="integrated_main_cqb_range5_threat3_damage4_kinetic_free_on_reload_thunder_god_protocol_chambers",
            ),
        ),
        traits=[
            FrameTrait(
                name="Full Metal Jacket",
                effects=MechanicalEffect(
                    special="reload_all_loading_if_no_attacks_or_forced_saves_end_turn",
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
]


IPSN_WEAPONS: list[MechWeaponDefinition] = [
    MechWeaponDefinition(
        id="hand_cannon",
        name="Hand Cannon",
        size="aux",
        weapon_type="cqb",
        damage_type="kinetic",
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
        ranges=[
            WeaponRange(range_type="range", value=5),
            WeaponRange(range_type="threat", value=3),
        ],
        damage=[WeaponDamage(damage_type="kinetic", flat=4)],
    ),
]


IPSN_SYSTEMS: list[MechSystemDefinition] = [
    MechSystemDefinition(
        id="ipsn_breaching_charges",
        name="Breaching Charges",
        sp_cost=2,
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
        unique=True,
        tags=[SystemTag(tag="mod")],
        effects=MechanicalEffect(
            special="after_reload_next_loading_attack_bonus_1d6_explosive_and_hull_save_or_prone",
        ),
    ),
    MechSystemDefinition(
        id="ipsn_uncle_cc",
        name="UNCLE-class C/C",
        sp_cost=3,
        unique=True,
        system_type="ai",
        tags=[SystemTag(tag="ai"), SystemTag(tag="mod")],
        effects=MechanicalEffect(
            special="grant_ai_control_to_weapon_once_per_turn_free_attack_plus_2_difficulty_cooldown_until_next_turn",
        ),
    ),
]


ALL_FRAMES: list[MechFrameDefinition] = GMS_FRAMES + IPSN_FRAMES
ALL_WEAPONS: list[MechWeaponDefinition] = GMS_WEAPONS + IPSN_WEAPONS
ALL_SYSTEMS: list[MechSystemDefinition] = GMS_SYSTEMS + IPSN_SYSTEMS

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
