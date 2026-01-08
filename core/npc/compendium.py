"""NPC compendium for typed Lancer mechanics.

This module provides NPC templates for various manufacturers/factions.
NPCs can reuse player compendium gear without license requirements.
"""

from core.npc.models import (
    NPCTemplate,
    NPCStats,
    NPCStatsBase,
    NPCTierScaling,
    NPCAbility,
    NPCGear,
)
from core.npc.enums import NPCTier, NPCClass
from core.shared.effects import (
    MechanicalEffect,
    DamageMultiplierEffect,
    AccuracyModifier,
    Resistance,
    DirectDamage,
    ActionGrant,
)
from core.npc import special_classes


def _make_grunt_scaling() -> NPCTierScaling:
    """Create standard grunt tier scaling."""
    return NPCTierScaling(
        hp_multiplier=1.0,
        hp_adder_tier_2=10,
        hp_adder_tier_3=20,
        evasion_adder_tier_2=1,
        evasion_adder_tier_3=2,
        e_defense_adder_tier_2=1,
        e_defense_adder_tier_3=2,
        armor_adder_tier_2=0,
        armor_adder_tier_3=1,
        save_adder_tier_2=1,
        save_adder_tier_3=2,
    )


def _make_elite_scaling() -> NPCTierScaling:
    """Create standard elite tier scaling (slightly higher base stats)."""
    return NPCTierScaling(
        hp_multiplier=1.5,
        hp_adder_tier_2=15,
        hp_adder_tier_3=30,
        evasion_adder_tier_2=1,
        evasion_adder_tier_3=2,
        e_defense_adder_tier_2=1,
        e_defense_adder_tier_3=2,
        armor_adder_tier_2=0,
        armor_adder_tier_3=1,
        save_adder_tier_2=1,
        save_adder_tier_3=2,
    )


def _make_boss_scaling() -> NPCTierScaling:
    """Create boss-tier scaling (much higher HP, better defenses)."""
    return NPCTierScaling(
        hp_multiplier=2.0,
        hp_adder_tier_2=30,
        hp_adder_tier_3=50,
        evasion_adder_tier_2=2,
        evasion_adder_tier_3=3,
        e_defense_adder_tier_2=2,
        e_defense_adder_tier_3=3,
        armor_adder_tier_2=1,
        armor_adder_tier_3=2,
        save_adder_tier_2=2,
        save_adder_tier_3=3,
    )


def _make_specialist_scaling() -> NPCTierScaling:
    """Create specialist tier scaling (focused on specific stats)."""
    return NPCTierScaling(
        hp_multiplier=1.0,
        hp_adder_tier_2=10,
        hp_adder_tier_3=20,
        evasion_adder_tier_2=1,
        evasion_adder_tier_3=2,
        e_defense_adder_tier_2=2,
        e_defense_adder_tier_3=4,
        armor_adder_tier_2=0,
        armor_adder_tier_3=1,
        save_adder_tier_2=1,
        save_adder_tier_3=2,
    )


GMS_GRUNT_T1 = NPCTemplate(
    id="gms_grunt_t1",
    name="GMS Grunt",
    description="Standard General Massive Systems assault mech used by corporate security.",
    npc_class="grunt",
    tier="tier_1",
    role="striker",
    victory_count=0.25,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_1",
            hp_base=10,
            evasion_base=8,
            e_defense_base=8,
            armor_base=0,
            speed_base=4,
            sensor_range=10,
            save_bonus=0,
        ),
        scaling=_make_grunt_scaling(),
    ),
    abilities=[
        NPCAbility(
            id="overwatch",
            name="Overwatch",
            trigger="on_turn_start",
            effect=MechanicalEffect(
                action_grants=[
                    ActionGrant(
                        action_type="reaction",
                        name="Rifle Overwatch",
                        trigger="on_move",
                    )
                ]
            ),
        ),
    ],
    gear=[
        NPCGear(weapon_id="gms_assault_rifle"),
        NPCGear(weapon_id="gms_tactical_knife"),
    ],
)

GMS_GRUNT_T2 = NPCTemplate(
    id="gms_grunt_t2",
    name="GMS Grunt Elite",
    description="Upgraded GMS assault mech with enhanced systems.",
    npc_class="grunt",
    tier="tier_2",
    role="controller",
    victory_count=0.25,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_1",
            hp_base=12,
            evasion_base=9,
            e_defense_base=9,
            armor_base=0,
            speed_base=5,
            sensor_range=10,
            save_bonus=1,
        ),
        scaling=_make_grunt_scaling(),
    ),
    abilities=[
        NPCAbility(
            id="overwatch",
            name="Overwatch",
            trigger="on_turn_start",
        ),
        NPCAbility(
            id="coordinated_fire",
            name="Coordinated Fire",
            trigger="on_kill",
            effect=MechanicalEffect(accuracy_mods=[AccuracyModifier(value=1)]),
        ),
    ],
    gear=[
        NPCGear(weapon_id="gms_assault_rifle"),
        NPCGear(weapon_id="gms_heavy_kinetic_hammer"),
    ],
)

GMS_ELITE_T2 = NPCTemplate(
    id="gms_elite_t2",
    name="GMS Vanguard",
    description="Heavily armed and armored GMS assault mech.",
    npc_class="elite",
    tier="tier_2",
    role="controller",
    victory_count=0.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_2",
            hp_base=15,
            evasion_base=8,
            e_defense_base=8,
            armor_base=1,
            speed_base=4,
            sensor_range=10,
            save_bonus=1,
        ),
        scaling=_make_elite_scaling(),
    ),
    abilities=[
        NPCAbility(
            id="shield_wall",
            name="Shield Wall",
            trigger="on_adjacent",
            effect=MechanicalEffect(resistances=[Resistance(damage_type="kinetic")]),
            uses_per_combat=2,
        ),
        NPCAbility(
            id="overwatch",
            name="Overwatch",
            trigger="on_turn_start",
        ),
    ],
    gear=[
        NPCGear(weapon_id="gms_heavy_kinetic_hammer"),
        NPCGear(system_id="gms_standard_pattern_issue_shield"),
    ],
)

IPSN_GRUNT_T1 = NPCTemplate(
    id="ipsn_grunt_t1",
    name="IPS-N Raider",
    description="IPS-N industrial mech repurposed for combat.",
    npc_class="grunt",
    tier="tier_1",
    role="striker",
    victory_count=0.25,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_1",
            hp_base=12,
            evasion_base=6,
            e_defense_base=8,
            armor_base=0,
            speed_base=4,
            sensor_range=10,
            save_bonus=0,
        ),
        scaling=_make_grunt_scaling(),
    ),
    abilities=[],
    gear=[
        NPCGear(weapon_id="ipsn_mjolnir"),
        NPCGear(weapon_id="ipsn_plasma_cutter"),
    ],
)

IPSN_BOSS_T3 = NPCTemplate(
    id="ipsn_boss_t3",
    name="IPS-N Dreadnought",
    description="Massive IPS-N heavy assault mech with devastating firepower.",
    npc_class="boss",
    tier="tier_3",
    role="defender",
    victory_count=1.0,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_3",
            hp_base=25,
            evasion_base=6,
            e_defense_base=8,
            armor_base=2,
            speed_base=3,
            sensor_range=10,
            save_bonus=2,
        ),
        scaling=_make_boss_scaling(),
    ),
    abilities=[
        NPCAbility(
            id="artillery_strike",
            name="Artillery Strike",
            trigger="on_turn_start",
            effect=MechanicalEffect(
                direct_damages=[
                    DirectDamage(
                        flat=3,
                        damage_type="explosive",
                        ap=True,
                    )
                ]
            ),
            uses_per_combat=2,
        ),
        NPCAbility(
            id="siege_mode",
            name="Siege Mode",
            trigger="on_take_damage",
            effect=MechanicalEffect(
                damage_multipliers=[DamageMultiplierEffect(multiplier=1.5)]
            ),
        ),
        NPCAbility(
            id="overwatch",
            name="Overwatch",
            trigger="on_turn_start",
        ),
    ],
    gear=[
        NPCGear(weapon_id="ipsn_railgun"),
        NPCGear(weapon_id="ipsn_missile_barrel"),
    ],
)

SSC_SPECIALIST_T2 = NPCTemplate(
    id="ssc_specialist_t2",
    name="SSC Specter",
    description="Stealth-oriented SSC mech designed for ambush tactics.",
    npc_class="specialist",
    tier="tier_2",
    role="controller",
    victory_count=0.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_1",
            hp_base=8,
            evasion_base=12,
            e_defense_base=10,
            armor_base=0,
            speed_base=5,
            sensor_range=15,
            save_bonus=1,
        ),
        scaling=_make_specialist_scaling(),
    ),
    abilities=[
        NPCAbility(
            id="ambush",
            name="Ambush",
            trigger="on_hit",
            effect=MechanicalEffect(
                damage_multipliers=[DamageMultiplierEffect(multiplier=2.0)]
            ),
            uses_per_combat=3,
        ),
        NPCAbility(
            id="invisibility",
            name="Invisibility",
            trigger="on_turn_start",
            effect=MechanicalEffect(
                action_grants=[
                    ActionGrant(
                        action_type="quick",
                        name="Hide",
                    )
                ]
            ),
        ),
    ],
    gear=[
        NPCGear(weapon_id="ssc_needle_rifle"),
        NPCGear(system_id="ssc_personalized_ecm"),
    ],
)

HORUS_ELITE_T3 = NPCTemplate(
    id="horus_elite_t3",
    name="HORUS Harbinger",
    description="Mysterious and unpredictable HORUS war machine.",
    npc_class="elite",
    tier="tier_3",
    role="defender",
    victory_count=0.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_2",
            hp_base=15,
            evasion_base=10,
            e_defense_base=12,
            armor_base=1,
            speed_base=5,
            sensor_range=20,
            save_bonus=2,
        ),
        scaling=_make_elite_scaling(),
    ),
    abilities=[
        NPCAbility(
            id="system_failure",
            name="System Failure",
            trigger="on_attacked",
            effect=MechanicalEffect(
                direct_damages=[
                    DirectDamage(
                        flat=2,
                        damage_type="energy",
                        ap=True,
                    )
                ]
            ),
            uses_per_combat=3,
        ),
        NPCAbility(
            id="reality_distortion",
            name="Reality Distortion",
            trigger="on_turn_start",
            effect=MechanicalEffect(accuracy_mods=[AccuracyModifier(value=2)]),
        ),
    ],
    gear=[
        NPCGear(weapon_id="horus_autogun"),
        NPCGear(system_id="horus_mantis_sinks"),
    ],
)

HA_BOSS_T3 = NPCTemplate(
    id="ha_boss_t3",
    name="HA Emperor",
    description="Harrison Armory's ultimate combat platform.",
    npc_class="boss",
    tier="tier_3",
    role="defender",
    victory_count=1.0,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_3",
            hp_base=30,
            evasion_base=8,
            e_defense_base=10,
            armor_base=3,
            speed_base=4,
            sensor_range=10,
            save_bonus=3,
        ),
        scaling=_make_boss_scaling(),
    ),
    abilities=[
        NPCAbility(
            id="devastating_fire",
            name="Devastating Fire",
            trigger="on_hit",
            effect=MechanicalEffect(
                damage_multipliers=[DamageMultiplierEffect(multiplier=1.5)]
            ),
        ),
        NPCAbility(
            id="armor_plating",
            name="Armor Plating",
            trigger="on_take_damage",
            effect=MechanicalEffect(resistances=[Resistance(damage_type="kinetic")]),
            uses_per_combat=2,
        ),
        NPCAbility(
            id="overwatch",
            name="Overwatch",
            trigger="on_turn_start",
        ),
    ],
    gear=[
        NPCGear(weapon_id="ha_sledgehammer"),
        NPCGear(weapon_id="ha_palus_knife"),
    ],
)

PR2_ACE_T1 = NPCTemplate(
    id="pr2_ace_t1",
    name="Ace",
    description="",
    npc_class="specialist",
    tier="tier_1",
    role="striker",
    victory_count=0.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_1",
            hp_base=10,
            evasion_base=12,
            e_defense_base=8,
            armor_base=0,
            speed_base=5,
            sensor_range=10,
            save_bonus=10,
            tech_attack=1,
        ),
        scaling=_make_specialist_scaling(),
    ),
    abilities=[],
    gear=[
        NPCGear(system_id="npc_ss_corpro_flight_system"),
        NPCGear(weapon_id="npc_missile_launcher"),
        NPCGear(system_id="npc_evasive_maneuvers"),
    ],
)

PR2_ASSAULT_T1 = NPCTemplate(
    id="pr2_assault_t1",
    name="Assault",
    description="",
    npc_class="specialist",
    tier="tier_1",
    role="striker",
    victory_count=0.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_1",
            hp_base=15,
            evasion_base=8,
            e_defense_base=8,
            armor_base=1,
            speed_base=4,
            sensor_range=8,
            save_bonus=10,
            tech_attack=1,
        ),
        scaling=_make_specialist_scaling(),
    ),
    abilities=[],
    gear=[
        NPCGear(weapon_id="npc_heavy_assault_rifle"),
        NPCGear(weapon_id="npc_combat_knife"),
    ],
)

PR2_ARCHER_T1 = NPCTemplate(
    id="pr2_archer_t1",
    name="Archer",
    description="",
    npc_class="specialist",
    tier="tier_1",
    role="controller",
    victory_count=0.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_1",
            hp_base=16,
            evasion_base=8,
            e_defense_base=8,
            armor_base=0,
            speed_base=5,
            sensor_range=15,
            save_bonus=11,
            tech_attack=2,
        ),
        scaling=_make_specialist_scaling(),
    ),
    abilities=[],
    gear=[
        NPCGear(weapon_id="npc_lmg"),
        NPCGear(system_id="npc_suppress"),
    ],
)

PR2_ASSASSIN_T1 = NPCTemplate(
    id="pr2_assassin_t1",
    name="Assassin",
    description="",
    npc_class="specialist",
    tier="tier_1",
    role="striker",
    victory_count=0.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_half",
            hp_base=15,
            evasion_base=12,
            e_defense_base=8,
            armor_base=0,
            speed_base=6,
            sensor_range=10,
            save_bonus=10,
            tech_attack=1,
        ),
        scaling=_make_specialist_scaling(),
    ),
    abilities=[],
    gear=[
        NPCGear(system_id="npc_kai_bioplating"),
        NPCGear(weapon_id="npc_heated_blade"),
        NPCGear(system_id="npc_assassins_mark"),
    ],
)

PR2_BASTION_T1 = NPCTemplate(
    id="pr2_bastion_t1",
    name="Bastion",
    description="",
    npc_class="specialist",
    tier="tier_1",
    role="defender",
    victory_count=0.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_2",
            hp_base=10,
            evasion_base=8,
            e_defense_base=8,
            armor_base=3,
            speed_base=4,
            sensor_range=5,
            save_bonus=8,
            tech_attack=0,
        ),
        scaling=_make_specialist_scaling(),
    ),
    abilities=[],
    gear=[
        NPCGear(weapon_id="npc_rotary_grenade_launcher"),
        NPCGear(weapon_id="npc_heavy_assault_shield"),
        NPCGear(system_id="npc_shieldwall"),
        NPCGear(system_id="npc_guardian"),
    ],
)

PR2_BERSERKER_T1 = NPCTemplate(
    id="pr2_berserker_t1",
    name="Berserker",
    description="",
    npc_class="specialist",
    tier="tier_1",
    role="striker",
    victory_count=0.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_1",
            hp_base=12,
            evasion_base=8,
            e_defense_base=6,
            armor_base=1,
            speed_base=5,
            sensor_range=5,
            save_bonus=10,
            tech_attack=0,
        ),
        scaling=_make_specialist_scaling(),
    ),
    abilities=[],
    gear=[
        NPCGear(weapon_id="npc_chain_axe"),
        NPCGear(system_id="npc_active_defense"),
        NPCGear(system_id="npc_aggression"),
    ],
)

PR2_BREACHER_T1 = NPCTemplate(
    id="pr2_breacher_t1",
    name="Breacher",
    description="",
    npc_class="specialist",
    tier="tier_1",
    role="striker",
    victory_count=0.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_1",
            hp_base=18,
            evasion_base=9,
            e_defense_base=7,
            armor_base=1,
            speed_base=3,
            sensor_range=5,
            save_bonus=10,
            tech_attack=0,
        ),
        scaling=_make_specialist_scaling(),
    ),
    abilities=[],
    gear=[
        NPCGear(weapon_id="npc_dual_shotguns"),
        NPCGear(system_id="npc_breach_ram"),
    ],
)

PR2_CATAPHRACT_T1 = NPCTemplate(
    id="pr2_cataphract_t1",
    name="Cataphract",
    description="",
    npc_class="specialist",
    tier="tier_1",
    role="striker",
    victory_count=0.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_1",
            hp_base=15,
            evasion_base=10,
            e_defense_base=8,
            armor_base=0,
            speed_base=8,
            sensor_range=5,
            save_bonus=12,
            tech_attack=0,
        ),
        scaling=_make_specialist_scaling(),
    ),
    abilities=[],
    gear=[
        NPCGear(weapon_id="npc_ram_cannon"),
        NPCGear(system_id="npc_trample"),
    ],
)

PR2_DEMOLISHER_T1 = NPCTemplate(
    id="pr2_demolisher_t1",
    name="Demolisher",
    description="",
    npc_class="specialist",
    tier="tier_1",
    role="defender",
    victory_count=0.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_2",
            hp_base=15,
            evasion_base=6,
            e_defense_base=7,
            armor_base=2,
            speed_base=2,
            sensor_range=10,
            save_bonus=10,
            tech_attack=0,
        ),
        scaling=_make_specialist_scaling(),
    ),
    abilities=[],
    gear=[
        NPCGear(weapon_id="npc_demolisher_hammer"),
        NPCGear(system_id="npc_entrench"),
        NPCGear(system_id="npc_shock_armor"),
    ],
)

PR2_ENGINEER_T1 = NPCTemplate(
    id="pr2_engineer_t1",
    name="Engineer",
    description="",
    npc_class="specialist",
    tier="tier_1",
    role="controller",
    victory_count=0.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_1",
            hp_base=20,
            evasion_base=7,
            e_defense_base=10,
            armor_base=0,
            speed_base=3,
            sensor_range=15,
            save_bonus=10,
            tech_attack=1,
        ),
        scaling=_make_specialist_scaling(),
    ),
    abilities=[],
    gear=[
        NPCGear(weapon_id="npc_flak_cannon"),
        NPCGear(system_id="npc_deployable_turret"),
    ],
)

PR2_PRIEST_T1 = NPCTemplate(
    id="pr2_priest_t1",
    name="Priest",
    description="",
    npc_class="specialist",
    tier="tier_1",
    role="controller",
    victory_count=0.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_half",
            hp_base=10,
            evasion_base=10,
            e_defense_base=12,
            armor_base=0,
            speed_base=5,
            sensor_range=10,
            save_bonus=11,
            tech_attack=2,
        ),
        scaling=_make_specialist_scaling(),
    ),
    abilities=[],
    gear=[
        NPCGear(system_id="npc_abjure"),
        NPCGear(system_id="npc_dispersal_shield"),
        NPCGear(system_id="npc_hardened_target"),
        NPCGear(system_id="npc_investiture"),
    ],
)

PR2_RAINMAKER_T1 = NPCTemplate(
    id="pr2_rainmaker_t1",
    name="Rainmaker",
    description="",
    npc_class="specialist",
    tier="tier_1",
    role="controller",
    victory_count=0.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_1",
            hp_base=10,
            evasion_base=8,
            e_defense_base=8,
            armor_base=1,
            speed_base=3,
            sensor_range=15,
            save_bonus=10,
            tech_attack=2,
        ),
        scaling=_make_specialist_scaling(),
    ),
    abilities=[],
    gear=[
        NPCGear(weapon_id="npc_missile_pods"),
        NPCGear(system_id="npc_javelin_rockets"),
    ],
)

PR2_RONIN_T1 = NPCTemplate(
    id="pr2_ronin_t1",
    name="Ronin",
    description="",
    npc_class="specialist",
    tier="tier_1",
    role="striker",
    victory_count=0.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_1",
            hp_base=18,
            evasion_base=10,
            e_defense_base=7,
            armor_base=0,
            speed_base=5,
            sensor_range=10,
            save_bonus=10,
            tech_attack=0,
        ),
        scaling=_make_specialist_scaling(),
    ),
    abilities=[],
    gear=[
        NPCGear(weapon_id="npc_carbon_fiber_sword"),
        NPCGear(system_id="npc_perfect_parry"),
    ],
)

PR2_SCOURER_T1 = NPCTemplate(
    id="pr2_scourer_t1",
    name="Scourer",
    description="",
    npc_class="specialist",
    tier="tier_1",
    role="striker",
    victory_count=0.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_1",
            hp_base=10,
            evasion_base=8,
            e_defense_base=8,
            armor_base=2,
            speed_base=4,
            sensor_range=10,
            save_bonus=10,
            tech_attack=0,
        ),
        scaling=_make_specialist_scaling(),
    ),
    abilities=[],
    gear=[
        NPCGear(weapon_id="npc_thermal_lance"),
        NPCGear(system_id="npc_focus_down"),
        NPCGear(system_id="npc_cooling_module"),
    ],
)

PR2_SNIPER_T1 = NPCTemplate(
    id="pr2_sniper_t1",
    name="Sniper",
    description="",
    npc_class="specialist",
    tier="tier_1",
    role="striker",
    victory_count=0.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_1",
            hp_base=10,
            evasion_base=10,
            e_defense_base=8,
            armor_base=0,
            speed_base=4,
            sensor_range=15,
            save_bonus=11,
            tech_attack=1,
        ),
        scaling=_make_specialist_scaling(),
    ),
    abilities=[],
    gear=[
        NPCGear(weapon_id="npc_anti_material_rifle"),
        NPCGear(system_id="npc_snipers_mark"),
    ],
)

PR2_SPECTRE_T1 = NPCTemplate(
    id="pr2_spectre_t1",
    name="Spectre",
    description="",
    npc_class="specialist",
    tier="tier_1",
    role="striker",
    victory_count=0.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_1",
            hp_base=10,
            evasion_base=10,
            e_defense_base=8,
            armor_base=0,
            speed_base=4,
            sensor_range=5,
            save_bonus=10,
            tech_attack=1,
        ),
        scaling=_make_specialist_scaling(),
    ),
    abilities=[],
    gear=[
        NPCGear(weapon_id="npc_monowire_sword"),
        NPCGear(system_id="npc_prowl"),
        NPCGear(system_id="npc_tactical_cloak"),
    ],
)

PR2_WITCH_T1 = NPCTemplate(
    id="pr2_witch_t1",
    name="Witch",
    description="",
    npc_class="specialist",
    tier="tier_1",
    role="controller",
    victory_count=0.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_half",
            hp_base=12,
            evasion_base=10,
            e_defense_base=13,
            armor_base=0,
            speed_base=6,
            sensor_range=15,
            save_bonus=12,
            tech_attack=3,
        ),
        scaling=_make_specialist_scaling(),
    ),
    abilities=[],
    gear=[
        NPCGear(system_id="npc_tear_down"),
        NPCGear(system_id="npc_blind"),
        NPCGear(system_id="npc_predatory_logic"),
    ],
)

PR2_AEGIS_T1 = NPCTemplate(
    id="pr2_aegis_t1",
    name="Aegis",
    description="",
    npc_class="specialist",
    tier="tier_1",
    role="defender",
    victory_count=0.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_1",
            hp_base=15,
            evasion_base=8,
            e_defense_base=10,
            armor_base=2,
            speed_base=4,
            sensor_range=8,
            save_bonus=10,
            tech_attack=0,
        ),
        scaling=_make_specialist_scaling(),
    ),
    abilities=[],
    gear=[
        NPCGear(weapon_id="gms_assault_rifle"),
        NPCGear(system_id="gms_shield_type_1"),
        NPCGear(system_id="ipsn_magnetic_shield"),
    ],
)

PR2_BARRICADE_T1 = NPCTemplate(
    id="pr2_barricade_t1",
    name="Barricade",
    description="",
    npc_class="specialist",
    tier="tier_1",
    role="defender",
    victory_count=0.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_1",
            hp_base=18,
            evasion_base=6,
            e_defense_base=8,
            armor_base=3,
            speed_base=3,
            sensor_range=5,
            save_bonus=10,
            tech_attack=0,
        ),
        scaling=_make_specialist_scaling(),
    ),
    abilities=[],
    gear=[
        NPCGear(weapon_id="gms_heavy_kinetic_hammer"),
        NPCGear(system_id="gms_jericho_cover"),
        NPCGear(system_id="ipsn_breaching_charges"),
    ],
)

PR2_BOMBARD_T1 = NPCTemplate(
    id="pr2_bombard_t1",
    name="Bombard",
    description="",
    npc_class="specialist",
    tier="tier_1",
    role="controller",
    victory_count=0.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_1",
            hp_base=12,
            evasion_base=8,
            e_defense_base=10,
            armor_base=0,
            speed_base=4,
            sensor_range=15,
            save_bonus=10,
            tech_attack=2,
        ),
        scaling=_make_specialist_scaling(),
    ),
    abilities=[],
    gear=[
        NPCGear(weapon_id="howitzer"),
        NPCGear(system_id="gms_hex_charges"),
        NPCGear(system_id="ssc_personalized_ecm"),
    ],
)

PR2_GOLIATH_T1 = NPCTemplate(
    id="pr2_goliath_t1",
    name="Goliath",
    description="",
    npc_class="specialist",
    tier="tier_1",
    role="defender",
    victory_count=0.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_2",
            hp_base=25,
            evasion_base=5,
            e_defense_base=6,
            armor_base=3,
            speed_base=2,
            sensor_range=5,
            save_bonus=8,
            tech_attack=0,
        ),
        scaling=_make_specialist_scaling(),
    ),
    abilities=[],
    gear=[
        NPCGear(weapon_id="gms_heavy_kinetic_hammer"),
        NPCGear(system_id="ha_heavy_plating"),
        NPCGear(system_id="ipsn_overpressure_valve"),
    ],
)

PR2_HIVE_T1 = NPCTemplate(
    id="pr2_hive_t1",
    name="Hive",
    description="",
    npc_class="specialist",
    tier="tier_1",
    role="controller",
    victory_count=0.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_1",
            hp_base=14,
            evasion_base=8,
            e_defense_base=10,
            armor_base=0,
            speed_base=5,
            sensor_range=12,
            save_bonus=10,
            tech_attack=1,
        ),
        scaling=_make_specialist_scaling(),
    ),
    abilities=[],
    gear=[
        NPCGear(weapon_id="missile_rack"),
        NPCGear(system_id="horus_hive_drone"),
        NPCGear(system_id="horus_hunter_micronet"),
    ],
)

PR2_HORNET_T1 = NPCTemplate(
    id="pr2_hornet_t1",
    name="Hornet",
    description="",
    npc_class="specialist",
    tier="tier_1",
    role="striker",
    victory_count=0.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_half",
            hp_base=8,
            evasion_base=14,
            e_defense_base=10,
            armor_base=0,
            speed_base=6,
            sensor_range=10,
            save_bonus=10,
            tech_attack=1,
        ),
        scaling=_make_specialist_scaling(),
    ),
    abilities=[],
    gear=[
        NPCGear(weapon_id="needle_rifle"),
        NPCGear(system_id="ssc_evasive_maneuvers"),
        NPCGear(system_id="ssc_personalized_ecm"),
    ],
)

PR2_MIRAGE_T1 = NPCTemplate(
    id="pr2_mirage_t1",
    name="Mirage",
    description="",
    npc_class="specialist",
    tier="tier_1",
    role="striker",
    victory_count=0.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_1",
            hp_base=10,
            evasion_base=12,
            e_defense_base=10,
            armor_base=0,
            speed_base=5,
            sensor_range=10,
            save_bonus=10,
            tech_attack=1,
        ),
        scaling=_make_specialist_scaling(),
    ),
    abilities=[],
    gear=[
        NPCGear(weapon_id="monowire_sword"),
        NPCGear(system_id="ssc_prowl"),
        NPCGear(system_id="ssc_tactical_cloak"),
    ],
)

PR2_OPERATOR_T1 = NPCTemplate(
    id="pr2_operator_t1",
    name="Operator",
    description="",
    npc_class="specialist",
    tier="tier_1",
    role="controller",
    victory_count=0.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_1",
            hp_base=10,
            evasion_base=10,
            e_defense_base=12,
            armor_base=0,
            speed_base=4,
            sensor_range=15,
            save_bonus=11,
            tech_attack=3,
        ),
        scaling=_make_specialist_scaling(),
    ),
    abilities=[],
    gear=[
        NPCGear(system_id="horus_ice_out_drone"),
        NPCGear(system_id="horus_predictive_logic"),
        NPCGear(system_id="ssc_black_witch_light"),
    ],
)

PR2_PYRO_T1 = NPCTemplate(
    id="pr2_pyro_t1",
    name="Pyro",
    description="",
    npc_class="specialist",
    tier="tier_1",
    role="striker",
    victory_count=0.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_1",
            hp_base=12,
            evasion_base=8,
            e_defense_base=8,
            armor_base=0,
            speed_base=5,
            sensor_range=8,
            save_bonus=10,
            tech_attack=1,
        ),
        scaling=_make_specialist_scaling(),
    ),
    abilities=[],
    gear=[
        NPCGear(weapon_id="charged_blade"),
        NPCGear(system_id="ha_cooling_module"),
        NPCGear(system_id="ha_focus_down"),
    ],
)

PR2_SCOUT_T1 = NPCTemplate(
    id="pr2_scout_t1",
    name="Scout",
    description="",
    npc_class="specialist",
    tier="tier_1",
    role="striker",
    victory_count=0.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_1",
            hp_base=10,
            evasion_base=12,
            e_defense_base=8,
            armor_base=0,
            speed_base=6,
            sensor_range=20,
            save_bonus=10,
            tech_attack=1,
        ),
        scaling=_make_specialist_scaling(),
    ),
    abilities=[],
    gear=[
        NPCGear(weapon_id="sniper_rifle"),
        NPCGear(system_id="gms_sensor_boost"),
        NPCGear(system_id="gms_type_1_flight_system"),
    ],
)

PR2_SEEDER_T1 = NPCTemplate(
    id="pr2_seeder_t1",
    name="Seeder",
    description="",
    npc_class="specialist",
    tier="tier_1",
    role="controller",
    victory_count=0.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_1",
            hp_base=12,
            evasion_base=8,
            e_defense_base=10,
            armor_base=1,
            speed_base=4,
            sensor_range=12,
            save_bonus=10,
            tech_attack=1,
        ),
        scaling=_make_specialist_scaling(),
    ),
    abilities=[],
    gear=[
        NPCGear(weapon_id="mortar"),
        NPCGear(system_id="gms_flash_mine"),
        NPCGear(system_id="gms_pattern_bolt"),
    ],
)

PR2_SENTINEL_T1 = NPCTemplate(
    id="pr2_sentinel_t1",
    name="Sentinel",
    description="",
    npc_class="specialist",
    tier="tier_1",
    role="defender",
    victory_count=0.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_1",
            hp_base=16,
            evasion_base=8,
            e_defense_base=10,
            armor_base=2,
            speed_base=4,
            sensor_range=8,
            save_bonus=11,
            tech_attack=0,
        ),
        scaling=_make_specialist_scaling(),
    ),
    abilities=[],
    gear=[
        NPCGear(weapon_id="rotary_grenade_launcher"),
        NPCGear(system_id="gms_shield_type_1"),
        NPCGear(system_id="ipsn_magnetic_shield"),
    ],
)

PR2_SUPPORT_T1 = NPCTemplate(
    id="pr2_support_t1",
    name="Support",
    description="",
    npc_class="specialist",
    tier="tier_1",
    role="supporter",
    victory_count=0.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_1",
            hp_base=14,
            evasion_base=8,
            e_defense_base=11,
            armor_base=1,
            speed_base=4,
            sensor_range=12,
            save_bonus=12,
            tech_attack=2,
        ),
        scaling=_make_specialist_scaling(),
    ),
    abilities=[],
    gear=[
        NPCGear(system_id="ssc_dispersal_shield"),
        NPCGear(system_id="ssc_investiture"),
        NPCGear(system_id="gms_nano_patch"),
    ],
)

PR2_NPC_TEMPLATES: list[NPCTemplate] = [
    PR2_ACE_T1,
    PR2_AEGIS_T1,
    PR2_ASSAULT_T1,
    PR2_ARCHER_T1,
    PR2_ASSASSIN_T1,
    PR2_BARRICADE_T1,
    PR2_BASTION_T1,
    PR2_BERSERKER_T1,
    PR2_BOMBARD_T1,
    PR2_BREACHER_T1,
    PR2_CATAPHRACT_T1,
    PR2_DEMOLISHER_T1,
    PR2_ENGINEER_T1,
    PR2_GOLIATH_T1,
    PR2_HIVE_T1,
    PR2_HORNET_T1,
    PR2_MIRAGE_T1,
    PR2_OPERATOR_T1,
    PR2_PRIEST_T1,
    PR2_PYRO_T1,
    PR2_RAINMAKER_T1,
    PR2_RONIN_T1,
    PR2_SCOUT_T1,
    PR2_SCOURER_T1,
    PR2_SEEDER_T1,
    PR2_SENTINEL_T1,
    PR2_SNIPER_T1,
    PR2_SPECTRE_T1,
    PR2_SUPPORT_T1,
    PR2_WITCH_T1,
]

NPC_TEMPLATES: list[NPCTemplate] = [
    GMS_GRUNT_T1,
    GMS_GRUNT_T2,
    GMS_ELITE_T2,
    IPSN_GRUNT_T1,
    IPSN_BOSS_T3,
    SSC_SPECIALIST_T2,
    HORUS_ELITE_T3,
    HA_BOSS_T3,
    *PR2_NPC_TEMPLATES,
]


NPC_TEMPLATES_BY_ID: dict[str, NPCTemplate] = {
    template.id: template for template in NPC_TEMPLATES
}

NPC_SPECIAL_CLASSES_BY_ID: dict[str, special_classes.SpecialNPCTemplate] = {
    template.id: template for template in special_classes.NPC_SPECIAL_CLASSES
}


def get_npc_template(template_id: str) -> NPCTemplate | None:
    """Get an NPC template by ID.

    Args:
        template_id: The unique identifier for the template

    Returns:
        The template if found, None otherwise
    """
    return NPC_TEMPLATES_BY_ID.get(template_id)


def get_special_class_template(
    template_id: str,
) -> special_classes.SpecialNPCTemplate | None:
    """Get a special NPC class template by ID.

    Args:
        template_id: The unique identifier for the template

    Returns:
        The special class template if found, None otherwise
    """
    return NPC_SPECIAL_CLASSES_BY_ID.get(template_id)


def get_any_template(
    template_id: str,
) -> NPCTemplate | special_classes.SpecialNPCTemplate | None:
    """Get any NPC template (regular or special class) by ID.

    Args:
        template_id: The unique identifier for the template

    Returns:
        The template if found, None otherwise
    """
    template = get_npc_template(template_id)
    if template is not None:
        return template
    return get_special_class_template(template_id)


def get_templates_by_class(npc_class: NPCClass) -> list[NPCTemplate]:
    """Get all NPC templates of a specific class.

    Args:
        npc_class: The class to filter by

    Returns:
        List of templates with the specified class
    """
    return [t for t in NPC_TEMPLATES if t.npc_class == npc_class]


def get_templates_by_tier(tier: NPCTier) -> list[NPCTemplate]:
    """Get all NPC templates of a specific tier.

    Args:
        tier: The tier to filter by

    Returns:
        List of templates with the specified tier
    """
    return [t for t in NPC_TEMPLATES if t.tier == tier]


def get_all_special_classes() -> list[special_classes.SpecialNPCTemplate]:
    """Get all special NPC class templates.

    Returns:
        List of all 15 special class templates
    """
    return list(special_classes.NPC_SPECIAL_CLASSES)
