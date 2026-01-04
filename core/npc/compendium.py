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
            trigger="on_damaged",
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
            trigger="on_damaged",
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


NPC_TEMPLATES: list[NPCTemplate] = [
    GMS_GRUNT_T1,
    GMS_GRUNT_T2,
    GMS_ELITE_T2,
    IPSN_GRUNT_T1,
    IPSN_BOSS_T3,
    SSC_SPECIALIST_T2,
    HORUS_ELITE_T3,
    HA_BOSS_T3,
]


NPC_TEMPLATES_BY_ID: dict[str, NPCTemplate] = {
    template.id: template for template in NPC_TEMPLATES
}


def get_npc_template(template_id: str) -> NPCTemplate | None:
    """Get an NPC template by ID.

    Args:
        template_id: The unique identifier for the template

    Returns:
        The template if found, None otherwise
    """
    return NPC_TEMPLATES_BY_ID.get(template_id)


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
