"""Special NPC class templates for typed Lancer mechanics.

This module provides 15 special NPC class templates from PR2 459-480:
- Human: Simple human-scale enemies
- Infantry Squad: Squad-level groups with aggregation rules
- Monstrosity: Massive wildlife creatures
- Ultra: Boss-tier enemies with special traits
- Elite: Enhanced grunts with structure checks
- Grunt: Basic mass-produced enemies
- Veteran: Experienced fighters with bonus traits
- Exotic: Strange enemies with unique tech
- Drone: Autonomous mechs with no pilot
- Mercenary: Professional soldiers for hire
- Commander: Leaders with command abilities
- Pirate: Raiders from the fringe
- Spacer: Zero-G specialists
- Vehicle: In-atmosphere vehicles
- Ship: Space-capable vessels

Each class has a victory_count field for SITREP resolution:
- Human: 1.0
- Infantry Squad: 0.25 (4 squads = 1 victory)
- Monstrosity: 4.0
- Ultra: 4.0
- Elite: 0.5
- Grunt: 0.25
- Veteran: 0.25
- Exotic: 1.0 (configurable)
- Drone: 0.5
- Mercenary: 1.0
- Commander: 2.0
- Pirate: 0.5
- Spacer: 1.0
- Vehicle: 1.5
- Ship: 8.0
"""

from core.npc.models import (
    NPCStats,
    NPCStatsBase,
    NPCTierScaling,
    SpecialNPCTemplate,
    UltraTrait,
    VeteranTrait,
    ExoticModule,
    CommanderTrait,
    InfantrySquadStats,
)
from core.npc.enums import (
    NPCSpecialClass,
)
from core.shared.effects import (
    MechanicalEffect,
    Resistance,
)


def _make_ultra_scaling() -> NPCTierScaling:
    """Ultra tier scaling: +5 HP bonus, structure=4, stress=4."""
    return NPCTierScaling(
        hp_multiplier=1.0,
        hp_adder_tier_2=15,
        hp_adder_tier_3=30,
        evasion_adder_tier_2=2,
        evasion_adder_tier_3=3,
        e_defense_adder_tier_2=2,
        e_defense_adder_tier_3=3,
        armor_adder_tier_2=1,
        armor_adder_tier_3=2,
        save_adder_tier_2=2,
        save_adder_tier_3=3,
    )


def _make_elite_scaling() -> NPCTierScaling:
    """Elite tier scaling: structure=2, stress=2."""
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


def _make_veteran_scaling() -> NPCTierScaling:
    """Veteran tier scaling: +1 structure, +1 stress."""
    return NPCTierScaling(
        hp_multiplier=1.0,
        hp_adder_tier_2=12,
        hp_adder_tier_3=24,
        evasion_adder_tier_2=1,
        evasion_adder_tier_3=2,
        e_defense_adder_tier_2=1,
        e_defense_adder_tier_3=2,
        armor_adder_tier_2=0,
        armor_adder_tier_3=1,
        save_adder_tier_2=1,
        save_adder_tier_3=2,
    )


def _make_drone_scaling() -> NPCTierScaling:
    """Drone tier scaling: +5 HP from no pilot."""
    return NPCTierScaling(
        hp_multiplier=1.0,
        hp_adder_tier_2=12,
        hp_adder_tier_3=24,
        evasion_adder_tier_2=1,
        evasion_adder_tier_3=2,
        e_defense_adder_tier_2=1,
        e_defense_adder_tier_3=2,
        armor_adder_tier_2=0,
        armor_adder_tier_3=1,
        save_adder_tier_2=1,
        save_adder_tier_3=2,
    )


def _make_vehicle_scaling() -> NPCTierScaling:
    """Vehicle tier scaling."""
    return NPCTierScaling(
        hp_multiplier=1.2,
        hp_adder_tier_2=15,
        hp_adder_tier_3=30,
        evasion_adder_tier_2=1,
        evasion_adder_tier_3=2,
        e_defense_adder_tier_2=0,
        e_defense_adder_tier_3=1,
        armor_adder_tier_2=1,
        armor_adder_tier_3=2,
        save_adder_tier_2=1,
        save_adder_tier_3=2,
    )


def _make_ship_scaling() -> NPCTierScaling:
    """Ship tier scaling: +5 HP, min size 4."""
    return NPCTierScaling(
        hp_multiplier=1.5,
        hp_adder_tier_2=25,
        hp_adder_tier_3=50,
        evasion_adder_tier_2=1,
        evasion_adder_tier_3=2,
        e_defense_adder_tier_2=1,
        e_defense_adder_tier_3=2,
        armor_adder_tier_2=2,
        armor_adder_tier_3=3,
        save_adder_tier_2=2,
        save_adder_tier_3=3,
    )


SPECIAL_HUMAN = SpecialNPCTemplate(
    id="special_human_t1",
    name="Human",
    description="Human-scale enemies such as pilots or infantry.",
    npc_class="grunt",
    tier="tier_1",
    role="striker",
    special_class="human",
    victory_count=1.0,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_half",
            hp_base=5,
            evasion_base=10,
            e_defense_base=10,
            armor_base=0,
            speed_base=4,
            sensor_range=5,
            save_bonus=0,
        ),
        scaling=NPCTierScaling(),
    ),
    tags=["biological"],
)


SPECIAL_INFANTRY_SQUAD_T1 = SpecialNPCTemplate(
    id="special_infantry_squad_t1",
    name="Infantry Squad",
    description="Squad-level group of 5-10 soldiers. Cannot have more than 1 structure. Weak to external heat.",
    npc_class="grunt",
    tier="tier_1",
    role="controller",
    special_class="infantry_squad",
    victory_count=0.25,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_1",
            hp_base=20,
            evasion_base=8,
            e_defense_base=8,
            armor_base=0,
            speed_base=4,
            sensor_range=10,
            save_bonus=0,
        ),
        scaling=NPCTierScaling(),
    ),
    tags=["infantry", "biological"],
    infantry_squad_stats=InfantrySquadStats(squad_members=5, members_destroyed=0),
    structure_override=1,
    effects=MechanicalEffect(
        resistances=[Resistance(damage_type="all", applies_to="self", condition=None)],
    ),
)


SPECIAL_MONSTROSITY_T1 = SpecialNPCTemplate(
    id="special_monstrosity_t1",
    name="Monstrosity",
    description="Massive or horrifying predatory wildlife.",
    npc_class="boss",
    tier="tier_1",
    role="striker",
    special_class="monstrosity",
    victory_count=4.0,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_2",
            hp_base=30,
            evasion_base=10,
            e_defense_base=10,
            armor_base=2,
            speed_base=4,
            sensor_range=10,
            save_bonus=1,
        ),
        scaling=NPCTierScaling(
            hp_multiplier=1.5,
            hp_adder_tier_2=20,
            hp_adder_tier_3=40,
            armor_adder_tier_3=1,
        ),
    ),
    tags=["biological"],
)


SPECIAL_ULTRA_T1 = SpecialNPCTemplate(
    id="special_ultra_t1",
    name="Ultra",
    description="Boss-tier enemy. +5 HP, Juggernaut, Legendary (double-roll structure/stress), structure=4, stress=4.",
    npc_class="boss",
    tier="tier_1",
    role="striker",
    special_class="ultra",
    victory_count=4.0,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_2",
            hp_base=20,
            evasion_base=10,
            e_defense_base=10,
            armor_base=2,
            speed_base=4,
            sensor_range=10,
            save_bonus=1,
        ),
        scaling=_make_ultra_scaling(),
    ),
    tags=["ultra"],
    structure_override=4,
    stress_override=4,
    bonus_hp=5,
    ultra_traits=[
        UltraTrait(
            trait_type="berserker",
            description="+1 Accuracy on all melee attacks. Free melee/ram/grapple 1x/turn.",
        ),
        UltraTrait(
            trait_type="devastator",
            description="1/round when hit, all visible targets take 2/4/6 damage.",
        ),
        UltraTrait(
            trait_type="evasive",
            description="+4 evasion (max 20), but reduce structure to 3.",
        ),
        UltraTrait(
            trait_type="extra_deadly",
            description="First Critical Hit per turn does +1d6 bonus damage/tier.",
        ),
        UltraTrait(
            trait_type="fortress",
            description="Tech attacks +3 difficulty, systems saves +3 accuracy.",
        ),
        UltraTrait(
            trait_type="legion",
            description="+4 e-defense (max 20), +1 tech attack accuracy.",
        ),
        UltraTrait(
            trait_type="limitless",
            description="Can overcharge. Cost always 1d6 heat.",
        ),
        UltraTrait(
            trait_type="unstoppable",
            description="Immune to knockback, prone, and involuntary movement.",
        ),
        UltraTrait(
            trait_type="sight",
            description="No hiding in sensor range, ignores invisibility.",
        ),
        UltraTrait(
            trait_type="superior_construction",
            description="Resistance to one damage type. Loses +5 HP.",
        ),
        UltraTrait(
            trait_type="superior_frame",
            description="Immune to Slowed, Shredded, and Immobilized.",
        ),
        UltraTrait(
            trait_type="superior_reactor",
            description="Immune to Stunned and Exposed.",
        ),
        UltraTrait(
            trait_type="superior_targeting",
            description="Ignores cover when making ranged attacks.",
        ),
        UltraTrait(
            trait_type="supreme_maintenance",
            description="Immune to Jammed, free reload/repair.",
        ),
        UltraTrait(
            trait_type="supreme_skirmisher",
            description="Unlimited boost reactions.",
        ),
    ],
)


SPECIAL_ELITE_T1 = SpecialNPCTemplate(
    id="special_elite_t1",
    name="Elite",
    description="Enhanced grunt. structure=2, stress=2, 2 activations/round, can Critical Hit.",
    npc_class="elite",
    tier="tier_1",
    role="striker",
    special_class="elite",
    victory_count=0.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_1",
            hp_base=15,
            evasion_base=10,
            e_defense_base=10,
            armor_base=1,
            speed_base=5,
            sensor_range=10,
            save_bonus=1,
        ),
        scaling=_make_elite_scaling(),
    ),
    tags=["elite"],
    structure_override=2,
    stress_override=2,
)


SPECIAL_GRUNT_T1 = SpecialNPCTemplate(
    id="special_grunt_t1",
    name="Grunt",
    description="Mass-produced enemy. 1 HP, 1 structure, 1 stress. Destroyed by external heat.",
    npc_class="grunt",
    tier="tier_1",
    role="striker",
    special_class="grunt",
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
        scaling=NPCTierScaling(),
    ),
    tags=["grunt"],
    structure_override=1,
    stress_override=1,
)


SPECIAL_VETERAN_T1 = SpecialNPCTemplate(
    id="special_veteran_t1",
    name="Veteran",
    description="Experienced fighter. +1 structure, +1 stress, +1 accuracy on chosen stat.",
    npc_class="elite",
    tier="tier_1",
    role="striker",
    special_class="veteran",
    victory_count=0.25,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_1",
            hp_base=15,
            evasion_base=10,
            e_defense_base=10,
            armor_base=1,
            speed_base=5,
            sensor_range=10,
            save_bonus=1,
        ),
        scaling=_make_veteran_scaling(),
    ),
    tags=["veteran"],
    structure_override=2,
    stress_override=2,
    veteran_traits=[
        VeteranTrait(
            trait_type="deadly",
            description="Can Critical Hit, dealing +1d6 bonus damage on Critical Hit.",
        ),
        VeteranTrait(
            trait_type="hardened_target",
            description="Tech attacks +1 difficulty, systems saves +1 accuracy.",
        ),
        VeteranTrait(
            trait_type="limitless",
            description="Can use Overcharge. Cost always 1d6 heat.",
        ),
        VeteranTrait(
            trait_type="self_repair",
            description="1/scene: Full Action to heal to full HP and end all conditions.",
        ),
        VeteranTrait(
            trait_type="skirmisher",
            description="1/round can move or boost as reaction to enemy movement.",
        ),
    ],
)


SPECIAL_EXOTIC_T1 = SpecialNPCTemplate(
    id="special_exotic_t1",
    name="Exotic",
    description="Strange enemies with unique technology. Xenotech, Hardened Target.",
    npc_class="specialist",
    tier="tier_1",
    role="controller",
    special_class="exotic",
    victory_count=1.0,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_1",
            hp_base=12,
            evasion_base=10,
            e_defense_base=12,
            armor_base=0,
            speed_base=5,
            sensor_range=15,
            save_bonus=1,
        ),
        scaling=NPCTierScaling(),
    ),
    tags=["exotic"],
    exotic_modules=[
        ExoticModule(
            module_type="bio_integrated",
            description="Mech gains biological tag, loses heat capacity.",
        ),
        ExoticModule(
            module_type="blinkspace_carver",
            description="When the NPC moves, it teleports.",
        ),
        ExoticModule(
            module_type="extrusion",
            description="Partial entity, resistance to all damage but half weapon damage.",
        ),
        ExoticModule(
            module_type="living_weaponry",
            description="Immune to Jammed, regenerates ammo, heals 1d6 HP on reload.",
        ),
        ExoticModule(
            module_type="paracausal_weapon",
            description="One weapon's damage cannot be reduced.",
        ),
        ExoticModule(
            module_type="ouroboros_brand",
            description="1/round can force a re-roll of any d20.",
        ),
        ExoticModule(
            module_type="regenerator",
            description="Heals 1/4 HP at end of turn. Doesn't function if took energy damage.",
        ),
    ],
)


SPECIAL_DRONE_T1 = SpecialNPCTemplate(
    id="special_drone_t1",
    name="Drone",
    description="Autonomous mech. +5 HP, Impaired condition, vulnerable to tech.",
    npc_class="specialist",
    tier="tier_1",
    role="striker",
    special_class="drone",
    victory_count=0.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_half",
            hp_base=10,
            evasion_base=10,
            e_defense_base=10,
            armor_base=0,
            speed_base=5,
            sensor_range=10,
            save_bonus=0,
        ),
        scaling=_make_drone_scaling(),
    ),
    tags=["drone"],
    bonus_hp=5,
)


SPECIAL_MERCENARY_T1 = SpecialNPCTemplate(
    id="special_mercenary_t1",
    name="Mercenary",
    description="Professional soldier for hire. Opportunist bonus, special systems.",
    npc_class="elite",
    tier="tier_1",
    role="striker",
    special_class="mercenary",
    victory_count=1.0,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_1",
            hp_base=15,
            evasion_base=10,
            e_defense_base=10,
            armor_base=1,
            speed_base=5,
            sensor_range=10,
            save_bonus=1,
        ),
        scaling=_make_elite_scaling(),
    ),
    tags=["mercenary"],
)


SPECIAL_COMMANDER_T1 = SpecialNPCTemplate(
    id="special_commander_t1",
    name="Commander",
    description="Fleet/army commander. +1 structure, +1 stress, Command reaction.",
    npc_class="boss",
    tier="tier_1",
    role="controller",
    special_class="commander",
    victory_count=2.0,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_2",
            hp_base=20,
            evasion_base=8,
            e_defense_base=10,
            armor_base=2,
            speed_base=4,
            sensor_range=15,
            save_bonus=2,
        ),
        scaling=NPCTierScaling(
            hp_multiplier=1.2,
            hp_adder_tier_2=20,
            hp_adder_tier_3=40,
            save_adder_tier_3=1,
        ),
    ),
    tags=["commander"],
    structure_override=2,
    stress_override=2,
    commander_traits=[
        CommanderTrait(
            trait_type="bolster_network",
            description="Allies gain Hardened Target trait while commander is alive.",
        ),
        CommanderTrait(
            trait_type="retribution",
            description="Reaction: 1/round attack against character that damaged ally.",
        ),
        CommanderTrait(
            trait_type="press_on",
            description="Quick Action, recharge 4+: End Stunned/Jammed on ally.",
        ),
        CommanderTrait(
            trait_type="reposition",
            description="Reaction: 1/round ally can boost as reaction to commander's turn.",
        ),
        CommanderTrait(
            trait_type="rank_and_file",
            description="Allies adjacent to commander gain +1 Accuracy on all attacks.",
        ),
    ],
)


SPECIAL_PIRATE_T1 = SpecialNPCTemplate(
    id="special_pirate_t1",
    name="Pirate",
    description="Raider from the fringe. Deadly (+1d6 Critical), special modules.",
    npc_class="grunt",
    tier="tier_1",
    role="striker",
    special_class="pirate",
    victory_count=0.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_1",
            hp_base=12,
            evasion_base=10,
            e_defense_base=8,
            armor_base=0,
            speed_base=5,
            sensor_range=10,
            save_bonus=1,
        ),
        scaling=NPCTierScaling(),
    ),
    tags=["pirate"],
)


SPECIAL_SPACER_T1 = SpecialNPCTemplate(
    id="special_spacer_t1",
    name="Spacer",
    description="Zero-G specialist. Never slowed in zero-G, special modules.",
    npc_class="grunt",
    tier="tier_1",
    role="striker",
    special_class="spacer",
    victory_count=1.0,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_1",
            hp_base=12,
            evasion_base=10,
            e_defense_base=10,
            armor_base=0,
            speed_base=5,
            sensor_range=12,
            save_bonus=1,
        ),
        scaling=NPCTierScaling(),
    ),
    tags=["spacer"],
)


SPECIAL_VEHICLE_T1 = SpecialNPCTemplate(
    id="special_vehicle_t1",
    name="Vehicle",
    description="In-atmosphere military/civilian vehicle. Limited maneuverability, crew.",
    npc_class="boss",
    tier="tier_1",
    role="defender",
    special_class="vehicle",
    victory_count=1.5,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_2",
            hp_base=25,
            evasion_base=6,
            e_defense_base=8,
            armor_base=2,
            speed_base=4,
            sensor_range=10,
            save_bonus=1,
        ),
        scaling=_make_vehicle_scaling(),
    ),
    tags=["vehicle"],
    vehicle_type=["flier", "transport", "treads", "hover"],
)


SPECIAL_SHIP_T1 = SpecialNPCTemplate(
    id="special_ship_t1",
    name="Ship",
    description="Space-capable vessel. Min size 4, +5 HP, no manipulators, limited melee.",
    npc_class="boss",
    tier="tier_1",
    role="defender",
    special_class="ship",
    victory_count=8.0,
    stats=NPCStats(
        base=NPCStatsBase(
            size="size_4",
            hp_base=40,
            evasion_base=4,
            e_defense_base=6,
            armor_base=4,
            speed_base=3,
            sensor_range=20,
            save_bonus=2,
        ),
        scaling=_make_ship_scaling(),
    ),
    tags=["ship", "vehicle"],
    bonus_hp=5,
)


NPC_SPECIAL_CLASSES: list[SpecialNPCTemplate] = [
    SPECIAL_HUMAN,
    SPECIAL_INFANTRY_SQUAD_T1,
    SPECIAL_MONSTROSITY_T1,
    SPECIAL_ULTRA_T1,
    SPECIAL_ELITE_T1,
    SPECIAL_GRUNT_T1,
    SPECIAL_VETERAN_T1,
    SPECIAL_EXOTIC_T1,
    SPECIAL_DRONE_T1,
    SPECIAL_MERCENARY_T1,
    SPECIAL_COMMANDER_T1,
    SPECIAL_PIRATE_T1,
    SPECIAL_SPACER_T1,
    SPECIAL_VEHICLE_T1,
    SPECIAL_SHIP_T1,
]


VICTORY_COUNTS: dict[NPCSpecialClass, float] = {
    "human": 1.0,
    "infantry_squad": 0.25,
    "monstrosity": 4.0,
    "ultra": 4.0,
    "elite": 0.5,
    "grunt": 0.25,
    "veteran": 0.25,
    "exotic": 1.0,
    "drone": 0.5,
    "mercenary": 1.0,
    "commander": 2.0,
    "pirate": 0.5,
    "spacer": 1.0,
    "vehicle": 1.5,
    "ship": 8.0,
}


def get_special_class_template(
    special_class: NPCSpecialClass,
) -> SpecialNPCTemplate | None:
    """Get a special NPC class template by type.

    Args:
        special_class: The type of special class to look up

    Returns:
        The template if found, None otherwise
    """
    for template in NPC_SPECIAL_CLASSES:
        if template.special_class == special_class:
            return template
    return None


def get_special_class_by_id(template_id: str) -> SpecialNPCTemplate | None:
    """Get a special NPC class template by ID.

    Args:
        template_id: The unique identifier for the template

    Returns:
        The template if found, None otherwise
    """
    for template in NPC_SPECIAL_CLASSES:
        if template.id == template_id:
            return template
    return None


def get_victory_count(special_class: NPCSpecialClass) -> float:
    """Get the victory count for a special class type.

    Args:
        special_class: The type of special class

    Returns:
        The victory count value (0.25-8.0)
    """
    return VICTORY_COUNTS.get(special_class, 1.0)


def calculate_victory_points_from_templates(
    templates: list[SpecialNPCTemplate],
) -> float:
    """Calculate total victory points from a list of special class templates.

    Args:
        templates: List of special NPC class templates

    Returns:
        Total victory points
    """
    total = 0.0
    for template in templates:
        total += template.victory_count
    return total


def get_ultra_traits() -> list[UltraTrait]:
    """Get all available Ultra traits.

    Returns:
        List of all Ultra trait definitions
    """
    ultra = get_special_class_template("ultra")
    return ultra.ultra_traits if ultra else []


def get_veteran_traits() -> list[VeteranTrait]:
    """Get all available Veteran traits.

    Returns:
        List of all Veteran trait definitions
    """
    veteran = get_special_class_template("veteran")
    return veteran.veteran_traits if veteran else []


def get_exotic_modules() -> list[ExoticModule]:
    """Get all available Exotic modules.

    Returns:
        List of all Exotic module definitions
    """
    exotic = get_special_class_template("exotic")
    return exotic.exotic_modules if exotic else []


def get_commander_traits() -> list[CommanderTrait]:
    """Get all available Commander traits.

    Returns:
        List of all Commander trait definitions
    """
    commander = get_special_class_template("commander")
    return commander.commander_traits if commander else []
