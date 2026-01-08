"""NPC-specific enumerations and literal types for typed Lancer mechanics."""

from typing import Literal


NPCTier = Literal[
    "tier_1",
    "tier_2",
    "tier_3",
]


NPCClass = Literal[
    "grunt",
    "elite",
    "boss",
    "specialist",
]


NPCAbilityTriggerType = Literal[
    "on_hit",
    "on_miss",
    "on_crit",
    "on_kill",
    "on_turn_start",
    "on_turn_end",
    "on_damaged",
    "on_attacked",
    "on_adjacent",
    "on_deploy",
    "on_destroyed",
    "on Initiative",
    "on_ally_killed",
    "on_hp_below_half",
    "on_damage_dealt",
]


NPCTierScalingType = Literal[
    "multiplier",
    "adder",
    "fixed_override",
]


NPCSpecialClass = Literal[
    "human",
    "infantry_squad",
    "monstrosity",
    "ultra",
    "elite",
    "grunt",
    "veteran",
    "exotic",
    "drone",
    "mercenary",
    "commander",
    "pirate",
    "spacer",
    "vehicle",
    "ship",
]


UltraTraitType = Literal[
    "berserker",
    "devastator",
    "evasive",
    "extra_deadly",
    "fortress",
    "legion",
    "limitless",
    "unstoppable",
    "sight",
    "superior_construction",
    "superior_frame",
    "superior_reactor",
    "superior_targeting",
    "supreme_maintenance",
    "supreme_skirmisher",
]


VeteranTraitType = Literal[
    "nhp_copilot",
    "acrobat",
    "deadly",
    "insulated",
    "self_repair",
    "feign_death",
    "hacker",
    "headshot",
    "hardened_target",
    "legendary",
    "lesser_sight",
    "limitless",
    "lightning_reflexes",
    "parting_gift",
    "rodeo_master",
    "shock_armor",
    "skirmisher",
    "slippery",
    "steel_jaw",
    "vipers_speed",
]


ExoticModuleType = Literal[
    "bio_integrated",
    "blinkspace_carver",
    "extrusion",
    "living_weaponry",
    "paracausal_weapon",
    "ouroboros_brand",
    "regenerator",
]


CommanderTraitType = Literal[
    "bolster_network",
    "retribution",
    "press_on",
    "reposition",
    "rank_and_file",
]


VehicleType = Literal[
    "flier",
    "transport",
    "treads",
    "hover",
]
