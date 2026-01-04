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
