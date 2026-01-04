"""NPC model definitions for typed Lancer mechanics.

This module provides schemas for NPC templates, stats, abilities,
and gear. NPCs use tier-based scaling and reuse player compendium
gear without license requirements.
"""

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.enums import SizeClass
from core.shared.effects import MechanicalEffect
from core.npc.enums import NPCTier, NPCClass, NPCAbilityTriggerType


class NPCStatsBase(FrozenModel):
    """Base stats for an NPC at tier 1 (baseline).

    NPCs use simplified stats compared to player mechs.
    Stats are scaled by tier using multipliers and adders.
    """

    size: SizeClass = "size_1"
    hp_base: int = Field(default=10, ge=1)
    evasion_base: int = Field(default=8, ge=0)
    e_defense_base: int = Field(default=8, ge=0)
    armor_base: int = Field(default=0, ge=0)
    speed_base: int = Field(default=4, ge=0)
    sensor_range: int = Field(default=10, ge=0)
    save_bonus: int = Field(default=0, ge=0)
    tech_attack: int = Field(default=0, ge=0)


class NPCTierScaling(FrozenModel):
    """Per-stat tier scaling rules for NPCs.

    Lancer NPCs scale using both multipliers and per-tier adders.
    - Multipliers apply to HP and damage values
    - Adders apply to defense values (evasion, e-defense, armor)
    """

    hp_multiplier: float = 1.0
    hp_adder_tier_2: int = 0
    hp_adder_tier_3: int = 0
    evasion_adder_tier_2: int = 1
    evasion_adder_tier_3: int = 2
    e_defense_adder_tier_2: int = 1
    e_defense_adder_tier_3: int = 2
    armor_adder_tier_2: int = 0
    armor_adder_tier_3: int = 1
    save_adder_tier_2: int = 1
    save_adder_tier_3: int = 2
    speed_adder_tier_2: int = 0
    speed_adder_tier_3: int = 0
    sensor_adder_tier_2: int = 0
    sensor_adder_tier_3: int = 5


class NPCStats(FrozenModel):
    """Complete NPC stats with tier scaling configuration.

    Contains both base stats (at tier 1) and scaling rules for
    tier 2 and tier 3.
    """

    base: NPCStatsBase
    scaling: NPCTierScaling = Field(default_factory=NPCTierScaling)


class NPCAbility(FrozenModel):
    """NPC special ability using MechanicalEffect for consistency.

    Abilities can trigger on various combat events and may have
    limited uses per combat.
    """

    id: str
    name: str
    trigger: NPCAbilityTriggerType
    effect: MechanicalEffect = Field(default_factory=MechanicalEffect)
    uses_per_combat: int | None = None  # None = unlimited


class NPCGear(FrozenModel):
    """Simplified gear slot for NPCs.

    NPCs can use player compendium weapons/systems without
    license requirements.
    """

    weapon_id: str | None = None
    system_id: str | None = None
    effect: MechanicalEffect = Field(default_factory=MechanicalEffect)


class NPCTemplate(FrozenModel):
    """Template for creating NPCs with class, tier, and abilities.

    Templates define the complete mechanical profile of an NPC type.
    Multiple instances can be created from a single template.
    """

    id: str
    name: str
    description: str = ""
    npc_class: NPCClass
    tier: NPCTier = "tier_1"
    stats: NPCStats
    abilities: list[NPCAbility] = Field(default_factory=list)
    gear: list[NPCGear] = Field(default_factory=list)
    effects: MechanicalEffect = Field(default_factory=MechanicalEffect)
    tags: list[str] = Field(default_factory=list)


class NPCTemplateSet(FrozenModel):
    """Collection of NPC templates for a manufacturer/faction."""

    manufacturer: str
    templates: list[NPCTemplate] = Field(default_factory=list)
