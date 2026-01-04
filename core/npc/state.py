"""NPC state definitions for typed Lancer mechanics.

This module provides NPC-specific combat state including stat scaling
by tier and integration with the broader combat system.
"""

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.enums import SizeClass
from core.npc.enums import NPCTier, NPCClass
from core.npc.models import (
    NPCTemplate,
    NPCStats,
    NPCStatsBase,
    NPCTierScaling,
)


class NPCCombatStats(FrozenModel):
    """Computed combat-ready stats for an NPC instance.

    These are the final stats after tier scaling has been applied.
    """

    size: SizeClass
    hp_max: int = Field(..., ge=0)
    evasion: int = Field(..., ge=0)
    e_defense: int = Field(..., ge=0)
    armor: int = Field(default=0, ge=0)
    speed: int = Field(default=0, ge=0)
    sensor_range: int = Field(default=0, ge=0)
    tech_attack: int = Field(default=0)
    save_bonus: int = Field(default=0)


def scale_npc_stats(
    stats: NPCStats,
    tier: NPCTier,
) -> NPCCombatStats:
    """Scale NPC base stats by tier using multipliers and adders.

    Lancer NPC scaling:
    - HP: multiplier applied first, then tier adder
    - Defense (evasion, e-defense): base + tier adder
    - Armor: base + tier adder (tier 3 only)
    - Save bonus: base + tier adder

    Args:
        stats: The NPC base stats and scaling configuration
        tier: The tier to scale to

    Returns:
        Combat-ready stats with tier scaling applied
    """
    base = stats.base
    scaling = stats.scaling

    if tier == "tier_1":
        hp_multiplier = 1.0
        hp_adder = 0
        eva_adder = 0
        edef_adder = 0
        armor_adder = 0
        save_adder = 0
        speed_adder = 0
        sensor_adder = 0
    elif tier == "tier_2":
        hp_multiplier = scaling.hp_multiplier
        hp_adder = scaling.hp_adder_tier_2
        eva_adder = scaling.evasion_adder_tier_2
        edef_adder = scaling.e_defense_adder_tier_2
        armor_adder = scaling.armor_adder_tier_2
        save_adder = scaling.save_adder_tier_2
        speed_adder = scaling.speed_adder_tier_2
        sensor_adder = scaling.sensor_adder_tier_2
    else:  # tier_3
        hp_multiplier = scaling.hp_multiplier
        hp_adder = scaling.hp_adder_tier_3
        eva_adder = scaling.evasion_adder_tier_3
        edef_adder = scaling.e_defense_adder_tier_3
        armor_adder = scaling.armor_adder_tier_3
        save_adder = scaling.save_adder_tier_3
        speed_adder = scaling.speed_adder_tier_3
        sensor_adder = scaling.sensor_adder_tier_3

    hp_max = int(base.hp_base * hp_multiplier) + hp_adder

    return NPCCombatStats(
        size=base.size,
        hp_max=hp_max,
        evasion=base.evasion_base + eva_adder,
        e_defense=base.e_defense_base + edef_adder,
        armor=base.armor_base + armor_adder,
        speed=base.speed_base + speed_adder,
        sensor_range=base.sensor_range + sensor_adder,
        tech_attack=base.tech_attack,
        save_bonus=base.save_bonus + save_adder,
    )


def convert_to_combat_stats(npc_stats: NPCCombatStats) -> dict:
    """Convert NPCCombatStats to dict compatible with CombatStats.

    Used when creating CombatantState from NPCState.
    """
    return {
        "size": npc_stats.size,
        "hp_max": npc_stats.hp_max,
        "evasion": npc_stats.evasion,
        "e_defense": npc_stats.e_defense,
        "armor": npc_stats.armor,
        "speed": npc_stats.speed,
        "sensor_range": npc_stats.sensor_range,
        "tech_attack": npc_stats.tech_attack,
    }


class NPCState(FrozenModel):
    """Full NPC instance state for use in combat.

    Contains both the template reference and the scaled combat stats.
    """

    id: str
    name: str
    npc_class: NPCClass
    tier: NPCTier
    template_id: str | None = None
    stats: NPCCombatStats
    abilities_used: set[str] = Field(default_factory=set)
    deploy_abilities_remaining: int | None = None
    custom_effects: list[str] = Field(default_factory=list)

    @classmethod
    def from_template(
        cls,
        template: NPCTemplate,
        instance_id: str,
        name: str | None = None,
    ) -> "NPCState":
        """Create an NPC instance from a template.

        Args:
            template: The NPC template to instantiate
            instance_id: Unique ID for this NPC instance
            name: Optional override for NPC name (defaults to template name)

        Returns:
            A new NPCState with tier-scaled stats
        """
        scaled_stats = scale_npc_stats(template.stats, template.tier)
        return cls(
            id=instance_id,
            name=name or template.name,
            npc_class=template.npc_class,
            tier=template.tier,
            template_id=template.id,
            stats=scaled_stats,
            abilities_used=set(),
        )

    @property
    def hp_current(self) -> int:
        """Get current HP (for CombatResources creation)."""
        return self.stats.hp_max

    @property
    def structure_current(self) -> int:
        """Get structure points based on tier.

        NPCs typically have:
        - Tier 1: 1 structure
        - Tier 2: 2 structures
        - Tier 3: 3 structures
        """
        tier_structure_map = {"tier_1": 1, "tier_2": 2, "tier_3": 3}
        return tier_structure_map.get(self.tier, 1)


class NPCLoadout(FrozenModel):
    """NPC weapon/system loadout for template definition."""

    weapons: list[str] = Field(default_factory=list)
    systems: list[str] = Field(default_factory=list)
