"""Encounter building helpers for GM Toolkit.

This module provides type-safe primitives for encounter difficulty scaling
and enemy force management per PR2 350-361 (SITREP rules).

Features:
- Difficulty scaling (trivial → extreme with linear multipliers)
- Player party power estimation based on license level
- Enemy force recommendations tied to SITREP type
- Victory point calculation for any NPC template
- Integration with existing SitrepTemplate from core.shared.scenario
"""

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel
from core.npc.models import NPCTemplate, SpecialNPCTemplate, NPCTier
from core.shared.scenario import SitrepTemplate, SITREP_TEMPLATES


EncounterDifficulty = Literal["trivial", "easy", "standard", "hard", "extreme"]

DIFFICULTY_MULTIPLIERS: dict[EncounterDifficulty, float] = {
    "trivial": 0.5,
    "easy": 0.75,
    "standard": 1.0,
    "hard": 1.5,
    "extreme": 2.0,
}

RESERVE_PATTERN_MULTIPLIERS: dict[str, dict[str, float]] = {
    "none": {"initial": 0.0, "reserve": 0.0},
    "half": {"initial": 0.5, "reserve": 0.5},
    "normal": {"initial": 1.0, "reserve": 0.0},
    "double": {"initial": 0.0, "reserve": 2.0},
    "increasing": {"initial": 1.0, "reserve": 1.0},
}

TIER_MULTIPLIERS: dict[NPCTier, float] = {
    "tier_1": 1.0,
    "tier_2": 1.5,
    "tier_3": 2.0,
}


class PlayerPartyPower(FrozenModel):
    """Player party power estimation for encounter scaling.

    Power is calculated as: player_count × (1 + avg_license_level × 0.1)
    This gives roughly 2x power for a level 12 party vs level 0.

    Attributes:
        player_count: Number of players in the party (1-6)
        avg_license_level: Average license level across all players (0-12)
    """

    player_count: int = Field(..., ge=1, le=6)
    avg_license_level: float = Field(..., ge=0, le=12)

    @property
    def base_power(self) -> float:
        """Calculate base party power.

        Returns:
            Base power value used for encounter scaling
        """
        return self.player_count * (1 + self.avg_license_level * 0.1)


class EnemyForceRecommendation(FrozenModel):
    """Recommended enemy force for an encounter.

    Provides victory point targets split between initial deployment
    and reserves based on the SITREP type.

    Attributes:
        target_victory_points: Total victory points for the encounter
        initial_victory_points: Victory points for initially deployed enemies
        reserve_victory_points: Victory points for reserve enemies
        suggested_tier: Recommended NPC tier for the encounter
        recommended_template_ids: Optional list of recommended template IDs
    """

    target_victory_points: float = Field(..., ge=0.0)
    initial_victory_points: float = Field(..., ge=0.0)
    reserve_victory_points: float = Field(..., ge=0.0)
    suggested_tier: NPCTier = Field(default="tier_1")
    recommended_template_ids: list[str] = Field(default_factory=list)


def estimate_party_power(
    player_count: int, avg_license_level: float
) -> PlayerPartyPower:
    """Estimate player party power for encounter scaling.

    Args:
        player_count: Number of players in the party
        avg_license_level: Average license level across all players (0-12)

    Returns:
        PlayerPartyPower with calculated base power
    """
    return PlayerPartyPower(
        player_count=player_count,
        avg_license_level=avg_license_level,
    )


def calculate_enemy_force(
    difficulty: EncounterDifficulty,
    sitrep_type: str,
    player_power: PlayerPartyPower,
    npc_templates: list[NPCTemplate] | None = None,
) -> EnemyForceRecommendation:
    """Calculate recommended enemy force for an encounter.

    Combines difficulty multiplier with SITREP reserve pattern to determine
    how enemy forces should be split between initial deployment and reserves.

    Args:
        difficulty: Encounter difficulty level (trivial → extreme)
        sitrep_type: SITREP mission type (escort, control, extract, hold_out, gauntlet, recon)
        player_power: Estimated player party power
        npc_templates: Optional list of NPC templates to recommend

    Returns:
        EnemyForceRecommendation with victory point targets

    Raises:
        ValueError: If sitrep_type is not a valid SITREP template
    """
    sitrep = SITREP_TEMPLATES.get(sitrep_type)
    if sitrep is None:
        raise ValueError(f"Unknown SITREP type: {sitrep_type}")

    difficulty_mult = DIFFICULTY_MULTIPLIERS[difficulty]
    reserve_mults = RESERVE_PATTERN_MULTIPLIERS[sitrep.reserve_pattern]

    base_vp = player_power.base_power * difficulty_mult

    total_multiplier = reserve_mults["initial"] + reserve_mults["reserve"]
    total_vp = base_vp * total_multiplier

    return EnemyForceRecommendation(
        target_victory_points=total_vp,
        initial_victory_points=base_vp * reserve_mults["initial"],
        reserve_victory_points=base_vp * reserve_mults["reserve"],
        suggested_tier="tier_1",
        recommended_template_ids=[t.id for t in npc_templates] if npc_templates else [],
    )


def calculate_total_victory_points(
    templates: list[NPCTemplate | SpecialNPCTemplate],
) -> float:
    """Calculate total victory points from a list of NPC templates.

    Victory points scale with NPC tier:
    - tier_1: 1.0x base value
    - tier_2: 1.5x base value
    - tier_3: 2.0x base value

    Args:
        templates: List of NPC templates (regular or special class)

    Returns:
        Total victory points (sum of base × tier multiplier)
    """
    total = 0.0
    for template in templates:
        tier_mult = TIER_MULTIPLIERS.get(template.tier, 1.0)
        total += template.victory_count * tier_mult
    return total


def get_sitrep_force_multipliers(sitrep_type: str) -> dict[str, float]:
    """Get force multipliers for a specific SITREP type.

    Args:
        sitrep_type: SITREP type identifier (escort, control, extract, hold_out, gauntlet, recon)

    Returns:
        Dict with 'initial' and 'reserve' multipliers

    Raises:
        ValueError: If sitrep_type is not a valid SITREP template
    """
    sitrep = SITREP_TEMPLATES.get(sitrep_type)
    if sitrep is None:
        raise ValueError(f"Unknown SITREP type: {sitrep_type}")
    return RESERVE_PATTERN_MULTIPLIERS[sitrep.reserve_pattern]


def get_sitrep_template(sitrep_type: str) -> SitrepTemplate | None:
    """Get a SITREP template by type.

    Args:
        sitrep_type: The SITREP type identifier

    Returns:
        The matching SitrepTemplate, or None if not found
    """
    return SITREP_TEMPLATES.get(sitrep_type)
