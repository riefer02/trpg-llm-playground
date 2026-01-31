"""Difficulty scaling formulas for mission generation and AI behavior.

Provides type-safe scaling functions that map pilot license level (LL0-LL12)
to encounter difficulty, NPC tier selection, and AI aggression parameters.

Scaling Formulas
----------------

1. Encounter Difficulty (trivial → extreme)
   Pilot license level determines baseline encounter difficulty:
   - LL0-2: trivial (0.5x multiplier)
   - LL3-5: easy (0.75x multiplier)
   - LL6-8: standard (1.0x multiplier)
   - LL9-11: hard (1.5x multiplier)
   - LL12: extreme (2.0x multiplier)

2. NPC Tier Distribution
   Higher license levels increase proportion of higher-tier NPCs:
   - LL0-3: 100% tier_1, 0% tier_2, 0% tier_3
   - LL4-7: 50% tier_1, 50% tier_2, 0% tier_3
   - LL8-12: 25% tier_1, 50% tier_2, 25% tier_3

   This is implemented via template filtering in mission generator.

3. AI Aggression (0.0-1.0)
   Influences tactical decision-making:
   - 0.0: Cautious (prioritize survival, avoid risk)
   - 0.5: Balanced (default tactical behavior)
   - 1.0: Aggressive (prioritize damage, accept risks)

   Formula: aggression = min(1.0, max(0.0, (pilot_level - 3) / 9))
   This starts at 0.0 for LL0-3, increases linearly to 1.0 at LL12.

These formulas ensure progressive difficulty scaling while maintaining
balance across the full license level range.
"""

from typing import Literal
from core.gm_toolkit.encounter_builder import EncounterDifficulty
from core.npc.enums import NPCTier


def get_encounter_difficulty(pilot_level: int) -> EncounterDifficulty:
    """Map pilot license level to encounter difficulty.

    Args:
        pilot_level: Pilot license level (0-12)

    Returns:
        Encounter difficulty level (trivial, easy, standard, hard, extreme)
    """
    if pilot_level <= 2:
        return "trivial"
    elif pilot_level <= 5:
        return "easy"
    elif pilot_level <= 8:
        return "standard"
    elif pilot_level <= 11:
        return "hard"
    else:  # LL12
        return "extreme"


def get_npc_tier_distribution(
    pilot_level: int,
) -> dict[NPCTier, float]:
    """Get NPC tier distribution percentages for given pilot level.

    Returns a dict mapping tier to proportion (0.0-1.0) of that tier
    in the enemy force composition.

    Args:
        pilot_level: Pilot license level (0-12)

    Returns:
        Dict with keys "tier_1", "tier_2", "tier_3" and proportion values
        that sum to 1.0.
    """
    if pilot_level <= 3:
        return {"tier_1": 1.0, "tier_2": 0.0, "tier_3": 0.0}
    elif pilot_level <= 7:
        return {"tier_1": 0.5, "tier_2": 0.5, "tier_3": 0.0}
    else:  # LL8-12
        return {"tier_1": 0.25, "tier_2": 0.5, "tier_3": 0.25}


def get_ai_aggression(pilot_level: int) -> float:
    """Calculate AI aggression factor based on pilot level.

    Args:
        pilot_level: Pilot license level (0-12)

    Returns:
        Aggression factor between 0.0 (cautious) and 1.0 (aggressive)
    """
    # Start cautious at low levels, become aggressive at high levels
    # LL0-3: 0.0, LL12: 1.0, linear progression
    aggression = min(1.0, max(0.0, (pilot_level - 3) / 9.0))
    return round(aggression, 2)


def get_mission_difficulty_stars(pilot_level: int, mission_index: int = 0) -> int:
    """Calculate mission difficulty stars (1-3) for display.

    Higher pilot levels unlock higher difficulty missions.
    Mission index (0-2) provides variation within a batch.

    Args:
        pilot_level: Pilot license level (0-12)
        mission_index: Index within mission batch (0-2)

    Returns:
        Difficulty stars (1-3)
    """
    # Base difficulty increases with pilot level
    base = min(3, max(1, pilot_level // 4 + 1))
    # Add variation based on mission index
    difficulty = min(3, max(1, base + mission_index))
    return difficulty


def get_encounter_difficulty_for_mission(
    pilot_level: int,
    mission_index: int = 0,
) -> EncounterDifficulty:
    """Get encounter difficulty for a specific mission in a batch.

    Base difficulty determined by pilot level, with mission index
    increasing difficulty by one step (e.g., trivial → easy).

    Args:
        pilot_level: Pilot license level (0-12)
        mission_index: Index within mission batch (0-2)

    Returns:
        Encounter difficulty level
    """
    base_difficulty = get_encounter_difficulty(pilot_level)
    difficulty_order: list[EncounterDifficulty] = [
        "trivial",
        "easy",
        "standard",
        "hard",
        "extreme",
    ]
    try:
        base_idx = difficulty_order.index(base_difficulty)
        # Increase difficulty by mission index, capped at extreme
        new_idx = min(len(difficulty_order) - 1, base_idx + mission_index)
        return difficulty_order[new_idx]
    except ValueError:
        # Should never happen
        return base_difficulty
