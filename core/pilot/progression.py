"""Pilot progression rules and level chart for Lancer TTRPG.

This module encodes the mechanical progression table for license levels.
"""

from pydantic import BaseModel, Field


LEVEL_CAP = 12


class LevelProgression(BaseModel):
    """Progression values for a given license level."""

    level: int = Field(..., ge=0, le=LEVEL_CAP)
    grit: int = Field(..., ge=0)
    license_points: int = Field(..., ge=0)
    total_mech_skill_points: int = Field(..., ge=0)
    total_talent_points: int = Field(..., ge=0)
    core_bonuses: int = Field(..., ge=0)
    pilot_trigger_points: int = Field(..., ge=0)

    model_config = {"frozen": True}


class PilotProgressionRules(BaseModel):
    """Rule-of-thumb progression parameters used to build the level chart."""

    level_cap: int = LEVEL_CAP
    starting_trigger_points: int = 8
    trigger_points_per_level: int = 2
    trigger_rank_min: int = 2
    trigger_rank_max: int = 6
    starting_mech_skill_points: int = 2
    mech_skill_points_per_level: int = 1
    mech_skill_rank_max: int = 6
    starting_talent_points: int = 3
    talent_points_per_level: int = 1
    license_points_per_level: int = 1
    core_bonus_every_levels: int = 3

    model_config = {"frozen": True}


DEFAULT_PILOT_PROGRESSION = PilotProgressionRules()


LICENSE_LEVEL_TABLE: list[LevelProgression] = [
    LevelProgression(
        level=0,
        grit=0,
        license_points=0,
        total_mech_skill_points=2,
        total_talent_points=3,
        core_bonuses=0,
        pilot_trigger_points=8,
    ),
    LevelProgression(
        level=1,
        grit=1,
        license_points=1,
        total_mech_skill_points=3,
        total_talent_points=4,
        core_bonuses=0,
        pilot_trigger_points=10,
    ),
    LevelProgression(
        level=2,
        grit=1,
        license_points=2,
        total_mech_skill_points=4,
        total_talent_points=5,
        core_bonuses=0,
        pilot_trigger_points=12,
    ),
    LevelProgression(
        level=3,
        grit=2,
        license_points=3,
        total_mech_skill_points=5,
        total_talent_points=6,
        core_bonuses=1,
        pilot_trigger_points=14,
    ),
    LevelProgression(
        level=4,
        grit=2,
        license_points=4,
        total_mech_skill_points=6,
        total_talent_points=7,
        core_bonuses=1,
        pilot_trigger_points=16,
    ),
    LevelProgression(
        level=5,
        grit=3,
        license_points=5,
        total_mech_skill_points=7,
        total_talent_points=8,
        core_bonuses=1,
        pilot_trigger_points=18,
    ),
    LevelProgression(
        level=6,
        grit=3,
        license_points=6,
        total_mech_skill_points=8,
        total_talent_points=9,
        core_bonuses=2,
        pilot_trigger_points=20,
    ),
    LevelProgression(
        level=7,
        grit=4,
        license_points=7,
        total_mech_skill_points=9,
        total_talent_points=10,
        core_bonuses=2,
        pilot_trigger_points=22,
    ),
    LevelProgression(
        level=8,
        grit=4,
        license_points=8,
        total_mech_skill_points=10,
        total_talent_points=11,
        core_bonuses=2,
        pilot_trigger_points=24,
    ),
    LevelProgression(
        level=9,
        grit=5,
        license_points=9,
        total_mech_skill_points=11,
        total_talent_points=12,
        core_bonuses=3,
        pilot_trigger_points=26,
    ),
    LevelProgression(
        level=10,
        grit=5,
        license_points=10,
        total_mech_skill_points=12,
        total_talent_points=13,
        core_bonuses=3,
        pilot_trigger_points=28,
    ),
    LevelProgression(
        level=11,
        grit=6,
        license_points=11,
        total_mech_skill_points=13,
        total_talent_points=14,
        core_bonuses=3,
        pilot_trigger_points=30,
    ),
    LevelProgression(
        level=12,
        grit=6,
        license_points=12,
        total_mech_skill_points=14,
        total_talent_points=15,
        core_bonuses=4,
        pilot_trigger_points=32,
    ),
]


def get_level_progression(level: int) -> LevelProgression:
    """Return the progression values for a given license level."""
    for row in LICENSE_LEVEL_TABLE:
        if row.level == level:
            return row
    raise ValueError(f"Invalid license level: {level}")
