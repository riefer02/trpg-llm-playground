"""GM Toolkit for encounter building and world generation.

This module provides type-safe primitives for:
- Encounter difficulty scaling (trivial → extreme)
- Player party power estimation
- Enemy force recommendations based on SITREP type
- Victory point calculation for NPC templates
- World type and environment generation (20 types each)
- Environmental hazard mechanics per PR2 362-381
"""

from core.gm_toolkit.encounter_builder import (
    EncounterDifficulty,
    PlayerPartyPower,
    EnemyForceRecommendation,
    estimate_party_power,
    calculate_enemy_force,
    calculate_total_victory_points,
    get_sitrep_force_multipliers,
)

from core.gm_toolkit.world_generation import (
    WorldType,
    NaturalFeature,
    AnthropocentricFeature,
    EnvironmentalHazard,
    WorldDetails,
    WORLD_TYPE_NAMES,
    NATURAL_FEATURE_NAMES,
    ANTHROPOCENTRIC_FEATURE_NAMES,
    ENVIRONMENTAL_HAZARD_NAMES,
    generate_world,
    generate_world_type,
    generate_natural_features,
    generate_anthropocentric_feature,
    generate_environmental_hazards,
    get_world_type_name,
    get_natural_feature_name,
    get_anthropocentric_feature_name,
    get_environmental_hazard_name,
)

__all__ = [
    "EncounterDifficulty",
    "PlayerPartyPower",
    "EnemyForceRecommendation",
    "estimate_party_power",
    "calculate_enemy_force",
    "calculate_total_victory_points",
    "get_sitrep_force_multipliers",
    "WorldType",
    "NaturalFeature",
    "AnthropocentricFeature",
    "EnvironmentalHazard",
    "WorldDetails",
    "WORLD_TYPE_NAMES",
    "NATURAL_FEATURE_NAMES",
    "ANTHROPOCENTRIC_FEATURE_NAMES",
    "ENVIRONMENTAL_HAZARD_NAMES",
    "generate_world",
    "generate_world_type",
    "generate_natural_features",
    "generate_anthropocentric_feature",
    "generate_environmental_hazards",
    "get_world_type_name",
    "get_natural_feature_name",
    "get_anthropocentric_feature_name",
    "get_environmental_hazard_name",
]
