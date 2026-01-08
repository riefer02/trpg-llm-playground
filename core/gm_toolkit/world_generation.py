"""World and Environment Generation for GM Toolkit.

This module provides type-safe primitives for random world generation
and environmental hazard mechanics per PR2 362-381.

Features:
- World Type generation (20 types: barren, temperate, inhospitable)
- Natural Feature generation (20 features: storms, tectonics, moons, etc.)
- Anthropocentric Feature generation (20 features: settlements, bases, etc.)
- Environmental Hazard mechanics (20 hazards: extreme cold/heat, storms, etc.)
- Seeded RNG for reproducible generation
"""

from typing import Literal
from random import Random
from pydantic import Field
from core.shared.models import FrozenModel


WorldType = Literal[
    "barren_radiation",
    "barren_ice_stone",
    "barren_mineral_thunderstorms",
    "barren_ice_subglacial",
    "barren_white_dunes",
    "temperate_hazy_mist",
    "temperate_biomes_clouds",
    "temperate_plains_desert",
    "temperate_archipelagos",
    "temperate_healthy",
    "inhospitable_choking",
    "inhospitable_nitrogen_oceans",
    "inhospitable_lava_lightning",
    "inhospitable_gamma_burst",
    "inhospitable_impact_cracked",
    "temperate_moon",
    "icy_moon",
    "ocean_world",
    "barren_geometric_features",
    "ancient_symmetry",
]

WORLD_TYPE_NAMES: dict[WorldType, str] = {
    "barren_radiation": "Barren World (Radiation)",
    "barren_ice_stone": "Barren World (Ice/Stone)",
    "barren_mineral_thunderstorms": "Barren World (Mineral/Thunderstorms)",
    "barren_ice_subglacial": "Barren World (Ice/Subglacial)",
    "barren_white_dunes": "Barren World (White Dunes)",
    "temperate_hazy_mist": "Temperate World (Hazy Mist)",
    "temperate_biomes_clouds": "Temperate World (Biomes/Clouds)",
    "temperate_plains_desert": "Temperate World (Plains/Desert)",
    "temperate_archipelagos": "Temperate World (Archipelagos)",
    "temperate_healthy": "Temperate World (Healthy)",
    "inhospitable_choking": "Inhospitable World (Choking Atmosphere)",
    "inhospitable_nitrogen_oceans": "Inhospitable World (Nitrogen Oceans)",
    "inhospitable_lava_lightning": "Inhospitable World (Lava/Lightning)",
    "inhospitable_gamma_burst": "Inhospitable World (Gamma Burst)",
    "inhospitable_impact_cracked": "Inhospitable World (Impact Cracked)",
    "temperate_moon": "Temperate Moon",
    "icy_moon": "Icy Moon",
    "ocean_world": "Ocean World",
    "barren_geometric_features": "Barren World (Geometric Features)",
    "ancient_symmetry": "Ancient World (Terrible Symmetry)",
}


NaturalFeature = Literal[
    "hundred_year_storms",
    "active_tectonics",
    "inert_core",
    "monobiome",
    "worldscar",
    "royal_court",
    "twin_suns",
    "ringed",
    "remote",
    "cosmopolitan",
    "hecatoncheires",
    "epochal_sunset",
    "monument_shame",
    "quarantined",
    "breathable_atmosphere",
    "high_gravity",
    "low_gravity",
    "hard_sun",
    "dreamland",
    "dust_echoes",
]

NATURAL_FEATURE_NAMES: dict[NaturalFeature, str] = {
    "hundred_year_storms": "100 Year Storms",
    "active_tectonics": "Active Tectonics",
    "inert_core": "Inert Core",
    "monobiome": "Monobiome",
    "worldscar": "Worldscar",
    "royal_court": "Royal Court (Many Moons)",
    "twin_suns": "Twin Suns",
    "ringed": "Ringed",
    "remote": "Remote",
    "cosmopolitan": "Cosmopolitan",
    "hecatoncheires": "Hecatoncheires (Massive Mountains)",
    "epochal_sunset": "Epochal Sunset",
    "monument_shame": "Monument of Shame",
    "quarantined": "Quarantined",
    "breathable_atmosphere": "Breathable Atmosphere",
    "high_gravity": "High Gravity (+1-2G)",
    "low_gravity": "Low Gravity",
    "hard_sun": "Hard Sun",
    "dreamland": "Dreamland",
    "dust_echoes": "Dust and Echoes (Ruins)",
}


AnthropocentricFeature = Literal[
    "colony_initial",
    "colony_first_gen",
    "colony_stable",
    "outpost_ff_team",
    "outpost_omninet",
    "outpost_sigint",
    "outpost_astrocartography",
    "outpost_garrison",
    "installation_research",
    "installation_proving_ground",
    "installation_deep_field_relay",
    "installation_embassy",
    "installation_cs_campus",
    "base_naval_command",
]

ANTHROPOCENTRIC_FEATURE_NAMES: dict[AnthropocentricFeature, str] = {
    "colony_initial": "Colonial Settlement (Initial)",
    "colony_first_gen": "Colonial Settlement (First Gen)",
    "colony_stable": "Colonial Settlement (Stable)",
    "outpost_ff_team": "Outpost (Union FF Team)",
    "outpost_omninet": "Outpost (Omninet Relay)",
    "outpost_sigint": "Outpost (Union SigInt)",
    "outpost_astrocartography": "Outpost (Astrocartography)",
    "outpost_garrison": "Outpost (Union Garrison)",
    "installation_research": "Installation (Research Facility)",
    "installation_proving_ground": "Installation (Proving Ground)",
    "installation_deep_field_relay": "Installation (Deep Field Relay)",
    "installation_embassy": "Installation (Union Embassy)",
    "installation_cs_campus": "Installation (Corpro-State Campus)",
    "base_naval_command": "Base (Union Naval System Command)",
}


EnvironmentalHazard = Literal[
    "dangerous_flora_fauna",
    "extreme_cold",
    "extreme_heat",
    "thin_atmosphere",
    "extreme_sun",
    "corrosive_atmosphere",
    "particulate_storm",
    "electric_storm",
    "disruptive_storm",
    "dangerous_storm",
    "earthquakes",
    "ocean_world",
    "molten_world",
    "primordial_world",
    "low_gravity_hazard",
    "high_gravity_hazard",
    "tomb_world",
    "spire_world",
    "sinking_world",
    "holy_world",
]

ENVIRONMENTAL_HAZARD_NAMES: dict[EnvironmentalHazard, str] = {
    "dangerous_flora_fauna": "Dangerous Flora/Fauna",
    "extreme_cold": "Extreme Cold",
    "extreme_heat": "Extreme Heat",
    "thin_atmosphere": "Thin Atmosphere",
    "extreme_sun": "Extreme Sun",
    "corrosive_atmosphere": "Corrosive Atmosphere",
    "particulate_storm": "Particulate Storm",
    "electric_storm": "Electric Storm",
    "disruptive_storm": "Disruptive Storm",
    "dangerous_storm": "Dangerous Storm",
    "earthquakes": "Earthquakes",
    "ocean_world": "Ocean World",
    "molten_world": "Molten World",
    "primordial_world": "Primordial World",
    "low_gravity_hazard": "Low Gravity",
    "high_gravity_hazard": "High Gravity",
    "tomb_world": "Tomb World",
    "spire_world": "Spire World",
    "sinking_world": "Sinking World",
    "holy_world": "Holy World",
}


class WorldDetails(FrozenModel):
    """Complete world generation result.

    Attributes:
        world_type: The generated world type
        natural_features: List of natural features (1-3)
        anthropocentric_features: Optional anthropocentric feature
        environmental_hazards: List of environmental hazards (1-3)
    """

    world_type: WorldType
    natural_features: list[NaturalFeature] = Field(..., min_length=1, max_length=3)
    anthropocentric_feature: AnthropocentricFeature | None = None
    environmental_hazards: list[EnvironmentalHazard] = Field(
        ..., min_length=1, max_length=3
    )


class SeededGenerator(FrozenModel):
    """Seeded RNG generator for reproducible world generation.

    Attributes:
        seed: Integer seed for RNG
        _rng: Internal Random instance (not serialized)
    """

    seed: int
    _rng: Random | None = None

    def _get_rng(self) -> Random:
        """Get or create the internal RNG instance."""
        if self._rng is None:
            self._rng = Random(self.seed)
        return self._rng


def generate_world(seed: int) -> WorldDetails:
    """Generate a complete world with all features.

    Args:
        seed: Integer seed for reproducible generation

    Returns:
        WorldDetails with all world features populated
    """
    rng = Random(seed)

    world_type = rng.choice(list(WORLD_TYPE_NAMES.keys()))

    natural_features = rng.sample(
        list(NATURAL_FEATURE_NAMES.keys()), k=rng.randint(1, 3)
    )

    anthropocentric_feature: AnthropocentricFeature | None = None
    if rng.random() < 0.7:
        anthropocentric_feature = rng.choice(list(ANTHROPOCENTRIC_FEATURE_NAMES.keys()))

    environmental_hazards = rng.sample(
        list(ENVIRONMENTAL_HAZARD_NAMES.keys()), k=rng.randint(1, 3)
    )

    return WorldDetails(
        world_type=world_type,
        natural_features=natural_features,
        anthropocentric_feature=anthropocentric_feature,
        environmental_hazards=environmental_hazards,
    )


def generate_world_type(seed: int) -> WorldType:
    """Generate a random world type.

    Args:
        seed: Integer seed for reproducible generation

    Returns:
        A random WorldType literal
    """
    rng = Random(seed)
    return rng.choice(list(WORLD_TYPE_NAMES.keys()))


def generate_natural_features(
    seed: int, count: int | None = None
) -> list[NaturalFeature]:
    """Generate random natural features.

    Args:
        seed: Integer seed for reproducible generation
        count: Number of features to generate (1-3, default random)

    Returns:
        List of NaturalFeature literals
    """
    rng = Random(seed)
    if count is None:
        count = rng.randint(1, 3)
    return rng.sample(list(NATURAL_FEATURE_NAMES.keys()), k=min(count, 20))


def generate_anthropocentric_feature(seed: int) -> AnthropocentricFeature | None:
    """Generate a random anthropocentric feature.

    70% chance to return a feature, 30% chance to return None (uninhabited).

    Args:
        seed: Integer seed for reproducible generation

    Returns:
        An AnthropocentricFeature literal, or None if uninhabited
    """
    rng = Random(seed)
    if rng.random() < 0.3:
        return None
    return rng.choice(list(ANTHROPOCENTRIC_FEATURE_NAMES.keys()))


def generate_environmental_hazards(
    seed: int, count: int | None = None
) -> list[EnvironmentalHazard]:
    """Generate random environmental hazards.

    Args:
        seed: Integer seed for reproducible generation
        count: Number of hazards to generate (1-3, default random)

    Returns:
        List of EnvironmentalHazard literals
    """
    rng = Random(seed)
    if count is None:
        count = rng.randint(1, 3)
    return rng.sample(list(ENVIRONMENTAL_HAZARD_NAMES.keys()), k=min(count, 20))


def get_world_type_name(world_type: WorldType) -> str:
    """Get the display name for a world type.

    Args:
        world_type: The world type literal

    Returns:
        Human-readable name for the world type
    """
    return WORLD_TYPE_NAMES.get(world_type, "Unknown")


def get_natural_feature_name(feature: NaturalFeature) -> str:
    """Get the display name for a natural feature.

    Args:
        feature: The natural feature literal

    Returns:
        Human-readable name for the feature
    """
    return NATURAL_FEATURE_NAMES.get(feature, "Unknown")


def get_anthropocentric_feature_name(feature: AnthropocentricFeature) -> str:
    """Get the display name for an anthropocentric feature.

    Args:
        feature: The anthropocentric feature literal

    Returns:
        Human-readable name for the feature
    """
    return ANTHROPOCENTRIC_FEATURE_NAMES.get(feature, "Unknown")


def get_environmental_hazard_name(hazard: EnvironmentalHazard) -> str:
    """Get the display name for an environmental hazard.

    Args:
        hazard: The environmental hazard literal

    Returns:
        Human-readable name for the hazard
    """
    return ENVIRONMENTAL_HAZARD_NAMES.get(hazard, "Unknown")
