"""Tests for World and Environment Generation module.

Tests cover:
- World Type generation and validation
- Natural Feature generation
- Anthropocentric Feature generation
- Environmental Hazard generation
- Seeded RNG reproducibility
- WorldDetails model validation
"""

import pytest
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


class TestWorldTypeLiterals:
    """Tests for WorldType literal definitions."""

    def test_world_type_count(self):
        """Verify exactly 20 world types are defined."""
        assert len(WORLD_TYPE_NAMES) == 20

    def test_world_type_keys_are_literals(self):
        """Verify world types match expected patterns."""
        expected_types = [
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
        assert set(WORLD_TYPE_NAMES.keys()) == set(expected_types)

    def test_barren_world_types_exist(self):
        """Verify barren world types are defined."""
        assert "barren_radiation" in WORLD_TYPE_NAMES
        assert "barren_ice_stone" in WORLD_TYPE_NAMES
        assert "barren_mineral_thunderstorms" in WORLD_TYPE_NAMES
        assert "barren_ice_subglacial" in WORLD_TYPE_NAMES
        assert "barren_white_dunes" in WORLD_TYPE_NAMES
        assert "barren_geometric_features" in WORLD_TYPE_NAMES

    def test_temperate_world_types_exist(self):
        """Verify temperate world types are defined."""
        assert "temperate_hazy_mist" in WORLD_TYPE_NAMES
        assert "temperate_biomes_clouds" in WORLD_TYPE_NAMES
        assert "temperate_plains_desert" in WORLD_TYPE_NAMES
        assert "temperate_archipelagos" in WORLD_TYPE_NAMES
        assert "temperate_healthy" in WORLD_TYPE_NAMES
        assert "temperate_moon" in WORLD_TYPE_NAMES

    def test_inhospitable_world_types_exist(self):
        """Verify inhospitable world types are defined."""
        assert "inhospitable_choking" in WORLD_TYPE_NAMES
        assert "inhospitable_nitrogen_oceans" in WORLD_TYPE_NAMES
        assert "inhospitable_lava_lightning" in WORLD_TYPE_NAMES
        assert "inhospitable_gamma_burst" in WORLD_TYPE_NAMES
        assert "inhospitable_impact_cracked" in WORLD_TYPE_NAMES

    def test_special_world_types_exist(self):
        """Verify special world types are defined."""
        assert "icy_moon" in WORLD_TYPE_NAMES
        assert "ocean_world" in WORLD_TYPE_NAMES
        assert "ancient_symmetry" in WORLD_TYPE_NAMES


class TestNaturalFeatureLiterals:
    """Tests for NaturalFeature literal definitions."""

    def test_natural_feature_count(self):
        """Verify exactly 20 natural features are defined."""
        assert len(NATURAL_FEATURE_NAMES) == 20

    def test_natural_feature_keys_are_literals(self):
        """Verify natural features match expected patterns."""
        expected_features = [
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
        assert set(NATURAL_FEATURE_NAMES.keys()) == set(expected_features)

    def test_astronomical_features_exist(self):
        """Verify astronomical features are defined."""
        assert "hundred_year_storms" in NATURAL_FEATURE_NAMES
        assert "worldscar" in NATURAL_FEATURE_NAMES
        assert "royal_court" in NATURAL_FEATURE_NAMES
        assert "twin_suns" in NATURAL_FEATURE_NAMES
        assert "ringed" in NATURAL_FEATURE_NAMES

    def test_gravity_features_exist(self):
        """Verify gravity features are defined."""
        assert "high_gravity" in NATURAL_FEATURE_NAMES
        assert "low_gravity" in NATURAL_FEATURE_NAMES

    def test_atmosphere_features_exist(self):
        """Verify atmosphere features are defined."""
        assert "breathable_atmosphere" in NATURAL_FEATURE_NAMES
        assert "inert_core" in NATURAL_FEATURE_NAMES
        assert "hard_sun" in NATURAL_FEATURE_NAMES


class TestAnthropocentricFeatureLiterals:
    """Tests for AnthropocentricFeature literal definitions."""

    def test_anthropocentric_feature_count(self):
        """Verify exactly 14 anthropocentric features are defined."""
        assert len(ANTHROPOCENTRIC_FEATURE_NAMES) == 14

    def test_anthropocentric_feature_keys_are_literals(self):
        """Verify features match expected patterns."""
        expected_features = [
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
        assert set(ANTHROPOCENTRIC_FEATURE_NAMES.keys()) == set(expected_features)

    def test_colony_features_exist(self):
        """Verify colony features are defined."""
        assert "colony_initial" in ANTHROPOCENTRIC_FEATURE_NAMES
        assert "colony_first_gen" in ANTHROPOCENTRIC_FEATURE_NAMES
        assert "colony_stable" in ANTHROPOCENTRIC_FEATURE_NAMES

    def test_outpost_features_exist(self):
        """Verify outpost features are defined."""
        assert "outpost_ff_team" in ANTHROPOCENTRIC_FEATURE_NAMES
        assert "outpost_omninet" in ANTHROPOCENTRIC_FEATURE_NAMES
        assert "outpost_sigint" in ANTHROPOCENTRIC_FEATURE_NAMES
        assert "outpost_astrocartography" in ANTHROPOCENTRIC_FEATURE_NAMES
        assert "outpost_garrison" in ANTHROPOCENTRIC_FEATURE_NAMES

    def test_installation_features_exist(self):
        """Verify installation features are defined."""
        assert "installation_research" in ANTHROPOCENTRIC_FEATURE_NAMES
        assert "installation_proving_ground" in ANTHROPOCENTRIC_FEATURE_NAMES
        assert "installation_deep_field_relay" in ANTHROPOCENTRIC_FEATURE_NAMES
        assert "installation_embassy" in ANTHROPOCENTRIC_FEATURE_NAMES
        assert "installation_cs_campus" in ANTHROPOCENTRIC_FEATURE_NAMES

    def test_base_feature_exists(self):
        """Verify base feature is defined."""
        assert "base_naval_command" in ANTHROPOCENTRIC_FEATURE_NAMES


class TestEnvironmentalHazardLiterals:
    """Tests for EnvironmentalHazard literal definitions."""

    def test_environmental_hazard_count(self):
        """Verify exactly 20 environmental hazards are defined."""
        assert len(ENVIRONMENTAL_HAZARD_NAMES) == 20

    def test_environmental_hazard_keys_are_literals(self):
        """Verify hazards match expected patterns."""
        expected_hazards = [
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
        assert set(ENVIRONMENTAL_HAZARD_NAMES.keys()) == set(expected_hazards)

    def test_temperature_hazards_exist(self):
        """Verify temperature hazards are defined."""
        assert "extreme_cold" in ENVIRONMENTAL_HAZARD_NAMES
        assert "extreme_heat" in ENVIRONMENTAL_HAZARD_NAMES

    def test_atmosphere_hazards_exist(self):
        """Verify atmosphere hazards are defined."""
        assert "thin_atmosphere" in ENVIRONMENTAL_HAZARD_NAMES
        assert "extreme_sun" in ENVIRONMENTAL_HAZARD_NAMES
        assert "corrosive_atmosphere" in ENVIRONMENTAL_HAZARD_NAMES

    def test_storm_hazards_exist(self):
        """Verify storm hazards are defined."""
        assert "particulate_storm" in ENVIRONMENTAL_HAZARD_NAMES
        assert "electric_storm" in ENVIRONMENTAL_HAZARD_NAMES
        assert "disruptive_storm" in ENVIRONMENTAL_HAZARD_NAMES
        assert "dangerous_storm" in ENVIRONMENTAL_HAZARD_NAMES

    def test_gravity_hazards_exist(self):
        """Verify gravity hazards are defined."""
        assert "low_gravity_hazard" in ENVIRONMENTAL_HAZARD_NAMES
        assert "high_gravity_hazard" in ENVIRONMENTAL_HAZARD_NAMES

    def test_extreme_world_hazards_exist(self):
        """Verify extreme world hazards are defined."""
        assert "ocean_world" in ENVIRONMENTAL_HAZARD_NAMES
        assert "molten_world" in ENVIRONMENTAL_HAZARD_NAMES
        assert "primordial_world" in ENVIRONMENTAL_HAZARD_NAMES
        assert "tomb_world" in ENVIRONMENTAL_HAZARD_NAMES
        assert "spire_world" in ENVIRONMENTAL_HAZARD_NAMES
        assert "sinking_world" in ENVIRONMENTAL_HAZARD_NAMES


class TestGenerateWorldType:
    """Tests for generate_world_type function."""

    def test_returns_valid_world_type(self):
        """Verify generate_world_type returns a valid WorldType."""
        world_type = generate_world_type(12345)
        assert world_type in WORLD_TYPE_NAMES

    def test_same_seed_produces_same_result(self):
        """Verify seeded RNG produces reproducible results."""
        result1 = generate_world_type(42)
        result2 = generate_world_type(42)
        assert result1 == result2

    def test_different_seeds_can_produce_different_results(self):
        """Verify different seeds can produce different world types."""
        results = {generate_world_type(i) for i in range(100)}
        assert len(results) > 1


class TestGenerateNaturalFeatures:
    """Tests for generate_natural_features function."""

    def test_returns_list(self):
        """Verify generate_natural_features returns a list."""
        features = generate_natural_features(12345)
        assert isinstance(features, list)

    def test_returns_valid_features(self):
        """Verify returned features are valid NaturalFeatures."""
        features = generate_natural_features(12345)
        for feature in features:
            assert feature in NATURAL_FEATURE_NAMES

    def test_same_seed_produces_same_result(self):
        """Verify seeded RNG produces reproducible results."""
        result1 = generate_natural_features(42)
        result2 = generate_natural_features(42)
        assert result1 == result2

    def test_count_parameter(self):
        """Verify count parameter controls number of features."""
        features_1 = generate_natural_features(12345, count=1)
        features_3 = generate_natural_features(12345, count=3)
        assert len(features_1) == 1
        assert len(features_3) == 3

    def test_default_count_range(self):
        """Verify default count is between 1 and 3."""
        for i in range(100):
            features = generate_natural_features(i)
            assert 1 <= len(features) <= 3

    def test_no_duplicates(self):
        """Verify returned features contain no duplicates."""
        features = generate_natural_features(12345)
        assert len(features) == len(set(features))


class TestGenerateAnthropocentricFeature:
    """Tests for generate_anthropocentric_feature function."""

    def test_returns_valid_feature_or_none(self):
        """Verify function returns valid feature or None."""
        result = generate_anthropocentric_feature(12345)
        assert result is None or result in ANTHROPOCENTRIC_FEATURE_NAMES

    def test_same_seed_produces_same_result(self):
        """Verify seeded RNG produces reproducible results."""
        result1 = generate_anthropocentric_feature(42)
        result2 = generate_anthropocentric_feature(42)
        assert result1 == result2

    def test_can_return_none(self):
        """Verify function can return None (uninhabited)."""
        results = {generate_anthropocentric_feature(i) for i in range(100)}
        assert None in results


class TestGenerateEnvironmentalHazards:
    """Tests for generate_environmental_hazards function."""

    def test_returns_list(self):
        """Verify function returns a list."""
        hazards = generate_environmental_hazards(12345)
        assert isinstance(hazards, list)

    def test_returns_valid_hazards(self):
        """Verify returned hazards are valid EnvironmentalHazards."""
        hazards = generate_environmental_hazards(12345)
        for hazard in hazards:
            assert hazard in ENVIRONMENTAL_HAZARD_NAMES

    def test_same_seed_produces_same_result(self):
        """Verify seeded RNG produces reproducible results."""
        result1 = generate_environmental_hazards(42)
        result2 = generate_environmental_hazards(42)
        assert result1 == result2

    def test_count_parameter(self):
        """Verify count parameter controls number of hazards."""
        hazards_1 = generate_environmental_hazards(12345, count=1)
        hazards_3 = generate_environmental_hazards(12345, count=3)
        assert len(hazards_1) == 1
        assert len(hazards_3) == 3

    def test_default_count_range(self):
        """Verify default count is between 1 and 3."""
        for i in range(100):
            hazards = generate_environmental_hazards(i)
            assert 1 <= len(hazards) <= 3

    def test_no_duplicates(self):
        """Verify returned hazards contain no duplicates."""
        hazards = generate_environmental_hazards(12345)
        assert len(hazards) == len(set(hazards))


class TestGenerateWorld:
    """Tests for generate_world function."""

    def test_returns_world_details(self):
        """Verify generate_world returns a WorldDetails."""
        world = generate_world(12345)
        assert isinstance(world, WorldDetails)

    def test_world_type_valid(self):
        """Verify world has valid world type."""
        world = generate_world(12345)
        assert world.world_type in WORLD_TYPE_NAMES

    def test_natural_features_valid(self):
        """Verify world has valid natural features."""
        world = generate_world(12345)
        assert len(world.natural_features) >= 1
        assert len(world.natural_features) <= 3
        for feature in world.natural_features:
            assert feature in NATURAL_FEATURE_NAMES

    def test_anthropocentric_feature_valid(self):
        """Verify world's anthropocentric feature is valid."""
        world = generate_world(12345)
        assert world.anthropocentric_feature is None or (
            world.anthropocentric_feature in ANTHROPOCENTRIC_FEATURE_NAMES
        )

    def test_environmental_hazards_valid(self):
        """Verify world has valid environmental hazards."""
        world = generate_world(12345)
        assert len(world.environmental_hazards) >= 1
        assert len(world.environmental_hazards) <= 3
        for hazard in world.environmental_hazards:
            assert hazard in ENVIRONMENTAL_HAZARD_NAMES

    def test_same_seed_produces_same_result(self):
        """Verify seeded RNG produces reproducible results."""
        world1 = generate_world(42)
        world2 = generate_world(42)
        assert world1 == world2

    def test_different_seeds_produce_different_worlds(self):
        """Verify different seeds produce different worlds."""
        worlds = {generate_world(i).world_type for i in range(50)}
        assert len(worlds) > 1


class TestWorldDetailsModel:
    """Tests for WorldDetails Pydantic model."""

    def test_valid_world_details(self):
        """Verify WorldDetails can be created with valid data."""
        world = WorldDetails(
            world_type="temperate_healthy",
            natural_features=["breathable_atmosphere", "monobiome"],
            anthropocentric_feature="colony_stable",
            environmental_hazards=["extreme_cold", "particulate_storm"],
        )
        assert world.world_type == "temperate_healthy"
        assert len(world.natural_features) == 2
        assert world.anthropocentric_feature == "colony_stable"
        assert len(world.environmental_hazards) == 2

    def test_minimal_world_details(self):
        """Verify WorldDetails can be created with minimal data."""
        world = WorldDetails(
            world_type="barren_radiation",
            natural_features=["inert_core"],
            environmental_hazards=["tomb_world"],
        )
        assert world.world_type == "barren_radiation"
        assert len(world.natural_features) == 1
        assert world.anthropocentric_feature is None
        assert len(world.environmental_hazards) == 1

    def test_world_details_frozen(self):
        """Verify WorldDetails is frozen (immutable)."""
        world = WorldDetails(
            world_type="ocean_world",
            natural_features=["cosmopolitan"],
            environmental_hazards=["ocean_world"],
        )
        with pytest.raises(Exception):
            world.world_type = "temperate_healthy"


class TestDisplayNameFunctions:
    """Tests for display name helper functions."""

    def test_get_world_type_name(self):
        """Verify get_world_type_name returns correct name."""
        name = get_world_type_name("temperate_healthy")
        assert "Temperate World (Healthy)" in name

    def test_get_natural_feature_name(self):
        """Verify get_natural_feature_name returns correct name."""
        name = get_natural_feature_name("twin_suns")
        assert "Twin Suns" in name

    def test_get_anthropocentric_feature_name(self):
        """Verify get_anthropocentric_feature_name returns correct name."""
        name = get_anthropocentric_feature_name("colony_stable")
        assert "Colonial Settlement (Stable)" in name

    def test_get_environmental_hazard_name(self):
        """Verify get_environmental_hazard_name returns correct name."""
        name = get_environmental_hazard_name("extreme_cold")
        assert "Extreme Cold" in name

    def test_unknown_type_returns_unknown(self):
        """Verify unknown types return 'Unknown'."""
        assert get_world_type_name("unknown_type") == "Unknown"
        assert get_natural_feature_name("unknown_feature") == "Unknown"
        assert get_anthropocentric_feature_name("unknown_feature") == "Unknown"
        assert get_environmental_hazard_name("unknown_hazard") == "Unknown"


class TestSeededReproducibility:
    """Tests for seeded RNG reproducibility across multiple calls."""

    def test_generate_world_reproducible(self):
        """Verify multiple generate_world calls with same seed are identical."""
        for seed in [1, 42, 100, 9999, 123456]:
            world1 = generate_world(seed)
            world2 = generate_world(seed)
            assert world1 == world2

    def test_component_generation_reproducible(self):
        """Verify component generation functions are reproducible."""
        for seed in [1, 42, 100]:
            assert generate_world_type(seed) == generate_world_type(seed)
            assert generate_natural_features(seed) == generate_natural_features(seed)
            assert generate_environmental_hazards(
                seed
            ) == generate_environmental_hazards(seed)
            assert generate_anthropocentric_feature(
                seed
            ) == generate_anthropocentric_feature(seed)

    def test_different_seeds_different_worlds(self):
        """Verify different seeds produce different world types."""
        world_types = {generate_world_type(i) for i in range(1000)}
        assert len(world_types) > 5

    def test_different_seeds_different_features(self):
        """Verify different seeds produce different feature combinations."""
        feature_sets = {tuple(generate_natural_features(i)) for i in range(100)}
        assert len(feature_sets) > 10

    def test_different_seeds_different_hazards(self):
        """Verify different seeds produce different hazard combinations."""
        hazard_sets = {tuple(generate_environmental_hazards(i)) for i in range(100)}
        assert len(hazard_sets) > 10
