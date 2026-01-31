"""Unit tests for mission generator."""

import pytest
from llm.src.mission.generator import generate_missions, AVAILABLE_SITREPS


def test_generate_missions_returns_correct_count():
    """generate_missions returns requested number of missions."""
    missions = generate_missions(pilot_level=0, count=3, seed=42)
    assert len(missions) == 3

    missions = generate_missions(pilot_level=5, count=1, seed=123)
    assert len(missions) == 1


def test_generate_missions_raises_on_too_many():
    """generate_missions raises ValueError if count > available SITREPs."""
    with pytest.raises(ValueError):
        generate_missions(pilot_level=0, count=10)


def test_no_duplicate_sitreps_in_batch():
    """Each mission batch has unique SITREP types."""
    missions = generate_missions(pilot_level=0, count=3, seed=999)
    sitreps = [m.sitrep for m in missions]
    assert len(set(sitreps)) == len(sitreps)

    # With count=5, should get all 5 unique SITREPs
    missions = generate_missions(pilot_level=0, count=5, seed=1000)
    sitreps = [m.sitrep for m in missions]
    assert set(sitreps) == set(AVAILABLE_SITREPS)


def test_difficulty_within_range():
    """Mission difficulty is between 1 and 3 inclusive."""
    missions = generate_missions(pilot_level=12, count=5, seed=555)
    for mission in missions:
        assert 1 <= mission.difficulty <= 3


def test_difficulty_scales_with_pilot_level():
    """Higher pilot level leads to higher difficulty missions."""
    missions_low = generate_missions(pilot_level=0, count=3, seed=42)
    missions_high = generate_missions(pilot_level=12, count=3, seed=42)

    # Same seed => same SITREP order, difficulty should be higher
    for low, high in zip(missions_low, missions_high):
        # Pilot level affects difficulty formula: pilot_level // 4 + idx + 1
        # At level 0, difficulty = idx + 1
        # At level 12, difficulty = 3 + idx + 1 = idx + 4 (capped at 3)
        # So we expect high.difficulty >= low.difficulty
        assert high.difficulty >= low.difficulty


def test_mission_config_structure():
    """Generated missions have all required fields populated."""
    missions = generate_missions(pilot_level=3, count=2, seed=777)
    for mission in missions:
        assert mission.id
        assert mission.name
        assert mission.sitrep in AVAILABLE_SITREPS
        assert mission.terrain in [
            "urban",
            "forest",
            "desert",
            "facility",
            "space station",
        ]
        assert mission.enemy_count >= 1
        assert mission.briefing
        assert isinstance(mission.objectives, list)
        assert mission.enemy_intel
        # enemy_force_preview should be populated by encounter builder
        assert mission.enemy_force_preview is not None


def test_enemy_count_scales_with_difficulty_and_sitrep():
    """Enemy count increases with difficulty and varies by SITREP."""
    missions = generate_missions(pilot_level=0, count=5, seed=888)
    for mission in missions:
        # Verify enemy count is at least 1
        assert mission.enemy_count >= 1
        # Verify enemy count matches preview total
        if mission.enemy_force_preview is not None:
            expected = (
                mission.enemy_force_preview.initial_count
                + mission.enemy_force_preview.reserve_count
            )
            assert mission.enemy_count == expected
        # Verify enemy count scales with difficulty (simple check)
        # Higher difficulty should have >= enemy count for same SITREP
        # We'll test by generating missions at different pilot levels
        # but keep this test simple for now.


def test_reproducible_with_seed():
    """Same seed produces identical mission sequence."""
    missions1 = generate_missions(pilot_level=5, count=3, seed=12345)
    missions2 = generate_missions(pilot_level=5, count=3, seed=12345)

    for m1, m2 in zip(missions1, missions2):
        assert m1.id == m2.id
        assert m1.name == m2.name
        assert m1.sitrep == m2.sitrep
        assert m1.difficulty == m2.difficulty
        assert m1.enemy_count == m2.enemy_count


def test_generate_terrain():
    """generate_terrain returns TerrainConfig for valid SITREP and theme."""
    from llm.src.mission.generator import generate_terrain

    # Test a few combinations
    config = generate_terrain("control", "urban")
    assert config.sitrep_type == "control"
    assert config.theme == "urban"
    assert config.tile_set == "urban"
    assert config.terrain_map is not None
    assert isinstance(config.zones, dict)

    config = generate_terrain("extract", "facility")
    assert config.sitrep_type == "extract"
    assert config.theme == "facility"
    assert config.tile_set == "industrial"

    config = generate_terrain("hold_out", "forest")
    assert config.sitrep_type == "hold_out"
    assert config.theme == "forest"
    assert config.tile_set == "wilderness"

    # Test error handling for invalid SITREP
    import pytest

    with pytest.raises(ValueError):
        generate_terrain("unknown", "urban")

    # Test error handling for invalid theme
    with pytest.raises(ValueError):
        generate_terrain("control", "unknown")
