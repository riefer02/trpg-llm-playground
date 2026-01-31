"""Unit tests for difficulty scaling module."""

import pytest
from llm.src.mission.difficulty import (
    get_encounter_difficulty,
    get_npc_tier_distribution,
    get_ai_aggression,
    get_mission_difficulty_stars,
    get_encounter_difficulty_for_mission,
)


class TestDifficultyScaling:
    """Test difficulty scaling formulas."""

    @pytest.mark.parametrize(
        "pilot_level,expected",
        [
            (0, "trivial"),
            (1, "trivial"),
            (2, "trivial"),
            (3, "easy"),
            (4, "easy"),
            (5, "easy"),
            (6, "standard"),
            (7, "standard"),
            (8, "standard"),
            (9, "hard"),
            (10, "hard"),
            (11, "hard"),
            (12, "extreme"),
        ],
    )
    def test_get_encounter_difficulty(self, pilot_level, expected):
        """Test pilot level to encounter difficulty mapping."""
        result = get_encounter_difficulty(pilot_level)
        assert result == expected

    def test_get_encounter_difficulty_out_of_range(self):
        """Test that extreme is returned for levels > 12."""
        assert get_encounter_difficulty(13) == "extreme"
        assert get_encounter_difficulty(100) == "extreme"
        # Negative levels should also map to trivial (clamped by formula)
        assert get_encounter_difficulty(-1) == "trivial"

    @pytest.mark.parametrize(
        "pilot_level,expected_dist",
        [
            (0, {"tier_1": 1.0, "tier_2": 0.0, "tier_3": 0.0}),
            (3, {"tier_1": 1.0, "tier_2": 0.0, "tier_3": 0.0}),
            (4, {"tier_1": 0.5, "tier_2": 0.5, "tier_3": 0.0}),
            (7, {"tier_1": 0.5, "tier_2": 0.5, "tier_3": 0.0}),
            (8, {"tier_1": 0.25, "tier_2": 0.5, "tier_3": 0.25}),
            (12, {"tier_1": 0.25, "tier_2": 0.5, "tier_3": 0.25}),
        ],
    )
    def test_get_npc_tier_distribution(self, pilot_level, expected_dist):
        """Test NPC tier distribution scaling."""
        result = get_npc_tier_distribution(pilot_level)
        assert result == expected_dist
        # Ensure proportions sum to 1.0
        assert sum(result.values()) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "pilot_level,expected_aggression",
        [
            (0, 0.0),  # clamped to 0.0
            (3, 0.0),  # threshold
            (4, 0.11),  # (4-3)/9 ≈ 0.111... rounded to 2 decimals
            (6, 0.33),  # (6-3)/9 rounded
            (9, 0.67),  # (9-3)/9 = 0.666... rounded
            (12, 1.0),
            (15, 1.0),  # clamped to 1.0
        ],
    )
    def test_get_ai_aggression(self, pilot_level, expected_aggression):
        """Test AI aggression scaling."""
        result = get_ai_aggression(pilot_level)
        assert result == pytest.approx(expected_aggression, abs=0.005)
        assert 0.0 <= result <= 1.0

    @pytest.mark.parametrize(
        "pilot_level,mission_index,expected_stars",
        [
            (0, 0, 1),
            (0, 1, 2),
            (0, 2, 3),
            (3, 0, 1),  # pilot_level//4 = 0, base=1
            (4, 0, 2),  # pilot_level//4 = 1, base=2
            (7, 0, 2),
            (8, 0, 3),
            (12, 0, 3),
            (12, 1, 3),  # capped at 3
            (12, 2, 3),
        ],
    )
    def test_get_mission_difficulty_stars(
        self, pilot_level, mission_index, expected_stars
    ):
        """Test mission difficulty stars calculation."""
        result = get_mission_difficulty_stars(pilot_level, mission_index)
        assert result == expected_stars
        assert 1 <= result <= 3

    @pytest.mark.parametrize(
        "pilot_level,mission_index,expected_difficulty",
        [
            (0, 0, "trivial"),
            (0, 1, "easy"),
            (0, 2, "standard"),
            (3, 0, "easy"),
            (3, 1, "standard"),
            (3, 2, "hard"),
            (6, 0, "standard"),
            (6, 1, "hard"),
            (6, 2, "extreme"),
            (9, 0, "hard"),
            (9, 1, "extreme"),
            (9, 2, "extreme"),  # capped at extreme
            (12, 0, "extreme"),
            (12, 1, "extreme"),
        ],
    )
    def test_get_encounter_difficulty_for_mission(
        self, pilot_level, mission_index, expected_difficulty
    ):
        """Test mission-index adjusted encounter difficulty."""
        result = get_encounter_difficulty_for_mission(pilot_level, mission_index)
        assert result == expected_difficulty
