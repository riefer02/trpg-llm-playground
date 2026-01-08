"""Tests for encounter building helpers (Priority 58).

Tests cover:
- Difficulty multipliers
- Player party power estimation
- SITREP force multipliers
- Enemy force calculation
- Victory point calculation with tier scaling
"""

import unittest
from core.gm_toolkit import (
    EncounterDifficulty,
    PlayerPartyPower,
    EnemyForceRecommendation,
    estimate_party_power,
    calculate_enemy_force,
    calculate_total_victory_points,
    get_sitrep_force_multipliers,
)
from core.npc.compendium import (
    GMS_GRUNT_T1,
    GMS_GRUNT_T2,
    GMS_ELITE_T2,
    IPSN_BOSS_T3,
    SSC_SPECIALIST_T2,
    PR2_ACE_T1,
    PR2_AEGIS_T1,
)
from core.npc.special_classes import (
    SPECIAL_ULTRA_T1,
    SPECIAL_GRUNT_T1,
    SPECIAL_HUMAN,
)


class TestDifficultyMultipliers(unittest.TestCase):
    """Test encounter difficulty scaling multipliers."""

    def test_difficulty_multipliers_exist(self):
        """Verify all difficulty levels have multipliers."""
        from core.gm_toolkit.encounter_builder import DIFFICULTY_MULTIPLIERS

        self.assertIn("trivial", DIFFICULTY_MULTIPLIERS)
        self.assertIn("easy", DIFFICULTY_MULTIPLIERS)
        self.assertIn("standard", DIFFICULTY_MULTIPLIERS)
        self.assertIn("hard", DIFFICULTY_MULTIPLIERS)
        self.assertIn("extreme", DIFFICULTY_MULTIPLIERS)

    def test_linear_scaling(self):
        """Verify linear scaling from trivial (0.5x) to extreme (2.0x)."""
        from core.gm_toolkit.encounter_builder import DIFFICULTY_MULTIPLIERS

        self.assertEqual(DIFFICULTY_MULTIPLIERS["trivial"], 0.5)
        self.assertEqual(DIFFICULTY_MULTIPLIERS["easy"], 0.75)
        self.assertEqual(DIFFICULTY_MULTIPLIERS["standard"], 1.0)
        self.assertEqual(DIFFICULTY_MULTIPLIERS["hard"], 1.5)
        self.assertEqual(DIFFICULTY_MULTIPLIERS["extreme"], 2.0)


class TestPlayerPartyPower(unittest.TestCase):
    """Test player party power estimation."""

    def test_level_0_power(self):
        """Level 0 party has base power = player_count."""
        party = PlayerPartyPower(player_count=4, avg_license_level=0)
        self.assertEqual(party.base_power, 4.0)

    def test_level_6_power(self):
        """Level 6 party has ~1.6x base power."""
        party = PlayerPartyPower(player_count=4, avg_license_level=6)
        self.assertAlmostEqual(party.base_power, 6.4)

    def test_level_12_power(self):
        """Level 12 party has ~2.2x base power."""
        party = PlayerPartyPower(player_count=4, avg_license_level=12)
        self.assertAlmostEqual(party.base_power, 8.8)

    def test_single_player_power(self):
        """Single player party has correct power scaling."""
        party = PlayerPartyPower(player_count=1, avg_license_level=6)
        self.assertAlmostEqual(party.base_power, 1.6)

    def test_estimate_party_power_function(self):
        """Test the convenience function."""
        party = estimate_party_power(player_count=3, avg_license_level=4)
        self.assertEqual(party.player_count, 3)
        self.assertEqual(party.avg_license_level, 4)
        self.assertAlmostEqual(party.base_power, 4.2)


class TestSitrepForceMultipliers(unittest.TestCase):
    """Test SITREP reserve pattern to force multiplier mapping."""

    def test_extract_no_initial(self):
        """Extract has 0 initial, 2x reserve (all held in reserve)."""
        multipliers = get_sitrep_force_multipliers("extract")
        self.assertEqual(multipliers["initial"], 0.0)
        self.assertEqual(multipliers["reserve"], 2.0)

    def test_escort_increasing(self):
        """Escort has 1x initial, 1x reserve (increasing pattern)."""
        multipliers = get_sitrep_force_multipliers("escort")
        self.assertEqual(multipliers["initial"], 1.0)
        self.assertEqual(multipliers["reserve"], 1.0)

    def test_control_normal(self):
        """Control has 1x initial, 0 reserve (no reserves)."""
        multipliers = get_sitrep_force_multipliers("control")
        self.assertEqual(multipliers["initial"], 1.0)
        self.assertEqual(multipliers["reserve"], 0.0)

    def test_gauntlet_half_half(self):
        """Gauntlet has 50% initial, 50% reserve."""
        multipliers = get_sitrep_force_multipliers("gauntlet")
        self.assertEqual(multipliers["initial"], 0.5)
        self.assertEqual(multipliers["reserve"], 0.5)

    def test_holdout_half_half(self):
        """Holdout has 50% initial, 50% reserve."""
        multipliers = get_sitrep_force_multipliers("hold_out")
        self.assertEqual(multipliers["initial"], 0.5)
        self.assertEqual(multipliers["reserve"], 0.5)

    def test_recon_normal(self):
        """Recon has 1x initial, 0 reserve (normal pattern)."""
        multipliers = get_sitrep_force_multipliers("recon")
        self.assertEqual(multipliers["initial"], 1.0)
        self.assertEqual(multipliers["reserve"], 0.0)


class TestEnemyForceCalculation(unittest.TestCase):
    """Test enemy force recommendation calculation."""

    def test_standard_encounter(self):
        """Standard difficulty with 4 players level 0."""
        party = estimate_party_power(player_count=4, avg_license_level=0)
        force = calculate_enemy_force("standard", "control", party)

        self.assertEqual(force.target_victory_points, 4.0)
        self.assertEqual(force.initial_victory_points, 4.0)
        self.assertEqual(force.reserve_victory_points, 0.0)

    def test_hard_encounter(self):
        """Hard difficulty doubles the target."""
        party = estimate_party_power(player_count=4, avg_license_level=0)
        force = calculate_enemy_force("hard", "control", party)

        self.assertEqual(force.target_victory_points, 6.0)
        self.assertEqual(force.initial_victory_points, 6.0)

    def test_extract_reserve_split(self):
        """Extract has 2x total force, all in reserve (0 initial, 8.0 reserve)."""
        party = estimate_party_power(player_count=4, avg_license_level=0)
        force = calculate_enemy_force("standard", "extract", party)

        self.assertEqual(force.target_victory_points, 8.0)
        self.assertEqual(force.initial_victory_points, 0.0)
        self.assertEqual(force.reserve_victory_points, 8.0)

    def test_gauntlet_half_split(self):
        """Gauntlet splits 50/50 between initial and reserve."""
        party = estimate_party_power(player_count=4, avg_license_level=0)
        force = calculate_enemy_force("standard", "gauntlet", party)

        self.assertEqual(force.target_victory_points, 4.0)
        self.assertEqual(force.initial_victory_points, 2.0)
        self.assertEqual(force.reserve_victory_points, 2.0)

    def test_experienced_party(self):
        """Level 6 party (power 6.4) with hard difficulty."""
        party = estimate_party_power(player_count=4, avg_license_level=6)
        force = calculate_enemy_force("hard", "control", party)

        self.assertAlmostEqual(force.target_victory_points, 9.6)
        self.assertAlmostEqual(force.initial_victory_points, 9.6)

    def test_invalid_sitrep_raises(self):
        """Invalid SITREP type raises ValueError."""
        party = estimate_party_power(player_count=4, avg_license_level=0)

        with self.assertRaises(ValueError):
            calculate_enemy_force("standard", "invalid_sitrep", party)

    def test_with_template_recommendations(self):
        """Templates are included in recommendations."""
        party = estimate_party_power(player_count=4, avg_license_level=0)
        templates = [GMS_GRUNT_T1, PR2_ACE_T1]
        force = calculate_enemy_force("standard", "control", party, templates)

        self.assertIn("gms_grunt_t1", force.recommended_template_ids)
        self.assertIn("pr2_ace_t1", force.recommended_template_ids)


class TestVictoryPointCalculation(unittest.TestCase):
    """Test victory point calculation with tier scaling."""

    def test_regular_npc_tier_1(self):
        """Tier 1 grunt has base victory count (0.25)."""
        total = calculate_total_victory_points([GMS_GRUNT_T1])
        self.assertEqual(total, 0.25)

    def test_regular_npc_tier_2(self):
        """Tier 2 grunt has 1.5x victory count (0.375)."""
        total = calculate_total_victory_points([GMS_GRUNT_T2])
        self.assertEqual(total, 0.375)

    def test_regular_npc_tier_3_boss(self):
        """Tier 3 boss has 2x victory count (2.0)."""
        total = calculate_total_victory_points([IPSN_BOSS_T3])
        self.assertEqual(total, 2.0)

    def test_special_class_npc(self):
        """Special class Ultra has 4.0 victory count."""
        total = calculate_total_victory_points([SPECIAL_ULTRA_T1])
        self.assertEqual(total, 4.0)

    def test_mixed_templates(self):
        """Mixed templates sum correctly."""
        templates = [
            GMS_GRUNT_T1,  # 0.25
            GMS_ELITE_T2,  # 0.75
            IPSN_BOSS_T3,  # 2.0
            SPECIAL_ULTRA_T1,  # 4.0
        ]
        total = calculate_total_victory_points(templates)
        self.assertEqual(total, 7.0)

    def test_empty_list(self):
        """Empty list returns 0 victory points."""
        total = calculate_total_victory_points([])
        self.assertEqual(total, 0.0)

    def test_multiple_same_template(self):
        """Multiple instances of same template add up."""
        templates = [GMS_GRUNT_T1] * 4
        total = calculate_total_victory_points(templates)
        self.assertEqual(total, 1.0)

    def test_special_class_grunt(self):
        """Special Grunt also has 0.25 victory count."""
        total = calculate_total_victory_points([SPECIAL_GRUNT_T1])
        self.assertEqual(total, 0.25)

    def test_special_class_human(self):
        """Human has 1.0 victory count."""
        total = calculate_total_victory_points([SPECIAL_HUMAN])
        self.assertEqual(total, 1.0)


class TestTierMultipliers(unittest.TestCase):
    """Test tier scaling multipliers."""

    def test_tier_multipliers_exist(self):
        """Verify tier multipliers are defined."""
        from core.gm_toolkit.encounter_builder import TIER_MULTIPLIERS

        self.assertEqual(TIER_MULTIPLIERS["tier_1"], 1.0)
        self.assertEqual(TIER_MULTIPLIERS["tier_2"], 1.5)
        self.assertEqual(TIER_MULTIPLIERS["tier_3"], 2.0)


if __name__ == "__main__":
    unittest.main()
