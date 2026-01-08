"""Tests for invisibility mechanics (Priority 56).

Tests cover:
- Detection checks for invisible targets
- 50% miss chance resolution
- Invisibility breaking conditions
- Heat signature calculations
"""

import unittest
from core.shared.invisibility import (
    detect_invisible_target,
    resolve_invisibility_miss_chance,
    check_invisibility_broken,
    calculate_heat_signature,
    can_always_hide_while_invisible,
    get_invisibility_miss_chance_description,
    INVISIBILITY_BREAK_CONDITIONS,
    InvisibilityDetectionAttempt,
    InvisibilityBreakCondition,
)


class TestInvisibilityDetection(unittest.TestCase):
    """Tests for detect_invisible_target function."""

    def test_pilot_detection_success(self):
        """Pilot detection succeeds on 1d20+skill >= 10."""
        attempt = InvisibilityDetectionAttempt(
            is_pilot_detection=True,
            pilot_skill_bonus=2,
        )
        result = detect_invisible_target(attempt, roll_result=12)
        self.assertTrue(result.detection_successful)
        self.assertEqual(12, result.detection_roll)
        self.assertEqual(14, result.detector_total)

    def test_pilot_detection_failure(self):
        """Pilot detection fails on 1d20+skill < 10."""
        attempt = InvisibilityDetectionAttempt(
            is_pilot_detection=True,
            pilot_skill_bonus=0,
        )
        result = detect_invisible_target(attempt, roll_result=5)
        self.assertFalse(result.detection_successful)
        self.assertEqual(5, result.detection_roll)
        self.assertEqual(5, result.detector_total)

    def test_mech_detection_systems_vs_systems(self):
        """Mech detection: Systems vs Systems contested check."""
        attempt = InvisibilityDetectionAttempt(
            detector_systems_bonus=3,
            target_systems_bonus=2,
            target_has_stealth_system=False,
            target_heat=5,  # Moderate heat for 0 bonus
        )
        result = detect_invisible_target(attempt, roll_result=15, target_roll_result=15)
        self.assertTrue(result.detection_successful)
        self.assertEqual(18, result.detector_total)
        self.assertEqual(17, result.target_total)

    def test_mech_detection_systems_vs_agility(self):
        """Mech detection uses better of Systems or Agility for target."""
        attempt = InvisibilityDetectionAttempt(
            detector_systems_bonus=4,
            target_systems_bonus=0,
            target_agility_bonus=3,
            target_heat=5,  # Moderate heat for 0 bonus
        )
        result = detect_invisible_target(attempt, roll_result=15, target_roll_result=15)
        self.assertTrue(result.detection_successful)
        self.assertEqual(19, result.detector_total)
        self.assertEqual(18, result.target_total)

    def test_stealth_system_bonus(self):
        """Stealth system gives target +2 bonus."""
        attempt = InvisibilityDetectionAttempt(
            detector_systems_bonus=4,
            target_systems_bonus=1,
            target_has_stealth_system=True,
            target_heat=5,  # Moderate heat for 0 bonus
        )
        result = detect_invisible_target(attempt, roll_result=10, target_roll_result=15)
        self.assertFalse(result.detection_successful)
        self.assertEqual(14, result.detector_total)
        self.assertEqual(18, result.target_total)


class TestInvisibilityMissChance(unittest.TestCase):
    """Tests for resolve_invisibility_miss_chance function."""

    def test_not_invisible_no_miss(self):
        """Non-invisible targets don't have miss chance applied."""
        result = resolve_invisibility_miss_chance(
            target_is_invisible=False,
        )
        self.assertFalse(result.miss_applies)
        self.assertEqual("Target is not invisible", result.reason)

    def test_invisible_with_ignore_no_miss(self):
        """Attacker that ignores invisibility doesn't cause miss."""
        result = resolve_invisibility_miss_chance(
            target_is_invisible=True,
            attacker_ignores_invisibility=True,
        )
        self.assertFalse(result.miss_applies)
        self.assertEqual("Attacker ignores invisibility", result.reason)

    def test_invisible_miss_on_1(self):
        """1d2 roll of 1 means miss."""
        result = resolve_invisibility_miss_chance(
            target_is_invisible=True,
            roll_result=1,
        )
        self.assertTrue(result.miss_applies)
        self.assertEqual("miss", result.miss_result)

    def test_invisible_hit_on_2(self):
        """1d2 roll of 2 means hit."""
        result = resolve_invisibility_miss_chance(
            target_is_invisible=True,
            roll_result=2,
        )
        self.assertFalse(result.miss_applies)
        self.assertEqual("hit", result.miss_result)

    def test_miss_chance_description(self):
        """Miss chance description is human-readable."""
        description = get_invisibility_miss_chance_description()
        self.assertIn("50%", description)
        self.assertIn("miss", description.lower())
        self.assertIn("invisible", description.lower())


class TestInvisibilityBreaking(unittest.TestCase):
    """Tests for check_invisibility_broken function."""

    def test_not_invisible_no_break(self):
        """Not invisible means invisibility not broken."""
        result = check_invisibility_broken(
            invisibility_source="stealth_hardsuit",
            current_conditions=[],
        )
        self.assertFalse(result.invisibility_broken)

    def test_stealth_hardsuit_breaks_on_damage(self):
        """Stealth Hardsuit breaks on damage."""
        result = check_invisibility_broken(
            invisibility_source="stealth_hardsuit",
            current_conditions=["invisible"],
            took_damage=True,
        )
        self.assertTrue(result.invisibility_broken)
        self.assertEqual("takes_damage", result.break_trigger)

    def test_stealth_hardsuit_breaks_on_attack(self):
        """Stealth Hardsuit breaks on attack."""
        result = check_invisibility_broken(
            invisibility_source="stealth_hardsuit",
            current_conditions=["invisible"],
            took_attack_action=True,
        )
        self.assertTrue(result.invisibility_broken)
        self.assertEqual("attacks", result.break_trigger)

    def test_integrated_cloak_breaks_end_of_turn(self):
        """Integrated Cloak breaks at end of turn."""
        result = check_invisibility_broken(
            invisibility_source="integrated_cloak",
            current_conditions=["invisible"],
            is_end_of_turn=True,
        )
        self.assertTrue(result.invisibility_broken)
        self.assertEqual("end_of_turn", result.break_trigger)

    def test_integrated_cloak_not_break_on_damage(self):
        """Integrated Cloak doesn't break on damage."""
        result = check_invisibility_broken(
            invisibility_source="integrated_cloak",
            current_conditions=["invisible"],
            took_damage=True,
        )
        self.assertFalse(result.invisibility_broken)

    def test_spectre_permanent_no_break_triggers(self):
        """Spectre permanent invisibility breaks on nothing."""
        result = check_invisibility_broken(
            invisibility_source="spectre_permanent",
            current_conditions=["invisible"],
            took_damage=True,
            took_attack_action=True,
            is_end_of_turn=True,
        )
        self.assertFalse(result.invisibility_broken)

    def test_flash_cloak_breaks_on_move(self):
        """Flash Cloak breaks on move."""
        result = check_invisibility_broken(
            invisibility_source="flash_cloak",
            current_conditions=["invisible"],
            took_move_action=True,
        )
        self.assertTrue(result.invisibility_broken)
        self.assertEqual("moves", result.break_trigger)

    def test_unknown_source_no_break(self):
        """Unknown invisibility source doesn't break unexpectedly."""
        result = check_invisibility_broken(
            invisibility_source="unknown_system",
            current_conditions=["invisible"],
            took_attack_action=True,
        )
        self.assertFalse(result.invisibility_broken)

    def test_mirage_breaks_on_attack_and_tech(self):
        """Mirage invisibility breaks on attack and tech action."""
        result = check_invisibility_broken(
            invisibility_source="mirage_invisibility",
            current_conditions=["invisible"],
            took_attack_action=True,
        )
        self.assertTrue(result.invisibility_broken)

        result = check_invisibility_broken(
            invisibility_source="mirage_invisibility",
            current_conditions=["invisible"],
            took_tech_action=True,
        )
        self.assertTrue(result.invisibility_broken)


class TestHeatSignature(unittest.TestCase):
    """Tests for calculate_heat_signature function."""

    def test_cold_signature(self):
        """Low heat (<=10%) gives cold signature with -2 bonus."""
        result = calculate_heat_signature(
            current_heat=1,
            heat_capacity=10,
        )
        self.assertEqual("cold", result.signature_level)
        self.assertEqual(-2, result.detection_difficulty_modifier)

    def test_low_signature(self):
        """Low heat (11-30%) gives low signature with -1 bonus."""
        result = calculate_heat_signature(
            current_heat=3,
            heat_capacity=10,
        )
        self.assertEqual("low", result.signature_level)
        self.assertEqual(-1, result.detection_difficulty_modifier)

    def test_moderate_signature(self):
        """Moderate heat (31-60%) gives normal detection."""
        result = calculate_heat_signature(
            current_heat=5,
            heat_capacity=10,
        )
        self.assertEqual("moderate", result.signature_level)
        self.assertEqual(0, result.detection_difficulty_modifier)

    def test_high_signature(self):
        """High heat (61-85%) gives +2 bonus to detector."""
        result = calculate_heat_signature(
            current_heat=7,
            heat_capacity=10,
        )
        self.assertEqual("high", result.signature_level)
        self.assertEqual(2, result.detection_difficulty_modifier)

    def test_critical_signature(self):
        """Critical heat (>85%) gives +4 bonus to detector."""
        result = calculate_heat_signature(
            current_heat=9,
            heat_capacity=10,
        )
        self.assertEqual("critical", result.signature_level)
        self.assertEqual(4, result.detection_difficulty_modifier)

    def test_heat_masking_bonus(self):
        """Heat masking provides additional -2 bonus."""
        result = calculate_heat_signature(
            current_heat=5,
            heat_capacity=10,
            has_heat_masking=True,
        )
        self.assertEqual(-2, result.detection_difficulty_modifier)

    def test_zero_heat_capacity(self):
        """Zero heat capacity handled gracefully."""
        result = calculate_heat_signature(
            current_heat=0,
            heat_capacity=0,
        )
        self.assertEqual("cold", result.signature_level)
        self.assertEqual(-2, result.detection_difficulty_modifier)

    def test_description_content(self):
        """Description contains useful information."""
        result = calculate_heat_signature(
            current_heat=8,
            heat_capacity=10,
        )
        self.assertIn("heat", result.description.lower())


class TestCanHideWhileInvisible(unittest.TestCase):
    """Tests for can_always_hide_while_invisible function."""

    def test_invisible_not_engaged_can_hide(self):
        """Invisible and not engaged = can always hide."""
        can_hide, reason = can_always_hide_while_invisible(
            is_invisible=True,
            is_engaged=False,
        )
        self.assertTrue(can_hide)
        self.assertIn("can always hide", reason)

    def test_invisible_engaged_cannot_hide(self):
        """Invisible but engaged = cannot hide."""
        can_hide, reason = can_always_hide_while_invisible(
            is_invisible=True,
            is_engaged=True,
        )
        self.assertFalse(can_hide)
        self.assertIn("ENGAGED", reason.upper())

    def test_not_invisible_cannot_hide_without_cover(self):
        """Not invisible requires cover to hide."""
        can_hide, reason = can_always_hide_while_invisible(
            is_invisible=False,
            is_engaged=False,
        )
        self.assertFalse(can_hide)
        self.assertIn("cover required", reason)


class TestInvisibilityBreakConditions(unittest.TestCase):
    """Tests for INVISIBILITY_BREAK_CONDITIONS dictionary."""

    def test_stealth_hardsuit_defined(self):
        """Stealth Hardsuit break conditions are defined."""
        conditions = INVISIBILITY_BREAK_CONDITIONS.get("stealth_hardsuit")
        self.assertIsNotNone(conditions)
        self.assertTrue(conditions.breaks_on_damage)
        self.assertTrue(conditions.breaks_on_attack)

    def test_integrated_cloak_defined(self):
        """Integrated Cloak break conditions are defined."""
        conditions = INVISIBILITY_BREAK_CONDITIONS.get("integrated_cloak")
        self.assertIsNotNone(conditions)
        self.assertFalse(conditions.breaks_on_damage)
        self.assertTrue(conditions.breaks_end_of_turn)

    def test_spectre_permanent_defined(self):
        """Spectre permanent invisibility breaks on nothing."""
        conditions = INVISIBILITY_BREAK_CONDITIONS.get("spectre_permanent")
        self.assertIsNotNone(conditions)
        self.assertFalse(conditions.breaks_on_damage)
        self.assertFalse(conditions.breaks_on_attack)
        self.assertFalse(conditions.breaks_end_of_turn)


if __name__ == "__main__":
    unittest.main()
