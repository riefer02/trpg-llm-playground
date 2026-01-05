"""Tests for prone status mechanics.

Tests cover:
- ProneStatus state tracking model
- StandUpAttempt input model
- StandUpResult output model
- can_stand_up() prerequisite checks
- attempt_stand_up() full resolution
- get_prone_penalties() mechanical effects
- is_prone_movement_difficult() terrain interaction
- can_be_knocked_prone() knock prone checks
- apply_prone() condition application
- remove_prone() condition removal
"""

import unittest
from core.shared.prone import (
    ProneStatus,
    StandUpAttempt,
    StandUpResult,
    ProneEffects,
    KnockProneAttempt,
    KnockProneResult,
    can_stand_up,
    attempt_stand_up,
    get_prone_effects,
    is_prone_movement_difficult,
    get_attack_accuracy_bonus_from_prone,
    can_be_knocked_prone,
    apply_prone,
    remove_prone,
    is_slowed_from_prone,
    get_slowed_effects,
    is_movement_restricted_by_slowed,
    create_prone_status,
)
from core.shared.enums import StatusType


class TestProneStatus(unittest.TestCase):
    """Tests for ProneStatus model."""

    def test_default_status_not_prone(self):
        """Default status has is_prone=False."""
        status = ProneStatus()
        self.assertFalse(status.is_prone)
        self.assertFalse(status.is_slowed)
        self.assertFalse(status.counts_as_difficult_terrain)
        self.assertEqual(status.attackers_accuracy_bonus, 0)

    def test_prone_status_values(self):
        """Active prone status has correct values."""
        status = ProneStatus(
            is_prone=True,
            is_slowed=True,
            counts_as_difficult_terrain=True,
            attackers_accuracy_bonus=1,
        )
        self.assertTrue(status.is_prone)
        self.assertTrue(status.is_slowed)
        self.assertTrue(status.counts_as_difficult_terrain)
        self.assertEqual(status.attackers_accuracy_bonus, 1)


class TestStandUpAttempt(unittest.TestCase):
    """Tests for StandUpAttempt model."""

    def test_default_attempt(self):
        """Default attempt has sensible values."""
        attempt = StandUpAttempt()
        self.assertFalse(attempt.is_prone)
        self.assertFalse(attempt.is_immobilized)
        self.assertTrue(attempt.has_full_action_available)
        self.assertFalse(attempt.is_stunned)
        self.assertFalse(attempt.is_shutdown)

    def test_stand_attempt_prone(self):
        """Attempt when character is prone."""
        attempt = StandUpAttempt(
            is_prone=True,
            has_full_action_available=True,
        )
        self.assertTrue(attempt.is_prone)


class TestStandUpResult(unittest.TestCase):
    """Tests for StandUpResult model."""

    def test_successful_result(self):
        """Successful stand up requires new_status when creating directly."""
        result = StandUpResult(
            stand_successful=True,
            prone_cleared=True,
            slowed_cleared=True,
            new_status=ProneStatus(),
        )
        self.assertTrue(result.stand_successful)
        self.assertTrue(result.prone_cleared)
        self.assertTrue(result.slowed_cleared)
        self.assertIsNotNone(result.new_status)

    def test_failed_result(self):
        """Failed stand up has correct defaults."""
        result = StandUpResult(
            stand_successful=False,
            reason="Immobilized characters cannot stand up",
        )
        self.assertFalse(result.stand_successful)
        self.assertFalse(result.prone_cleared)
        self.assertFalse(result.slowed_cleared)
        self.assertIsNone(result.new_status)


class TestCanStandUp(unittest.TestCase):
    """Tests for can_stand_up() function."""

    def test_can_stand_up_normal(self):
        """Normal prone character can stand up."""
        can_stand, reason = can_stand_up(
            is_prone=True,
            is_immobilized=False,
            has_full_action_available=True,
            is_stunned=False,
            is_shutdown=False,
        )
        self.assertTrue(can_stand)
        self.assertEqual(reason, "Stand up action available")

    def test_cannot_stand_not_prone(self):
        """Character not prone cannot stand up."""
        can_stand, reason = can_stand_up(
            is_prone=False,
            is_immobilized=False,
            has_full_action_available=True,
            is_stunned=False,
            is_shutdown=False,
        )
        self.assertFalse(can_stand)
        self.assertEqual(reason, "Character is not prone")

    def test_cannot_stand_immobilized(self):
        """Immobilized character cannot stand up."""
        can_stand, reason = can_stand_up(
            is_prone=True,
            is_immobilized=True,
            has_full_action_available=True,
            is_stunned=False,
            is_shutdown=False,
        )
        self.assertFalse(can_stand)
        self.assertEqual(reason, "Immobilized characters cannot stand up")

    def test_cannot_stand_no_action(self):
        """Character without full action cannot stand up."""
        can_stand, reason = can_stand_up(
            is_prone=True,
            is_immobilized=False,
            has_full_action_available=False,
            is_stunned=False,
            is_shutdown=False,
        )
        self.assertFalse(can_stand)
        self.assertEqual(reason, "No full action available for stand up")

    def test_cannot_stand_stunned(self):
        """Stunned character cannot stand up."""
        can_stand, reason = can_stand_up(
            is_prone=True,
            is_immobilized=False,
            has_full_action_available=True,
            is_stunned=True,
            is_shutdown=False,
        )
        self.assertFalse(can_stand)
        self.assertEqual(reason, "Stunned characters cannot take actions")

    def test_cannot_stand_shutdown(self):
        """Shutdown mech cannot stand up."""
        can_stand, reason = can_stand_up(
            is_prone=True,
            is_immobilized=False,
            has_full_action_available=True,
            is_stunned=False,
            is_shutdown=True,
        )
        self.assertFalse(can_stand)
        self.assertEqual(reason, "Shutdown mechs cannot take actions")


class TestAttemptStandUp(unittest.TestCase):
    """Tests for attempt_stand_up() function."""

    def test_stand_up_succeeds(self):
        """Stand up action succeeds when conditions met."""
        attempt = StandUpAttempt(
            is_prone=True,
            is_immobilized=False,
            has_full_action_available=True,
        )
        result = attempt_stand_up(attempt, current_turn=3)
        self.assertTrue(result.stand_successful)
        self.assertTrue(result.prone_cleared)
        self.assertTrue(result.slowed_cleared)
        self.assertIsNotNone(result.new_status)
        self.assertFalse(result.new_status.is_prone)
        self.assertFalse(result.new_status.is_slowed)

    def test_stand_up_fails_immobilized(self):
        """Stand up fails when character is immobilized."""
        attempt = StandUpAttempt(
            is_prone=True,
            is_immobilized=True,
            has_full_action_available=True,
        )
        result = attempt_stand_up(attempt)
        self.assertFalse(result.stand_successful)
        self.assertFalse(result.prone_cleared)
        self.assertFalse(result.slowed_cleared)
        self.assertIsNone(result.new_status)
        self.assertIn("Immobilized", result.reason)

    def test_stand_up_fails_no_action(self):
        """Stand up fails when no full action available."""
        attempt = StandUpAttempt(
            is_prone=True,
            is_immobilized=False,
            has_full_action_available=False,
        )
        result = attempt_stand_up(attempt)
        self.assertFalse(result.stand_successful)
        self.assertIsNone(result.new_status)

    def test_stand_up_fails_not_prone(self):
        """Stand up fails when character is not prone."""
        attempt = StandUpAttempt(
            is_prone=False,
            is_immobilized=False,
            has_full_action_available=True,
        )
        result = attempt_stand_up(attempt)
        self.assertFalse(result.stand_successful)
        self.assertIn("not prone", result.reason)

    def test_stand_up_fails_stunned(self):
        """Stand up fails when character is stunned."""
        attempt = StandUpAttempt(
            is_prone=True,
            is_immobilized=False,
            has_full_action_available=True,
            is_stunned=True,
        )
        result = attempt_stand_up(attempt)
        self.assertFalse(result.stand_successful)
        self.assertIn("Stunned", result.reason)


class TestGetProneEffects(unittest.TestCase):
    """Tests for get_prone_effects() function."""

    def test_prone_effects_values(self):
        """Prone effects have correct mechanical values."""
        effects = get_prone_effects()
        self.assertEqual(effects.attackers_accuracy_bonus, 1)
        self.assertEqual(effects.movement_difficulty_modifier, 1)
        self.assertTrue(effects.is_slowed)
        self.assertTrue(effects.is_difficult_terrain)
        self.assertTrue(effects.requires_full_action_to_stand)
        self.assertTrue(effects.cannot_stand_if_immobilized)

    def test_prone_effects_not_slowed_standalone(self):
        """Prone effects indicate the target is slowed."""
        effects = get_prone_effects()
        self.assertTrue(effects.is_slowed)


class TestIsProneMovementDifficult(unittest.TestCase):
    """Tests for is_prone_movement_difficult() function."""

    def test_prone_is_difficult_terrain(self):
        """Prone character treats movement as difficult terrain."""
        result = is_prone_movement_difficult(is_prone=True)
        self.assertTrue(result)

    def test_not_prone_not_difficult(self):
        """Non-prone character doesn't have difficult terrain from prone."""
        result = is_prone_movement_difficult(is_prone=False)
        self.assertFalse(result)


class TestGetAttackAccuracyBonusFromProne(unittest.TestCase):
    """Tests for get_attack_accuracy_bonus_from_prone() function."""

    def test_prone_target_gives_bonus(self):
        """Attackers get +1 accuracy against prone target."""
        bonus = get_attack_accuracy_bonus_from_prone(is_prone=True)
        self.assertEqual(bonus, 1)

    def test_non_prone_target_no_bonus(self):
        """Non-prone targets don't give accuracy bonus."""
        bonus = get_attack_accuracy_bonus_from_prone(is_prone=False)
        self.assertEqual(bonus, 0)


class TestCanBeKnockedProne(unittest.TestCase):
    """Tests for can_be_knocked_prone() function."""

    def test_can_be_knocked_prone_normal(self):
        """Normal character can be knocked prone."""
        can_become, reason = can_be_knocked_prone(
            target_is_prone=False,
            target_is_flying=False,
        )
        self.assertTrue(can_become)
        self.assertEqual(reason, "Target can be knocked prone")

    def test_cannot_knock_already_prone(self):
        """Already prone character cannot be knocked prone again."""
        can_become, reason = can_be_knocked_prone(
            target_is_prone=True,
            target_is_flying=False,
        )
        self.assertFalse(can_become)
        self.assertEqual(reason, "Target is already prone")

    def test_cannot_knock_flying_prone(self):
        """Flying character cannot be knocked prone."""
        can_become, reason = can_be_knocked_prone(
            target_is_prone=False,
            target_is_flying=True,
        )
        self.assertFalse(can_become)
        self.assertEqual(reason, "Flying characters cannot be knocked prone")


class TestApplyProne(unittest.TestCase):
    """Tests for apply_prone() function."""

    def test_apply_prone_success(self):
        """Applying prone adds both prone and slowed conditions."""
        conditions: list[StatusType] = []
        success, reason, updated = apply_prone(conditions)
        self.assertTrue(success)
        self.assertIn("prone", updated)
        self.assertIn("slowed", updated)

    def test_apply_prone_already_prone(self):
        """Cannot apply prone if already prone."""
        conditions: list[StatusType] = ["prone", "slowed"]
        success, reason, updated = apply_prone(conditions)
        self.assertFalse(success)
        self.assertIn("already prone", reason)

    def test_apply_prone_adds_both_conditions(self):
        """Prone application adds prone AND slowed."""
        conditions: list[StatusType] = []
        apply_prone(conditions)
        self.assertEqual(len(conditions), 2)
        self.assertIn("prone", conditions)
        self.assertIn("slowed", conditions)


class TestRemoveProne(unittest.TestCase):
    """Tests for remove_prone() function."""

    def test_remove_prone_success(self):
        """Removing prone removes both prone and slowed conditions."""
        conditions: list[StatusType] = ["prone", "slowed", "impaired"]
        success, reason, updated = remove_prone(conditions)
        self.assertTrue(success)
        self.assertNotIn("prone", updated)
        self.assertNotIn("slowed", updated)
        self.assertIn("impaired", updated)

    def test_remove_prone_not_prone(self):
        """Cannot remove prone if not prone."""
        conditions: list[StatusType] = ["slowed"]
        success, reason, updated = remove_prone(conditions)
        self.assertFalse(success)
        self.assertEqual(reason, "Target is not prone")

    def test_remove_prone_preserves_other_conditions(self):
        """Removing prone preserves other conditions."""
        conditions: list[StatusType] = ["prone", "slowed", "shredded"]
        success, reason, updated = remove_prone(conditions)
        self.assertTrue(success)
        self.assertNotIn("prone", updated)
        self.assertNotIn("slowed", updated)
        self.assertIn("shredded", updated)


class TestIsSlowedFromProne(unittest.TestCase):
    """Tests for is_slowed_from_prone() function."""

    def test_prone_is_slowed(self):
        """Prone character has slowed condition."""
        result = is_slowed_from_prone(is_prone=True, conditions=[])
        self.assertTrue(result)

    def test_slowed_condition_from_list(self):
        """Slowed condition from list is detected."""
        result = is_slowed_from_prone(is_prone=False, conditions=["slowed"])
        self.assertTrue(result)

    def test_not_slowed(self):
        """Character without prone or slowed is not slowed."""
        result = is_slowed_from_prone(is_prone=False, conditions=["impaired"])
        self.assertFalse(result)


class TestGetSlowedEffects(unittest.TestCase):
    """Tests for get_slowed_effects() function."""

    def test_slowed_effects_values(self):
        """Slowed effects have correct mechanical values."""
        effects = get_slowed_effects()
        self.assertEqual(effects.attackers_accuracy_bonus, 0)
        self.assertEqual(effects.movement_difficulty_modifier, 0)
        self.assertTrue(effects.is_slowed)
        self.assertFalse(effects.is_difficult_terrain)
        self.assertFalse(effects.requires_full_action_to_stand)


class TestIsMovementRestrictedBySlowed(unittest.TestCase):
    """Tests for is_movement_restricted_by_slowed() function."""

    def test_slowed_restricts_movement(self):
        """Slowed condition restricts movement."""
        result = is_movement_restricted_by_slowed(is_slowed=True)
        self.assertTrue(result)

    def test_not_slowed_no_restriction(self):
        """Not slowed means no movement restriction."""
        result = is_movement_restricted_by_slowed(is_slowed=False)
        self.assertFalse(result)


class TestCreateProneStatus(unittest.TestCase):
    """Tests for create_prone_status() function."""

    def test_not_prone_status(self):
        """Creating status for non-prone character."""
        status = create_prone_status(is_prone=False)
        self.assertFalse(status.is_prone)
        self.assertFalse(status.is_slowed)
        self.assertFalse(status.counts_as_difficult_terrain)

    def test_prone_status(self):
        """Creating status for prone character."""
        status = create_prone_status(is_prone=True)
        self.assertTrue(status.is_prone)
        self.assertTrue(status.is_slowed)
        self.assertTrue(status.counts_as_difficult_terrain)
        self.assertEqual(status.attackers_accuracy_bonus, 1)


class TestProneModels(unittest.TestCase):
    """Tests for prone model validation and properties."""

    def test_prone_status_frozen(self):
        """ProneStatus is immutable."""
        status = ProneStatus(is_prone=True)
        with self.assertRaises(Exception):
            status.is_prone = False

    def test_stand_up_attempt_frozen(self):
        """StandUpAttempt is immutable."""
        attempt = StandUpAttempt(is_prone=True)
        with self.assertRaises(Exception):
            attempt.is_prone = False

    def test_stand_up_result_frozen(self):
        """StandUpResult is immutable."""
        result = StandUpResult(stand_successful=True)
        with self.assertRaises(Exception):
            result.stand_successful = False

    def test_prone_status_defaults(self):
        """ProneStatus has correct default values."""
        status = ProneStatus()
        self.assertFalse(status.is_prone)
        self.assertFalse(status.is_slowed)
        self.assertFalse(status.counts_as_difficult_terrain)
        self.assertEqual(status.attackers_accuracy_bonus, 0)

    def test_stand_up_result_defaults(self):
        """StandUpResult defaults reflect outcome is not yet resolved."""
        result = StandUpResult(stand_successful=True)
        self.assertFalse(result.prone_cleared)
        self.assertFalse(result.slowed_cleared)
        self.assertIsNone(result.new_status)


class TestProneIntegration(unittest.TestCase):
    """Integration tests for prone mechanics workflow."""

    def test_full_stand_up_workflow(self):
        """Complete workflow: become prone, then stand up."""
        conditions: list[StatusType] = []

        apply_success, apply_reason, updated_conditions = apply_prone(conditions)
        self.assertTrue(apply_success)
        self.assertIn("prone", updated_conditions)
        self.assertIn("slowed", updated_conditions)

        attempt = StandUpAttempt(
            is_prone=True,
            is_immobilized=False,
            has_full_action_available=True,
        )
        stand_result = attempt_stand_up(attempt)
        self.assertTrue(stand_result.stand_successful)

        remove_success, remove_reason, final_conditions = remove_prone(
            updated_conditions
        )
        self.assertTrue(remove_success)
        self.assertNotIn("prone", final_conditions)
        self.assertNotIn("slowed", final_conditions)

    def test_cannot_stand_while_immobilized_workflow(self):
        """Cannot stand up while immobilized (even with full action)."""
        conditions: list[StatusType] = ["prone", "slowed", "immobilized"]

        attempt = StandUpAttempt(
            is_prone=True,
            is_immobilized=True,
            has_full_action_available=True,
        )
        result = attempt_stand_up(attempt)
        self.assertFalse(result.stand_successful)
        self.assertIn("Immobilized", result.reason)

    def test_prone_effects_on_attack(self):
        """Prone target grants +1 accuracy to attackers."""
        bonus = get_attack_accuracy_bonus_from_prone(is_prone=True)
        self.assertEqual(bonus, 1)

        effects = get_prone_effects()
        self.assertEqual(effects.attackers_accuracy_bonus, 1)


if __name__ == "__main__":
    unittest.main()
