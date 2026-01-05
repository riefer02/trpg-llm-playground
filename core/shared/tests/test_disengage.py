"""Tests for disengage action mechanics.

Tests cover:
- DisengageStatus state tracking model
- DisengageAttempt input model
- DisengageResult output model
- can_disengage() prerequisite checks
- attempt_disengage() full resolution
- get_engagement_state() engagement detection
- is_movement_ignoring_engagement() movement phase checks
- check_reaction_prevented() reaction blocking logic
- end_disengage_at_turn_end() duration management
- can_disengage_from_grapple() grapple interaction
"""

import unittest
from core.shared.disengage import (
    DisengageStatus,
    DisengageAttempt,
    DisengageResult,
    EngagementCheck,
    can_disengage,
    attempt_disengage,
    get_engagement_state,
    is_movement_ignoring_engagement,
    check_reaction_prevented,
    end_disengage_at_turn_end,
    can_disengage_from_grapple,
)


class TestDisengageStatus(unittest.TestCase):
    """Tests for DisengageStatus model."""

    def test_default_status_inactive(self):
        """Default status has is_active=False."""
        status = DisengageStatus()
        self.assertFalse(status.is_active)
        self.assertIsNone(status.started_at_turn)
        self.assertFalse(status.ignores_engagement)
        self.assertFalse(status.prevents_reactions)

    def test_active_status_with_turn(self):
        """Active status tracks turn number."""
        status = DisengageStatus(
            is_active=True,
            started_at_turn=5,
            ignores_engagement=True,
            prevents_reactions=True,
        )
        self.assertTrue(status.is_active)
        self.assertEqual(status.started_at_turn, 5)
        self.assertTrue(status.ignores_engagement)
        self.assertTrue(status.prevents_reactions)


class TestDisengageAttempt(unittest.TestCase):
    """Tests for DisengageAttempt model."""

    def test_default_attempt(self):
        """Default attempt has sensible values."""
        attempt = DisengageAttempt()
        self.assertFalse(attempt.is_engaged)
        self.assertTrue(attempt.has_full_action_available)
        self.assertFalse(attempt.is_stunned)
        self.assertFalse(attempt.is_shutdown)
        self.assertFalse(attempt.is_prone)

    def test_engaged_attempt(self):
        """Attempt when character is engaged."""
        attempt = DisengageAttempt(
            is_engaged=True,
            has_full_action_available=True,
        )
        self.assertTrue(attempt.is_engaged)


class TestDisengageResult(unittest.TestCase):
    """Tests for DisengageResult model."""

    def test_successful_result(self):
        """Successful disengage has correct defaults."""
        result = DisengageResult(
            disengage_successful=True,
        )
        self.assertTrue(result.disengage_successful)
        self.assertTrue(result.ignores_engagement)
        self.assertTrue(result.prevents_reactions)
        self.assertTrue(result.duration_turn_end)

    def test_failed_result(self):
        """Failed disengage has effect flags set (action would grant these if successful).

        The effect flags default to True because they describe what the action grants.
        The disengage_successful flag indicates whether the action itself could be taken.
        """
        result = DisengageResult(
            disengage_successful=False,
        )
        self.assertFalse(result.disengage_successful)
        self.assertTrue(
            result.duration_turn_end
        )  # Duration describes what action grants


class TestCanDisengage(unittest.TestCase):
    """Tests for can_disengage() function."""

    def test_can_disengage_normal(self):
        """Normal character can disengage."""
        can_dis, reason = can_disengage(
            has_full_action_available=True,
            is_stunned=False,
            is_shutdown=False,
        )
        self.assertTrue(can_dis)
        self.assertEqual(reason, "Disengage action available")

    def test_cannot_disengage_stunned(self):
        """Stunned character cannot disengage."""
        can_dis, reason = can_disengage(
            has_full_action_available=True,
            is_stunned=True,
            is_shutdown=False,
        )
        self.assertFalse(can_dis)
        self.assertEqual(reason, "Stunned characters cannot take actions")

    def test_cannot_disengage_shutdown(self):
        """Shutdown mech cannot disengage."""
        can_dis, reason = can_disengage(
            has_full_action_available=True,
            is_stunned=False,
            is_shutdown=True,
        )
        self.assertFalse(can_dis)
        self.assertEqual(reason, "Shutdown mechs cannot take actions")

    def test_cannot_disengage_no_action(self):
        """Character without full action cannot disengage."""
        can_dis, reason = can_disengage(
            has_full_action_available=False,
            is_stunned=False,
            is_shutdown=False,
        )
        self.assertFalse(can_dis)
        self.assertEqual(reason, "No full action available for disengage")


class TestAttemptDisengage(unittest.TestCase):
    """Tests for attempt_disengage() function."""

    def test_disengage_succeeds(self):
        """Disengage action succeeds when conditions met."""
        attempt = DisengageAttempt(
            is_engaged=True,
            has_full_action_available=True,
        )
        result = attempt_disengage(attempt, current_turn=3)
        self.assertTrue(result.disengage_successful)
        self.assertIsNotNone(result.new_status)
        self.assertTrue(result.new_status.is_active)
        self.assertEqual(result.new_status.started_at_turn, 3)

    def test_disengage_fails_stunned(self):
        """Disengage fails when character is stunned."""
        attempt = DisengageAttempt(
            is_engaged=True,
            has_full_action_available=True,
            is_stunned=True,
        )
        result = attempt_disengage(attempt)
        self.assertFalse(result.disengage_successful)
        self.assertIsNone(result.new_status)

    def test_disengage_fails_shutdown(self):
        """Disengage fails when mech is shutdown."""
        attempt = DisengageAttempt(
            is_engaged=False,
            has_full_action_available=True,
            is_shutdown=True,
        )
        result = attempt_disengage(attempt)
        self.assertFalse(result.disengage_successful)
        self.assertIsNone(result.new_status)

    def test_disengage_fails_no_action(self):
        """Disengage fails when no full action available."""
        attempt = DisengageAttempt(
            is_engaged=False,
            has_full_action_available=False,
        )
        result = attempt_disengage(attempt)
        self.assertFalse(result.disengage_successful)
        self.assertIsNone(result.new_status)

    def test_disengage_result_ignores_engagement(self):
        """Disengage result has ignores_engagement=True."""
        attempt = DisengageAttempt(has_full_action_available=True)
        result = attempt_disengage(attempt)
        self.assertTrue(result.ignores_engagement)

    def test_disengage_result_prevents_reactions(self):
        """Disengage result has prevents_reactions=True."""
        attempt = DisengageAttempt(has_full_action_available=True)
        result = attempt_disengage(attempt)
        self.assertTrue(result.prevents_reactions)

    def test_disengage_result_duration(self):
        """Disengage result has duration_turn_end=True."""
        attempt = DisengageAttempt(has_full_action_available=True)
        result = attempt_disengage(attempt)
        self.assertTrue(result.duration_turn_end)

    def test_disengage_while_engaged(self):
        """Disengage can be used while engaged."""
        attempt = DisengageAttempt(
            is_engaged=True,
            has_full_action_available=True,
        )
        result = attempt_disengage(attempt)
        self.assertTrue(result.disengage_successful)
        self.assertIn("while engaged", result.reason)

    def test_disengage_not_engaged(self):
        """Disengage can be used when not engaged."""
        attempt = DisengageAttempt(
            is_engaged=False,
            has_full_action_available=True,
        )
        result = attempt_disengage(attempt)
        self.assertTrue(result.disengage_successful)
        self.assertIn("not engaged", result.reason)


class TestGetEngagementState(unittest.TestCase):
    """Tests for get_engagement_state() function."""

    def test_not_engaged_no_hostiles(self):
        """Not engaged when no adjacent hostiles."""
        check = get_engagement_state(adjacent_hostiles=0)
        self.assertFalse(check.is_engaged)
        self.assertEqual(check.adjacent_hostiles, 0)
        self.assertEqual(check.engagement_details, "Not engaged")

    def test_engaged_with_hostile(self):
        """Engaged when adjacent to hostile."""
        check = get_engagement_state(adjacent_hostiles=1)
        self.assertTrue(check.is_engaged)
        self.assertEqual(check.adjacent_hostiles, 1)
        self.assertIn("1 hostile", check.engagement_details)

    def test_engaged_multiple_hostiles(self):
        """Engaged when adjacent to multiple hostiles."""
        check = get_engagement_state(adjacent_hostiles=3)
        self.assertTrue(check.is_engaged)
        self.assertEqual(check.adjacent_hostiles, 3)
        self.assertIn("3 hostile", check.engagement_details)

    def test_immune_to_engagement(self):
        """Immune character not engaged."""
        check = get_engagement_state(
            adjacent_hostiles=2,
            is_immune_to_engagement=True,
        )
        self.assertFalse(check.is_engaged)
        self.assertEqual(check.engagement_details, "Immune to engagement")


class TestIsMovementIgnoringEngagement(unittest.TestCase):
    """Tests for is_movement_ignoring_engagement() function."""

    def test_no_ignoring_without_status(self):
        """Movement doesn't ignore engagement without status."""
        result = is_movement_ignoring_engagement(disengage_status=None)
        self.assertFalse(result)

    def test_disengage_ignores_engagement(self):
        """Disengage status causes movement to ignore engagement."""
        status = DisengageStatus(
            is_active=True,
            ignores_engagement=True,
        )
        result = is_movement_ignoring_engagement(disengage_status=status)
        self.assertTrue(result)

    def test_invisible_ignores_engagement(self):
        """Invisibility causes movement to ignore engagement."""
        result = is_movement_ignoring_engagement(
            disengage_status=None,
            is_invisible=True,
        )
        self.assertTrue(result)

    def test_teleporting_ignores_engagement(self):
        """Teleporting ignores engagement."""
        result = is_movement_ignoring_engagement(
            disengage_status=None,
            is_teleporting=True,
        )
        self.assertTrue(result)

    def test_involuntary_movement_ignores_engagement(self):
        """Involuntary movement ignores engagement."""
        result = is_movement_ignoring_engagement(
            disengage_status=None,
            is_involuntary_movement=True,
        )
        self.assertTrue(result)

    def test_inactive_disengage_status(self):
        """Inactive disengage status doesn't ignore engagement."""
        status = DisengageStatus(
            is_active=False,
            ignores_engagement=False,
        )
        result = is_movement_ignoring_engagement(disengage_status=status)
        self.assertFalse(result)

    def test_combined_ignoring(self):
        """Multiple sources can cause ignoring engagement."""
        result = is_movement_ignoring_engagement(
            disengage_status=None,
            is_invisible=True,
            is_teleporting=True,
        )
        self.assertTrue(result)


class TestCheckReactionPrevented(unittest.TestCase):
    """Tests for check_reaction_prevented() function."""

    def test_no_prevention_without_status(self):
        """Reactions not prevented without status."""
        prevented, reason = check_reaction_prevented(disengage_status=None)
        self.assertFalse(prevented)
        self.assertEqual(reason, "Reactions not prevented")

    def test_disengage_prevents_reactions(self):
        """Disengage status prevents reactions."""
        status = DisengageStatus(
            is_active=True,
            prevents_reactions=True,
        )
        prevented, reason = check_reaction_prevented(disengage_status=status)
        self.assertTrue(prevented)
        self.assertIn("Disengage", reason)

    def test_hidden_prevents_reactions(self):
        """Hidden condition prevents reactions from engagement."""
        prevented, reason = check_reaction_prevented(
            disengage_status=None,
            is_hidden=True,
        )
        self.assertTrue(prevented)
        self.assertIn("Hidden", reason)

    def test_invisible_prevents_reactions(self):
        """Invisibility prevents reactions."""
        prevented, reason = check_reaction_prevented(
            disengage_status=None,
            is_invisible=True,
        )
        self.assertTrue(prevented)
        self.assertIn("Invisibility", reason)

    def test_teleporting_prevents_reactions(self):
        """Teleporting prevents reactions."""
        prevented, reason = check_reaction_prevented(
            disengage_status=None,
            is_teleporting=True,
        )
        self.assertTrue(prevented)
        self.assertIn("Teleporting", reason)

    def test_involuntary_movement_prevents_reactions(self):
        """Involuntary movement prevents reactions."""
        prevented, reason = check_reaction_prevented(
            disengage_status=None,
            is_involuntary_movement=True,
        )
        self.assertTrue(prevented)
        self.assertIn("Involuntary", reason)

    def test_inactive_disengage_status(self):
        """Inactive disengage status doesn't prevent reactions."""
        status = DisengageStatus(
            is_active=False,
            prevents_reactions=False,
        )
        prevented, reason = check_reaction_prevented(disengage_status=status)
        self.assertFalse(prevented)


class TestEndDisengageAtTurnEnd(unittest.TestCase):
    """Tests for end_disengage_at_turn_end() function."""

    def test_ends_active_disengage(self):
        """Active disengage is ended at turn end."""
        status = DisengageStatus(
            is_active=True,
            started_at_turn=5,
            ignores_engagement=True,
            prevents_reactions=True,
        )
        new_status = end_disengage_at_turn_end(status, current_turn=5)
        self.assertFalse(new_status.is_active)
        self.assertFalse(new_status.ignores_engagement)
        self.assertFalse(new_status.prevents_reactions)

    def test_different_turn_no_change(self):
        """Disengage doesn't end on different turn."""
        status = DisengageStatus(
            is_active=True,
            started_at_turn=3,
            ignores_engagement=True,
            prevents_reactions=True,
        )
        new_status = end_disengage_at_turn_end(status, current_turn=5)
        self.assertTrue(new_status.is_active)

    def test_inactive_disengage_no_change(self):
        """Inactive disengage remains inactive."""
        status = DisengageStatus(is_active=False)
        new_status = end_disengage_at_turn_end(status, current_turn=5)
        self.assertFalse(new_status.is_active)


class TestCanDisengageFromGrapple(unittest.TestCase):
    """Tests for can_disengage_from_grapple() function."""

    def test_not_grappled_can_disengage(self):
        """Not grappled character can disengage."""
        can_dis, reason = can_disengage_from_grapple(
            is_grappling=False,
            is_same_size_or_larger=False,
        )
        self.assertTrue(can_dis)

    def test_grappled_same_size_cannot(self):
        """Grappled with same/larger target cannot disengage."""
        can_dis, reason = can_disengage_from_grapple(
            is_grappling=True,
            is_same_size_or_larger=True,
        )
        self.assertFalse(can_dis)
        self.assertIn("quick action", reason)

    def test_grappled_smaller_can_disengage(self):
        """Grappled with smaller target can end grapple."""
        can_dis, reason = can_disengage_from_grapple(
            is_grappling=True,
            is_same_size_or_larger=False,
        )
        self.assertTrue(can_dis)
        self.assertIn("free action", reason)


class TestDisengageModels(unittest.TestCase):
    """Tests for disengage model validation."""

    def test_disengage_status_frozen(self):
        """DisengageStatus is immutable."""
        status = DisengageStatus(is_active=True)
        with self.assertRaises(Exception):
            status.is_active = False

    def test_disengage_attempt_frozen(self):
        """DisengageAttempt is immutable."""
        attempt = DisengageAttempt(is_engaged=True)
        with self.assertRaises(Exception):
            attempt.is_engaged = False

    def test_disengage_result_frozen(self):
        """DisengageResult is immutable."""
        result = DisengageResult(disengage_successful=True)
        with self.assertRaises(Exception):
            result.disengage_successful = False

    def test_disengage_status_defaults(self):
        """DisengageStatus has correct default values."""
        status = DisengageStatus()
        self.assertFalse(status.is_active)
        self.assertIsNone(status.started_at_turn)
        self.assertFalse(status.ignores_engagement)
        self.assertFalse(status.prevents_reactions)

    def test_engagement_check_defaults(self):
        """EngagementCheck has correct default values."""
        check = EngagementCheck(is_engaged=False)
        self.assertEqual(check.adjacent_hostiles, 0)
        self.assertEqual(check.engagement_details, "")


if __name__ == "__main__":
    unittest.main()
