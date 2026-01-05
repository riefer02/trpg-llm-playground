"""Tests for flying status mechanics.

Tests cover:
- FlyingStatus state tracking model
- FlightAttempt input model
- FlightResult output model
- can_takeoff() prerequisite checks
- attempt_takeoff() full resolution
- can_land() prerequisite checks
- attempt_landing() landing and fall mechanics
- calculate_fall_damage() damage calculation
- get_flight_effects() mechanical effects
- can_be_knocked_prone_while_flying() interaction with prone
- should_fall_from_flying() status-triggered fall
- create_flying_status() factory function
- is_hover_mode() mode checks
- get_altitude_accuracy_bonus() elevation bonus
"""

import unittest
from core.shared.flying import (
    FlyingStatus,
    FlightAttempt,
    FlightResult,
    LandingAttempt,
    LandingResult,
    FlightEffects,
    FallDamageResult,
    can_takeoff,
    attempt_takeoff,
    can_land,
    attempt_landing,
    calculate_fall_damage,
    get_flight_effects,
    is_flying,
    can_be_knocked_prone_while_flying,
    should_fall_from_flying,
    create_flying_status,
    is_hover_mode,
    get_altitude_accuracy_bonus,
)
from core.mech.combat_rules import FlightRules, DEFAULT_MECH_COMBAT_RULES


class TestFlyingStatus(unittest.TestCase):
    """Tests for FlyingStatus model."""

    def test_default_status_not_flying(self):
        """Default status has is_flying=False."""
        status = FlyingStatus()
        self.assertFalse(status.is_flying)
        self.assertEqual(status.altitude_level, 0)
        self.assertFalse(status.is_hover)
        self.assertEqual(status.movement_mode, "ground")

    def test_flying_status_values(self):
        """Active flying status has correct values."""
        status = FlyingStatus(
            is_flying=True,
            altitude_level=2,
            is_hover=False,
            movement_mode="flight",
        )
        self.assertTrue(status.is_flying)
        self.assertEqual(status.altitude_level, 2)
        self.assertFalse(status.is_hover)
        self.assertEqual(status.movement_mode, "flight")

    def test_hover_status_values(self):
        """Hover status has correct values."""
        status = FlyingStatus(
            is_flying=True,
            altitude_level=1,
            is_hover=True,
            movement_mode="hover",
        )
        self.assertTrue(status.is_flying)
        self.assertTrue(status.is_hover)
        self.assertEqual(status.movement_mode, "hover")

    def test_flying_status_immutable(self):
        """FlyingStatus is immutable (frozen)."""
        status = FlyingStatus(is_flying=True, altitude_level=1)
        with self.assertRaises(Exception):
            status.is_flying = False


class TestFlightAttempt(unittest.TestCase):
    """Tests for FlightAttempt model."""

    def test_flight_attempt_defaults(self):
        """FlightAttempt has sensible defaults."""
        attempt = FlightAttempt()
        self.assertFalse(attempt.is_flying)
        self.assertFalse(attempt.is_stunned)
        self.assertFalse(attempt.is_shutdown)
        self.assertTrue(attempt.has_movement_available)
        self.assertEqual(attempt.target_altitude, 1)


class TestFlightResult(unittest.TestCase):
    """Tests for FlightResult model."""

    def test_flight_result_success(self):
        """Successful flight result."""
        result = FlightResult(
            takeoff_successful=True,
            is_flying=True,
            altitude_level=2,
            reason="Takeoff successful",
        )
        self.assertTrue(result.takeoff_successful)
        self.assertTrue(result.is_flying)

    def test_flight_result_failure(self):
        """Failed flight result."""
        result = FlightResult(
            takeoff_successful=False,
            reason="Already flying",
        )
        self.assertFalse(result.takeoff_successful)
        self.assertFalse(result.is_flying)


class TestLandingAttempt(unittest.TestCase):
    """Tests for LandingAttempt model."""

    def test_landing_attempt_defaults(self):
        """LandingAttempt has sensible defaults."""
        attempt = LandingAttempt()
        self.assertFalse(attempt.is_flying)
        self.assertEqual(attempt.current_altitude, 0)
        self.assertFalse(attempt.is_hover)
        self.assertFalse(attempt.is_immobilized)
        self.assertTrue(attempt.has_surface_below)


class TestLandingResult(unittest.TestCase):
    """Tests for LandingResult model."""

    def test_landing_result_success(self):
        """Successful landing result."""
        result = LandingResult(
            landed_successfully=True,
            fell=False,
            fall_damage=0,
            became_prone=False,
        )
        self.assertTrue(result.landed_successfully)
        self.assertFalse(result.fell)

    def test_landing_result_with_fall(self):
        """Landing result with fall and damage."""
        result = LandingResult(
            landed_successfully=False,
            fell=True,
            fall_damage=3,
            became_prone=True,
            reason="Fell from altitude 3",
        )
        self.assertFalse(result.landed_successfully)
        self.assertTrue(result.fell)
        self.assertEqual(result.fall_damage, 3)
        self.assertTrue(result.became_prone)


class TestFallDamageResult(unittest.TestCase):
    """Tests for FallDamageResult model."""

    def test_fall_damage_zero_altitude(self):
        """No damage from zero altitude."""
        result = calculate_fall_damage(0)
        self.assertEqual(result.damage_taken, 0)
        self.assertEqual(result.fell_from_altitude, 0)

    def test_fall_damage_from_altitude(self):
        """Damage scales with altitude."""
        result = calculate_fall_damage(3)
        self.assertEqual(result.damage_taken, 3)
        self.assertEqual(result.fell_from_altitude, 3)


class TestCanTakeoff(unittest.TestCase):
    """Tests for can_takeoff function."""

    def test_can_takeoff_grounded(self):
        """Can takeoff when grounded and ready."""
        can, reason = can_takeoff(
            is_flying=False,
            is_stunned=False,
            is_shutdown=False,
            has_movement_available=True,
        )
        self.assertTrue(can)
        self.assertEqual(reason, "Takeoff action available")

    def test_cannot_takeoff_already_flying(self):
        """Cannot takeoff when already flying."""
        can, reason = can_takeoff(
            is_flying=True,
            is_stunned=False,
            is_shutdown=False,
            has_movement_available=True,
        )
        self.assertFalse(can)
        self.assertEqual(reason, "Character is already flying")

    def test_cannot_takeoff_stunned(self):
        """Cannot takeoff when stunned."""
        can, reason = can_takeoff(
            is_flying=False,
            is_stunned=True,
            is_shutdown=False,
            has_movement_available=True,
        )
        self.assertFalse(can)
        self.assertEqual(reason, "Stunned characters cannot take actions")

    def test_cannot_takeoff_shutdown(self):
        """Cannot takeoff when shutdown."""
        can, reason = can_takeoff(
            is_flying=False,
            is_stunned=False,
            is_shutdown=True,
            has_movement_available=True,
        )
        self.assertFalse(can)
        self.assertEqual(reason, "Shutdown mechs cannot take actions")

    def test_cannot_takeoff_no_movement(self):
        """Cannot takeoff without movement available."""
        can, reason = can_takeoff(
            is_flying=False,
            is_stunned=False,
            is_shutdown=False,
            has_movement_available=False,
        )
        self.assertFalse(can)
        self.assertEqual(reason, "No movement action available for takeoff")


class TestAttemptTakeoff(unittest.TestCase):
    """Tests for attempt_takeoff function."""

    def test_attempt_takeoff_success(self):
        """Successful takeoff to flight."""
        attempt = FlightAttempt(
            is_flying=False,
            is_stunned=False,
            is_shutdown=False,
            has_movement_available=True,
            target_altitude=2,
        )
        result = attempt_takeoff(attempt)
        self.assertTrue(result.takeoff_successful)
        self.assertTrue(result.is_flying)
        self.assertEqual(result.altitude_level, 2)
        self.assertIsNotNone(result.new_status)

    def test_attempt_takeoff_failure_already_flying(self):
        """Failed takeoff when already flying."""
        attempt = FlightAttempt(
            is_flying=True,
            is_stunned=False,
            is_shutdown=False,
            has_movement_available=True,
        )
        result = attempt_takeoff(attempt)
        self.assertFalse(result.takeoff_successful)
        self.assertFalse(result.is_flying)

    def test_attempt_takeoff_failure_stunned(self):
        """Failed takeoff when stunned."""
        attempt = FlightAttempt(
            is_flying=False,
            is_stunned=True,
            is_shutdown=False,
            has_movement_available=True,
        )
        result = attempt_takeoff(attempt)
        self.assertFalse(result.takeoff_successful)
        self.assertIn("Stunned", result.reason)


class TestCanLand(unittest.TestCase):
    """Tests for can_land function."""

    def test_can_land_valid_surface(self):
        """Can land with valid surface below."""
        can, reason = can_land(
            is_flying=True,
            current_altitude=2,
            has_surface_below=True,
            surface_is_valid=True,
        )
        self.assertTrue(can)
        self.assertEqual(reason, "Landing action available")

    def test_cannot_land_not_flying(self):
        """Cannot land when not flying."""
        can, reason = can_land(
            is_flying=False,
            current_altitude=0,
            has_surface_below=True,
            surface_is_valid=True,
        )
        self.assertFalse(can)
        self.assertEqual(reason, "Character is not flying")

    def test_cannot_land_no_surface(self):
        """Cannot land with no surface below."""
        can, reason = can_land(
            is_flying=True,
            current_altitude=2,
            has_surface_below=False,
            surface_is_valid=True,
        )
        self.assertFalse(can)
        self.assertEqual(reason, "No surface below for landing")

    def test_cannot_land_invalid_surface(self):
        """Cannot land on invalid surface."""
        can, reason = can_land(
            is_flying=True,
            current_altitude=2,
            has_surface_below=True,
            surface_is_valid=False,
        )
        self.assertFalse(can)
        self.assertEqual(reason, "Surface is not valid for landing")


class TestAttemptLanding(unittest.TestCase):
    """Tests for attempt_landing function."""

    def test_attempt_landing_success(self):
        """Successful landing on valid surface."""
        attempt = LandingAttempt(
            is_flying=True,
            current_altitude=1,
            is_hover=False,
            is_immobilized=False,
            is_stunned=False,
            is_shutdown=False,
            has_surface_below=True,
            surface_is_valid=True,
        )
        result = attempt_landing(attempt)
        self.assertTrue(result.landed_successfully)
        self.assertFalse(result.fell)
        self.assertFalse(result.became_prone)

    def test_attempt_landing_fall_immobilized(self):
        """Fall when immobilized while flying."""
        attempt = LandingAttempt(
            is_flying=True,
            current_altitude=2,
            is_hover=False,
            is_immobilized=True,
            is_stunned=False,
            is_shutdown=False,
            has_surface_below=True,
            surface_is_valid=True,
        )
        result = attempt_landing(attempt)
        self.assertFalse(result.landed_successfully)
        self.assertTrue(result.fell)
        self.assertEqual(result.fall_damage, 2)
        self.assertTrue(result.became_prone)

    def test_attempt_landing_fall_stunned(self):
        """Fall when stunned while flying."""
        attempt = LandingAttempt(
            is_flying=True,
            current_altitude=3,
            is_hover=False,
            is_immobilized=False,
            is_stunned=True,
            is_shutdown=False,
            has_surface_below=True,
            surface_is_valid=True,
        )
        result = attempt_landing(attempt)
        self.assertFalse(result.landed_successfully)
        self.assertTrue(result.fell)
        self.assertEqual(result.fall_damage, 3)

    def test_attempt_landing_not_flying(self):
        """Landing when not flying returns error."""
        attempt = LandingAttempt(
            is_flying=False,
            current_altitude=0,
            is_hover=False,
            is_immobilized=False,
            is_stunned=False,
            is_shutdown=False,
            has_surface_below=True,
            surface_is_valid=True,
        )
        result = attempt_landing(attempt)
        self.assertFalse(result.landed_successfully)
        self.assertIn("not flying", result.reason)

    def test_attempt_landing_hover_stationary(self):
        """Hover mode allows stationary landing without falling."""
        attempt = LandingAttempt(
            is_flying=True,
            current_altitude=1,
            is_hover=True,
            is_immobilized=False,
            is_stunned=False,
            is_shutdown=False,
            has_surface_below=True,
            surface_is_valid=True,
            flight_rules=FlightRules(hover_allows_stationary=True),
        )
        result = attempt_landing(attempt)
        self.assertTrue(result.landed_successfully)
        self.assertFalse(result.fell)


class TestGetFlightEffects(unittest.TestCase):
    """Tests for get_flight_effects function."""

    def test_ground_effects(self):
        """Non-flying character has no flight effects."""
        effects = get_flight_effects(
            is_flying=False,
            altitude_level=0,
            is_hover=False,
        )
        self.assertFalse(effects.ignores_obstructions)
        self.assertFalse(effects.cannot_be_knocked_prone)
        self.assertEqual(effects.altitude_modifier, 0)

    def test_flight_effects_at_altitude(self):
        """Flying at altitude ignores obstructions."""
        effects = get_flight_effects(
            is_flying=True,
            altitude_level=2,
            is_hover=False,
        )
        self.assertTrue(effects.ignores_obstructions)
        self.assertTrue(effects.cannot_be_knocked_prone)
        self.assertEqual(effects.altitude_modifier, 2)

    def test_hover_effects(self):
        """Hover mode has hover allows stationary."""
        effects = get_flight_effects(
            is_flying=True,
            altitude_level=1,
            is_hover=True,
        )
        self.assertTrue(effects.hover_allows_stationary)
        self.assertEqual(effects.movement_mode, "hover")


class TestIsFlying(unittest.TestCase):
    """Tests for is_flying function."""

    def test_is_flying_true_flight_mode(self):
        """Returns True for flight mode."""
        self.assertTrue(is_flying(True, "flight"))

    def test_is_flying_true_hover_mode(self):
        """Returns True for hover mode."""
        self.assertTrue(is_flying(True, "hover"))

    def test_is_flying_false_ground_mode(self):
        """Returns False for ground mode."""
        self.assertFalse(is_flying(False, "ground"))

    def test_is_flying_false_teleport_mode(self):
        """Returns False for teleport mode."""
        self.assertFalse(is_flying(False, "teleport"))


class TestCanBeKnockedProneWhileFlying(unittest.TestCase):
    """Tests for can_be_knocked_prone_while_flying function."""

    def test_cannot_knock_flying_prone(self):
        """Cannot knock prone while flying."""
        can, reason = can_be_knocked_prone_while_flying(is_flying=True)
        self.assertFalse(can)
        self.assertEqual(reason, "Flying characters cannot be knocked prone")

    def test_can_knock_grounded_prone(self):
        """Can knock prone while grounded."""
        can, reason = can_be_knocked_prone_while_flying(is_flying=False)
        self.assertTrue(can)
        self.assertEqual(reason, "Character can be knocked prone")


class TestShouldFallFromFlying(unittest.TestCase):
    """Tests for should_fall_from_flying function."""

    def test_should_fall_immobilized(self):
        """Should fall when immobilized while flying."""
        should, reason = should_fall_from_flying(
            is_flying=True,
            is_immobilized=True,
            is_stunned=False,
            is_shutdown=False,
        )
        self.assertTrue(should)
        self.assertIn("status condition", reason)

    def test_should_fall_stunned(self):
        """Should fall when stunned while flying."""
        should, reason = should_fall_from_flying(
            is_flying=True,
            is_immobilized=False,
            is_stunned=True,
            is_shutdown=False,
        )
        self.assertTrue(should)

    def test_should_fall_shutdown(self):
        """Should fall when shutdown while flying."""
        should, reason = should_fall_from_flying(
            is_flying=True,
            is_immobilized=False,
            is_stunned=False,
            is_shutdown=True,
        )
        self.assertTrue(should)

    def test_should_not_fall_healthy(self):
        """Should not fall when healthy while flying."""
        should, reason = should_fall_from_flying(
            is_flying=True,
            is_immobilized=False,
            is_stunned=False,
            is_shutdown=False,
        )
        self.assertFalse(should)
        self.assertIn("remains flying", reason)

    def test_should_not_fall_not_flying(self):
        """Should not fall when not flying."""
        should, reason = should_fall_from_flying(
            is_flying=False,
            is_immobilized=True,
            is_stunned=False,
            is_shutdown=False,
        )
        self.assertFalse(should)
        self.assertEqual(reason, "Character is not flying")


class TestCreateFlyingStatus(unittest.TestCase):
    """Tests for create_flying_status function."""

    def test_create_ground_status(self):
        """Create ground status."""
        status = create_flying_status(is_flying=False)
        self.assertFalse(status.is_flying)
        self.assertEqual(status.altitude_level, 0)
        self.assertEqual(status.movement_mode, "ground")

    def test_create_flight_status(self):
        """Create flight status."""
        status = create_flying_status(
            is_flying=True,
            altitude_level=2,
            is_hover=False,
        )
        self.assertTrue(status.is_flying)
        self.assertEqual(status.altitude_level, 2)
        self.assertEqual(status.movement_mode, "flight")

    def test_create_hover_status(self):
        """Create hover status."""
        status = create_flying_status(
            is_flying=True,
            altitude_level=1,
            is_hover=True,
        )
        self.assertTrue(status.is_flying)
        self.assertTrue(status.is_hover)
        self.assertEqual(status.movement_mode, "hover")


class TestIsHoverMode(unittest.TestCase):
    """Tests for is_hover_mode function."""

    def test_is_hover_true(self):
        """Returns True for hover mode."""
        self.assertTrue(is_hover_mode("hover"))

    def test_is_hover_false_flight(self):
        """Returns False for flight mode."""
        self.assertFalse(is_hover_mode("flight"))

    def test_is_hover_false_ground(self):
        """Returns False for ground mode."""
        self.assertFalse(is_hover_mode("ground"))


class TestGetAltitudeAccuracyBonus(unittest.TestCase):
    """Tests for get_altitude_accuracy_bonus function."""

    def test_no_bonus_ground(self):
        """No accuracy bonus on ground."""
        self.assertEqual(get_altitude_accuracy_bonus(0), 0)

    def test_bonus_one_altitude(self):
        """+1 accuracy at altitude 1."""
        self.assertEqual(get_altitude_accuracy_bonus(1), 1)

    def test_bonus_scales_with_altitude(self):
        """Bonus scales with altitude."""
        self.assertEqual(get_altitude_accuracy_bonus(2), 2)
        self.assertEqual(get_altitude_accuracy_bonus(3), 3)

    def test_bonus_capped(self):
        """Bonus capped at 3."""
        self.assertEqual(get_altitude_accuracy_bonus(5), 3)


if __name__ == "__main__":
    unittest.main()
