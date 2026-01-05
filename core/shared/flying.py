"""Flying status resolution helpers for Lancer combat.

This module provides type-safe helpers for flying status mechanics including:
- Takeoff from ground to flight
- Landing from flight to ground
- Fall damage when falling from flight
- Prone interaction (cannot be knocked prone while flying)
- Hover mode behavior
- Altitude-based modifiers

Flying Rules (per PR2):
- Flying units cannot be knocked prone
- Flying units must move minimum spaces or fall (if immobilized/stunned)
- Landing requires a valid surface
- Fall damage based on altitude fallen

The module integrates with FlightRules from combat_rules.py for configurable behavior.
"""

from __future__ import annotations

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.enums import StatusType
from core.mech.combat_rules import FlightRules, DEFAULT_MECH_COMBAT_RULES
from core.mech.statuses import (
    StatusDefinition,
    get_status_definition,
)


class FlyingStatus(FrozenModel):
    """State tracking for flying status effects.

    Tracks whether a character is flying, current altitude level, and hover mode.
    The caller is responsible for:
    - Setting is_flying when character takes off
    - Tracking current altitude_level (0 = ground, 1+ = elevated)
    - Handling hover mode for units that can hover
    - Clearing flying status on landing
    """

    is_flying: bool = False
    altitude_level: int = Field(default=0, ge=0)
    is_hover: bool = False
    movement_mode: Literal["ground", "flight", "hover", "teleport"] = "ground"


class FlightAttempt(FrozenModel):
    """Input for takeoff action resolution.

    Per PR2, taking off from ground to flight requires a movement action.
    Some flight systems allow takeoff as part of move/boost.
    """

    is_flying: bool = False
    is_stunned: bool = False
    is_shutdown: bool = False
    has_movement_available: bool = True
    target_altitude: int = Field(default=1, ge=0)
    flight_rules: FlightRules = Field(
        default_factory=lambda: DEFAULT_MECH_COMBAT_RULES.flight
    )


class FlightResult(FrozenModel):
    """Result of takeoff action resolution."""

    takeoff_successful: bool
    is_flying: bool = False
    altitude_level: int = 0
    new_status: FlyingStatus | None = None
    reason: str = ""


class LandingAttempt(FrozenModel):
    """Input for landing action resolution.

    Per PR2, landing from flight requires a valid surface below.
    If immobilized or stunned while flying, unit falls.
    """

    is_flying: bool = False
    current_altitude: int = Field(default=0, ge=0)
    is_hover: bool = False
    is_immobilized: bool = False
    is_stunned: bool = False
    is_shutdown: bool = False
    has_surface_below: bool = True
    surface_is_valid: bool = True
    flight_rules: FlightRules = Field(
        default_factory=lambda: DEFAULT_MECH_COMBAT_RULES.flight
    )


class LandingResult(FrozenModel):
    """Result of landing action resolution."""

    landed_successfully: bool
    fell: bool = False
    fall_damage: int = 0
    became_prone: bool = False
    new_status: FlyingStatus | None = None
    reason: str = ""


class FlightEffects(FrozenModel):
    """Resolved mechanical effects of flying status.

    Per PR2 flight rules:
    - Ignore obstructions at altitude 1+
    - Cannot be knocked prone
    - May have altitude-based modifiers
    - Hover mode allows stationary flight
    """

    ignores_obstructions: bool = False
    cannot_be_knocked_prone: bool = True
    altitude_modifier: int = 0
    hover_allows_stationary: bool = False
    movement_mode: Literal["ground", "flight", "hover", "teleport"] = "ground"


class FallDamageResult(FrozenModel):
    """Result of fall damage calculation."""

    damage_taken: int
    fell_from_altitude: int
    structure_damage: bool = False


def can_takeoff(
    is_flying: bool,
    is_stunned: bool,
    is_shutdown: bool,
    has_movement_available: bool,
) -> tuple[bool, str]:
    """Check if a character can initiate flight (takeoff).

    Per PR2 flight rules:
    - Cannot takeoff while already flying
    - Cannot takeoff while stunned or shutdown
    - Requires a movement action

    Args:
        is_flying: Whether the character is already flying
        is_stunned: Whether the character has the stunned condition
        is_shutdown: Whether the character is shutdown
        has_movement_available: Whether the character has movement available

    Returns:
        Tuple of (can_takeoff, reason)
    """
    if is_flying:
        return False, "Character is already flying"

    if is_stunned:
        return False, "Stunned characters cannot take actions"

    if is_shutdown:
        return False, "Shutdown mechs cannot take actions"

    if not has_movement_available:
        return False, "No movement action available for takeoff"

    return True, "Takeoff action available"


def attempt_takeoff(
    attempt: FlightAttempt,
    current_turn: int = 1,
) -> FlightResult:
    """Resolve a takeoff action from ground to flight.

    Per PR2 flight rules:
    - Taking off requires a movement action
    - Some flight systems allow takeoff as part of move/boost
    - Target altitude determines flight level

    Args:
        attempt: FlightAttempt with all required parameters
        current_turn: The current turn number for status tracking

    Returns:
        FlightResult with outcome details
    """
    can_take, reason = can_takeoff(
        is_flying=attempt.is_flying,
        is_stunned=attempt.is_stunned,
        is_shutdown=attempt.is_shutdown,
        has_movement_available=attempt.has_movement_available,
    )

    if not can_take:
        return FlightResult(
            takeoff_successful=False,
            is_flying=False,
            altitude_level=0,
            new_status=None,
            reason=reason,
        )

    new_status = FlyingStatus(
        is_flying=True,
        altitude_level=max(1, attempt.target_altitude),
        is_hover=False,
        movement_mode="flight",
    )

    return FlightResult(
        takeoff_successful=True,
        is_flying=True,
        altitude_level=new_status.altitude_level,
        new_status=new_status,
        reason=f"Takeoff successful: now flying at altitude {new_status.altitude_level}",
    )


def can_land(
    is_flying: bool,
    current_altitude: int,
    has_surface_below: bool,
    surface_is_valid: bool,
) -> tuple[bool, str]:
    """Check if a character can land.

    Per PR2 flight rules:
    - Landing requires a valid surface below
    - Cannot land in mid-air without surface

    Args:
        is_flying: Whether the character is currently flying
        current_altitude: Current altitude level
        has_surface_below: Whether there is a surface to land on
        surface_is_valid: Whether the surface is valid for landing

    Returns:
        Tuple of (can_land, reason)
    """
    if not is_flying:
        return False, "Character is not flying"

    if current_altitude <= 0:
        return False, "Character is already on ground"

    if not has_surface_below:
        return False, "No surface below for landing"

    if not surface_is_valid:
        return False, "Surface is not valid for landing"

    return True, "Landing action available"


def attempt_landing(
    attempt: LandingAttempt,
    current_turn: int = 1,
) -> LandingResult:
    """Resolve a landing action from flight to ground.

    Per PR2 flight rules:
    - Landing requires a valid surface
    - If immobilized/stunned while flying, unit falls
    - Fall damage based on altitude fallen
    - Falling from flight may cause prone condition

    Args:
        attempt: LandingAttempt with all required parameters
        current_turn: The current turn number for status tracking

    Returns:
        LandingResult with outcome details
    """
    if not attempt.is_flying:
        return LandingResult(
            landed_successfully=False,
            fell=False,
            fall_damage=0,
            became_prone=False,
            new_status=None,
            reason="Character is not flying",
        )

    if attempt.is_hover and attempt.flight_rules.hover_allows_stationary:
        return LandingResult(
            landed_successfully=True,
            fell=False,
            fall_damage=0,
            became_prone=False,
            new_status=FlyingStatus(
                is_flying=False,
                altitude_level=0,
                is_hover=False,
                movement_mode="ground",
            ),
            reason="Hover mode: landed without falling",
        )

    if attempt.is_immobilized or attempt.is_stunned or attempt.is_shutdown:
        fall_result = calculate_fall_damage(attempt.current_altitude)
        became_prone = attempt.current_altitude >= 1
        return LandingResult(
            landed_successfully=False,
            fell=True,
            fall_damage=fall_result.damage_taken,
            became_prone=became_prone,
            new_status=FlyingStatus(
                is_flying=False,
                altitude_level=0,
                is_hover=False,
                movement_mode="ground",
            ),
            reason=f"Fell from flight: {fall_result.damage_taken} damage"
            + (" and became prone" if became_prone else ""),
        )

    can_land_result, reason = can_land(
        is_flying=attempt.is_flying,
        current_altitude=attempt.current_altitude,
        has_surface_below=attempt.has_surface_below,
        surface_is_valid=attempt.surface_is_valid,
    )

    if not can_land_result:
        return LandingResult(
            landed_successfully=False,
            fell=True,
            fall_damage=calculate_fall_damage(attempt.current_altitude).damage_taken,
            became_prone=attempt.current_altitude >= 1,
            new_status=None,
            reason=reason,
        )

    return LandingResult(
        landed_successfully=True,
        fell=False,
        fall_damage=0,
        became_prone=False,
        new_status=FlyingStatus(
            is_flying=False,
            altitude_level=0,
            is_hover=False,
            movement_mode="ground",
        ),
        reason="Landing successful",
    )


def calculate_fall_damage(fall_from_altitude: int) -> FallDamageResult:
    """Calculate fall damage based on altitude fallen.

    Per PR2 falling rules:
    - Falling from altitude causes damage
    - Damage based on spaces fallen
    - Falls are resolved at end of turn

    Args:
        fall_from_altitude: Altitude level fallen from (1 = 1 space, etc.)

    Returns:
        FallDamageResult with damage calculation
    """
    damage = max(0, fall_from_altitude)
    return FallDamageResult(
        damage_taken=damage,
        fell_from_altitude=fall_from_altitude,
        structure_damage=False,
    )


def get_flight_effects(
    is_flying: bool,
    altitude_level: int,
    is_hover: bool,
    flight_rules: FlightRules = DEFAULT_MECH_COMBAT_RULES.flight,
) -> FlightEffects:
    """Get the mechanical effects of flying status.

    Per PR2 flight rules:
    - Flying units ignore obstructions at altitude 1+
    - Cannot be knocked prone while flying
    - Hover mode allows stationary flight
    - Altitude may provide accuracy bonuses

    Args:
        is_flying: Whether the character is flying
        altitude_level: Current altitude (0 = ground)
        is_hover: Whether in hover mode
        flight_rules: Flight rules configuration

    Returns:
        FlightEffects with all mechanical modifiers
    """
    if not is_flying:
        return FlightEffects(
            ignores_obstructions=False,
            cannot_be_knocked_prone=False,
            altitude_modifier=0,
            hover_allows_stationary=False,
            movement_mode="ground",
        )

    ignores_obstructions = flight_rules.ignore_obstructions and altitude_level >= 1

    altitude_modifier = 0
    if altitude_level >= 1:
        altitude_modifier = altitude_level

    return FlightEffects(
        ignores_obstructions=ignores_obstructions,
        cannot_be_knocked_prone=True,
        altitude_modifier=altitude_modifier,
        hover_allows_stationary=flight_rules.hover_allows_stationary,
        movement_mode="hover" if is_hover else "flight",
    )


def is_flying(
    is_flying_status: bool,
    movement_mode: Literal["ground", "flight", "hover", "teleport"],
) -> bool:
    """Check if a character is in flying state.

    Args:
        is_flying_status: Explicit flying status flag
        movement_mode: Current movement mode

    Returns:
        True if character is flying
    """
    return is_flying_status or movement_mode in ("flight", "hover")


def can_be_knocked_prone_while_flying(
    is_flying: bool,
) -> tuple[bool, str]:
    """Check if a flying character can be knocked prone.

    Per PR2 ~3765-3769 and flight rules:
    - Flying units cannot be knocked prone

    Args:
        is_flying: Whether the target is flying

    Returns:
        Tuple of (can_be_knocked_prone, reason)
    """
    if is_flying:
        return False, "Flying characters cannot be knocked prone"

    return True, "Character can be knocked prone"


def should_fall_from_flying(
    is_flying: bool,
    is_immobilized: bool,
    is_stunned: bool,
    is_shutdown: bool,
    flight_rules: FlightRules = DEFAULT_MECH_COMBAT_RULES.flight,
) -> tuple[bool, str]:
    """Check if a flying character should fall due to status.

    Per PR2 flight rules:
    - Flying units fall if immobilized or stunned

    Args:
        is_flying: Whether the character is flying
        is_immobilized: Whether the character has the immobilized condition
        is_stunned: Whether the character has the stunned condition
        is_shutdown: Whether the character is shutdown
        flight_rules: Flight rules configuration

    Returns:
        Tuple of (should_fall, reason)
    """
    if not is_flying:
        return False, "Character is not flying"

    if not flight_rules.falls_if_immobilized_or_stunned:
        return False, "Flight rules do not cause fall from status"

    if is_immobilized or is_stunned or is_shutdown:
        return True, "Character falls due to status condition"

    return False, "Character remains flying"


def create_flying_status(
    is_flying: bool,
    altitude_level: int = 0,
    is_hover: bool = False,
    movement_mode: Literal["ground", "flight", "hover", "teleport"] | None = None,
) -> FlyingStatus:
    """Create a FlyingStatus model from flight state.

    Args:
        is_flying: Whether the character is flying
        altitude_level: Current altitude (0 = ground)
        is_hover: Whether in hover mode
        movement_mode: Current movement mode (auto-derived if not provided)

    Returns:
        FlyingStatus with appropriate values
    """
    if not is_flying:
        return FlyingStatus(
            is_flying=False,
            altitude_level=0,
            is_hover=False,
            movement_mode="ground",
        )

    effective_movement_mode = movement_mode or ("hover" if is_hover else "flight")

    return FlyingStatus(
        is_flying=True,
        altitude_level=altitude_level,
        is_hover=is_hover,
        movement_mode=effective_movement_mode,
    )


def is_hover_mode(
    movement_mode: Literal["ground", "flight", "hover", "teleport"],
) -> bool:
    """Check if character is in hover mode.

    Args:
        movement_mode: Current movement mode

    Returns:
        True if in hover mode
    """
    return movement_mode == "hover"


def get_altitude_accuracy_bonus(altitude_level: int) -> int:
    """Get accuracy bonus from higher altitude.

    Per PR2 elevation rules:
    - Higher elevation provides accuracy bonus

    Args:
        altitude_level: Current altitude level

    Returns:
        Accuracy bonus (0 if on ground)
    """
    if altitude_level <= 0:
        return 0
    return min(altitude_level, 3)
