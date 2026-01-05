"""Disengage action resolution helpers for Lancer combat.

This module provides type-safe helpers for the Disengage action including:
- Disengage action: Full action to ignore engagement and reactions
- Engagement detection: Check if character is engaged with hostiles
- State tracking: DisengageStatus for duration management

Disengage Rules (per PR2 ~3778, 3843, 4289-4291):
- Full action
- Until end of turn: movement ignores engagement, does not provoke reactions
- Disengage always succeeds if full action is available
- Cannot disengage while stunned or otherwise unable to take actions

Engagement Rules (per PR2 ~3818-3819, 3953-3954):
- Adjacent to hostile character = engaged
- Engaged: +1 difficulty on ranged attacks
- Engaged with same/larger target: must stop moving, cannot continue movement
"""

from __future__ import annotations

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.enums import StatusType


class DisengageStatus(FrozenModel):
    """State tracking for disengage effect duration.

    Tracks whether a character is currently benefiting from disengage.
    The caller is responsible for:
    - Setting is_active when disengage is used
    - Tracking turn number for duration
    - Clearing is_active at end of turn
    """

    is_active: bool = False
    started_at_turn: int | None = None
    ignores_engagement: bool = False
    prevents_reactions: bool = False


class DisengageAttempt(FrozenModel):
    """Input for disengage action resolution.

    Per PR2 ~4289-4291, disengage is a full action that causes movement
    to ignore engagement and reactions until end of turn.
    """

    is_engaged: bool = False
    has_full_action_available: bool = True
    is_stunned: bool = False
    is_shutdown: bool = False
    is_prone: bool = False


class DisengageResult(FrozenModel):
    """Result of disengage action resolution.

    Per PR2 ~4289-4291:
    "Until the end of your current turn, your movement ignores engagement
    and does not provoke reactions, such as overwatch."
    """

    disengage_successful: bool
    ignores_engagement: bool = True
    prevents_reactions: bool = True
    duration_turn_end: bool = True
    new_status: DisengageStatus | None = None
    reason: str = ""


class EngagementCheck(FrozenModel):
    """Result of engagement detection check."""

    is_engaged: bool
    adjacent_hostiles: int = 0
    engagement_details: str = ""


def can_disengage(
    has_full_action_available: bool,
    is_stunned: bool,
    is_shutdown: bool,
) -> tuple[bool, str]:
    """Check if a character can perform the Disengage action.

    Per PR2 ~4289-4291, disengage is a full action. A character cannot
    take actions if stunned or shutdown.

    Args:
        has_full_action_available: Whether the character has a full action available
        is_stunned: Whether the character has the stunned status
        is_shutdown: Whether the character is shutdown

    Returns:
        Tuple of (can_disengage, reason)
    """
    if is_stunned:
        return False, "Stunned characters cannot take actions"

    if is_shutdown:
        return False, "Shutdown mechs cannot take actions"

    if not has_full_action_available:
        return False, "No full action available for disengage"

    return True, "Disengage action available"


def attempt_disengage(
    attempt: DisengageAttempt,
    current_turn: int = 1,
) -> DisengageResult:
    """Resolve a disengage action.

    Per PR2 ~4289-4291:
    "When you disengage, you attempt to extricate yourself safely from a
    dangerous situation, make a steady and measured retreat, or your mech's
    agility is such that you can slip in and out of threat ranges faster
    than the enemy can strike. Until the end of your current turn, your
    movement ignores engagement and does not provoke reactions, such as overwatch."

    Disengage is always successful if the character can take actions.

    Args:
        attempt: DisengageAttempt with all required parameters
        current_turn: The current turn number for status tracking

    Returns:
        DisengageResult with outcome details
    """
    can_disengage_result, reason = can_disengage(
        has_full_action_available=attempt.has_full_action_available,
        is_stunned=attempt.is_stunned,
        is_shutdown=attempt.is_shutdown,
    )

    if not can_disengage_result:
        return DisengageResult(
            disengage_successful=False,
            ignores_engagement=False,
            prevents_reactions=False,
            duration_turn_end=False,
            new_status=None,
            reason=reason,
        )

    new_status = DisengageStatus(
        is_active=True,
        started_at_turn=current_turn,
        ignores_engagement=True,
        prevents_reactions=True,
    )

    engaged_status = "while engaged" if attempt.is_engaged else "not engaged"
    return DisengageResult(
        disengage_successful=True,
        ignores_engagement=True,
        prevents_reactions=True,
        duration_turn_end=True,
        new_status=new_status,
        reason=f"Disengage successful ({engaged_status}): movement ignores engagement and reactions until end of turn",
    )


def get_engagement_state(
    adjacent_hostiles: int,
    is_immune_to_engagement: bool = False,
) -> EngagementCheck:
    """Check if a character is engaged based on adjacent hostile count.

    Per PR2 ~3818-3819, 3953-3954:
    "If you move adjacent to a hostile character, you become engaged. Being
    engaged gives penalties to ranged attacks (+1 difficulty)..."

    Args:
        adjacent_hostiles: Number of hostile characters in adjacent spaces
        is_immune_to_engagement: Whether character ignores engagement effects

    Returns:
        EngagementCheck with engagement state details
    """
    if is_immune_to_engagement:
        return EngagementCheck(
            is_engaged=False,
            adjacent_hostiles=adjacent_hostiles,
            engagement_details="Immune to engagement",
        )

    is_engaged = adjacent_hostiles > 0

    if is_engaged:
        detail = f"Engaged with {adjacent_hostiles} hostile character(s)"
    else:
        detail = "Not engaged"

    return EngagementCheck(
        is_engaged=is_engaged,
        adjacent_hostiles=adjacent_hostiles,
        engagement_details=detail,
    )


def is_movement_ignoring_engagement(
    disengage_status: DisengageStatus | None,
    is_invisible: bool = False,
    is_teleporting: bool = False,
    is_involuntary_movement: bool = False,
) -> bool:
    """Check if current movement ignores engagement.

    Per PR2:
    - Disengage: movement ignores engagement until end of turn
    - Invisibility: movement ignores engagement
    - Teleporting: ignores engagement entirely
    - Involuntary movement: ignores engagement

    Args:
        disengage_status: Current disengage status if active
        is_invisible: Whether character is invisible
        is_teleporting: Whether movement is teleportation
        is_involuntary_movement: Whether movement is involuntary

    Returns:
        True if movement ignores engagement
    """
    if (
        disengage_status
        and disengage_status.is_active
        and disengage_status.ignores_engagement
    ):
        return True

    if is_invisible:
        return True

    if is_teleporting:
        return True

    if is_involuntary_movement:
        return True

    return False


def check_reaction_prevented(
    disengage_status: DisengageStatus | None,
    is_hidden: bool = False,
    is_invisible: bool = False,
    is_teleporting: bool = False,
    is_involuntary_movement: bool = False,
) -> tuple[bool, str]:
    """Check if reactions are prevented during this movement.

    Per PR2:
    - Disengage: movement does not provoke reactions
    - Teleporting: does not provoke reactions
    - Involuntary movement: does not provoke reactions
    - Hidden: enemies ignore engagement (reactions don't trigger from engagement)

    Args:
        disengage_status: Current disengage status if active
        is_hidden: Whether character has hidden condition
        is_invisible: Whether character is invisible
        is_teleporting: Whether movement is teleportation
        is_involuntary_movement: Whether movement is involuntary

    Returns:
        Tuple of (reactions_prevented, reason)
    """
    if (
        disengage_status
        and disengage_status.is_active
        and disengage_status.prevents_reactions
    ):
        return True, "Disengage prevents reactions during movement"

    if is_hidden:
        return (
            True,
            "Hidden prevents enemies from engaging (no reactions from engagement)",
        )

    if is_invisible:
        return True, "Invisibility prevents reactions during movement"

    if is_teleporting:
        return True, "Teleporting does not provoke reactions"

    if is_involuntary_movement:
        return True, "Involuntary movement does not provoke reactions"

    return False, "Reactions not prevented"


def end_disengage_at_turn_end(
    disengage_status: DisengageStatus,
    current_turn: int,
) -> DisengageStatus:
    """End disengage status at end of turn.

    Per PR2 ~4289-4291, disengage effect lasts "until the end of your current turn."

    Args:
        disengage_status: Current disengage status
        current_turn: The current turn number

    Returns:
        Updated DisengageStatus with is_active cleared
    """
    if disengage_status.started_at_turn == current_turn and disengage_status.is_active:
        return DisengageStatus(
            is_active=False,
            started_at_turn=disengage_status.started_at_turn,
            ignores_engagement=False,
            prevents_reactions=False,
        )

    return disengage_status


def can_disengage_from_grapple(
    is_grappling: bool,
    is_same_size_or_larger: bool,
) -> tuple[bool, str]:
    """Check if character can disengage while grappled.

    Per PR2 ~4161-4168, grapple rules state:
    "Both parties are engaged... Neither party can boost or take reactions."

    Grapple ending rules:
    - Attacker can end grapple as free action
    - Defender can end grapple as quick action with successful contested HULL check

    Args:
        is_grappling: Whether character is currently grappling
        is_same_size_or_larger: Whether grappled target is same size or larger

    Returns:
        Tuple of (can_disengage, reason)
    """
    if not is_grappling:
        return True, "Not grappled, can disengage"

    if is_same_size_or_larger:
        return (
            False,
            "Cannot disengage while grappled with same/larger target (quick action required)",
        )

    return True, "Can end grapple as free action"
