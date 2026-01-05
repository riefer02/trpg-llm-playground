"""Prone and Slowed status resolution helpers for Lancer combat.

This module provides type-safe helpers for prone status mechanics including:
- Prone status effects (attackers +1 accuracy, auto-slowed, difficult terrain)
- Stand up from prone (full action, cannot stand while immobilized)
- Prone/Slowed interaction rules

Prone Rules (per PR2):
- PRONE: Attackers get +1 accuracy, target is Slowed, counts as difficult terrain
- STAND UP: Full action to stand, cannot stand while Immobilized
- SLOWED: Only regular move, max voluntary movement = 0

The prone status auto-grants the slowed condition (per status definition).
"""

from __future__ import annotations

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.enums import StatusType
from core.mech.statuses import (
    StatusDefinition,
    STATUS_DEFINITIONS_BY_ID,
    get_status_definition,
)


class ProneStatus(FrozenModel):
    """State tracking for prone status effects.

    Tracks whether a character is prone and the auto-granted slowed condition.
    The caller is responsible for:
    - Setting is_prone when character becomes prone
    - Tracking the auto-slowed condition
    - Clearing both at end of turn or when standing up
    """

    is_prone: bool = False
    is_slowed: bool = False
    counts_as_difficult_terrain: bool = False
    attackers_accuracy_bonus: int = 0


class StandUpAttempt(FrozenModel):
    """Input for stand up action resolution.

    Per PR2, standing up from prone is a full action. A character cannot
    stand up if immobilized (cannot move voluntarily).
    """

    is_prone: bool = False
    is_immobilized: bool = False
    has_full_action_available: bool = True
    is_stunned: bool = False
    is_shutdown: bool = False


class StandUpResult(FrozenModel):
    """Result of stand up action resolution.

    Per PR2 ~3765-3769:
    "You can stand up from being prone as a full action. You cannot stand up
    if you are immobilized."
    """

    stand_successful: bool
    prone_cleared: bool = False
    slowed_cleared: bool = False
    new_status: ProneStatus | None = None
    reason: str = ""


class ProneEffects(FrozenModel):
    """Resolved mechanical effects of being prone.

    Per PR2 ~3765-3769:
    - Attackers get +1 accuracy bonus
    - Target is Slowed (auto-granted)
    - Movement counts as difficult terrain
    """

    attackers_accuracy_bonus: int = 1
    movement_difficulty_modifier: int = 1  # Difficult terrain = +1 difficulty
    is_slowed: bool = True
    is_difficult_terrain: bool = True
    requires_full_action_to_stand: bool = True
    cannot_stand_if_immobilized: bool = True


class KnockProneAttempt(FrozenModel):
    """Input for attempting to knock a character prone."""

    target_is_prone: bool = False
    target_is_flying: bool = False  # Flying mechs cannot be knocked prone
    target_size: (
        Literal["size_half", "size_1", "size_2", "size_3", "size_4", "size_5"] | None
    ) = None
    attack_hit: bool = True


class KnockProneResult(FrozenModel):
    """Result of attempting to knock a character prone."""

    can_become_prone: bool
    became_prone: bool = False
    reason: str = ""


def can_stand_up(
    is_prone: bool,
    is_immobilized: bool,
    has_full_action_available: bool,
    is_stunned: bool,
    is_shutdown: bool,
) -> tuple[bool, str]:
    """Check if a character can perform the Stand Up action.

    Per PR2 ~3765-3769:
    "You can stand up from being prone as a full action. You cannot stand up
    if you are immobilized (see page 4168)."

    Args:
        is_prone: Whether the character is currently prone
        is_immobilized: Whether the character has the immobilized condition
        has_full_action_available: Whether the character has a full action available
        is_stunned: Whether the character has the stunned condition
        is_shutdown: Whether the character is shutdown

    Returns:
        Tuple of (can_stand_up, reason)
    """
    if not is_prone:
        return False, "Character is not prone"

    if is_immobilized:
        return False, "Immobilized characters cannot stand up"

    if is_stunned:
        return False, "Stunned characters cannot take actions"

    if is_shutdown:
        return False, "Shutdown mechs cannot take actions"

    if not has_full_action_available:
        return False, "No full action available for stand up"

    return True, "Stand up action available"


def attempt_stand_up(
    attempt: StandUpAttempt,
    current_turn: int = 1,
) -> StandUpResult:
    """Resolve a stand up action from prone.

    Per PR2 ~3765-3769:
    "You can stand up from being prone as a full action. You cannot stand up
    if you are immobilized (see page 4168)."

    Standing up removes both the prone status and the auto-granted slowed condition.

    Args:
        attempt: StandUpAttempt with all required parameters
        current_turn: The current turn number for status tracking

    Returns:
        StandUpResult with outcome details
    """
    can_stand, reason = can_stand_up(
        is_prone=attempt.is_prone,
        is_immobilized=attempt.is_immobilized,
        has_full_action_available=attempt.has_full_action_available,
        is_stunned=attempt.is_stunned,
        is_shutdown=attempt.is_shutdown,
    )

    if not can_stand:
        return StandUpResult(
            stand_successful=False,
            prone_cleared=False,
            slowed_cleared=False,
            new_status=None,
            reason=reason,
        )

    new_status = ProneStatus(
        is_prone=False,
        is_slowed=False,
        counts_as_difficult_terrain=False,
        attackers_accuracy_bonus=0,
    )

    return StandUpResult(
        stand_successful=True,
        prone_cleared=True,
        slowed_cleared=True,
        new_status=new_status,
        reason="Stand up successful: prone and slowed conditions cleared",
    )


def get_prone_effects() -> ProneEffects:
    """Get the mechanical effects of being prone.

    Per PR2 ~3765-3769:
    - Attackers get +1 accuracy bonus
    - Target is Slowed (auto-granted per status definition)
    - Movement counts as difficult terrain
    - Standing up requires a full action

    Returns:
        ProneEffects with all mechanical modifiers
    """
    definition = get_status_definition("prone")

    return ProneEffects(
        attackers_accuracy_bonus=1,
        movement_difficulty_modifier=1,
        is_slowed=True,
        is_difficult_terrain=True,
        requires_full_action_to_stand=True,
        cannot_stand_if_immobilized=True,
    )


def is_prone_movement_difficult(is_prone: bool) -> bool:
    """Check if movement is treated as difficult terrain due to prone.

    Per PR2 ~3765-3769:
    "This counts as difficult terrain for the purpose of movement."

    Args:
        is_prone: Whether the character is prone

    Returns:
        True if movement counts as difficult terrain
    """
    return is_prone


def get_attack_accuracy_bonus_from_prone(is_prone: bool) -> int:
    """Get the accuracy bonus attackers get against a prone target.

    Per PR2 ~3765-3769:
    "Attackers get +1 accuracy against prone targets."

    Args:
        is_prone: Whether the target is prone

    Returns:
        Accuracy bonus for attackers (0 or 1)
    """
    if not is_prone:
        return 0
    return 1


def can_be_knocked_prone(
    target_is_prone: bool,
    target_is_flying: bool = False,
) -> tuple[bool, str]:
    """Check if a character can become prone.

    Per PR2 ~3765-3769 and flying rules:
    - Cannot be prone while already prone
    - Cannot be knocked prone while flying

    Args:
        target_is_prone: Whether the target is already prone
        target_is_flying: Whether the target is flying

    Returns:
        Tuple of (can_become_prone, reason)
    """
    if target_is_prone:
        return False, "Target is already prone"

    if target_is_flying:
        return False, "Flying characters cannot be knocked prone"

    return True, "Target can be knocked prone"


def apply_prone(
    target_conditions: list[StatusType],
) -> tuple[bool, str, list[StatusType]]:
    """Apply prone status and auto-grant slowed condition.

    Per PR2 ~3765-3769:
    "When you become prone, you also gain the slowed condition."

    This helper applies both conditions in the correct order.

    Args:
        target_conditions: Current conditions list to modify

    Returns:
        Tuple of (application_successful, reason, updated_conditions)
    """
    can_become, reason = can_be_knocked_prone(
        target_is_prone="prone" in target_conditions,
    )

    if not can_become:
        return False, reason, target_conditions

    target_conditions.append("prone")
    target_conditions.append("slowed")

    return True, "Prone and slowed conditions applied", target_conditions


def remove_prone(
    target_conditions: list[StatusType],
) -> tuple[bool, str, list[StatusType]]:
    """Remove prone status and associated slowed condition.

    Per PR2 ~3765-3769:
    Standing up from prone removes both conditions.

    Args:
        target_conditions: Current conditions list to modify

    Returns:
        Tuple of (removal_successful, reason, updated_conditions)
    """
    if "prone" not in target_conditions:
        return False, "Target is not prone", target_conditions

    if "prone" in target_conditions:
        target_conditions.remove("prone")

    if "slowed" in target_conditions:
        target_conditions.remove("slowed")

    return True, "Prone and slowed conditions removed", target_conditions


def is_slowed_from_prone(is_prone: bool, conditions: list[StatusType]) -> bool:
    """Check if a character has the slowed condition from being prone.

    Per PR2 ~3765-3769:
    "When you become prone, you also gain the slowed condition."

    Args:
        is_prone: Whether the character is prone
        conditions: Current conditions on the character

    Returns:
        True if the character has slowed (either from prone or other source)
    """
    return is_prone or "slowed" in conditions


def get_slowed_effects() -> ProneEffects:
    """Get the mechanical effects of being slowed.

    Per PR2 ~4168-4172:
    "Slowed: You can only take regular moves. Your maximum voluntary movement is 0."

    Returns:
        ProneEffects with slowed-specific modifiers
    """
    return ProneEffects(
        attackers_accuracy_bonus=0,
        movement_difficulty_modifier=0,
        is_slowed=True,
        is_difficult_terrain=False,
        requires_full_action_to_stand=False,
        cannot_stand_if_immobilized=True,
    )


def is_movement_restricted_by_slowed(
    is_slowed: bool,
) -> bool:
    """Check if movement is restricted by the slowed condition.

    Per PR2 ~4168-4172:
    "Slowed: You can only take regular moves. Your maximum voluntary movement is 0."

    Args:
        is_slowed: Whether the character has the slowed condition

    Returns:
        True if movement is restricted
    """
    return is_slowed


def create_prone_status(is_prone: bool) -> ProneStatus:
    """Create a ProneStatus model from prone state.

    Args:
        is_prone: Whether the character is prone

    Returns:
        ProneStatus with appropriate values
    """
    if not is_prone:
        return ProneStatus(
            is_prone=False,
            is_slowed=False,
            counts_as_difficult_terrain=False,
            attackers_accuracy_bonus=0,
        )

    return ProneStatus(
        is_prone=True,
        is_slowed=True,
        counts_as_difficult_terrain=True,
        attackers_accuracy_bonus=1,
    )
