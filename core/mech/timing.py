"""Turn phase and action timing models for mech combat.

Provides primitives for:
- Turn phases (start/normal/end)
- Prepared action tracking and lockout
- Per-round reaction limits (brace/overwatch)
- Protocol timing enforcement (start-of-turn only)
"""

from typing import Literal
from pydantic import Field
from core.shared.models import FrozenModel

from core.shared.enums import ActionType


TurnPhase = Literal["start", "normal", "end"]


class PreparedActionState(FrozenModel):
    """State for a prepared action held by a combatant.

    After using the Prepare action, the combatant holds a prepared action
    and cannot take other actions/reactions/movement until the trigger occurs
    or the prepared action expires at the start of their next turn.
    """

    held_action_id: str = Field(
        ..., description="ID of the action to execute when triggered"
    )
    held_action_type: ActionType = Field(..., description="Type of the prepared action")
    trigger_condition: str = Field(
        ..., description="Description of the trigger condition"
    )
    created_on_turn: int = Field(..., ge=1, description="Round number when prepared")
    expires_on_turn: int = Field(
        ..., ge=1, description="Turn number when prepared action expires"
    )
    blocks_actions: bool = Field(
        default=True, description="Whether this blocks other actions"
    )
    blocks_reactions: bool = Field(
        default=True, description="Whether this blocks reactions"
    )
    blocks_movement: bool = Field(
        default=True, description="Whether this blocks regular movement"
    )


class PerRoundReactionUse(FrozenModel):
    """Tracking for per-round reaction usage."""

    action_id: str = Field(
        ..., description="Reaction action ID (e.g., 'brace', 'overwatch')"
    )
    uses_remaining: int = Field(..., ge=0, description="Uses remaining this round")


class TurnPhaseState(FrozenModel):
    """Current turn phase state for a combatant's turn."""

    current_phase: TurnPhase = Field(default="start", description="Current turn phase")
    protocol_activated: bool = Field(
        default=False, description="Whether protocol has been activated this turn"
    )
    protocol_id: str | None = Field(
        default=None, description="ID of active protocol if any"
    )


class PhasedActionUse(FrozenModel):
    """An action taken during a specific turn phase."""

    action_id: str = Field(..., description="Action ID")
    action_type: ActionType = Field(..., description="Action type")
    timing: TurnPhase | None = Field(
        default=None, description="Phase when action was taken"
    )
    is_protocol: bool = Field(
        default=False, description="Whether this is a protocol activation"
    )


class ActionTimingValidationSettings(FrozenModel):
    """Validation settings for action timing enforcement.

    Use strict_mode=True for tactical combat (errors block invalid actions).
    Use strict_mode=False for narrative play (warnings only, actions proceed).
    """

    strict_mode: bool = Field(
        default=True,
        description="If True, timing violations are errors that block actions",
    )
    allow_protocol_outside_start: bool = Field(
        default=False, description="If True, protocols allowed in any phase"
    )
    allow_actions_while_prepared: bool = Field(
        default=False, description="If True, prepared action lockout is advisory"
    )
    allow_reactions_while_prepared: bool = Field(
        default=False, description="If True, reactions allowed while prepared"
    )
    allow_movement_while_prepared: bool = Field(
        default=False, description="If True, movement allowed while prepared"
    )


DEFAULT_TIMING_VALIDATION = ActionTimingValidationSettings(strict_mode=True)


class TimingValidationResult(FrozenModel):
    """Result of timing validation for an action."""

    valid: bool = Field(..., description="Whether the action timing is valid")
    errors: list[str] = Field(
        default_factory=list, description="Error messages if invalid"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Warning messages (non-blocking)"
    )


def validate_protocol_timing(
    action_id: str,
    is_protocol: bool,
    current_phase: TurnPhase,
    settings: ActionTimingValidationSettings | None = None,
) -> TimingValidationResult:
    """Validate that a protocol is used at the correct time.

    Protocols can only be activated at the start of a turn (before other actions).

    Args:
        action_id: The action being taken
        is_protocol: Whether this action is a protocol activation
        current_phase: The current turn phase
        settings: Validation settings (uses defaults if None)

    Returns:
        Validation result indicating if timing is valid
    """
    if settings is None:
        settings = DEFAULT_TIMING_VALIDATION

    if not is_protocol:
        return TimingValidationResult(valid=True)

    if settings.allow_protocol_outside_start:
        return TimingValidationResult(valid=True)

    if current_phase != "start":
        is_valid = False
        if settings.strict_mode:
            return TimingValidationResult(
                valid=False,
                errors=["Protocols can only be activated at the start of your turn"],
            )
        else:
            return TimingValidationResult(
                valid=False,
                warnings=[
                    "Protocols are typically activated at the start of your turn"
                ],
            )

    return TimingValidationResult(valid=True)


def validate_action_while_prepared(
    action_id: str,
    action_type: ActionType,
    prepared_state: PreparedActionState | None,
    settings: ActionTimingValidationSettings | None = None,
) -> TimingValidationResult:
    """Validate that an action is allowed while a prepared action is held.

    When a combatant has a prepared action, they cannot take other actions,
    reactions, or regular movement until the trigger occurs or the prepared
    action expires.

    Args:
        action_id: The action being taken
        action_type: Type of action being taken
        prepared_state: Current prepared action state (None if no prepared action)
        settings: Validation settings (uses defaults if None)

    Returns:
        Validation result indicating if timing is valid
    """
    if settings is None:
        settings = DEFAULT_TIMING_VALIDATION

    if prepared_state is None:
        return TimingValidationResult(valid=True)

    errors = []
    warnings = []

    if prepared_state.blocks_actions:
        if action_type == "reaction":
            if not settings.allow_reactions_while_prepared:
                msg = "Cannot take reactions while a prepared action is held"
                if settings.strict_mode:
                    errors.append(msg)
                else:
                    warnings.append(msg)
        else:
            msg = f"Cannot take action {action_id} while a prepared action is held"
            if settings.strict_mode:
                errors.append(msg)
            else:
                warnings.append(msg)

    if prepared_state.blocks_movement and action_type == "move":
        msg = "Cannot move while a prepared action is held"
        if settings.strict_mode:
            errors.append(msg)
        else:
            warnings.append(msg)

    is_valid = len(errors) == 0
    return TimingValidationResult(valid=is_valid, errors=errors, warnings=warnings)


def validate_per_round_reaction(
    action_id: str,
    current_round: int,
    actor_id: str,
    reaction_counts_by_actor: dict[str, dict[str, int]],
    max_per_round: int,
) -> TimingValidationResult:
    """Validate that a per-round reaction hasn't exceeded its limit.

    Some reactions like Brace and Overwatch can only be used once per round.

    Args:
        action_id: The reaction action ID
        current_round: The current round number
        actor_id: The actor taking the action
        reaction_counts_by_actor: Tracking dict {actor_id: {action_id: count}}
        max_per_round: Maximum uses per round for this reaction

    Returns:
        Validation result indicating if timing is valid
    """
    actor_counts = reaction_counts_by_actor.get(actor_id, {})
    current_count = actor_counts.get(action_id, 0)

    if current_count >= max_per_round:
        return TimingValidationResult(
            valid=False,
            errors=[
                f"Reaction {action_id} already used {current_count} time(s) this round (max {max_per_round})"
            ],
        )

    return TimingValidationResult(valid=True)
