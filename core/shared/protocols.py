"""Protocol activation system for Lancer TTRPG.

Implements protocol mechanics per PR2 4406, 5091-5092:

Protocols:
- Free action that can only be activated/deactivated at start of turn
- Two types: activation_only and toggle
- Duration types: scene, start_of_next_turn, end_of_next_turn, turns
- Can apply buffs, conditions, accuracy modifiers, or trigger AI control

Integration:
- Uses timing.validate_protocol_timing() for start-of-turn validation
- Integrates with ai.apply_cede_control() for SEKHMET-style AI protocols
- Tracks active protocols in CombatantState.active_protocols

Resolution Pattern:
1. resolve_protocol_activation() - Validates start-of-turn, applies activation effects
2. resolve_protocol_deactivation() - Validates toggle is active, applies deactivation
3. apply_protocol_effects() - Updates combatant state with buffs/conditions
"""

from __future__ import annotations

from typing import Literal, Any
from pydantic import Field
from core.shared.models import FrozenModel

from core.mech.timing import (
    TurnPhase,
    validate_protocol_timing,
    ActionTimingValidationSettings,
    DEFAULT_TIMING_VALIDATION,
)


ProtocolEffectType = Literal[
    "buff", "condition", "ai_control", "accuracy_mod", "resource_change"
]

ProtocolDurationType = Literal[
    "scene", "start_of_next_turn", "end_of_next_turn", "turns"
]


class ProtocolDuration(FrozenModel):
    """Duration tracking for protocol effects.

    Attributes:
        effect_id: Unique identifier for this effect instance
        duration_type: How the effect expires
        turns_remaining: Only used when duration_type="turns"
    """

    effect_id: str = Field(..., description="Unique identifier for this effect")
    duration_type: ProtocolDurationType = Field(
        ..., description="When the effect expires"
    )
    turns_remaining: int | None = Field(
        default=None, ge=1, description="Turns remaining (for turns duration_type)"
    )


class ProtocolActivationInput(FrozenModel):
    """Input for protocol activation.

    Attributes:
        actor_id: Combatant activating the protocol
        protocol_id: ID of the protocol being activated
        protocol_name: Human-readable name of the protocol
        effect_type: Type of effect this protocol applies
        effect_data: Details of the effect (condition, buff values, etc.)
        duration_type: How long the effect lasts
        duration_turns: Turns remaining if duration_type="turns"
        target_id: Target of the protocol (None if self-target)
        is_toggle: Whether this protocol can be deactivated
        deactivation_effect: Effect applied when protocol is deactivated
    """

    actor_id: str = Field(..., description="Combatant activating the protocol")
    protocol_id: str = Field(..., description="ID of the protocol being activated")
    protocol_name: str = Field(..., description="Human-readable name of the protocol")
    effect_type: ProtocolEffectType = Field(..., description="Type of effect")
    effect_data: dict[str, Any] = Field(
        default_factory=dict, description="Effect details"
    )
    duration_type: ProtocolDurationType = Field(
        default="start_of_next_turn", description="How long effect lasts"
    )
    duration_turns: int | None = Field(
        default=None, ge=1, description="Turns remaining for turns duration"
    )
    target_id: str | None = Field(
        default=None, description="Target of the protocol (None for self)"
    )
    is_toggle: bool = Field(
        default=False, description="Whether this protocol can be deactivated"
    )
    deactivation_effect: dict[str, Any] | None = Field(
        default=None, description="Effect applied on deactivation"
    )


class ProtocolDeactivationInput(FrozenModel):
    """Input for protocol deactivation (toggle protocols only).

    Attributes:
        actor_id: Combatant deactivating the protocol
        protocol_id: ID of the protocol being deactivated
        protocol_name: Human-readable name
    """

    actor_id: str = Field(..., description="Combatant deactivating the protocol")
    protocol_id: str = Field(..., description="ID of the protocol being deactivated")
    protocol_name: str = Field(..., description="Human-readable name of the protocol")


class ProtocolResult(FrozenModel):
    """Result of protocol activation or deactivation.

    Attributes:
        success: Whether the operation succeeded
        operation: "activation" or "deactivation"
        protocol_id: ID of the protocol
        protocol_name: Human-readable name
        effects_applied: List of effect descriptions applied
        duration_tracking: Duration tracking info for state updates
        validation_errors: Errors if operation failed
    """

    success: bool = Field(default=True, description="Whether operation succeeded")
    operation: Literal["activation", "deactivation"] = Field(
        ..., description="Type of operation"
    )
    protocol_id: str = Field(..., description="ID of the protocol")
    protocol_name: str = Field(..., description="Human-readable name")
    effects_applied: list[str] = Field(
        default_factory=list, description="Effect descriptions applied"
    )
    duration_tracking: list[ProtocolDuration] = Field(
        default_factory=list, description="Duration tracking info"
    )
    validation_errors: list[str] = Field(default_factory=list)


class ProtocolState(FrozenModel):
    """State tracking for an active protocol.

    Attributes:
        protocol_id: ID of the protocol
        protocol_name: Human-readable name
        effect_type: Type of effect applied
        effect_data: Effect details
        target_id: Target of the protocol
        is_toggle: Whether this protocol can be deactivated
        deactivation_effect: Effect applied on deactivation
        duration: Duration tracking
    """

    protocol_id: str = Field(..., description="ID of the protocol")
    protocol_name: str = Field(..., description="Human-readable name")
    effect_type: ProtocolEffectType = Field(..., description="Type of effect")
    effect_data: dict[str, Any] = Field(
        default_factory=dict, description="Effect details"
    )
    target_id: str | None = Field(default=None, description="Target of the protocol")
    is_toggle: bool = Field(
        default=False, description="Whether this protocol can be deactivated"
    )
    deactivation_effect: dict[str, Any] | None = Field(
        default=None, description="Effect applied on deactivation"
    )
    duration: ProtocolDuration = Field(..., description="Duration tracking")


def resolve_protocol_activation(
    input: ProtocolActivationInput,
    current_phase: TurnPhase,
    has_ai_control: bool = False,
    settings: ActionTimingValidationSettings | None = None,
) -> ProtocolResult:
    """Resolve protocol activation per PR2 4406, 5091-5092.

    Protocols can only be activated at the start of your turn.
    They apply free action effects that persist until their duration expires.

    For AI protocols (SEKHMET pattern):
    - Activation cedes control to AI via ai.apply_cede_control()
    - Deactivation applies stun + restores pilot control

    Args:
        input: Protocol activation input
        current_phase: Current turn phase (must be "start" for protocols)
        has_ai_control: Whether actor currently has AI control capability
        settings: Validation settings

    Returns:
        ProtocolResult with activation outcome
    """
    if settings is None:
        settings = DEFAULT_TIMING_VALIDATION

    timing_result = validate_protocol_timing(
        action_id=input.protocol_id,
        is_protocol=True,
        current_phase=current_phase,
        settings=settings,
    )

    if not timing_result.valid:
        return ProtocolResult(
            success=False,
            operation="activation",
            protocol_id=input.protocol_id,
            protocol_name=input.protocol_name,
            validation_errors=timing_result.errors,
        )

    effects_applied: list[str] = []
    duration_tracking: list[ProtocolDuration] = []

    if input.effect_type == "condition":
        condition = input.effect_data.get("condition", "")
        target = input.target_id if input.target_id else input.actor_id
        effects_applied.append(f"Applied {condition} to {target}")
    elif input.effect_type == "buff":
        buff_type = input.effect_data.get("type", "")
        value = input.effect_data.get("value", 0)
        target = input.target_id if input.target_id else input.actor_id
        effects_applied.append(f"Applied {buff_type} buff (+{value}) to {target}")
    elif input.effect_type == "accuracy_mod":
        value = input.effect_data.get("value", 0)
        attack_types = input.effect_data.get("attack_types", ["all"])
        effects_applied.append(
            f"Accuracy {'+' if value > 0 else ''}{value} on {attack_types}"
        )
    elif input.effect_type == "resource_change":
        resource = input.effect_data.get("resource", "")
        amount = input.effect_data.get("amount", 0)
        direction = input.effect_data.get("direction", "gain")
        effects_applied.append(f"{direction.title()} {abs(amount)} {resource}")
    elif input.effect_type == "ai_control":
        effects_applied.append("AI control ceded (SEKHMET protocol)")

    duration = ProtocolDuration(
        effect_id=f"{input.protocol_id}:{input.actor_id}",
        duration_type=input.duration_type,
        turns_remaining=input.duration_turns,
    )
    duration_tracking.append(duration)

    return ProtocolResult(
        success=True,
        operation="activation",
        protocol_id=input.protocol_id,
        protocol_name=input.protocol_name,
        effects_applied=effects_applied,
        duration_tracking=duration_tracking,
    )


def resolve_protocol_deactivation(
    input: ProtocolDeactivationInput,
    is_protocol_active: bool,
    deactivation_effect: dict[str, Any] | None,
    current_phase: TurnPhase,
    settings: ActionTimingValidationSettings | None = None,
) -> ProtocolResult:
    """Resolve protocol deactivation (toggle protocols only).

    Toggle protocols can be deactivated at the start of your turn.
    Some protocols (like SEKHMET) apply effects on deactivation (e.g., stun).

    Args:
        input: Protocol deactivation input
        is_protocol_active: Whether the protocol is currently active
        deactivation_effect: Effect to apply on deactivation
        current_phase: Current turn phase
        settings: Validation settings

    Returns:
        ProtocolResult with deactivation outcome
    """
    if settings is None:
        settings = DEFAULT_TIMING_VALIDATION

    validation_errors: list[str] = []

    if not is_protocol_active:
        validation_errors.append(f"Protocol {input.protocol_name} is not active")

    timing_result = validate_protocol_timing(
        action_id=input.protocol_id,
        is_protocol=True,
        current_phase=current_phase,
        settings=settings,
    )

    if not timing_result.valid:
        validation_errors.extend(timing_result.errors)

    if validation_errors:
        return ProtocolResult(
            success=False,
            operation="deactivation",
            protocol_id=input.protocol_id,
            protocol_name=input.protocol_name,
            validation_errors=validation_errors,
        )

    effects_applied: list[str] = []

    if deactivation_effect:
        effect_type = deactivation_effect.get("type", "")
        if effect_type == "condition":
            condition = deactivation_effect.get("condition", "")
            effects_applied.append(f"Applied {condition} on deactivation")
        elif effect_type == "stun":
            effects_applied.append("Applied stunned condition on deactivation")
        else:
            effects_applied.append(f"Applied {effect_type} on deactivation")

    return ProtocolResult(
        success=True,
        operation="deactivation",
        protocol_id=input.protocol_id,
        protocol_name=input.protocol_name,
        effects_applied=effects_applied,
    )


def apply_protocol_state(
    active_protocols: dict[str, ProtocolState],
    result: ProtocolResult,
    input: ProtocolActivationInput,
) -> dict[str, ProtocolState]:
    """Apply protocol activation result to protocol state dict.

    Args:
        active_protocols: Current active protocols
        result: Protocol activation result
        input: Original activation input

    Returns:
        Updated active_protocols dict
    """
    if not result.success:
        return active_protocols

    if result.operation == "deactivation":
        active_protocols.pop(result.protocol_id, None)
        return active_protocols

    duration = (
        result.duration_tracking[0]
        if result.duration_tracking
        else ProtocolDuration(
            effect_id=f"{input.protocol_id}:{input.actor_id}",
            duration_type=input.duration_type,
            turns_remaining=input.duration_turns,
        )
    )

    protocol_state = ProtocolState(
        protocol_id=input.protocol_id,
        protocol_name=input.protocol_name,
        effect_type=input.effect_type,
        effect_data=input.effect_data,
        target_id=input.target_id,
        is_toggle=input.is_toggle,
        deactivation_effect=input.deactivation_effect,
        duration=duration,
    )

    active_protocols[input.protocol_id] = protocol_state
    return active_protocols


def decrement_protocol_durations(
    active_protocols: dict[str, ProtocolState],
) -> tuple[dict[str, ProtocolState], list[str]]:
    """Decrement protocol durations at end of turn.

    For "turns" duration_type: decrements turns_remaining
    For "start_of_next_turn" or "end_of_next_turn": expires after next turn
    For "scene": never expires automatically

    Args:
        active_protocols: Current active protocols

    Returns:
        Tuple of (updated_protocols, expired_protocol_ids)
    """
    updated_protocols: dict[str, ProtocolState] = {}
    expired: list[str] = []

    for protocol_id, protocol in active_protocols.items():
        duration = protocol.duration
        duration_type = duration.duration_type

        if duration_type == "turns":
            if duration.turns_remaining is not None and duration.turns_remaining > 1:
                new_remaining = duration.turns_remaining - 1
                updated_duration = duration.model_copy(
                    update={"turns_remaining": new_remaining}
                )
                updated_protocols[protocol_id] = protocol.model_copy(
                    update={"duration": updated_duration}
                )
            else:
                expired.append(protocol_id)
        elif duration_type in ("start_of_next_turn", "end_of_next_turn"):
            expired.append(protocol_id)
        else:
            updated_protocols[protocol_id] = protocol

    return updated_protocols, expired


def get_protocol_effects_for_combatant(
    active_protocols: dict[str, ProtocolState],
    combatant_id: str,
) -> list[dict[str, Any]]:
    """Get all active protocol effects for a combatant.

    Args:
        active_protocols: Current active protocols
        combatant_id: ID of the combatant

    Returns:
        List of effect dicts to apply to the combatant
    """
    effects: list[dict[str, Any]] = []

    for protocol in active_protocols.values():
        if protocol.effect_type == "condition":
            if protocol.target_id is None or protocol.target_id == combatant_id:
                effects.append(
                    {
                        "effect_type": "condition",
                        "condition": protocol.effect_data.get("condition", ""),
                        "source": protocol.protocol_id,
                        "duration": protocol.duration.duration_type,
                    }
                )
        elif protocol.effect_type == "buff":
            effects.append(
                {
                    "effect_type": "buff",
                    "buff_type": protocol.effect_data.get("type", ""),
                    "value": protocol.effect_data.get("value", 0),
                    "source": protocol.protocol_id,
                    "duration": protocol.duration.duration_type,
                }
            )
        elif protocol.effect_type == "accuracy_mod":
            effects.append(
                {
                    "effect_type": "accuracy_mod",
                    "value": protocol.effect_data.get("value", 0),
                    "attack_types": protocol.effect_data.get("attack_types", ["all"]),
                    "source": protocol.protocol_id,
                    "duration": protocol.duration.duration_type,
                }
            )
        elif protocol.effect_type == "resource_change":
            effects.append(
                {
                    "effect_type": "resource_change",
                    "resource": protocol.effect_data.get("resource", ""),
                    "amount": protocol.effect_data.get("amount", 0),
                    "direction": protocol.effect_data.get("direction", "gain"),
                    "source": protocol.protocol_id,
                    "duration": protocol.duration.duration_type,
                }
            )

    return effects


def check_protocol_active(
    active_protocols: dict[str, ProtocolState],
    protocol_id: str,
) -> bool:
    """Check if a specific protocol is currently active.

    Args:
        active_protocols: Current active protocols
        protocol_id: ID of the protocol to check

    Returns:
        True if the protocol is active
    """
    return protocol_id in active_protocols


def get_active_protocol_ids(
    active_protocols: dict[str, ProtocolState],
) -> list[str]:
    """Get list of all active protocol IDs.

    Args:
        active_protocols: Current active protocols

    Returns:
        List of active protocol IDs
    """
    return list(active_protocols.keys())


class ProtocolValidationSettings(FrozenModel):
    """Validation settings for protocol operations.

    Attributes:
        strict_mode: If True, timing violations are errors
        allow_protocol_outside_start: If True, protocols allowed in any phase
        max_protocols_per_turn: Maximum protocols that can be activated per turn
    """

    strict_mode: bool = Field(default=True)
    allow_protocol_outside_start: bool = Field(default=False)
    max_protocols_per_turn: int = Field(default=10, ge=1)


DEFAULT_PROTOCOL_VALIDATION = ProtocolValidationSettings()


def validate_protocol_count(
    active_protocols: dict[str, ProtocolState],
    settings: ProtocolValidationSettings | None = None,
) -> tuple[bool, list[str]]:
    """Validate that activating another protocol doesn't exceed limits.

    Args:
        active_protocols: Currently active protocols
        settings: Validation settings

    Returns:
        Tuple of (is_valid, error_messages)
    """
    if settings is None:
        settings = DEFAULT_PROTOCOL_VALIDATION

    current_count = len(active_protocols)
    if current_count > settings.max_protocols_per_turn:
        return False, [
            f"Cannot activate more protocols: already at maximum {settings.max_protocols_per_turn}"
        ]

    return True, []
