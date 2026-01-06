"""Tests for protocol activation system."""

from __future__ import annotations

from typing import Literal
import pytest
from pydantic import Field
from core.shared.models import FrozenModel
from core.shared.protocols import (
    ProtocolActivationInput,
    ProtocolDeactivationInput,
    ProtocolResult,
    ProtocolState,
    ProtocolDuration,
    ProtocolEffectType,
    ProtocolDurationType,
    resolve_protocol_activation,
    resolve_protocol_deactivation,
    apply_protocol_state,
    decrement_protocol_durations,
    get_protocol_effects_for_combatant,
    check_protocol_active,
    get_active_protocol_ids,
    validate_protocol_count,
    ProtocolValidationSettings,
    DEFAULT_PROTOCOL_VALIDATION,
)
from core.mech.timing import TurnPhase


class TestResolveProtocolActivation:
    """Tests for resolve_protocol_activation function."""

    def test_protocol_activation_at_start_of_turn_valid(self):
        """Test that protocol activation at start of turn is valid."""
        input = ProtocolActivationInput(
            actor_id="mech1",
            protocol_id="stable_steady",
            protocol_name="Stable Steady",
            effect_type="buff",
            effect_data={"type": "accuracy", "value": 1},
            duration_type="start_of_next_turn",
        )
        result = resolve_protocol_activation(input, "start")

        assert result.success
        assert result.operation == "activation"
        assert result.protocol_id == "stable_steady"
        assert len(result.effects_applied) == 1

    def test_protocol_activation_during_normal_phase_invalid(self):
        """Test that protocol activation during normal phase is invalid."""
        input = ProtocolActivationInput(
            actor_id="mech1",
            protocol_id="shield_protocol",
            protocol_name="Shield Protocol",
            effect_type="buff",
            effect_data={"type": "accuracy", "value": -2},
            duration_type="start_of_next_turn",
        )
        result = resolve_protocol_activation(input, "normal")

        assert not result.success
        assert len(result.validation_errors) > 0
        assert "start of your turn" in result.validation_errors[0].lower()

    def test_protocol_activation_during_end_phase_invalid(self):
        """Test that protocol activation during end phase is invalid."""
        input = ProtocolActivationInput(
            actor_id="mech1",
            protocol_id="stable_steady",
            protocol_name="Stable Steady",
            effect_type="buff",
            effect_data={"type": "accuracy", "value": 1},
            duration_type="start_of_next_turn",
        )
        result = resolve_protocol_activation(input, "end")

        assert not result.success
        assert len(result.validation_errors) > 0

    def test_protocol_activation_with_condition_effect(self):
        """Test protocol activation with condition effect (e.g., immobilized)."""
        input = ProtocolActivationInput(
            actor_id="mech1",
            protocol_id="stable_steady",
            protocol_name="Stable Steady",
            effect_type="condition",
            effect_data={"condition": "immobilized"},
            duration_type="start_of_next_turn",
        )
        result = resolve_protocol_activation(input, "start")

        assert result.success
        assert "immobilized" in result.effects_applied[0]

    def test_protocol_activation_with_accuracy_mod(self):
        """Test protocol activation with accuracy modifier."""
        input = ProtocolActivationInput(
            actor_id="mech1",
            protocol_id="stable_steady",
            protocol_name="Stable Steady",
            effect_type="accuracy_mod",
            effect_data={"value": 1, "attack_types": ["rifle"]},
            duration_type="start_of_next_turn",
        )
        result = resolve_protocol_activation(input, "start")

        assert result.success
        assert "+1" in result.effects_applied[0]
        assert "rifle" in result.effects_applied[0]

    def test_protocol_activation_with_resource_change(self):
        """Test protocol activation with resource change (e.g., heat gain)."""
        input = ProtocolActivationInput(
            actor_id="mech1",
            protocol_id="shield_protocol",
            protocol_name="Shield Protocol",
            effect_type="resource_change",
            effect_data={"resource": "heat", "amount": 1, "direction": "gain"},
            duration_type="start_of_next_turn",
        )
        result = resolve_protocol_activation(input, "start")

        assert result.success
        assert "1 heat" in result.effects_applied[0]
        assert "Gain" in result.effects_applied[0]

    def test_protocol_activation_with_ai_control(self):
        """Test protocol activation with AI control (SEKHMET pattern)."""
        input = ProtocolActivationInput(
            actor_id="mech1",
            protocol_id="sekhmat_protocol",
            protocol_name="SEKHMET Protocol",
            effect_type="ai_control",
            effect_data={},
            duration_type="scene",
        )
        result = resolve_protocol_activation(input, "start")

        assert result.success
        assert "AI control ceded" in result.effects_applied[0]

    def test_protocol_activation_with_turns_duration(self):
        """Test protocol activation with turns duration."""
        input = ProtocolActivationInput(
            actor_id="mech1",
            protocol_id="test_protocol",
            protocol_name="Test Protocol",
            effect_type="buff",
            effect_data={"type": "defense", "value": 2},
            duration_type="turns",
            duration_turns=3,
        )
        result = resolve_protocol_activation(input, "start")

        assert result.success
        assert len(result.duration_tracking) == 1
        assert result.duration_tracking[0].duration_type == "turns"
        assert result.duration_tracking[0].turns_remaining == 3

    def test_protocol_activation_with_target(self):
        """Test protocol activation with specific target."""
        input = ProtocolActivationInput(
            actor_id="mech1",
            protocol_id="shield_protocol",
            protocol_name="Shield Protocol",
            effect_type="buff",
            effect_data={"type": "defense", "value": 2},
            duration_type="start_of_next_turn",
            target_id="ally1",
        )
        result = resolve_protocol_activation(input, "start")

        assert result.success
        assert "ally1" in result.effects_applied[0]


class TestResolveProtocolDeactivation:
    """Tests for resolve_protocol_deactivation function."""

    def test_protocol_deactivation_when_active(self):
        """Test successful deactivation of active toggle protocol."""
        input = ProtocolDeactivationInput(
            actor_id="mech1",
            protocol_id="sekhmat_protocol",
            protocol_name="SEKHMET Protocol",
        )
        result = resolve_protocol_deactivation(
            input,
            is_protocol_active=True,
            deactivation_effect={"type": "condition", "condition": "stunned"},
            current_phase="start",
        )

        assert result.success
        assert result.operation == "deactivation"
        assert "stunned" in result.effects_applied[0]

    def test_protocol_deactivation_when_inactive(self):
        """Test deactivation fails when protocol is not active."""
        input = ProtocolDeactivationInput(
            actor_id="mech1",
            protocol_id="sekhmat_protocol",
            protocol_name="SEKHMET Protocol",
        )
        result = resolve_protocol_deactivation(
            input,
            is_protocol_active=False,
            deactivation_effect=None,
            current_phase="start",
        )

        assert not result.success
        assert len(result.validation_errors) > 0
        assert "not active" in result.validation_errors[0].lower()

    def test_protocol_deactivation_wrong_phase(self):
        """Test deactivation fails when not at start of turn."""
        input = ProtocolDeactivationInput(
            actor_id="mech1",
            protocol_id="sekhmat_protocol",
            protocol_name="SEKHMET Protocol",
        )
        result = resolve_protocol_deactivation(
            input,
            is_protocol_active=True,
            deactivation_effect={"type": "stun"},
            current_phase="normal",
        )

        assert not result.success
        assert len(result.validation_errors) > 0

    def test_protocol_deactivation_no_effect(self):
        """Test deactivation without effect applies no effects."""
        input = ProtocolDeactivationInput(
            actor_id="mech1",
            protocol_id="simple_toggle",
            protocol_name="Simple Toggle",
        )
        result = resolve_protocol_deactivation(
            input,
            is_protocol_active=True,
            deactivation_effect=None,
            current_phase="start",
        )

        assert result.success
        assert len(result.effects_applied) == 0


class TestApplyProtocolState:
    """Tests for apply_protocol_state function."""

    def test_apply_protocol_activation(self):
        """Test applying protocol activation to state."""
        input = ProtocolActivationInput(
            actor_id="mech1",
            protocol_id="stable_steady",
            protocol_name="Stable Steady",
            effect_type="buff",
            effect_data={"type": "accuracy", "value": 1},
            duration_type="start_of_next_turn",
        )
        result = resolve_protocol_activation(input, "start")
        active_protocols = {}

        updated = apply_protocol_state(active_protocols, result, input)

        assert "stable_steady" in updated
        assert updated["stable_steady"].protocol_id == "stable_steady"
        assert updated["stable_steady"].effect_type == "buff"

    def test_apply_protocol_deactivation(self):
        """Test applying protocol deactivation to state."""
        initial = {
            "sekhmat_protocol": ProtocolState(
                protocol_id="sekhmat_protocol",
                protocol_name="SEKHMET Protocol",
                effect_type="ai_control",
                effect_data={},
                target_id=None,
                is_toggle=True,
                deactivation_effect={"type": "condition", "condition": "stunned"},
                duration=ProtocolDuration(
                    effect_id="sekhmat_protocol:mech1",
                    duration_type="scene",
                ),
            )
        }
        input = ProtocolDeactivationInput(
            actor_id="mech1",
            protocol_id="sekhmat_protocol",
            protocol_name="SEKHMET Protocol",
        )
        result = resolve_protocol_deactivation(
            input,
            is_protocol_active=True,
            deactivation_effect={"type": "condition", "condition": "stunned"},
            current_phase="start",
        )

        updated = apply_protocol_state(initial, result, input)

        assert "sekhmat_protocol" not in updated

    def test_apply_failed_activation_no_change(self):
        """Test failed activation leaves state unchanged."""
        input = ProtocolActivationInput(
            actor_id="mech1",
            protocol_id="stable_steady",
            protocol_name="Stable Steady",
            effect_type="buff",
            effect_data={"type": "accuracy", "value": 1},
            duration_type="start_of_next_turn",
        )
        result = resolve_protocol_activation(input, "normal")  # Will fail
        active_protocols = {"other_protocol": None}

        updated = apply_protocol_state(active_protocols, result, input)

        assert updated == active_protocols


class TestDecrementProtocolDurations:
    """Tests for decrement_protocol_durations function."""

    def test_decrement_turns_duration(self):
        """Test decrementing turns duration."""
        initial = {
            "test_protocol": ProtocolState(
                protocol_id="test_protocol",
                protocol_name="Test Protocol",
                effect_type="buff",
                effect_data={"type": "defense", "value": 1},
                target_id=None,
                is_toggle=False,
                deactivation_effect=None,
                duration=ProtocolDuration(
                    effect_id="test_protocol:mech1",
                    duration_type="turns",
                    turns_remaining=3,
                ),
            )
        }

        updated, expired = decrement_protocol_durations(initial)

        assert "test_protocol" in updated
        assert updated["test_protocol"].duration.turns_remaining == 2
        assert len(expired) == 0

    def test_expire_turns_duration_at_one(self):
        """Test turns duration expires when reaches 1."""
        initial = {
            "test_protocol": ProtocolState(
                protocol_id="test_protocol",
                protocol_name="Test Protocol",
                effect_type="buff",
                effect_data={"type": "defense", "value": 1},
                target_id=None,
                is_toggle=False,
                deactivation_effect=None,
                duration=ProtocolDuration(
                    effect_id="test_protocol:mech1",
                    duration_type="turns",
                    turns_remaining=1,
                ),
            )
        }

        updated, expired = decrement_protocol_durations(initial)

        assert "test_protocol" not in updated
        assert "test_protocol" in expired

    def test_expire_start_of_next_turn(self):
        """Test start_of_next_turn duration expires immediately."""
        initial = {
            "test_protocol": ProtocolState(
                protocol_id="test_protocol",
                protocol_name="Test Protocol",
                effect_type="buff",
                effect_data={"type": "defense", "value": 1},
                target_id=None,
                is_toggle=False,
                deactivation_effect=None,
                duration=ProtocolDuration(
                    effect_id="test_protocol:mech1",
                    duration_type="start_of_next_turn",
                ),
            )
        }

        updated, expired = decrement_protocol_durations(initial)

        assert "test_protocol" not in updated
        assert "test_protocol" in expired

    def test_expire_end_of_next_turn(self):
        """Test end_of_next_turn duration expires immediately."""
        initial = {
            "test_protocol": ProtocolState(
                protocol_id="test_protocol",
                protocol_name="Test Protocol",
                effect_type="buff",
                effect_data={"type": "defense", "value": 1},
                target_id=None,
                is_toggle=False,
                deactivation_effect=None,
                duration=ProtocolDuration(
                    effect_id="test_protocol:mech1",
                    duration_type="end_of_next_turn",
                ),
            )
        }

        updated, expired = decrement_protocol_durations(initial)

        assert "test_protocol" not in updated
        assert "test_protocol" in expired

    def test_scene_duration_not_expired(self):
        """Test scene duration is not affected by decrement."""
        initial = {
            "test_protocol": ProtocolState(
                protocol_id="test_protocol",
                protocol_name="Test Protocol",
                effect_type="buff",
                effect_data={"type": "defense", "value": 1},
                target_id=None,
                is_toggle=False,
                deactivation_effect=None,
                duration=ProtocolDuration(
                    effect_id="test_protocol:mech1",
                    duration_type="scene",
                ),
            )
        }

        updated, expired = decrement_protocol_durations(initial)

        assert "test_protocol" in updated
        assert len(expired) == 0

    def test_multiple_protocols_decrement(self):
        """Test multiple protocols are processed correctly."""
        initial = {
            "turns_protocol": ProtocolState(
                protocol_id="turns_protocol",
                protocol_name="Turns Protocol",
                effect_type="buff",
                effect_data={},
                target_id=None,
                is_toggle=False,
                deactivation_effect=None,
                duration=ProtocolDuration(
                    effect_id="turns_protocol:mech1",
                    duration_type="turns",
                    turns_remaining=2,
                ),
            ),
            "scene_protocol": ProtocolState(
                protocol_id="scene_protocol",
                protocol_name="Scene Protocol",
                effect_type="buff",
                effect_data={},
                target_id=None,
                is_toggle=False,
                deactivation_effect=None,
                duration=ProtocolDuration(
                    effect_id="scene_protocol:mech1",
                    duration_type="scene",
                ),
            ),
        }

        updated, expired = decrement_protocol_durations(initial)

        assert "turns_protocol" in updated
        assert updated["turns_protocol"].duration.turns_remaining == 1
        assert "scene_protocol" in updated
        assert len(expired) == 0


class TestGetProtocolEffectsForCombatant:
    """Tests for get_protocol_effects_for_combatant function."""

    def test_get_condition_effect(self):
        """Test getting condition effect for combatant."""
        active_protocols = {
            "test_protocol": ProtocolState(
                protocol_id="test_protocol",
                protocol_name="Test Protocol",
                effect_type="condition",
                effect_data={"condition": "immobilized"},
                target_id="mech1",
                is_toggle=False,
                deactivation_effect=None,
                duration=ProtocolDuration(
                    effect_id="test_protocol:mech1",
                    duration_type="start_of_next_turn",
                ),
            )
        }

        effects = get_protocol_effects_for_combatant(active_protocols, "mech1")

        assert len(effects) == 1
        assert effects[0]["effect_type"] == "condition"
        assert effects[0]["condition"] == "immobilized"

    def test_get_condition_effect_wrong_target(self):
        """Test condition effect not returned for wrong target."""
        active_protocols = {
            "test_protocol": ProtocolState(
                protocol_id="test_protocol",
                protocol_name="Test Protocol",
                effect_type="condition",
                effect_data={"condition": "immobilized"},
                target_id="other_mech",
                is_toggle=False,
                deactivation_effect=None,
                duration=ProtocolDuration(
                    effect_id="test_protocol:other_mech",
                    duration_type="start_of_next_turn",
                ),
            )
        }

        effects = get_protocol_effects_for_combatant(active_protocols, "mech1")

        assert len(effects) == 0

    def test_get_self_target_condition(self):
        """Test self-target condition (None) applies to actor."""
        active_protocols = {
            "test_protocol": ProtocolState(
                protocol_id="test_protocol",
                protocol_name="Test Protocol",
                effect_type="condition",
                effect_data={"condition": "impaired"},
                target_id=None,
                is_toggle=False,
                deactivation_effect=None,
                duration=ProtocolDuration(
                    effect_id="test_protocol:mech1",
                    duration_type="start_of_next_turn",
                ),
            )
        }

        effects = get_protocol_effects_for_combatant(active_protocols, "mech1")

        assert len(effects) == 1
        assert effects[0]["condition"] == "impaired"

    def test_get_accuracy_mod_effect(self):
        """Test getting accuracy modifier effect."""
        active_protocols = {
            "stable_steady": ProtocolState(
                protocol_id="stable_steady",
                protocol_name="Stable Steady",
                effect_type="accuracy_mod",
                effect_data={"value": 1, "attack_types": ["rifle"]},
                target_id=None,
                is_toggle=False,
                deactivation_effect=None,
                duration=ProtocolDuration(
                    effect_id="stable_steady:mech1",
                    duration_type="start_of_next_turn",
                ),
            )
        }

        effects = get_protocol_effects_for_combatant(active_protocols, "mech1")

        assert len(effects) == 1
        assert effects[0]["effect_type"] == "accuracy_mod"
        assert effects[0]["value"] == 1
        assert "rifle" in effects[0]["attack_types"]

    def test_get_resource_change_effect(self):
        """Test getting resource change effect."""
        active_protocols = {
            "shield_protocol": ProtocolState(
                protocol_id="shield_protocol",
                protocol_name="Shield Protocol",
                effect_type="resource_change",
                effect_data={"resource": "heat", "amount": 1, "direction": "gain"},
                target_id=None,
                is_toggle=False,
                deactivation_effect=None,
                duration=ProtocolDuration(
                    effect_id="shield_protocol:mech1",
                    duration_type="start_of_next_turn",
                ),
            )
        }

        effects = get_protocol_effects_for_combatant(active_protocols, "mech1")

        assert len(effects) == 1
        assert effects[0]["effect_type"] == "resource_change"
        assert effects[0]["resource"] == "heat"
        assert effects[0]["amount"] == 1


class TestCheckProtocolActive:
    """Tests for check_protocol_active function."""

    def test_active_protocol(self):
        """Test checking active protocol returns True."""
        active_protocols = {
            "test_protocol": ProtocolState(
                protocol_id="test_protocol",
                protocol_name="Test Protocol",
                effect_type="buff",
                effect_data={},
                target_id=None,
                is_toggle=False,
                deactivation_effect=None,
                duration=ProtocolDuration(
                    effect_id="test_protocol:mech1",
                    duration_type="scene",
                ),
            )
        }

        assert check_protocol_active(active_protocols, "test_protocol")

    def test_inactive_protocol(self):
        """Test checking inactive protocol returns False."""
        active_protocols = {}

        assert not check_protocol_active(active_protocols, "test_protocol")


class TestGetActiveProtocolIds:
    """Tests for get_active_protocol_ids function."""

    def test_get_single_protocol(self):
        """Test getting active protocol IDs with single protocol."""
        active_protocols = {
            "test_protocol": ProtocolState(
                protocol_id="test_protocol",
                protocol_name="Test Protocol",
                effect_type="buff",
                effect_data={},
                target_id=None,
                is_toggle=False,
                deactivation_effect=None,
                duration=ProtocolDuration(
                    effect_id="test_protocol:mech1",
                    duration_type="scene",
                ),
            )
        }

        ids = get_active_protocol_ids(active_protocols)

        assert ids == ["test_protocol"]

    def test_get_multiple_protocols(self):
        """Test getting active protocol IDs with multiple protocols."""
        active_protocols = {
            "protocol1": ProtocolState(
                protocol_id="protocol1",
                protocol_name="Protocol 1",
                effect_type="buff",
                effect_data={},
                target_id=None,
                is_toggle=False,
                deactivation_effect=None,
                duration=ProtocolDuration(
                    effect_id="protocol1:mech1",
                    duration_type="scene",
                ),
            ),
            "protocol2": ProtocolState(
                protocol_id="protocol2",
                protocol_name="Protocol 2",
                effect_type="buff",
                effect_data={},
                target_id=None,
                is_toggle=False,
                deactivation_effect=None,
                duration=ProtocolDuration(
                    effect_id="protocol2:mech1",
                    duration_type="scene",
                ),
            ),
        }

        ids = get_active_protocol_ids(active_protocols)

        assert len(ids) == 2
        assert "protocol1" in ids
        assert "protocol2" in ids

    def test_get_no_protocols(self):
        """Test getting active protocol IDs with no protocols."""
        ids = get_active_protocol_ids({})

        assert ids == []


class TestValidateProtocolCount:
    """Tests for validate_protocol_count function."""

    def test_under_max_protocols(self):
        """Test validation passes when under max protocols."""
        active_protocols = {f"protocol{i}": None for i in range(5)}
        settings = ProtocolValidationSettings(max_protocols_per_turn=10)

        is_valid, errors = validate_protocol_count(active_protocols, settings)

        assert is_valid
        assert len(errors) == 0

    def test_at_max_protocols(self):
        """Test validation passes when at max protocols."""
        active_protocols = {f"protocol{i}": None for i in range(10)}
        settings = ProtocolValidationSettings(max_protocols_per_turn=10)

        is_valid, errors = validate_protocol_count(active_protocols, settings)

        assert is_valid
        assert len(errors) == 0

    def test_over_max_protocols(self):
        """Test validation fails when over max protocols."""
        active_protocols = {f"protocol{i}": None for i in range(11)}
        settings = ProtocolValidationSettings(max_protocols_per_turn=10)

        is_valid, errors = validate_protocol_count(active_protocols, settings)

        assert not is_valid
        assert len(errors) > 0
        assert "maximum" in errors[0].lower()

    def test_default_settings(self):
        """Test using default validation settings."""
        active_protocols = {f"protocol{i}": None for i in range(15)}

        is_valid, errors = validate_protocol_count(active_protocols)

        assert not is_valid
        assert len(errors) > 0


class TestProtocolDurationTypes:
    """Tests for different protocol duration types."""

    def test_start_of_next_turn_duration(self):
        """Test start_of_next_turn duration type."""
        input = ProtocolActivationInput(
            actor_id="mech1",
            protocol_id="test",
            protocol_name="Test",
            effect_type="buff",
            effect_data={},
            duration_type="start_of_next_turn",
        )
        result = resolve_protocol_activation(input, "start")

        assert result.success
        assert result.duration_tracking[0].duration_type == "start_of_next_turn"

    def test_end_of_next_turn_duration(self):
        """Test end_of_next_turn duration type."""
        input = ProtocolActivationInput(
            actor_id="mech1",
            protocol_id="test",
            protocol_name="Test",
            effect_type="buff",
            effect_data={},
            duration_type="end_of_next_turn",
        )
        result = resolve_protocol_activation(input, "start")

        assert result.success
        assert result.duration_tracking[0].duration_type == "end_of_next_turn"

    def test_scene_duration(self):
        """Test scene duration type."""
        input = ProtocolActivationInput(
            actor_id="mech1",
            protocol_id="test",
            protocol_name="Test",
            effect_type="ai_control",
            effect_data={},
            duration_type="scene",
        )
        result = resolve_protocol_activation(input, "start")

        assert result.success
        assert result.duration_tracking[0].duration_type == "scene"


class TestToggleProtocols:
    """Tests for toggle protocol functionality."""

    def test_toggle_protocol_activation(self):
        """Test activating a toggle protocol."""
        input = ProtocolActivationInput(
            actor_id="mech1",
            protocol_id="sekhmat",
            protocol_name="SEKHMET",
            effect_type="ai_control",
            effect_data={},
            duration_type="scene",
            is_toggle=True,
            deactivation_effect={"type": "condition", "condition": "stunned"},
        )
        result = resolve_protocol_activation(input, "start")

        assert result.success
        assert len(result.effects_applied) == 1

    def test_non_toggle_protocol_no_deactivation(self):
        """Test non-toggle protocol cannot be deactivated."""
        input = ProtocolActivationInput(
            actor_id="mech1",
            protocol_id="stable",
            protocol_name="Stable Steady",
            effect_type="buff",
            effect_data={},
            duration_type="start_of_next_turn",
            is_toggle=False,
        )
        result = resolve_protocol_activation(input, "start")

        assert result.success
        assert result.protocol_id == "stable"
