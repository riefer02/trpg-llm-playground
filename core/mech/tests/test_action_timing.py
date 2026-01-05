"""Tests for action timing, prepared actions, and protocol validation."""

import pytest
from typing import Literal
from core.mech.timing import (
    TurnPhase,
    PreparedActionState,
    ActionTimingValidationSettings,
    TimingValidationResult,
    validate_protocol_timing,
    validate_action_while_prepared,
    validate_per_round_reaction,
)
from core.mech.combat_state import (
    CombatantState,
    CombatStats,
    CombatResources,
    MechInventory,
    CombatTurn,
    ActionUse,
    CombatRound,
    MechCombatScenario,
)
from core.shared.enums import StatusType
from core.mech.combat_validation import validate_combat_scenario


class TestPreparedActionState:
    """Tests for PreparedActionState model."""

    def test_prepared_action_state_creation(self):
        """Test creating a prepared action state."""
        prepared = PreparedActionState(
            held_action_id="skirmish",
            held_action_type="quick",
            trigger_condition="enemy enters engagement range",
            created_on_turn=1,
            expires_on_turn=2,
        )
        assert prepared.held_action_id == "skirmish"
        assert prepared.held_action_type == "quick"
        assert prepared.blocks_actions is True
        assert prepared.blocks_reactions is True
        assert prepared.blocks_movement is True

    def test_prepared_action_state_custom_lockout(self):
        """Test creating a prepared action with custom lockout settings."""
        prepared = PreparedActionState(
            held_action_id="ram",
            held_action_type="quick",
            trigger_condition="enemy moves adjacent",
            created_on_turn=1,
            expires_on_turn=2,
            blocks_actions=False,
            blocks_reactions=True,
            blocks_movement=True,
        )
        assert prepared.blocks_actions is False
        assert prepared.blocks_reactions is True


class TestActionTimingValidationSettings:
    """Tests for ActionTimingValidationSettings."""

    def test_default_settings_strict(self):
        """Test that default settings are strict."""
        settings = ActionTimingValidationSettings()
        assert settings.strict_mode is True
        assert settings.allow_protocol_outside_start is False
        assert settings.allow_actions_while_prepared is False

    def test_narrative_settings(self):
        """Test settings for narrative play (non-strict)."""
        settings = ActionTimingValidationSettings(
            strict_mode=False,
            allow_protocol_outside_start=True,
            allow_actions_while_prepared=True,
            allow_reactions_while_prepared=True,
            allow_movement_while_prepared=True,
        )
        assert settings.strict_mode is False
        assert settings.allow_protocol_outside_start is True


class TestValidateProtocolTiming:
    """Tests for protocol timing validation."""

    def test_protocol_at_start_valid(self):
        """Test that protocol at start of turn is valid."""
        result = validate_protocol_timing(
            action_id="activate",
            is_protocol=True,
            current_phase="start",
        )
        assert result.valid is True
        assert len(result.errors) == 0

    def test_protocol_during_normal_invalid_strict(self):
        """Test that protocol during normal phase is invalid in strict mode."""
        settings = ActionTimingValidationSettings(strict_mode=True)
        result = validate_protocol_timing(
            action_id="activate",
            is_protocol=True,
            current_phase="normal",
            settings=settings,
        )
        assert result.valid is False
        assert len(result.errors) == 1
        assert "start of your turn" in result.errors[0]

    def test_protocol_during_normal_warning_narrative(self):
        """Test that protocol during normal phase is a warning in narrative mode."""
        settings = ActionTimingValidationSettings(strict_mode=False)
        result = validate_protocol_timing(
            action_id="activate",
            is_protocol=True,
            current_phase="normal",
            settings=settings,
        )
        assert result.valid is False
        assert len(result.warnings) == 1
        assert "typically" in result.warnings[0]

    def test_protocol_during_end_invalid_strict(self):
        """Test that protocol during end phase is invalid in strict mode."""
        settings = ActionTimingValidationSettings(strict_mode=True)
        result = validate_protocol_timing(
            action_id="activate",
            is_protocol=True,
            current_phase="end",
            settings=settings,
        )
        assert result.valid is False
        assert len(result.errors) == 1

    def test_non_protocol_action_always_valid(self):
        """Test that non-protocol actions are always valid."""
        result = validate_protocol_timing(
            action_id="skirmish",
            is_protocol=False,
            current_phase="normal",
        )
        assert result.valid is True

    def test_allow_protocol_outside_start_setting(self):
        """Test allow_protocol_outside_start setting."""
        settings = ActionTimingValidationSettings(
            allow_protocol_outside_start=True,
            strict_mode=True,
        )
        result = validate_protocol_timing(
            action_id="activate",
            is_protocol=True,
            current_phase="normal",
            settings=settings,
        )
        assert result.valid is True


class TestValidateActionWhilePrepared:
    """Tests for prepared action lockout validation."""

    def test_action_with_no_prepared_valid(self):
        """Test that actions are valid when no prepared action exists."""
        result = validate_action_while_prepared(
            action_id="skirmish",
            action_type="quick",
            prepared_state=None,
        )
        assert result.valid is True

    def test_quick_action_blocked_while_prepared_strict(self):
        """Test that quick actions are blocked while prepared in strict mode."""
        settings = ActionTimingValidationSettings(strict_mode=True)
        prepared = PreparedActionState(
            held_action_id="skirmish",
            held_action_type="quick",
            trigger_condition="enemy enters engagement",
            created_on_turn=1,
            expires_on_turn=2,
        )
        result = validate_action_while_prepared(
            action_id="boost",
            action_type="quick",
            prepared_state=prepared,
            settings=settings,
        )
        assert result.valid is False
        assert len(result.errors) == 1

    def test_quick_action_allowed_while_prepared_narrative(self):
        """Test that quick actions are allowed while prepared in narrative mode."""
        settings = ActionTimingValidationSettings(
            strict_mode=False,
            allow_actions_while_prepared=True,
        )
        prepared = PreparedActionState(
            held_action_id="skirmish",
            held_action_type="quick",
            trigger_condition="enemy enters engagement",
            created_on_turn=1,
            expires_on_turn=2,
        )
        result = validate_action_while_prepared(
            action_id="boost",
            action_type="quick",
            prepared_state=prepared,
            settings=settings,
        )
        assert result.valid is True

    def test_reaction_blocked_while_prepared_strict(self):
        """Test that reactions are blocked while prepared in strict mode."""
        settings = ActionTimingValidationSettings(strict_mode=True)
        prepared = PreparedActionState(
            held_action_id="skirmish",
            held_action_type="quick",
            trigger_condition="enemy enters engagement",
            created_on_turn=1,
            expires_on_turn=2,
        )
        result = validate_action_while_prepared(
            action_id="brace",
            action_type="reaction",
            prepared_state=prepared,
            settings=settings,
        )
        assert result.valid is False

    def test_reaction_allowed_while_prepared_with_setting(self):
        """Test that reactions are allowed while prepared with setting."""
        settings = ActionTimingValidationSettings(
            strict_mode=True,
            allow_reactions_while_prepared=True,
        )
        prepared = PreparedActionState(
            held_action_id="skirmish",
            held_action_type="quick",
            trigger_condition="enemy enters engagement",
            created_on_turn=1,
            expires_on_turn=2,
        )
        result = validate_action_while_prepared(
            action_id="brace",
            action_type="reaction",
            prepared_state=prepared,
            settings=settings,
        )
        assert result.valid is True

    def test_movement_blocked_while_prepared_strict(self):
        """Test that movement is blocked while prepared in strict mode."""
        settings = ActionTimingValidationSettings(strict_mode=True)
        prepared = PreparedActionState(
            held_action_id="skirmish",
            held_action_type="quick",
            trigger_condition="enemy enters engagement",
            created_on_turn=1,
            expires_on_turn=2,
        )
        result = validate_action_while_prepared(
            action_id="move",
            action_type="move",
            prepared_state=prepared,
            settings=settings,
        )
        assert result.valid is False


class TestValidatePerRoundReaction:
    """Tests for per-round reaction limit validation."""

    def test_reaction_under_limit_valid(self):
        """Test that reaction under the limit is valid."""
        reaction_counts = {"brace": 0}
        result = validate_per_round_reaction(
            action_id="brace",
            current_round=1,
            actor_id="mech1",
            reaction_counts_by_actor={"mech1": reaction_counts},
            max_per_round=1,
        )
        assert result.valid is True

    def test_reaction_at_limit_invalid(self):
        """Test that reaction at the limit is invalid."""
        reaction_counts = {"brace": 1}
        result = validate_per_round_reaction(
            action_id="brace",
            current_round=1,
            actor_id="mech1",
            reaction_counts_by_actor={"mech1": reaction_counts},
            max_per_round=1,
        )
        assert result.valid is False
        assert "already used" in result.errors[0]

    def test_reaction_over_limit_invalid(self):
        """Test that reaction over the limit is invalid."""
        reaction_counts = {"brace": 2}
        result = validate_per_round_reaction(
            action_id="brace",
            current_round=1,
            actor_id="mech1",
            reaction_counts_by_actor={"mech1": reaction_counts},
            max_per_round=1,
        )
        assert result.valid is False


class TestPreparedActionResolution:
    """Tests for prepared action resolution helpers."""

    def test_prepare_action_creates_state(self):
        """Test that prepare_action creates a PreparedActionState."""
        from core.mech.combat_resolution import prepare_action, PreparedActionResult

        combatant = _create_test_combatant("mech1")
        result = prepare_action(
            combatant=combatant,
            held_action_id="skirmish",
            held_action_type="quick",
            trigger_condition="enemy enters engagement range",
            current_round=1,
            expires_on_turn=2,
        )
        assert isinstance(result, PreparedActionResult)
        assert result.success is True
        assert result.prepared_action is not None
        assert result.prepared_action.held_action_id == "skirmish"

    def test_trigger_prepared_action_with_state(self):
        """Test triggering a prepared action."""
        from core.mech.combat_resolution import (
            trigger_prepared_action,
            PreparedActionTriggerResult,
        )

        prepared = PreparedActionState(
            held_action_id="skirmish",
            held_action_type="quick",
            trigger_condition="enemy enters engagement",
            created_on_turn=1,
            expires_on_turn=2,
        )
        combatant = _create_test_combatant("mech1")
        combatant = combatant.model_copy(update={"prepared_action": prepared})
        result = trigger_prepared_action(combatant)
        assert isinstance(result, PreparedActionTriggerResult)
        assert result.success is True
        assert result.executed_action_id == "skirmish"
        assert result.prepared_action_cleared is True

    def test_trigger_prepared_action_no_state(self):
        """Test triggering when no prepared action exists."""
        from core.mech.combat_resolution import (
            trigger_prepared_action,
            PreparedActionTriggerResult,
        )

        combatant = _create_test_combatant("mech1")
        result = trigger_prepared_action(combatant)
        assert isinstance(result, PreparedActionTriggerResult)
        assert result.success is False
        assert "No prepared action" in result.message


class TestPerRoundReactionResolution:
    """Tests for per-round reaction resolution helpers."""

    def test_consume_per_round_reaction_first_use(self):
        """Test consuming a per-round reaction for the first time."""
        from core.mech.combat_resolution import (
            consume_per_round_reaction,
            PerRoundReactionResult,
        )

        combatant = _create_test_combatant("mech1")
        result = consume_per_round_reaction(
            combatant=combatant,
            action_id="brace",
            max_per_round=1,
        )
        assert isinstance(result, PerRoundReactionResult)
        assert result.success is True
        assert result.reaction_consumed is True
        assert result.uses_remaining == 0

    def test_consume_per_round_reaction_at_limit(self):
        """Test consuming a per-round reaction at the limit."""
        from core.mech.combat_resolution import (
            consume_per_round_reaction,
            PerRoundReactionResult,
        )

        combatant = _create_test_combatant("mech1")
        combatant.per_round_reactions["brace"] = 1
        result = consume_per_round_reaction(
            combatant=combatant,
            action_id="brace",
            max_per_round=1,
        )
        assert isinstance(result, PerRoundReactionResult)
        assert result.success is False
        assert result.reaction_consumed is False


class TestValidateCombatScenarioWithTiming:
    """Integration tests for timing validation in combat scenarios."""

    def test_action_at_normal_phase_valid(self):
        """Test that a normal action at normal phase passes validation."""
        from core.mech.combat_state import ActionUse

        combatant = _create_test_combatant("mech1")
        turn = CombatTurn(
            actor_id="mech1",
            actions=[
                ActionUse(
                    action_id="activate",
                    action_type="quick",
                )
            ],
        )
        round_ = CombatRound(round_index=1, turns=[turn])
        scenario = MechCombatScenario(
            combatants=[combatant],
            rounds=[round_],
        )
        result = validate_combat_scenario(scenario)
        assert result.valid is True

    def test_prepare_action_tracks_state(self):
        """Test that prepare action creates prepared action state."""
        from core.mech.combat_resolution import prepare_action

        combatant = _create_test_combatant("mech1")
        result = prepare_action(
            combatant=combatant,
            held_action_id="skirmish",
            held_action_type="quick",
            trigger_condition="enemy enters engagement",
            current_round=1,
            expires_on_turn=2,
        )
        assert result.success is True
        assert result.prepared_action is not None

    def test_action_while_prepared_invalid(self):
        """Test that actions while prepared fail validation."""
        scenario = _create_test_scenario_with_prepared_action(
            actor_id="mech1",
            subsequent_action_id="boost",
            subsequent_action_type="quick",
        )
        result = validate_combat_scenario(scenario)
        assert result.valid is False
        error_codes = [issue.code for issue in result.issues]
        assert "prepared_action_lockout" in error_codes


def _create_test_combatant(combatant_id: str) -> CombatantState:
    """Create a test combatant."""
    return CombatantState(
        id=combatant_id,
        name=f"Test {combatant_id}",
        side="players",
        kind="mech",
        stats=CombatStats(
            size="size_2",
            hp_max=10,
            evasion=10,
            e_defense=10,
            armor=0,
            speed=4,
        ),
        resources=CombatResources(hp_current=10),
        inventory=MechInventory(),
    )


def _create_test_turn(
    actor_id: str,
    action_id: str,
    action_type: Literal["free", "quick", "full", "reaction", "protocol", "move"],
    current_phase: TurnPhase = "normal",
) -> CombatTurn:
    """Create a test turn with a single action."""
    return CombatTurn(
        actor_id=actor_id,
        actions=[
            ActionUse(
                action_id=action_id,
                action_type=action_type,
            )
        ],
    )


def _create_test_scenario_with_action(
    actor_id: str,
    action_id: str,
    action_type: Literal["free", "quick", "full", "reaction", "protocol", "move"],
    current_phase: TurnPhase,
) -> MechCombatScenario:
    """Create a test scenario with a single action at a specific phase."""
    combatant = _create_test_combatant(actor_id)
    turn = _create_test_turn(actor_id, action_id, action_type, current_phase)
    round_ = CombatRound(round_index=1, turns=[turn])
    return MechCombatScenario(
        combatants=[combatant],
        rounds=[round_],
    )


def _create_test_scenario_with_prepared_action(
    actor_id: str,
    subsequent_action_id: str,
    subsequent_action_type: Literal[
        "free", "quick", "full", "reaction", "protocol", "move"
    ],
) -> MechCombatScenario:
    """Create a test scenario with a prepared action and subsequent action."""
    prepared = PreparedActionState(
        held_action_id="skirmish",
        held_action_type="quick",
        trigger_condition="enemy enters engagement",
        created_on_turn=1,
        expires_on_turn=2,
    )
    combatant = _create_test_combatant(actor_id)
    combatant = combatant.model_copy(update={"prepared_action": prepared})
    turn = _create_test_turn(
        actor_id, subsequent_action_id, subsequent_action_type, "normal"
    )
    round_ = CombatRound(round_index=1, turns=[turn])
    return MechCombatScenario(
        combatants=[combatant],
        rounds=[round_],
    )
    turn = _create_test_turn(
        actor_id, subsequent_action_id, subsequent_action_type, "normal"
    )
    round_ = CombatRound(round_index=1, turns=[turn])
    return MechCombatScenario(
        combatants=[combatant],
        rounds=[round_],
    )
