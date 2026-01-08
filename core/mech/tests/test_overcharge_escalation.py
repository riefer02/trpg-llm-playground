"""Tests for overcharge escalation tracking."""

import pytest
from core.mech.combat_state import (
    CombatantState,
    CombatStats,
    CombatResources,
    OverchargeState,
)
from core.mech.combat_resolution import (
    compute_overcharge_escalation,
    use_overcharge,
    reset_overcharge,
    increment_overcharge_on_turn_start,
    OverchargeEscalationResult,
    OverchargeUsageResult,
    OverchargeResetResult,
)
from core.mech.validation.combat_validation import (
    _validate_overcharge_escalation,
    CombatValidationIssue,
)
from core.shared.enums import StatusType
from core.mech.rules import DEFAULT_OVERCHARGE_RULES


class TestOverchargeState:
    """Tests for OverchargeState model."""

    def test_initial_state(self):
        """Test that initial state has level 0 and 0 uses."""
        state = OverchargeState()
        assert state.current_level == 0
        assert state.uses_this_turn == 0
        assert state.can_overcharge is True

    def test_can_overcharge_false_after_use(self):
        """Test that can_overcharge becomes False after use."""
        state = OverchargeState(uses_this_turn=1)
        assert state.can_overcharge is False

    def test_next_cost_at_level_0(self):
        """Test next_cost at level 0 (1 heat)."""
        state = OverchargeState(current_level=0)
        assert state.next_cost == 1

    def test_next_cost_at_level_1(self):
        """Test next_cost at level 1 (1d3)."""
        state = OverchargeState(current_level=1)
        cost = state.next_cost
        from core.shared.dice import DiceExpression

        assert isinstance(cost, DiceExpression)
        assert str(cost) == "1d3"

    def test_next_cost_at_level_2(self):
        """Test next_cost at level 2 (1d6)."""
        state = OverchargeState(current_level=2)
        cost = state.next_cost
        from core.shared.dice import DiceExpression

        assert isinstance(cost, DiceExpression)
        assert str(cost) == "1d6"

    def test_next_cost_at_level_3(self):
        """Test next_cost at level 3 (1d6+4)."""
        state = OverchargeState(current_level=3)
        cost = state.next_cost
        from core.shared.dice import DiceExpression

        assert isinstance(cost, DiceExpression)
        assert str(cost) == "1d6+4"

    def test_level_constrained_to_max_3(self):
        """Test that level greater than 3 is rejected by validation."""
        with pytest.raises(Exception):
            OverchargeState(current_level=5)


class TestComputeOverchargeEscalation:
    """Tests for compute_overcharge_escalation helper."""

    def test_none_state_returns_defaults(self):
        """Test that None state returns initial values."""
        result = compute_overcharge_escalation(None)
        assert result.current_level == 0
        assert result.can_overcharge is True
        assert result.uses_this_turn == 0

    def test_with_existing_state(self):
        """Test with an existing state."""
        state = OverchargeState(current_level=2, uses_this_turn=1)
        result = compute_overcharge_escalation(state)
        assert result.current_level == 2
        assert result.can_overcharge is False
        assert result.uses_this_turn == 1


class TestUseOvercharge:
    """Tests for use_overcharge helper."""

    def test_initial_use_escalates_to_level_1(self):
        """Test that first use escalates from level 0 to 1."""
        state, result = use_overcharge(None)
        assert state.current_level == 1
        assert result.level_before == 0
        assert result.level_after == 1
        assert result.rolled_cost == 1
        assert result.uses_this_turn_after == 1

    def test_second_use_escalates_to_level_2(self):
        """Test that second use escalates from level 1 to 2."""
        state = OverchargeState(current_level=1)
        state, result = use_overcharge(state)
        assert state.current_level == 2
        assert result.level_before == 1
        assert result.level_after == 2

    def test_uses_increment(self):
        """Test that uses_this_turn increments."""
        state = OverchargeState(uses_this_turn=0)
        state, result = use_overcharge(state)
        assert result.uses_this_turn_before == 0
        assert result.uses_this_turn_after == 1

    def test_force_roll_deterministic(self):
        """Test that force_roll produces deterministic result."""
        state = OverchargeState(current_level=1)
        state, result = use_overcharge(state, force_roll=3)
        assert result.rolled_cost == 3

    def test_level_3_max(self):
        """Test that level caps at 3."""
        state = OverchargeState(current_level=3)
        state, result = use_overcharge(state)
        assert state.current_level == 3
        assert result.level_before == 3
        assert result.level_after == 3


class TestResetOvercharge:
    """Tests for reset_overcharge helper."""

    def test_reset_none_state(self):
        """Test resetting None state."""
        state, result = reset_overcharge(None)
        assert state.current_level == 0
        assert result.level_before == 0
        assert result.level_after == 0

    def test_reset_clears_level(self):
        """Test that reset clears the escalation level."""
        state = OverchargeState(current_level=2, uses_this_turn=1)
        state, result = reset_overcharge(state)
        assert state.current_level == 0
        assert result.level_before == 2
        assert result.level_after == 0

    def test_reset_clears_uses(self):
        """Test that reset clears uses_this_turn."""
        state = OverchargeState(current_level=2, uses_this_turn=1)
        _, result = reset_overcharge(state)
        assert result.uses_cleared is True


class TestIncrementOverchargeOnTurnStart:
    """Tests for increment_overcharge_on_turn_start helper."""

    def test_resets_uses_to_zero(self):
        """Test that uses_this_turn resets to 0."""
        state = OverchargeState(current_level=2, uses_this_turn=1)
        new_state = increment_overcharge_on_turn_start(state)
        assert new_state.uses_this_turn == 0
        assert new_state.current_level == 2

    def test_no_change_if_already_zero(self):
        """Test that no change if already 0 uses."""
        state = OverchargeState(current_level=2, uses_this_turn=0)
        new_state = increment_overcharge_on_turn_start(state)
        assert new_state is state

    def test_none_state_returns_initial(self):
        """Test that None state returns initial state."""
        new_state = increment_overcharge_on_turn_start(None)
        assert new_state.current_level == 0
        assert new_state.uses_this_turn == 0


class TestValidateOverchargeEscalation:
    """Tests for _validate_overcharge_escalation helper."""

    def test_non_overcharge_action_returns_empty(self):
        """Test that non-overcharge actions return no issues."""
        from core.mech.combat_state import ActionUse

        action = ActionUse(action_id="skirmish", action_type="full")
        issues = _validate_overcharge_escalation(action, None)
        assert issues == []

    def test_correct_cost_no_issues(self):
        """Test that correct cost produces no issues."""
        from core.mech.combat_state import ActionUse

        action = ActionUse(action_id="overcharge", action_type="free", heat_generated=1)
        issues = _validate_overcharge_escalation(action, None)
        assert issues == []

    def test_wrong_cost_strict_mode_error(self):
        """Test that wrong cost produces error in strict mode."""
        from core.mech.combat_state import ActionUse

        action = ActionUse(action_id="overcharge", action_type="free", heat_generated=5)
        issues = _validate_overcharge_escalation(action, None, strict_mode=True)
        assert len(issues) == 1
        assert issues[0].code == "overcharge_cost_mismatch"
        assert issues[0].severity == "error"

    def test_wrong_cost_narrative_mode_warning(self):
        """Test that wrong cost produces warning in narrative mode."""
        from core.mech.combat_state import ActionUse

        action = ActionUse(action_id="overcharge", action_type="free", heat_generated=5)
        issues = _validate_overcharge_escalation(action, None, strict_mode=False)
        assert len(issues) == 1
        assert issues[0].severity == "warning"

    def test_dice_expression_no_issues(self):
        """Test that dice expression costs don't produce issues."""
        from core.mech.combat_state import ActionUse

        action = ActionUse(
            action_id="overcharge", action_type="free", heat_generated=None
        )
        state = OverchargeState(current_level=1)
        issues = _validate_overcharge_escalation(action, state)
        assert issues == []


class TestOverchargeInCombatantState:
    """Tests for overcharge_state field in CombatantState."""

    def test_combatant_with_overcharge_state(self):
        """Test creating combatant with overcharge state."""
        state = OverchargeState(current_level=1, uses_this_turn=0)
        combatant = CombatantState(
            id="test_mech",
            name="Test Mech",
            side="players",
            kind="mech",
            stats=CombatStats(
                size="size_2",
                hp_max=10,
                evasion=8,
                e_defense=10,
            ),
            resources=CombatResources(hp_current=10),
            overcharge_state=state,
        )
        assert combatant.overcharge_state is not None
        assert combatant.overcharge_state.current_level == 1

    def test_combatant_default_no_overcharge_state(self):
        """Test that combatant defaults to None overcharge_state."""
        combatant = CombatantState(
            id="test_mech",
            name="Test Mech",
            side="players",
            kind="mech",
            stats=CombatStats(
                size="size_2",
                hp_max=10,
                evasion=8,
                e_defense=10,
            ),
            resources=CombatResources(hp_current=10),
        )
        assert combatant.overcharge_state is None


class TestOverchargeEscalationFlow:
    """Integration tests for overcharge escalation flow."""

    def test_full_escalation_sequence(self):
        """Test full escalation from level 0 to max."""
        state = None
        expected_levels = [1, 2, 3, 3]

        for i, expected_level in enumerate(expected_levels):
            state, result = use_overcharge(state)
            assert state.current_level == expected_level, (
                f"Step {i}: expected {expected_level}, got {state.current_level}"
            )

    def test_turn_start_resets_uses_not_level(self):
        """Test that turn start resets uses but not level."""
        state = OverchargeState(current_level=2, uses_this_turn=0)
        state, _ = use_overcharge(state)
        assert state.current_level == 3
        assert state.uses_this_turn == 1

        new_state = increment_overcharge_on_turn_start(state)
        assert new_state.current_level == 3
        assert new_state.uses_this_turn == 0

    def test_full_repair_resets_everything(self):
        """Test that full repair resets everything."""
        state = OverchargeState(current_level=2, uses_this_turn=1)
        state, result = reset_overcharge(state)
        assert state.current_level == 0
        assert state.uses_this_turn == 0
        assert result.level_before == 2
        assert result.level_after == 0
        assert result.uses_cleared is True
