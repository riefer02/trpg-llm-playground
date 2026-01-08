"""Tests for action economy primitives."""

import pytest
from core.mech.action_economy import (
    ActionEconomyState,
    ActionEconomyResult,
    OverchargeCostResult,
    OverchargeInput,
    OverchargeResult,
    compute_overcharge_cost,
    resolve_overcharge,
    validate_action_economy,
    use_full_action,
    use_quick_action,
    use_overcharge,
    use_reaction,
    reset_economy_for_new_turn,
    get_action_economy_summary,
)
from core.mech.combat_state import OverchargeState
from core.shared.enums import ActionType


class TestActionEconomyState:
    """Tests for ActionEconomyState model."""

    def test_initial_state(self):
        """Test that a new economy state has all counters at zero."""
        economy = ActionEconomyState()
        assert economy.full_actions_used == 0
        assert economy.quick_actions_used == 0
        assert economy.overcharge_used is False
        assert economy.reactions_used_this_turn == 0

    def test_full_actions_remaining(self):
        """Test full action remaining calculation."""
        economy = ActionEconomyState()
        assert economy.full_actions_remaining == 1

        economy = economy.model_copy(update={"full_actions_used": 1})
        assert economy.full_actions_remaining == 0

    def test_quick_actions_remaining(self):
        """Test quick action remaining calculation."""
        economy = ActionEconomyState()
        assert economy.quick_actions_remaining == 2

        economy = economy.model_copy(update={"quick_actions_used": 1})
        assert economy.quick_actions_remaining == 1

        economy = economy.model_copy(update={"quick_actions_used": 2})
        assert economy.quick_actions_remaining == 0

    def test_can_overcharge(self):
        """Test overcharge availability."""
        economy = ActionEconomyState()
        assert economy.can_overcharge is True

        economy = economy.model_copy(update={"overcharge_used": True})
        assert economy.can_overcharge is False

    def test_reactions_remaining_this_turn(self):
        """Test reaction remaining calculation."""
        economy = ActionEconomyState()
        assert economy.reactions_remaining_this_turn == 1

        economy = economy.model_copy(update={"reactions_used_this_turn": 1})
        assert economy.reactions_remaining_this_turn == 0


class TestOverchargeState:
    """Tests for OverchargeState model."""

    def test_initial_state(self):
        """Test that a new overcharge state starts at level 0."""
        state = OverchargeState()
        assert state.current_level == 0
        assert state.uses_this_turn == 0
        assert state.can_overcharge is True

    def test_next_cost_level_0(self):
        """Test next cost at level 0 is 1 heat."""
        state = OverchargeState(current_level=0)
        cost = state.next_cost
        assert cost == 1

    def test_escalation_on_use(self):
        """Test that using overcharge increments escalation level."""
        state = OverchargeState(current_level=0, uses_this_turn=0)
        assert state.can_overcharge is True

        state = state.model_copy(update={"uses_this_turn": 1})
        assert state.can_overcharge is False


class TestComputeOverchargeCost:
    """Tests for compute_overcharge_cost function."""

    def test_level_0_cost(self):
        """Test level 0 cost is 1 heat."""
        result = compute_overcharge_cost(0)
        assert result.level == 0
        assert result.modified_cost == 1

    def test_level_1_cost_dice(self):
        """Test level 1 cost is 1d3."""
        result = compute_overcharge_cost(1)
        assert result.level == 1
        assert result.roll_result is not None
        assert 1 <= result.roll_result <= 3

    def test_level_2_cost_dice(self):
        """Test level 2 cost is 1d6."""
        result = compute_overcharge_cost(2)
        assert result.level == 2
        assert result.roll_result is not None
        assert 1 <= result.roll_result <= 6

    def test_level_3_cost_dice_plus_4(self):
        """Test level 3 cost is 1d6+4."""
        result = compute_overcharge_cost(3)
        assert result.level == 3
        assert result.roll_result is not None
        assert 5 <= result.modified_cost <= 10


class TestResolveOvercharge:
    """Tests for resolve_overcharge function."""

    def test_overcharge_state_escalation(self):
        """Test overcharge state escalation mechanics."""
        state = OverchargeState(current_level=0, uses_this_turn=0)
        assert state.can_overcharge is True

        state = state.model_copy(update={"uses_this_turn": 1})
        assert state.can_overcharge is False

        state = state.model_copy(update={"current_level": 1, "uses_this_turn": 0})
        assert state.can_overcharge is True

    def test_overcharge_cost_by_level(self):
        """Test overcharge costs escalate by level."""
        level_0 = compute_overcharge_cost(0)
        assert level_0.modified_cost == 1

        level_1 = compute_overcharge_cost(1)
        assert level_1.modified_cost >= 1 and level_1.modified_cost <= 3

        level_2 = compute_overcharge_cost(2)
        assert level_2.modified_cost >= 1 and level_2.modified_cost <= 6


class TestValidateActionEconomy:
    """Tests for validate_action_economy function."""

    def test_full_action_available(self):
        """Test validation when full action is available."""
        economy = ActionEconomyState()
        result = validate_action_economy(economy, "full")
        assert result.can_take_action is True
        assert result.can_take_full_action is True

    def test_full_action_exhausted(self):
        """Test validation when full action is already used."""
        economy = ActionEconomyState(full_actions_used=1)
        result = validate_action_economy(economy, "full")
        assert result.can_take_action is False
        assert len(result.errors) > 0

    def test_quick_action_available(self):
        """Test validation when quick actions are available."""
        economy = ActionEconomyState()
        result = validate_action_economy(economy, "quick")
        assert result.can_take_action is True
        assert result.can_take_quick_action is True

    def test_quick_actions_exhausted(self):
        """Test validation when quick actions are exhausted."""
        economy = ActionEconomyState(quick_actions_used=2)
        result = validate_action_economy(economy, "quick")
        assert result.can_take_action is False
        assert "Quick actions exhausted" in result.errors[0]

    def test_reaction_available(self):
        """Test validation when reaction is available."""
        economy = ActionEconomyState()
        result = validate_action_economy(economy, "reaction")
        assert result.can_take_action is True
        assert result.can_take_reaction is True

    def test_reaction_exhausted(self):
        """Test validation when reaction is already used."""
        economy = ActionEconomyState(reactions_used_this_turn=1)
        result = validate_action_economy(economy, "reaction")
        assert result.can_take_action is False

    def test_overcharge_available(self):
        """Test validation for overcharge action."""
        economy = ActionEconomyState()
        result = validate_action_economy(economy, "free", is_overcharge=True)
        assert result.can_take_action is True
        assert result.can_overcharge is True

    def test_overcharge_already_used(self):
        """Test validation when overcharge is already used."""
        economy = ActionEconomyState(overcharge_used=True)
        result = validate_action_economy(economy, "free", is_overcharge=True)
        assert result.can_take_action is False


class TestActionEconomyModifiers:
    """Tests for action economy modifier functions."""

    def test_use_full_action(self):
        """Test using a full action."""
        economy = ActionEconomyState()
        updated = use_full_action(economy)
        assert updated.full_actions_used == 1

    def test_use_quick_action(self):
        """Test using a quick action."""
        economy = ActionEconomyState()
        updated = use_quick_action(economy)
        assert updated.quick_actions_used == 1

    def test_use_overcharge(self):
        """Test using overcharge."""
        economy = ActionEconomyState()
        updated = use_overcharge(economy)
        assert updated.overcharge_used is True

    def test_use_reaction(self):
        """Test using a reaction."""
        economy = ActionEconomyState()
        updated = use_reaction(economy)
        assert updated.reactions_used_this_turn == 1

    def test_reset_economy_for_new_turn(self):
        """Test resetting economy for new turn."""
        economy = ActionEconomyState(
            full_actions_used=1,
            quick_actions_used=2,
            overcharge_used=True,
            reactions_used_this_turn=1,
        )
        reset = reset_economy_for_new_turn(economy)
        assert reset.full_actions_used == 0
        assert reset.quick_actions_used == 0
        assert reset.overcharge_used is False
        assert reset.reactions_used_this_turn == 0


class TestGetActionEconomySummary:
    """Tests for get_action_economy_summary function."""

    def test_summary_format(self):
        """Test that summary returns expected format."""
        economy = ActionEconomyState()
        summary = get_action_economy_summary(economy)

        assert "full_actions" in summary
        assert "quick_actions" in summary
        assert "overcharge" in summary
        assert "reactions_this_turn" in summary
        assert "can_take_full" in summary
        assert "can_take_quick" in summary
        assert "can_overcharge" in summary
        assert "can_react" in summary

    def test_summary_after_usage(self):
        """Test summary reflects used actions."""
        economy = ActionEconomyState(
            full_actions_used=1,
            quick_actions_used=1,
            overcharge_used=True,
            reactions_used_this_turn=1,
        )
        summary = get_action_economy_summary(economy)

        assert summary["full_actions"] == "1/1"
        assert summary["quick_actions"] == "1/2"
        assert summary["overcharge"] == "used"
        assert summary["can_take_full"] is False
        assert summary["can_overcharge"] is False
