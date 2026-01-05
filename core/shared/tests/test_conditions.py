"""Tests for condition resolution helpers."""

import pytest
from core.shared.enums import StatusType
from core.shared.conditions import (
    CONDITIONS,
    ConditionApplicationResult,
    ConditionRemovalResult,
    ConditionEffectResult,
    resolve_condition_effects,
    can_apply_condition,
    apply_condition,
    remove_condition,
    clear_all_conditions,
    get_condition_difficulty_modifier,
    is_condition_active,
    get_active_conditions,
    get_condition_stacks,
    conditions_prevent_attacks,
    conditions_prevent_movement,
    conditions_affect_damage_multiplier,
    is_condition,
)


class TestResolveConditionEffects:
    """Tests for resolve_condition_effects."""

    def test_impaired_effects(self):
        """Test impaired condition effects."""
        effects = resolve_condition_effects("impaired")
        assert effects.condition == "impaired"
        assert effects.attack_difficulty_modifier == 1
        assert effects.save_difficulty_modifier == 1
        assert effects.check_difficulty_modifier == 1

    def test_shredded_effects(self):
        """Test shredded condition effects."""
        effects = resolve_condition_effects("shredded")
        assert effects.condition == "shredded"
        assert effects.ignore_armor is True
        assert effects.ignore_resistance is True

    def test_jammed_effects(self):
        """Test jammed condition effects."""
        effects = resolve_condition_effects("jammed")
        assert effects.condition == "jammed"
        assert effects.action_restrictions.get("disallow_tech_actions") is True
        assert effects.action_restrictions.get("disallow_reactions") is True

    def test_lock_on_effects(self):
        """Test lock_on condition effects."""
        effects = resolve_condition_effects("lock_on")
        assert effects.condition == "lock_on"
        assert effects.consumable_accuracy_bonus == 1

    def test_exposed_effects(self):
        """Test exposed condition effects."""
        effects = resolve_condition_effects("exposed")
        assert effects.condition == "exposed"
        assert effects.damage_multiplier == 2.0


class TestCanApplyCondition:
    """Tests for can_apply_condition."""

    def test_can_apply_new_condition(self):
        """Test can apply when target doesn't have condition."""
        can_apply, reason = can_apply_condition([], "impaired")
        assert can_apply is True
        assert reason == ""

    def test_can_apply_within_stacks(self):
        """Test can apply when below max stacks."""
        conditions: list[StatusType] = ["impaired"]
        can_apply, reason = can_apply_condition(conditions, "impaired", max_stacks=3)
        assert can_apply is True
        assert reason == ""

    def test_cannot_apply_at_max_stacks(self):
        """Test cannot apply when at max stacks."""
        conditions: list[StatusType] = ["impaired", "impaired"]
        can_apply, reason = can_apply_condition(conditions, "impaired", max_stacks=2)
        assert can_apply is False
        assert "max stacks" in reason

    def test_unlimited_stacks(self):
        """Test unlimited stacks allows infinite application."""
        conditions: list[StatusType] = ["impaired"] * 10
        can_apply, reason = can_apply_condition(conditions, "impaired")
        assert can_apply is True


class TestApplyCondition:
    """Tests for apply_condition."""

    def test_apply_new_condition(self):
        """Test applying a new condition."""
        conditions: list[StatusType] = []
        result = apply_condition(conditions, "impaired")
        assert result.applied is True
        assert result.condition == "impaired"
        assert result.stacks == 1
        assert "impaired" in conditions

    def test_apply_stacks_condition(self):
        """Test applying same condition multiple times."""
        conditions: list[StatusType] = []
        apply_condition(conditions, "impaired")
        apply_condition(conditions, "impaired")
        result = apply_condition(conditions, "impaired")
        assert result.applied is True
        assert result.stacks == 3
        assert conditions.count("impaired") == 3

    def test_apply_blocked_by_max_stacks(self):
        """Test application blocked at max stacks."""
        conditions: list[StatusType] = ["impaired", "impaired"]
        result = apply_condition(conditions, "impaired", max_stacks=2)
        assert result.applied is False
        assert len(conditions) == 2

    def test_apply_different_conditions(self):
        """Test applying multiple different conditions."""
        conditions: list[StatusType] = []
        result1 = apply_condition(conditions, "impaired")
        result2 = apply_condition(conditions, "shredded")
        assert result1.applied is True
        assert result2.applied is True
        assert len(conditions) == 2
        assert "impaired" in conditions
        assert "shredded" in conditions


class TestRemoveCondition:
    """Tests for remove_condition."""

    def test_remove_single_stack(self):
        """Test removing a single stack of a condition."""
        conditions: list[StatusType] = ["impaired", "impaired", "shredded"]
        result = remove_condition(conditions, "impaired")
        assert result.removed is True
        assert result.remaining_stacks == 1
        assert conditions.count("impaired") == 1
        assert "shredded" in conditions

    def test_remove_all_stacks(self):
        """Test removing all stacks of a condition."""
        conditions: list[StatusType] = ["impaired", "impaired", "impaired"]
        result = remove_condition(conditions, "impaired", count=99)
        assert result.removed is True
        assert result.remaining_stacks == 0
        assert "impaired" not in conditions

    def test_remove_nonexistent_condition(self):
        """Test removing a condition that doesn't exist."""
        conditions: list[StatusType] = []
        result = remove_condition(conditions, "impaired")
        assert result.removed is False
        assert "not present" in result.reason

    def test_remove_partial_stacks(self):
        """Test removing only some stacks."""
        conditions: list[StatusType] = ["impaired", "impaired", "impaired"]
        result = remove_condition(conditions, "impaired", count=2)
        assert result.removed is True
        assert result.remaining_stacks == 1
        assert conditions.count("impaired") == 1


class TestClearAllConditions:
    """Tests for clear_all_conditions."""

    def test_clear_all_conditions(self):
        """Test clearing all conditions."""
        conditions: list[StatusType] = ["impaired", "shredded", "jammed", "impaired"]
        results = clear_all_conditions(conditions)
        assert len(conditions) == 0
        assert len(results) == 3  # 3 unique conditions

    def test_clear_empty_conditions(self):
        """Test clearing empty condition list."""
        conditions: list[StatusType] = []
        results = clear_all_conditions(conditions)
        assert len(results) == 0


class TestGetConditionDifficultyModifier:
    """Tests for get_condition_difficulty_modifier."""

    def test_impaired_attack_modifier(self):
        """Test impaired gives +1 attack difficulty."""
        modifier = get_condition_difficulty_modifier("impaired", "attack")
        assert modifier == 1

    def test_impaired_save_modifier(self):
        """Test impaired gives +1 save difficulty."""
        modifier = get_condition_difficulty_modifier("impaired", "save")
        assert modifier == 1

    def test_impaired_skill_modifier(self):
        """Test impaired gives +1 skill check difficulty."""
        modifier = get_condition_difficulty_modifier("impaired", "skill")
        assert modifier == 1


class TestIsConditionActive:
    """Tests for is_condition_active."""

    def test_active_condition(self):
        """Test detecting active condition."""
        conditions: list[StatusType] = ["impaired", "shredded"]
        assert is_condition_active("impaired", conditions) is True

    def test_inactive_condition(self):
        """Test detecting inactive condition."""
        conditions: list[StatusType] = ["shredded", "jammed"]
        assert is_condition_active("impaired", conditions) is False


class TestGetActiveConditions:
    """Tests for get_active_conditions."""

    def test_get_active_conditions(self):
        """Test getting all active conditions."""
        conditions: list[StatusType] = ["impaired", "shredded", "impaired", "exposed"]
        active = get_active_conditions(conditions)
        assert set(active) == {"impaired", "shredded", "exposed"}

    def test_empty_conditions(self):
        """Test getting active conditions from empty list."""
        active = get_active_conditions([])
        assert active == []


class TestGetConditionStacks:
    """Tests for get_condition_stacks."""

    def test_count_stacks(self):
        """Test counting condition stacks."""
        conditions: list[StatusType] = [
            "impaired",
            "impaired",
            "shredded",
            "exposed",
            "exposed",
        ]
        stacks = get_condition_stacks(conditions)
        assert stacks["impaired"] == 2
        assert stacks["shredded"] == 1
        assert stacks["exposed"] == 2

    def test_empty_stacks(self):
        """Test counting stacks from empty conditions."""
        stacks = get_condition_stacks([])
        assert len(stacks) == 0


class TestConditionsPreventAttacks:
    """Tests for conditions_prevent_attacks."""

    def test_stunned_prevents_attacks(self):
        """Test stunned prevents attacks."""
        assert conditions_prevent_attacks(["stunned"]) is True

    def test_other_conditions_allow_attacks(self):
        """Test other conditions don't prevent attacks."""
        assert conditions_prevent_attacks(["impaired"]) is False
        assert conditions_prevent_attacks(["shredded"]) is False
        assert conditions_prevent_attacks(["jammed"]) is False


class TestConditionsPreventMovement:
    """Tests for conditions_prevent_movement."""

    def test_immobilized_prevents_movement(self):
        """Test immobilized prevents movement."""
        assert conditions_prevent_movement(["immobilized"]) is True

    def test_slowed_affects_movement(self):
        """Test slowed affects movement."""
        assert conditions_prevent_movement(["slowed"]) is True

    def test_no_movement_restriction(self):
        """Test conditions without movement restriction."""
        assert conditions_prevent_movement(["impaired"]) is False
        assert conditions_prevent_movement(["shredded"]) is False


class TestConditionsAffectDamageMultiplier:
    """Tests for conditions_affect_damage_multiplier."""

    def test_exposed_doubles_damage(self):
        """Test exposed doubles damage."""
        multiplier = conditions_affect_damage_multiplier(["exposed"])
        assert multiplier == 2.0

    def test_no_multiplier_without_exposed(self):
        """Test no multiplier without exposed."""
        multiplier = conditions_affect_damage_multiplier(["impaired", "shredded"])
        assert multiplier == 1.0

    def test_combined_multipliers(self):
        """Test multiple exposed conditions don't stack multipliers."""
        multiplier = conditions_affect_damage_multiplier(["exposed", "exposed"])
        assert multiplier == 2.0  # Not 4.0


class TestIsCondition:
    """Tests for is_condition helper."""

    def test_conditions_are_conditions(self):
        """Test that condition types are identified as conditions."""
        assert is_condition("impaired") is True
        assert is_condition("shredded") is True
        assert is_condition("jammed") is True
        assert is_condition("lock_on") is True
        assert is_condition("exposed") is True
        assert is_condition("immobilized") is True
        assert is_condition("slowed") is True
        assert is_condition("stunned") is True

    def test_statuses_are_not_conditions(self):
        """Test that non-condition statuses are not identified as conditions."""
        assert is_condition("braced") is False
        assert is_condition("engaged") is False
        assert is_condition("prone") is False


class TestConditionsSet:
    """Tests for CONDITIONS constant."""

    def test_conditions_contains_all_conditions(self):
        """Test that CONDITIONS set contains all condition types."""
        expected = {
            "impaired",
            "shredded",
            "jammed",
            "lock_on",
            "exposed",
            "immobilized",
            "slowed",
            "stunned",
            "hidden",
        }
        assert CONDITIONS == expected
