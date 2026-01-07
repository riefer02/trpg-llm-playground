"""Tests for roll mechanics and attack resolution."""

import pytest
from core.shared.rolls import (
    resolve_attack,
    AttackResolutionResult,
    AccuracyDifficulty,
    FlatBonus,
    RollModifiers,
    AttackRoll,
    SkillCheck,
    ContestedCheck,
)


class TestResolveAttack:
    """Tests for the resolve_attack function."""

    def test_basic_hit(self):
        """Test basic attack that hits."""
        result = resolve_attack(
            attack_bonus=3,
            target_defense=10,
            forced_roll=12,
        )

        assert result.hit is True
        assert result.roll == 12
        assert result.attack_bonus == 3
        assert result.total_accuracy == 15
        assert result.target_defense == 10
        assert result.net_accuracy == 0
        assert result.miss_by == 0

    def test_basic_miss(self):
        """Test basic attack that misses."""
        result = resolve_attack(
            attack_bonus=2,
            target_defense=15,
            forced_roll=8,
        )

        assert result.hit is False
        assert result.roll == 8
        assert result.total_accuracy == 10
        assert result.target_defense == 15
        assert result.miss_by == 5

    def test_exact_tie_hit_on_10(self):
        """Test that ties hit when roll >= 10."""
        result = resolve_attack(
            attack_bonus=5,
            target_defense=15,
            forced_roll=10,
        )

        assert result.hit is True
        assert result.total_accuracy == 15
        assert result.roll == 10

    def test_exact_tie_miss_below_10(self):
        """Test that ties miss when roll < 10."""
        result = resolve_attack(
            attack_bonus=5,
            target_defense=15,
            forced_roll=9,
        )

        assert result.hit is False
        assert result.total_accuracy == 14
        assert result.roll == 9
        assert result.miss_by == 1

    def test_critical_hit_natural_20(self):
        """Test that natural 20 is always a critical hit."""
        result = resolve_attack(
            attack_bonus=1,
            target_defense=25,
            forced_roll=20,
        )

        assert result.hit is True
        assert result.is_critical is True
        assert result.total_accuracy == 21

    def test_critical_hit_with_high_bonus(self):
        """Test critical hit with very high bonus."""
        result = resolve_attack(
            attack_bonus=10,
            target_defense=35,
            forced_roll=20,
        )

        assert result.hit is True
        assert result.is_critical is True
        assert result.total_accuracy == 30

    def test_accuracy_dice_single(self):
        """Test single accuracy die."""
        result = resolve_attack(
            attack_bonus=2,
            target_defense=12,
            accuracy_bonus=1,
            difficulty_bonus=0,
            forced_roll=5,
            forced_accuracy_rolls=[6],
        )

        assert result.hit is True
        assert result.accuracy_dice_rolls == [6]
        assert result.net_accuracy == 6
        assert result.total_accuracy == 13

    def test_accuracy_dice_multiple(self):
        """Test multiple accuracy dice (keep highest)."""
        result = resolve_attack(
            attack_bonus=2,
            target_defense=12,
            accuracy_bonus=2,
            difficulty_bonus=0,
            forced_roll=5,
            forced_accuracy_rolls=[3, 6],
        )

        assert result.hit is True
        assert result.accuracy_dice_rolls == [3, 6]
        assert result.net_accuracy == 6
        assert result.total_accuracy == 13

    def test_difficulty_dice_single(self):
        """Test single difficulty die."""
        result = resolve_attack(
            attack_bonus=5,
            target_defense=8,
            accuracy_bonus=0,
            difficulty_bonus=1,
            forced_roll=8,
            forced_difficulty_rolls=[4],
        )

        assert result.hit is True
        assert result.difficulty_dice_rolls == [4]
        assert result.net_accuracy == -4
        assert result.total_accuracy == 9

    def test_difficulty_dice_multiple(self):
        """Test multiple difficulty dice (keep lowest)."""
        result = resolve_attack(
            attack_bonus=5,
            target_defense=9,
            accuracy_bonus=0,
            difficulty_bonus=2,
            forced_roll=8,
            forced_difficulty_rolls=[3, 5],
        )

        assert result.hit is True
        assert result.difficulty_dice_rolls == [3, 5]
        assert result.net_accuracy == -3
        assert result.total_accuracy == 10

    def test_accuracy_difficulty_cancellation(self):
        """Test accuracy and difficulty cancellation."""
        result = resolve_attack(
            attack_bonus=3,
            target_defense=12,
            accuracy_bonus=2,
            difficulty_bonus=2,
            forced_roll=5,
            forced_accuracy_rolls=[4, 6],
            forced_difficulty_rolls=[2, 5],
        )

        assert result.hit is False
        assert result.accuracy_dice_rolls == [4, 6]
        assert result.difficulty_dice_rolls == [2, 5]
        assert result.net_accuracy == 0
        assert result.total_accuracy == 8
        assert result.miss_by == 4

    def test_more_accuracy_than_difficulty(self):
        """Test net accuracy when accuracy > difficulty."""
        result = resolve_attack(
            attack_bonus=2,
            target_defense=12,
            accuracy_bonus=3,
            difficulty_bonus=1,
            forced_roll=5,
            forced_accuracy_rolls=[2, 4, 6],
            forced_difficulty_rolls=[3],
        )

        assert result.hit is True
        assert result.net_accuracy == 6
        assert result.total_accuracy == 13

    def test_more_difficulty_than_accuracy(self):
        """Test net accuracy when difficulty > accuracy."""
        result = resolve_attack(
            attack_bonus=5,
            target_defense=6,
            accuracy_bonus=1,
            difficulty_bonus=2,
            forced_roll=6,
            forced_accuracy_rolls=[5],
            forced_difficulty_rolls=[2, 4],
        )

        assert result.hit is True
        assert result.net_accuracy == -4
        assert result.total_accuracy == 7

    def test_edge_case_roll_1(self):
        """Test roll of 1 (minimum)."""
        result = resolve_attack(
            attack_bonus=5,
            target_defense=7,
            forced_roll=1,
        )

        assert result.hit is False
        assert result.roll == 1
        assert result.total_accuracy == 6
        assert result.miss_by == 1

    def test_edge_case_roll_10(self):
        """Test roll of 10 (tie threshold)."""
        result = resolve_attack(
            attack_bonus=2,
            target_defense=12,
            forced_roll=10,
        )

        assert result.hit is True
        assert result.roll == 10
        assert result.total_accuracy == 12

    def test_edge_case_roll_9(self):
        """Test roll of 9 (just below tie threshold)."""
        result = resolve_attack(
            attack_bonus=3,
            target_defense=12,
            forced_roll=9,
        )

        assert result.hit is False
        assert result.roll == 9
        assert result.total_accuracy == 12
        assert result.miss_by == 0

    def test_high_defense_target(self):
        """Test attack against very high defense."""
        result = resolve_attack(
            attack_bonus=5,
            target_defense=20,
            accuracy_bonus=2,
            difficulty_bonus=0,
            forced_roll=15,
            forced_accuracy_rolls=[6],
        )

        assert result.hit is True
        assert result.total_accuracy == 26

    def test_zero_attack_bonus(self):
        """Test attack with no bonus."""
        result = resolve_attack(
            attack_bonus=0,
            target_defense=10,
            forced_roll=10,
        )

        assert result.hit is True
        assert result.total_accuracy == 10

    def test_zero_defense(self):
        """Test attack against zero defense."""
        result = resolve_attack(
            attack_bonus=1,
            target_defense=0,
            forced_roll=1,
        )

        assert result.hit is True
        assert result.miss_by == 0


class TestAttackResolutionResult:
    """Tests for AttackResolutionResult model."""

    def test_result_fields(self):
        """Test that result contains all expected fields."""
        result = resolve_attack(
            attack_bonus=3,
            target_defense=10,
            forced_roll=12,
        )

        assert isinstance(result, AttackResolutionResult)
        assert result.roll == 12
        assert result.attack_bonus == 3
        assert result.total_accuracy == 15
        assert result.target_defense == 10
        assert result.hit is True
        assert result.is_critical is False
        assert result.miss_by == 0

    def test_miss_by_calculation(self):
        """Test miss_by calculation."""
        result = resolve_attack(
            attack_bonus=2,
            target_defense=15,
            forced_roll=5,
        )

        assert result.hit is False
        assert result.miss_by == 8

    def test_miss_by_zero_on_hit(self):
        """Test miss_by is 0 when attack hits."""
        result = resolve_attack(
            attack_bonus=5,
            target_defense=10,
            forced_roll=8,
        )

        assert result.hit is True
        assert result.miss_by == 0


class TestAccuracyDifficulty:
    """Tests for AccuracyDifficulty model."""

    def test_net_accuracy_positive(self):
        """Test net accuracy when accuracy > difficulty."""
        ad = AccuracyDifficulty(accuracy=3, difficulty=1)
        assert ad.net == 2
        assert ad.direction == "accuracy"

    def test_net_difficulty_positive(self):
        """Test net difficulty when difficulty > accuracy."""
        ad = AccuracyDifficulty(accuracy=1, difficulty=3)
        assert ad.net == -2
        assert ad.direction == "difficulty"

    def test_equal_cancellation(self):
        """Test when accuracy equals difficulty."""
        ad = AccuracyDifficulty(accuracy=2, difficulty=2)
        assert ad.net == 0
        assert ad.direction == "none"

    def test_dice_count(self):
        """Test dice count calculation."""
        ad = AccuracyDifficulty(accuracy=3, difficulty=1)
        assert ad.dice_count == 2


class TestFlatBonus:
    """Tests for FlatBonus model."""

    def test_flat_bonus_creation(self):
        """Test FlatBonus creation."""
        bonus = FlatBonus(source="grit", value=3)
        assert bonus.source == "grit"
        assert bonus.value == 3


class TestRollModifiers:
    """Tests for RollModifiers model."""

    def test_roll_modifiers_default(self):
        """Test default RollModifiers."""
        modifiers = RollModifiers()
        assert modifiers.accuracy_difficulty.accuracy == 0
        assert modifiers.accuracy_difficulty.difficulty == 0
        assert modifiers.flat_bonus is None

    def test_roll_modifiers_with_bonus(self):
        """Test RollModifiers with flat bonus."""
        modifiers = RollModifiers(flat_bonus=FlatBonus(source="trigger", value=4))
        assert modifiers.flat_bonus is not None
        assert modifiers.flat_bonus.value == 4


class TestContestedCheck:
    """Tests for ContestedCheck model."""

    def test_contested_check_default_tie_breaker(self):
        """Test default tie breaker is attacker."""
        check = ContestedCheck(
            attacker=SkillCheck(target=10),
            defender=SkillCheck(target=8),
        )
        assert check.tie_breaker == "attacker"


class TestAttackRoll:
    """Tests for AttackRoll model."""

    def test_attack_roll_creation(self):
        """Test AttackRoll creation."""
        attack = AttackRoll(target=10)
        assert attack.roll_type == "attack"
        assert attack.target == 10


class TestSkillCheck:
    """Tests for SkillCheck model."""

    def test_skill_check_default(self):
        """Test default SkillCheck values."""
        check = SkillCheck()
        assert check.roll_type == "skill_check"
        assert check.target == 10
        assert check.is_difficult is False

    def test_skill_check_difficult(self):
        """Test difficult SkillCheck."""
        check = SkillCheck(target=12, is_difficult=True)
        assert check.is_difficult is True
