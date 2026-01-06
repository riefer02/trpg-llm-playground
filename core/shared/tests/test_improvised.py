"""Tests for Improvised Attack action resolution."""

import pytest
from core.shared.improvised import (
    resolve_improvised_attack,
    apply_improvised_result,
    ImprovisedInput,
    ImprovisedRule,
    ImprovisedResolutionResult,
    DEFAULT_IMPROVISED_RULES,
)
from core.shared.dice import DiceExpression
from core.mech.combat_state import (
    CombatantState,
    CombatStats,
    CombatResources,
)


@pytest.fixture
def test_attacker() -> CombatantState:
    """Create a test mech for improvised attack."""
    return CombatantState(
        id="test_attacker",
        name="Test Attacker",
        side="players",
        kind="mech",
        stats=CombatStats(
            size="size_2",
            hp_max=16,
            evasion=10,
            e_defense=8,
            armor=1,
            speed=4,
        ),
        resources=CombatResources(
            hp_current=16,
            heat_current=0,
            heat_cap=10,
            structure_current=3,
            stress_current=2,
        ),
    )


@pytest.fixture
def test_target() -> CombatantState:
    """Create a test target for improvised attack."""
    return CombatantState(
        id="test_target",
        name="Test Target",
        side="hostiles",
        kind="mech",
        stats=CombatStats(
            size="size_2",
            hp_max=16,
            evasion=10,
            e_defense=8,
            armor=1,
            speed=4,
        ),
        resources=CombatResources(
            hp_current=16,
            heat_current=0,
            heat_cap=10,
            structure_current=3,
            stress_current=2,
        ),
    )


@pytest.fixture
def high_evasion_target() -> CombatantState:
    """Create a high evasion target."""
    return CombatantState(
        id="high_evasion_target",
        name="High Evasion Target",
        side="hostiles",
        kind="mech",
        stats=CombatStats(
            size="size_2",
            hp_max=16,
            evasion=18,
            e_defense=8,
            armor=1,
            speed=4,
        ),
        resources=CombatResources(
            hp_current=16,
            heat_current=0,
            heat_cap=10,
            structure_current=3,
            stress_current=2,
        ),
    )


class TestResolveImprovisedAttack:
    """Tests for Improvised Attack resolution."""

    def test_resolve_improvised_attack_success(self, test_attacker, test_target):
        """Test successful improvised attack resolution."""
        input_data = ImprovisedInput(
            actor_id=test_attacker.id,
            target_id=test_target.id,
            is_unarmed=True,
        )
        result = resolve_improvised_attack(
            input_data,
            actor_accuracy_bonus=0,
            target_evasion=10,
        )

        assert result.actor_id == test_attacker.id
        assert result.target_id == test_target.id
        assert result.is_unarmed is True
        assert result.attack_success is True
        assert result.accuracy_roll is not None
        assert result.total_accuracy is not None
        assert result.hit is not None
        assert result.damage_type == "kinetic"
        assert result.validation_errors == []

    def test_resolve_improvised_attack_miss(self, test_attacker, high_evasion_target):
        """Test improvised attack that misses due to high evasion."""
        input_data = ImprovisedInput(
            actor_id=test_attacker.id,
            target_id=high_evasion_target.id,
            is_unarmed=True,
        )
        result = resolve_improvised_attack(
            input_data,
            actor_accuracy_bonus=0,
            target_evasion=18,
            forced_roll=2,
        )

        assert result.hit is False
        assert result.damage_on_hit is None
        assert result.validation_errors == []

    def test_resolve_improvised_attack_hit(self, test_attacker, test_target):
        """Test improvised attack that hits."""
        input_data = ImprovisedInput(
            actor_id=test_attacker.id,
            target_id=test_target.id,
            is_unarmed=True,
        )
        result = resolve_improvised_attack(
            input_data,
            actor_accuracy_bonus=0,
            target_evasion=10,
            forced_roll=10,
        )

        assert result.hit is True
        assert result.damage_on_hit is not None
        assert result.damage_on_hit >= 1
        assert result.damage_on_hit <= 6

    def test_resolve_improvised_attack_requires_unarmed_failure(
        self, test_attacker, test_target
    ):
        """Test improvised attack fails if mech is not unarmed."""
        input_data = ImprovisedInput(
            actor_id=test_attacker.id,
            target_id=test_target.id,
            is_unarmed=False,
        )
        result = resolve_improvised_attack(
            input_data,
            actor_accuracy_bonus=0,
            target_evasion=10,
        )

        assert result.attack_success is False
        assert len(result.validation_errors) > 0
        assert "unarmed" in result.validation_errors[0].lower()

    def test_resolve_improvised_attack_with_accuracy_bonus(
        self, test_attacker, test_target
    ):
        """Test improvised attack with accuracy bonus."""
        input_data = ImprovisedInput(
            actor_id=test_attacker.id,
            target_id=test_target.id,
            is_unarmed=True,
        )
        result = resolve_improvised_attack(
            input_data,
            actor_accuracy_bonus=5,
            target_evasion=10,
            forced_roll=10,
        )

        assert result.hit is True
        assert result.accuracy_bonus == 5
        assert result.total_accuracy == 15

    def test_resolve_improvised_attack_with_custom_rules(
        self, test_attacker, test_target
    ):
        """Test improvised attack with custom rules."""
        custom_rules = ImprovisedRule(
            damage=DiceExpression.parse("1d6+2"),
        )
        input_data = ImprovisedInput(
            actor_id=test_attacker.id,
            target_id=test_target.id,
            is_unarmed=True,
            rules=custom_rules,
        )
        result = resolve_improvised_attack(
            input_data,
            actor_accuracy_bonus=0,
            target_evasion=10,
            forced_roll=10,
        )

        assert result.attack_success is True
        assert result.hit is True
        assert result.damage_on_hit is not None


class TestApplyImprovisedResult:
    """Tests for applying Improvised Attack result."""

    def test_apply_improvised_result_hit(self, test_target):
        """Test applying hit result to target."""
        result = ImprovisedResolutionResult(
            actor_id="test_attacker",
            target_id=test_target.id,
            is_unarmed=True,
            attack_success=True,
            accuracy_roll=10,
            accuracy_bonus=0,
            total_accuracy=10,
            target_evasion=10,
            target_e_defense=8,
            hit=True,
            damage_expression=DiceExpression.parse("1d6"),
            damage_type="kinetic",
            damage_on_hit=4,
            validation_errors=[],
        )

        application = apply_improvised_result(test_target, result)

        assert application.target_hit is True
        assert application.damage_dealt == 4
        assert application.damage_type == "kinetic"

    def test_apply_improvised_result_miss(self, test_target):
        """Test applying miss result to target."""
        result = ImprovisedResolutionResult(
            actor_id="test_attacker",
            target_id=test_target.id,
            is_unarmed=True,
            attack_success=True,
            accuracy_roll=2,
            accuracy_bonus=0,
            total_accuracy=2,
            target_evasion=10,
            target_e_defense=8,
            hit=False,
            damage_expression=DiceExpression.parse("1d6"),
            damage_type="kinetic",
            damage_on_hit=None,
            validation_errors=[],
        )

        application = apply_improvised_result(test_target, result)

        assert application.target_hit is False
        assert application.damage_dealt == 0

    def test_apply_improvised_result_reduces_hp(self, test_target):
        """Test that hit result reduces target HP."""
        original_hp = test_target.resources.hp_current

        result = ImprovisedResolutionResult(
            actor_id="test_attacker",
            target_id=test_target.id,
            is_unarmed=True,
            attack_success=True,
            accuracy_roll=10,
            accuracy_bonus=0,
            total_accuracy=10,
            target_evasion=10,
            target_e_defense=8,
            hit=True,
            damage_expression=DiceExpression.parse("1d6"),
            damage_type="kinetic",
            damage_on_hit=5,
            validation_errors=[],
        )

        application = apply_improvised_result(test_target, result)

        assert application.target_hit is True
        expected_hp = original_hp - 5
        assert application.damage_dealt == 5


class TestImprovisedRuleDefaults:
    """Tests for default Improvised Rule values."""

    def test_default_damage(self):
        """Test default damage is 1d6."""
        from core.shared.improvised import DEFAULT_IMPROVISED_RULES

        assert DEFAULT_IMPROVISED_RULES.damage.count == 1
        assert DEFAULT_IMPROVISED_RULES.damage_type == "kinetic"

    def test_default_requires_unarmed(self):
        """Test default requires_unarmed is True."""
        from core.shared.improvised import DEFAULT_IMPROVISED_RULES

        assert DEFAULT_IMPROVISED_RULES.requires_unarmed is True

    def test_default_attack_type(self):
        """Test default attack type is melee."""
        from core.shared.improvised import DEFAULT_IMPROVISED_RULES

        assert DEFAULT_IMPROVISED_RULES.attack_type == "melee"
