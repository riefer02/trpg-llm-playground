"""Tests for repair system resolution."""

import pytest
from core.shared.repair import (
    resolve_rest,
    apply_rest_result,
    calculate_repair_capacity,
    can_spend_repair,
    calculate_repair_cost,
    RestInput,
    RestRule,
    RepairSpec,
)
from core.mech.combat_state import (
    CombatantState,
    CombatStats,
    CombatResources,
)


@pytest.fixture
def test_combatant() -> CombatantState:
    """Create a test combatant for repair tests."""
    return CombatantState(
        id="test_mech",
        name="Test Mech",
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
        resources=CombatResources(
            hp_current=5,
            heat_current=5,
            heat_cap=10,
            structure_current=3,
            stress_current=2,
            repairs_remaining=6,
        ),
        statuses=["exposed"],
        conditions=["impaired"],
    )


@pytest.fixture
def damaged_combatant() -> CombatantState:
    """Create a heavily damaged combatant."""
    return CombatantState(
        id="damaged_mech",
        name="Damaged Mech",
        side="players",
        kind="mech",
        stats=CombatStats(
            size="size_2",
            hp_max=15,
            evasion=8,
            e_defense=10,
            armor=1,
            speed=3,
        ),
        resources=CombatResources(
            hp_current=3,
            heat_current=8,
            heat_cap=10,
            structure_current=1,
            stress_current=3,
            repairs_remaining=4,
        ),
        statuses=["burn", "shredded"],
        conditions=["slowed", "jammed"],
    )


@pytest.fixture
def destroyed_combatant() -> CombatantState:
    """Create a combatant in destroyed state."""
    return CombatantState(
        id="destroyed_mech",
        name="Destroyed Mech",
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
        resources=CombatResources(
            hp_current=0,
            heat_current=0,
            heat_cap=10,
            structure_current=0,
            stress_current=4,
            repairs_remaining=0,
        ),
        statuses=[],
        conditions=[],
    )


@pytest.fixture
def low_repairs_combatant() -> CombatantState:
    """Create a combatant with few repairs remaining."""
    return CombatantState(
        id="low_repairs_mech",
        name="Low Repairs Mech",
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
        resources=CombatResources(
            hp_current=5,
            heat_current=5,
            heat_cap=10,
            structure_current=3,
            stress_current=2,
            repairs_remaining=1,
        ),
    )


class TestCalculateRepairCapacity:
    """Tests for repair capacity calculation."""

    def test_base_capacity(self):
        """Test base repair capacity without hull bonus."""
        assert calculate_repair_capacity(5, 0) == 5

    def test_hull_bonus_2(self):
        """Test repair capacity with 2 hull bonus."""
        assert calculate_repair_capacity(5, 2) == 6

    def test_hull_bonus_4(self):
        """Test repair capacity with 4 hull bonus."""
        assert calculate_repair_capacity(5, 4) == 7

    def test_hull_bonus_6(self):
        """Test repair capacity with 6 hull bonus."""
        assert calculate_repair_capacity(5, 6) == 8

    def test_odd_hull_bonus_floors(self):
        """Test that odd hull bonus floors to nearest even."""
        assert calculate_repair_capacity(5, 1) == 5
        assert calculate_repair_capacity(5, 3) == 6
        assert calculate_repair_capacity(5, 5) == 7


class TestCanSpendRepair:
    """Tests for repair spending validation."""

    def test_can_spend_exact(self):
        """Test spending exact amount of repairs."""
        assert can_spend_repair(4, 4) is True

    def test_can_spend_less(self):
        """Test spending less than available repairs."""
        assert can_spend_repair(6, 2) is True

    def test_cannot_spend_more(self):
        """Test that spending more than available fails."""
        assert can_spend_repair(3, 4) is False

    def test_cannot_spend_zero(self):
        """Test that spending zero repairs is valid."""
        assert can_spend_repair(0, 0) is True

    def test_cannot_spend_negative(self):
        """Test that spending negative repairs fails."""
        assert can_spend_repair(5, -1) is False


class TestCalculateRepairCost:
    """Tests for repair cost calculation."""

    def test_hp_repair_cost(self):
        """Test cost to repair HP."""
        assert calculate_repair_cost("hp") == 1

    def test_destroyed_weapon_cost(self):
        """Test cost to repair destroyed weapon."""
        assert calculate_repair_cost("destroyed_weapon") == 1

    def test_destroyed_system_cost(self):
        """Test cost to repair destroyed system."""
        assert calculate_repair_cost("destroyed_system") == 1

    def test_structure_cost(self):
        """Test cost to repair structure."""
        assert calculate_repair_cost("structure") == 2

    def test_stress_cost(self):
        """Test cost to repair reactor stress."""
        assert calculate_repair_cost("stress") == 2

    def test_destroyed_mech_cost(self):
        """Test cost to restore destroyed mech."""
        assert calculate_repair_cost("destroyed_mech") == 4


class TestResolveRest:
    """Tests for rest resolution (pure logic)."""

    def test_short_rest_basic(self):
        """Test basic short rest (1 hour) without repairs."""
        input_data = RestInput(
            actor_id="test_mech",
            duration_hours=1,
            repairs_to_spend=[],
            repair_cap=6,
            repairs_remaining=6,
        )
        result = resolve_rest(input_data)

        assert result.actor_id == "test_mech"
        assert result.duration_hours == 1
        assert result.is_full_repair is False
        assert result.heat_cleared is True
        assert result.repairs_spent == 0
        assert result.repair_cap_before == 6
        assert result.repair_cap_refreshed is False
        assert len(result.validation_errors) == 0

    def test_full_repair_detected(self):
        """Test that 10+ hours is detected as full repair."""
        input_data = RestInput(
            actor_id="test_mech",
            duration_hours=10,
            repairs_to_spend=[],
            repair_cap=6,
            repairs_remaining=6,
        )
        result = resolve_rest(input_data)

        assert result.is_full_repair is True
        assert result.repair_cap_refreshed is True
        assert result.core_power_regained is True
        assert len(result.limited_weapons_reset) > 0

    def test_spend_repair_for_hp(self):
        """Test spending repair to restore HP."""
        input_data = RestInput(
            actor_id="test_mech",
            duration_hours=1,
            repairs_to_spend=[
                RepairSpec(target_id="test_mech", repair_type="hp", repairs_spent=1)
            ],
            repair_cap=6,
            repairs_remaining=6,
        )
        result = resolve_rest(input_data)

        assert result.repairs_spent == 1
        assert len(result.repair_results) == 1
        assert result.repair_results[0].repair_type == "hp"
        assert result.repair_results[0].repairs_spent == 1
        assert "full HP" in result.repair_results[0].effect_applied

    def test_spend_repair_for_structure(self):
        """Test spending repairs to repair structure."""
        input_data = RestInput(
            actor_id="test_mech",
            duration_hours=1,
            repairs_to_spend=[
                RepairSpec(
                    target_id="test_mech", repair_type="structure", repairs_spent=2
                )
            ],
            repair_cap=6,
            repairs_remaining=6,
        )
        result = resolve_rest(input_data)

        assert result.repairs_spent == 2
        assert len(result.repair_results) == 1
        assert result.repair_results[0].repair_type == "structure"
        assert "1 structure" in result.repair_results[0].effect_applied

    def test_spend_repair_for_stress(self):
        """Test spending repairs to repair reactor stress."""
        input_data = RestInput(
            actor_id="test_mech",
            duration_hours=1,
            repairs_to_spend=[
                RepairSpec(target_id="test_mech", repair_type="stress", repairs_spent=2)
            ],
            repair_cap=6,
            repairs_remaining=6,
        )
        result = resolve_rest(input_data)

        assert result.repairs_spent == 2
        assert len(result.repair_results) == 1
        assert result.repair_results[0].repair_type == "stress"

    def test_spend_repair_for_destroyed_weapon(self):
        """Test spending repair to fix destroyed weapon."""
        input_data = RestInput(
            actor_id="test_mech",
            duration_hours=1,
            repairs_to_spend=[
                RepairSpec(
                    target_id="rifle", repair_type="destroyed_weapon", repairs_spent=1
                )
            ],
            repair_cap=6,
            repairs_remaining=6,
        )
        result = resolve_rest(input_data)

        assert result.repairs_spent == 1
        assert len(result.repair_results) == 1
        assert result.repair_results[0].repair_type == "destroyed_weapon"
        assert "destroyed weapon" in result.repair_results[0].effect_applied

    def test_spend_repair_for_destroyed_system(self):
        """Test spending repair to fix destroyed system."""
        input_data = RestInput(
            actor_id="test_mech",
            duration_hours=1,
            repairs_to_spend=[
                RepairSpec(
                    target_id="scanner", repair_type="destroyed_system", repairs_spent=1
                )
            ],
            repair_cap=6,
            repairs_remaining=6,
        )
        result = resolve_rest(input_data)

        assert result.repairs_spent == 1
        assert len(result.repair_results) == 1
        assert result.repair_results[0].repair_type == "destroyed_system"

    def test_spend_repair_for_destroyed_mech(self):
        """Test spending repairs to restore destroyed mech."""
        input_data = RestInput(
            actor_id="test_mech",
            duration_hours=1,
            repairs_to_spend=[
                RepairSpec(
                    target_id="destroyed_mech",
                    repair_type="destroyed_mech",
                    repairs_spent=4,
                )
            ],
            repair_cap=6,
            repairs_remaining=6,
        )
        result = resolve_rest(input_data)

        assert result.repairs_spent == 4
        assert len(result.repair_results) == 1
        assert result.repair_results[0].repair_type == "destroyed_mech"
        assert (
            "1 structure, 1 stress, full HP" in result.repair_results[0].effect_applied
        )

    def test_multiple_repairs_in_single_rest(self):
        """Test spending multiple repairs in one rest."""
        input_data = RestInput(
            actor_id="test_mech",
            duration_hours=1,
            repairs_to_spend=[
                RepairSpec(target_id="test_mech", repair_type="hp", repairs_spent=1),
                RepairSpec(
                    target_id="test_mech", repair_type="structure", repairs_spent=2
                ),
            ],
            repair_cap=6,
            repairs_remaining=6,
        )
        result = resolve_rest(input_data)

        assert result.repairs_spent == 3
        assert len(result.repair_results) == 2

    def test_insufficient_repairs_error(self):
        """Test error when trying to spend more repairs than available."""
        input_data = RestInput(
            actor_id="test_mech",
            duration_hours=1,
            repairs_to_spend=[
                RepairSpec(
                    target_id="test_mech", repair_type="destroyed_mech", repairs_spent=4
                )
            ],
            repair_cap=6,
            repairs_remaining=2,
        )
        result = resolve_rest(input_data)

        assert len(result.validation_errors) == 1
        assert "Insufficient repairs" in result.validation_errors[0]

    def test_wrong_repair_cost_error(self):
        """Test error when repair cost doesn't match expected."""
        input_data = RestInput(
            actor_id="test_mech",
            duration_hours=1,
            repairs_to_spend=[
                RepairSpec(
                    target_id="test_mech", repair_type="structure", repairs_spent=1
                )
            ],
            repair_cap=6,
            repairs_remaining=6,
        )
        result = resolve_rest(input_data)

        assert len(result.validation_errors) == 1
        assert "Wrong repair cost" in result.validation_errors[0]

    def test_full_repair_clears_conditions(self):
        """Test that full repair clears all clearable conditions."""
        input_data = RestInput(
            actor_id="test_mech",
            duration_hours=10,
            repairs_to_spend=[],
            repair_cap=6,
            repairs_remaining=6,
        )
        result = resolve_rest(input_data)

        assert result.is_full_repair is True
        assert len(result.conditions_ended) > 0
        assert "impaired" in result.conditions_ended
        assert "jammed" in result.conditions_ended

    def test_short_rest_does_not_clear_conditions(self):
        """Test that short rest does not clear conditions."""
        input_data = RestInput(
            actor_id="test_mech",
            duration_hours=1,
            repairs_to_spend=[],
            repair_cap=6,
            repairs_remaining=6,
        )
        result = resolve_rest(input_data)

        assert result.is_full_repair is False
        assert len(result.conditions_ended) == 0


class TestApplyRestResult:
    """Tests for applying rest results to combatant state."""

    def test_apply_short_rest_heat_clear(self, test_combatant: CombatantState):
        """Test that short rest clears heat."""
        combatant = test_combatant
        input_data = RestInput(
            actor_id="test_mech",
            duration_hours=1,
            repairs_to_spend=[],
            repair_cap=6,
            repairs_remaining=6,
        )
        result = resolve_rest(input_data)
        updated = apply_rest_result(combatant, result)

        assert updated.resources.heat_current == 0

    def test_apply_rest_reduces_repairs(self, test_combatant: CombatantState):
        """Test that repairs spent are deducted."""
        combatant = test_combatant
        input_data = RestInput(
            actor_id="test_mech",
            duration_hours=1,
            repairs_to_spend=[
                RepairSpec(target_id="test_mech", repair_type="hp", repairs_spent=1),
                RepairSpec(
                    target_id="test_mech", repair_type="structure", repairs_spent=2
                ),
            ],
            repair_cap=6,
            repairs_remaining=6,
        )
        result = resolve_rest(input_data)
        updated = apply_rest_result(combatant, result)

        assert updated.resources.repairs_remaining == 3

    def test_apply_full_repair_conditions_cleared(self, test_combatant: CombatantState):
        """Test that full repair clears conditions."""
        combatant = test_combatant
        input_data = RestInput(
            actor_id="test_mech",
            duration_hours=10,
            repairs_to_spend=[],
            repair_cap=6,
            repairs_remaining=6,
        )
        result = resolve_rest(input_data)
        updated = apply_rest_result(combatant, result)

        assert "impaired" not in updated.conditions

    def test_repair_cap_not_below_zero(self, low_repairs_combatant: CombatantState):
        """Test that repairs don't go below zero."""
        combatant = low_repairs_combatant
        input_data = RestInput(
            actor_id="low_repairs_mech",
            duration_hours=1,
            repairs_to_spend=[
                RepairSpec(
                    target_id="low_repairs_mech", repair_type="hp", repairs_spent=1
                )
            ],
            repair_cap=6,
            repairs_remaining=1,
        )
        result = resolve_rest(input_data)
        updated = apply_rest_result(combatant, result)

        assert updated.resources.repairs_remaining == 0


class TestRestRuleCustomization:
    """Tests for customizable rest rules."""

    def test_custom_repair_costs(self):
        """Test custom repair cost rules."""
        custom_rules = RestRule(
            repair_cost_hp=2,
            repair_cost_structure=3,
            repair_cost_destroyed_mech=6,
        )
        assert custom_rules.repair_cost_hp == 2
        assert custom_rules.repair_cost_structure == 3
        assert custom_rules.repair_cost_destroyed_mech == 6

    def test_no_heat_clear_on_rest(self):
        """Test rule to disable heat clearing on rest."""
        custom_rules = RestRule(heat_cleared_on_rest=False)
        input_data = RestInput(
            actor_id="test_mech",
            duration_hours=1,
            repairs_to_spend=[],
            repair_cap=6,
            repairs_remaining=6,
            rules=custom_rules,
        )
        result = resolve_rest(input_data)

        assert result.heat_cleared is False

    def test_custom_full_repair_threshold(self):
        """Test custom full repair hour threshold."""
        custom_rules = RestRule(min_hours_for_full_repair=8)
        input_data = RestInput(
            actor_id="test_mech",
            duration_hours=8,
            repairs_to_spend=[],
            repair_cap=6,
            repairs_remaining=6,
            rules=custom_rules,
        )
        result = resolve_rest(input_data)

        assert result.is_full_repair is True

    def test_no_condition_clearing(self):
        """Test rule to disable condition clearing on rest."""
        custom_rules = RestRule(can_end_conditions=False)
        input_data = RestInput(
            actor_id="test_mech",
            duration_hours=10,
            repairs_to_spend=[],
            repair_cap=6,
            repairs_remaining=6,
            rules=custom_rules,
        )
        result = resolve_rest(input_data)

        assert len(result.conditions_ended) == 0
