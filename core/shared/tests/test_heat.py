"""Tests for overheat resolution and meltdown handling."""

import pytest
from core.shared.heat import (
    resolve_overheat,
    apply_overheat_result,
    decrement_meltdown_countdown,
    trigger_meltdown,
    OverheatInput,
    MeltdownState,
)
from core.mech.combat_state import (
    CombatantState,
    CombatStats,
    CombatResources,
)
from core.shared.enums import StatusType


@pytest.fixture
def test_combatant() -> CombatantState:
    """Create a test combatant for overheat tests."""
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
            structure_current=3,
            stress_current=2,
            heat_current=5,
            heat_cap=10,
        ),
    )


class TestResolveOverheat:
    """Tests for overheat resolution."""

    def test_emergency_shunt(self):
        """Test that rolling 5-6 results in emergency shunt (impaired)."""
        result = resolve_overheat(
            OverheatInput(stress_marked=3, remaining_stress=2),
            force_roll=5,
        )
        assert result.outcome == "emergency_shunt"
        assert "impaired" in result.statuses_to_apply

    def test_power_plant_destabilize(self):
        """Test that rolling 2-4 results in exposed status."""
        result = resolve_overheat(
            OverheatInput(stress_marked=3, remaining_stress=2),
            force_roll=3,
        )
        assert result.outcome == "power_plant_destabilize"
        assert "exposed" in result.statuses_to_apply

    def test_meltdown_exposed(self):
        """Test that rolling 1 at 3+ stress results in exposed."""
        result = resolve_overheat(
            OverheatInput(stress_marked=3, remaining_stress=3),
            force_roll=1,
        )
        assert result.outcome == "meltdown"
        assert "exposed" in result.statuses_to_apply

    def test_meltdown_engineering_check(self):
        """Test that rolling 1 at 2 stress requires Engineering check."""
        result = resolve_overheat(
            OverheatInput(stress_marked=3, remaining_stress=2),
            force_roll=1,
        )
        assert result.outcome == "meltdown"
        assert "exposed" in result.statuses_to_apply
        assert result.engineering_check_request is not None
        assert result.engineering_check_request.save_type == "engineering"

    def test_meltdown_immediate(self):
        """Test that rolling 1 at 1 stress triggers immediate meltdown."""
        result = resolve_overheat(
            OverheatInput(stress_marked=3, remaining_stress=1),
            force_roll=1,
        )
        assert result.outcome == "meltdown"
        assert result.meltdown_state is not None
        assert result.meltdown_state.is_immediate is True

    def test_irreversible_meltdown(self):
        """Test that rolling 2+ 1s results in irreversible meltdown."""
        result = resolve_overheat(
            OverheatInput(stress_marked=3, remaining_stress=2),
            force_roll=1,
        )
        assert result.outcome == "meltdown"
        result2 = resolve_overheat(
            OverheatInput(stress_marked=3, remaining_stress=2),
            force_roll=1,
        )
        assert result2.outcome == "meltdown"


class TestApplyOverheatResult:
    """Tests for applying overheat results."""

    def test_apply_emergency_shunt(self, test_combatant):
        """Test applying emergency shunt status."""
        result = resolve_overheat(
            OverheatInput(stress_marked=3, remaining_stress=2),
            force_roll=5,
        )
        applied = apply_overheat_result(test_combatant, result)
        assert "impaired" in applied.statuses_applied
        assert applied.heat_current == 0  # Heat is cleared per PR2

    def test_apply_meltdown_state(self, test_combatant):
        """Test that meltdown state is preserved in application."""
        result = resolve_overheat(
            OverheatInput(stress_marked=3, remaining_stress=1),
            force_roll=1,
        )
        applied = apply_overheat_result(test_combatant, result)
        assert applied.meltdown_state is not None
        assert applied.meltdown_state.turns_remaining == 1


class TestMeltdownCountdown:
    """Tests for meltdown countdown processing."""

    def test_decrement_countdown(self, test_combatant):
        """Test decrementing meltdown countdown."""
        combatant_with_meltdown = test_combatant.model_copy(
            update={"meltdown_state": MeltdownState(turns_remaining=2)}
        )
        updated, triggered = decrement_meltdown_countdown(combatant_with_meltdown)
        assert updated.meltdown_state is not None
        assert updated.meltdown_state.turns_remaining == 1
        assert triggered is False

    def test_meltdown_triggers(self, test_combatant):
        """Test that meltdown triggers when countdown reaches 0."""
        combatant_with_meltdown = test_combatant.model_copy(
            update={"meltdown_state": MeltdownState(turns_remaining=1)}
        )
        updated, triggered = decrement_meltdown_countdown(combatant_with_meltdown)
        assert triggered is True
        assert updated.meltdown_state is None

    def test_no_meltdown(self, test_combatant):
        """Test that combatant without meltdown is unchanged."""
        updated, triggered = decrement_meltdown_countdown(test_combatant)
        assert updated.meltdown_state is None
        assert triggered is False


class TestTriggerMeltdown:
    """Tests for triggering immediate meltdown."""

    def test_trigger_meltdown_creates_wreckage(self, test_combatant):
        """Test that triggering meltdown creates wreckage object."""
        updated, wreckage = trigger_meltdown(test_combatant)
        assert wreckage is not None
        assert "out" in updated.statuses
        assert updated.resources.hp_current == 0
        assert updated.resources.structure_current == 0

    def test_meltdown_clears_meltdown_state(self, test_combatant):
        """Test that triggering meltdown clears the meltdown state."""
        combatant_with_meltdown = test_combatant.model_copy(
            update={"meltdown_state": MeltdownState(turns_remaining=1)}
        )
        updated, _ = trigger_meltdown(combatant_with_meltdown)
        assert updated.meltdown_state is None
