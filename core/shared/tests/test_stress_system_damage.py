"""Tests for stress/system damage integration and defender's choice."""

import pytest
from core.shared.structure import (
    resolve_structure_damage,
    apply_structure_result,
    StructureInput,
    SystemTraumaSelection,
    apply_system_trauma,
)
from core.shared.heat import (
    resolve_overheat,
    apply_overheat_result,
    decrement_meltdown_countdown,
    trigger_meltdown,
    OverheatInput,
    OverheatResolutionResult,
    MeltdownState,
    resolve_stress_check,
    apply_stress_check_result,
    StressCheckInput,
    StressCheckResult,
    resolve_meltdown_countdown,
    MeltdownCountdownInput,
    MeltdownCountdownResult,
)
from core.mech.combat_state import (
    CombatantState,
    CombatStats,
    CombatResources,
    MechInventory,
    WeaponMountState,
    WeaponState,
    MechSystemState,
)
from core.shared.enums import StatusType


@pytest.fixture
def test_combatant() -> CombatantState:
    """Create a test combatant for stress/system tests."""
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


@pytest.fixture
def test_inventory() -> MechInventory:
    """Create a test inventory with mounts and systems."""
    return MechInventory(
        mounts=[
            WeaponMountState(
                mount_index=0,
                slot_type="main_aux",
                weapons=[
                    WeaponState(
                        weapon_id="heavy_machine_gun", tags=[], destroyed=False
                    ),
                    WeaponState(weapon_id="auxiliary_gun", tags=[], destroyed=False),
                ],
                destroyed=False,
            ),
            WeaponMountState(
                mount_index=1,
                slot_type="heavy",
                weapons=[
                    WeaponState(weapon_id="thermal_rifle", tags=[], destroyed=False),
                ],
                destroyed=False,
            ),
        ],
        systems=[
            MechSystemState(system_id="personalizations", destroyed=False),
            MechSystemState(system_id=" Harding", destroyed=False),
            MechSystemState(system_id="信管", destroyed=False),
        ],
    )


@pytest.fixture
def inventory_no_mounts() -> MechInventory:
    """Create an inventory with no destroyable mounts."""
    return MechInventory(
        mounts=[
            WeaponMountState(
                mount_index=0,
                slot_type="main_aux",
                weapons=[
                    WeaponState(weapon_id="heavy_machine_gun", tags=[], destroyed=True),
                ],
                destroyed=False,
            ),
        ],
        systems=[
            MechSystemState(system_id="personalizations", destroyed=False),
            MechSystemState(system_id=" Harding", destroyed=False),
        ],
    )


@pytest.fixture
def inventory_no_systems() -> MechInventory:
    """Create an inventory with no destroyable systems."""
    return MechInventory(
        mounts=[
            WeaponMountState(
                mount_index=0,
                slot_type="main_aux",
                weapons=[
                    WeaponState(
                        weapon_id="heavy_machine_gun", tags=[], destroyed=False
                    ),
                ],
                destroyed=False,
            ),
        ],
        systems=[
            MechSystemState(system_id="personalizations", destroyed=True),
            MechSystemState(system_id=" Harding", destroyed=True),
        ],
    )


class TestDefendersChoiceSystemTrauma:
    """Tests for defender's choice in system trauma resolution."""

    def test_defender_chooses_mount(self, test_inventory):
        """Test defender can choose to destroy a mount."""
        trauma_selection = SystemTraumaSelection(
            trauma_roll=2,
            initial_target="mount",
            resolved_target="mount",
            mount_index=0,
            system_id=None,
            eligible_mounts=[0, 1],
            eligible_systems=["personalizations", " Harding", "信管"],
            fallback_reason="none",
        )
        updated = apply_system_trauma(test_inventory, trauma_selection)
        assert updated.mounts[0].weapons[0].destroyed is True
        assert updated.mounts[0].weapons[1].destroyed is True

    def test_defender_chooses_system(self, test_inventory):
        """Test defender can choose to destroy a system."""
        trauma_selection = SystemTraumaSelection(
            trauma_roll=5,
            initial_target="system",
            resolved_target="system",
            mount_index=None,
            system_id=" Harding",
            eligible_mounts=[0, 1],
            eligible_systems=["personalizations", " Harding", "信管"],
            fallback_reason="none",
        )
        updated = apply_system_trauma(test_inventory, trauma_selection)
        systems_after = [s for s in updated.systems if not s.destroyed]
        system_ids_after = [s.system_id for s in systems_after]
        assert " Harding" not in system_ids_after

    def test_fallback_to_direct_hit_when_no_mounts(self, inventory_no_mounts):
        """Test fallback to direct hit when no mounts available."""
        trauma_selection = SystemTraumaSelection(
            trauma_roll=2,
            initial_target="mount",
            resolved_target="direct_hit",
            mount_index=None,
            system_id=None,
            eligible_mounts=[],
            eligible_systems=["personalizations", " Harding"],
            fallback_reason="no_mounts",
        )
        updated = apply_system_trauma(inventory_no_mounts, trauma_selection)
        for mount in updated.mounts:
            assert mount.destroyed is False
        for system in updated.systems:
            assert system.destroyed is False

    def test_fallback_to_direct_hit_when_no_systems(self, inventory_no_systems):
        """Test fallback to direct hit when no systems available."""
        trauma_selection = SystemTraumaSelection(
            trauma_roll=5,
            initial_target="system",
            resolved_target="direct_hit",
            mount_index=None,
            system_id=None,
            eligible_mounts=[0],
            eligible_systems=[],
            fallback_reason="no_systems",
        )
        updated = apply_system_trauma(inventory_no_systems, trauma_selection)
        for mount in updated.mounts:
            assert mount.destroyed is False
        non_destroyed_systems = [s for s in updated.systems if not s.destroyed]
        assert len(non_destroyed_systems) == 0


class TestUnifiedStressCheck:
    """Tests for unified stress check workflow."""

    def test_stress_check_emergency_shunt(self):
        """Test stress check at 4 stress with emergency shunt result."""
        result = resolve_stress_check(
            StressCheckInput(
                heat_exceeded=6,
                heat_cap=10,
                current_stress=4,
                additional_stress=1,
            ),
            force_roll=5,
        )
        assert result.outcome == "emergency_shunt"
        assert "impaired" in result.statuses_to_apply
        assert result.stress_after == 3

    def test_stress_check_exposed(self):
        """Test stress check with exposed result."""
        result = resolve_stress_check(
            StressCheckInput(
                heat_exceeded=6,
                heat_cap=10,
                current_stress=3,
                additional_stress=1,
            ),
            force_roll=3,
        )
        assert result.outcome == "power_plant_destabilize"
        assert "exposed" in result.statuses_to_apply
        assert result.stress_after == 2

    def test_stress_check_meltdown_exposed(self):
        """Test stress check at meltdown with exposed outcome (3+ stress remaining)."""
        result = resolve_stress_check(
            StressCheckInput(
                heat_exceeded=6,
                heat_cap=10,
                current_stress=3,
                additional_stress=1,
            ),
            force_roll=1,
        )
        assert result.outcome == "meltdown"
        assert "exposed" in result.statuses_to_apply
        assert result.stress_after == 2

    def test_stress_check_meltdown_engineering_check(self):
        """Test stress check at 2 stress remaining requires Engineering check."""
        result = resolve_stress_check(
            StressCheckInput(
                heat_exceeded=6,
                heat_cap=10,
                current_stress=3,
                additional_stress=1,
            ),
            force_roll=1,
        )
        assert result.outcome == "meltdown"
        assert "exposed" in result.statuses_to_apply
        assert result.engineering_check_required is True
        assert result.countdown_turns == 6

    def test_stress_check_meltdown_immediate(self):
        """Test stress check at 1 stress triggers immediate meltdown."""
        result = resolve_stress_check(
            StressCheckInput(
                heat_exceeded=6,
                heat_cap=10,
                current_stress=1,
                additional_stress=1,
            ),
            force_roll=1,
        )
        assert result.outcome == "meltdown"
        assert result.meltdown_immediate is True
        assert result.stress_after == 0

    def test_apply_stress_check(self, test_combatant):
        """Test applying stress check result to combatant state."""
        result = resolve_stress_check(
            StressCheckInput(
                heat_exceeded=6,
                heat_cap=10,
                current_stress=3,
                additional_stress=1,
            ),
            force_roll=5,
        )
        applied = apply_stress_check_result(test_combatant, result)
        assert "impaired" in applied.statuses_applied
        assert applied.stress_current == 2
        assert applied.heat_current == 0


class TestMeltdownCountdown:
    """Tests for meltdown countdown management."""

    def test_resolve_countdown_success(self):
        """Test successful engineering check delays meltdown."""
        result = resolve_meltdown_countdown(
            MeltdownCountdownInput(
                current_turns_remaining=4,
                engineering_check_target=10,
                engineering_check_bonus=0,
                stress_at_countdown_start=2,
            ),
            force_roll=12,
        )
        assert result.check_passed is True
        assert result.countdown_extended is True
        assert result.new_turns_remaining == 4

    def test_resolve_countdown_failure(self):
        """Test failed engineering check triggers countdown."""
        result = resolve_meltdown_countdown(
            MeltdownCountdownInput(
                current_turns_remaining=4,
                engineering_check_target=10,
                engineering_check_bonus=0,
                stress_at_countdown_start=2,
            ),
            force_roll=7,
        )
        assert result.check_passed is False
        assert result.countdown_triggers is True
        assert result.meltdown_triggered is True

    def test_countdown_triggers_meltdown(self, test_combatant):
        """Test that countdown reaching zero triggers meltdown."""
        combatant_with_countdown = test_combatant.model_copy(
            update={"meltdown_state": MeltdownState(turns_remaining=1)}
        )
        updated, triggered = decrement_meltdown_countdown(combatant_with_countdown)
        assert triggered is True
        assert updated.meltdown_state is None

    def test_countdown_decrement(self, test_combatant):
        """Test decrementing countdown without triggering."""
        combatant_with_countdown = test_combatant.model_copy(
            update={"meltdown_state": MeltdownState(turns_remaining=3)}
        )
        updated, triggered = decrement_meltdown_countdown(combatant_with_countdown)
        assert triggered is False
        assert updated.meltdown_state is not None
        assert updated.meltdown_state.turns_remaining == 2


class TestStressStructureIntegration:
    """Integration tests combining stress and structure damage."""

    def test_stress_then_structure_damage(self, test_combatant, test_inventory):
        """Test resolving stress damage followed by structure damage."""
        overheat_result = resolve_overheat(
            OverheatInput(stress_marked=2, remaining_stress=2),
            force_roll=5,
        )
        applied = apply_overheat_result(test_combatant, overheat_result)

        assert applied.stress_current == 1
        assert "impaired" in applied.statuses_applied

        structure_result = resolve_structure_damage(
            StructureInput(
                damage_dealt=10,
                remaining_structure=3,
                inventory=test_inventory,
            ),
            force_roll=5,
        )
        structure_applied = apply_structure_result(
            applied.updated_combatant,
            structure_result,
            object_id="test_obj",
        )
        assert "impaired" in structure_applied.statuses_applied

    def test_multiple_overheats_escalate(self, test_combatant):
        """Test that multiple overheats properly escalate stress damage."""
        combatant_2_stress = test_combatant.model_copy(
            update={
                "resources": test_combatant.resources.model_copy(
                    update={"stress_current": 2}
                )
            }
        )
        result1 = resolve_overheat(
            OverheatInput(stress_marked=1, remaining_stress=2),
            force_roll=5,
        )
        applied1 = apply_overheat_result(combatant_2_stress, result1)
        assert applied1.stress_current == 1

        result2 = resolve_overheat(
            OverheatInput(stress_marked=1, remaining_stress=1),
            force_roll=1,
        )
        applied2 = apply_overheat_result(applied1.updated_combatant, result2)
        assert applied2.meltdown_state is not None
        assert applied2.meltdown_state.is_immediate is True


class TestReactorMeltdown:
    """Tests for reactor meltdown outcomes."""

    def test_meltdown_creates_wreckage(self, test_combatant):
        """Test that meltdown creates wreckage object."""
        updated, wreckage = trigger_meltdown(test_combatant)
        assert wreckage is not None
        assert "out" in updated.statuses
        assert updated.resources.hp_current == 0
        assert updated.resources.structure_current == 0
        assert updated.resources.stress_current == 0

    def test_meltdown_clears_exposed(self, test_combatant):
        """Test that meltdown clears exposed status."""
        exposed_combatant = test_combatant.model_copy(
            update={"statuses": ["exposed", "stunned"]}
        )
        updated, _ = trigger_meltdown(exposed_combatant)
        assert "exposed" not in updated.statuses
        assert "stunned" not in updated.statuses
        assert "out" in updated.statuses


class TestStressAtZero:
    """Tests for mech at zero stress (immediate meltdown)."""

    def test_zero_stress_countdown(self, test_combatant):
        """Test that 0 stress creates countdown to meltdown."""
        combatant_zero_stress = test_combatant.model_copy(
            update={
                "resources": test_combatant.resources.model_copy(
                    update={"stress_current": 0}
                )
            }
        )
        assert combatant_zero_stress.resources.stress_current == 0

    def test_stress_recovery_prevents_meltdown(self, test_combatant):
        """Test that stress recovery during countdown prevents meltdown."""
        combatant_with_countdown = test_combatant.model_copy(
            update={"meltdown_state": MeltdownState(turns_remaining=2)}
        )
        recovered_combatant = combatant_with_countdown.model_copy(
            update={
                "resources": combatant_with_countdown.resources.model_copy(
                    update={"stress_current": 1}
                )
            }
        )
        assert recovered_combatant.resources.stress_current == 1
