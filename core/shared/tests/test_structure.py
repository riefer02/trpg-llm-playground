"""Tests for structure damage resolution."""

import pytest
from core.shared.structure import (
    resolve_structure_damage,
    apply_structure_result,
    StructureInput,
    SystemTraumaSelection,
)
from core.shared.heat import MeltdownState
from core.mech.combat_state import (
    CombatantState,
    CombatStats,
    CombatResources,
    MechInventory,
    WeaponMountState,
    WeaponState,
    MechSystemState,
)
from core.shared.enums import SizeClass, StatusType


@pytest.fixture
def test_combatant() -> CombatantState:
    """Create a test combatant for structure tests."""
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
            MechSystemState(system_id=" Harding  ", destroyed=False),
        ],
    )


class TestResolveStructureDamage:
    """Tests for structure damage resolution."""

    def test_glancing_blow(self):
        """Test that rolling 5-6 results in glancing blow."""
        result = resolve_structure_damage(
            StructureInput(damage_dealt=6, remaining_structure=3),
            force_roll=5,
        )
        assert result.outcome == "glancing_blow"
        assert "impaired" in result.statuses_to_apply
        assert not result.mech_destroyed

    def test_system_trauma(self, test_inventory):
        """Test that rolling 2-4 results in system trauma."""
        result = resolve_structure_damage(
            StructureInput(
                damage_dealt=6, remaining_structure=3, inventory=test_inventory
            ),
            force_roll=3,
        )
        assert result.outcome == "system_trauma"
        assert result.system_trauma is not None
        assert result.system_trauma.initial_target in ["mount", "system"]

    def test_direct_hit_stunned(self):
        """Test that rolling 1 at 3+ structure results in stunned."""
        result = resolve_structure_damage(
            StructureInput(damage_dealt=6, remaining_structure=3),
            force_roll=1,
        )
        assert result.outcome == "direct_hit"
        assert "stunned" in result.statuses_to_apply

    def test_direct_hit_hull_check(self):
        """Test that rolling 1 at 2 structure requires Hull check."""
        result = resolve_structure_damage(
            StructureInput(damage_dealt=6, remaining_structure=2),
            force_roll=1,
        )
        assert result.outcome == "direct_hit"
        assert "stunned" in result.statuses_to_apply
        assert result.hull_check_request is not None
        assert result.hull_check_request.save_type == "hull"

    def test_direct_hit_destroyed(self):
        """Test that rolling 1 at 1 structure destroys the mech."""
        result = resolve_structure_damage(
            StructureInput(damage_dealt=6, remaining_structure=1),
            force_roll=1,
        )
        assert result.outcome == "direct_hit"
        assert result.mech_destroyed

    def test_crushing_hit(self):
        """Test that rolling 2+ 1s results in crushing hit (destroyed)."""
        result = resolve_structure_damage(
            StructureInput(damage_dealt=6, remaining_structure=3),
            force_roll=1,
        )
        assert result.outcome == "direct_hit"
        result2 = resolve_structure_damage(
            StructureInput(damage_dealt=6, remaining_structure=3),
            force_roll=1,
        )
        assert result2.outcome == "direct_hit"


class TestApplyStructureResult:
    """Tests for applying structure damage results."""

    def test_apply_glancing_blow(self, test_combatant):
        """Test applying glancing blow status."""
        result = resolve_structure_damage(
            StructureInput(damage_dealt=6, remaining_structure=3),
            force_roll=5,
        )
        applied = apply_structure_result(
            test_combatant,
            result,
            object_id="test_object",
        )
        assert "impaired" in applied.statuses_applied
        assert applied.mech_destroyed is False

    def test_apply_mech_destruction(self, test_combatant):
        """Test that destroyed mech creates battlefield object."""
        result = resolve_structure_damage(
            StructureInput(damage_dealt=6, remaining_structure=1),
            force_roll=1,
        )
        applied = apply_structure_result(
            test_combatant,
            result,
            object_id="wreckage_123",
        )
        assert applied.mech_destroyed is True
        assert applied.created_object is not None
        assert applied.created_object.id == "wreckage_123"
