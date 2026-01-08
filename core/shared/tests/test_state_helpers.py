"""Tests for state mutation helpers."""

import pytest
from core.shared.state_helpers import (
    set_hp,
    decrement_hp,
    increment_hp,
    set_heat,
    increment_heat,
    clear_heat,
    set_structure,
    decrement_structure,
    set_stress,
    decrement_stress,
    add_status,
    add_statuses,
    remove_status,
    clear_statuses,
    destroy_weapon,
    destroy_mount,
    destroy_system,
    consume_limited_charge,
    clear_per_round_reactions,
    increment_reaction_use,
    set_meltdown_state,
    decrement_meltdown_countdown,
    create_overcharge_state,
    use_overcharge,
    reset_overcharge_uses,
    set_effect_duration,
    advance_effect_to_next_turn,
    apply_damage,
    apply_heat_damage,
    apply_structure_damage,
    apply_overheat_result,
    StateUpdateResult,
)
from core.mech.combat_state import (
    CombatantState,
    CombatStats,
    CombatResources,
    MechInventory,
    WeaponMountState,
    WeaponState,
    MechSystemState,
    OverchargeState,
)
from core.shared.heat import MeltdownState
from core.shared.turn_end import TurnEndEffectState
from core.shared.effects import EffectDuration
from core.shared.enums import StatusType


@pytest.fixture
def test_combatant() -> CombatantState:
    """Create a test combatant for state helper tests."""
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
            armor=2,
            speed=4,
        ),
        resources=CombatResources(
            hp_current=8,
            structure_current=3,
            stress_current=1,
            heat_current=3,
            heat_cap=10,
        ),
    )


@pytest.fixture
def test_inventory() -> MechInventory:
    """Create a test inventory for destruction tests."""
    return MechInventory(
        mounts=[
            WeaponMountState(
                mount_index=0,
                slot_type="flexible",
                weapons=[
                    WeaponState(
                        weapon_id="rifle",
                        tags=[],
                        destroyed=False,
                        limited_charges_remaining=None,
                    ),
                    WeaponState(
                        weapon_id="pistol",
                        tags=[],
                        destroyed=False,
                        limited_charges_remaining=None,
                    ),
                ],
                destroyed=False,
            ),
            WeaponMountState(
                mount_index=1,
                slot_type="heavy",
                weapons=[
                    WeaponState(
                        weapon_id="cannon",
                        tags=[],
                        destroyed=False,
                        limited_charges_remaining=3,
                    ),
                ],
                destroyed=False,
            ),
        ],
        systems=[
            MechSystemState(
                system_id="scanner", destroyed=False, limited_charges_remaining=None
            ),
            MechSystemState(
                system_id="shield", destroyed=False, limited_charges_remaining=2
            ),
        ],
    )


class TestHpHelpers:
    """Tests for HP-related state helpers."""

    def test_set_hp(self, test_combatant: CombatantState):
        """Test setting HP to exact value."""
        result = set_hp(test_combatant, 5)
        assert result.resources.hp_current == 5
        assert test_combatant.resources.hp_current == 8  # Original unchanged

    def test_set_hp_clamped_to_zero(self, test_combatant: CombatantState):
        """Test that HP is clamped to 0."""
        result = set_hp(test_combatant, -5)
        assert result.resources.hp_current == 0

    def test_decrement_hp(self, test_combatant: CombatantState):
        """Test decreasing HP."""
        result = decrement_hp(test_combatant, 3)
        assert result.resources.hp_current == 5

    def test_decrement_hp_minimum_zero(self, test_combatant: CombatantState):
        """Test that HP doesn't go below 0."""
        result = decrement_hp(test_combatant, 20)
        assert result.resources.hp_current == 0

    def test_increment_hp(self, test_combatant: CombatantState):
        """Test increasing HP."""
        result = increment_hp(test_combatant, 2)
        assert result.resources.hp_current == 10

    def test_increment_hp_with_max(self, test_combatant: CombatantState):
        """Test increasing HP with max clamping."""
        result = increment_hp(test_combatant, 5, max_hp=10)
        assert result.resources.hp_current == 10


class TestHeatHelpers:
    """Tests for heat-related state helpers."""

    def test_set_heat(self, test_combatant: CombatantState):
        """Test setting heat to exact value."""
        result = set_heat(test_combatant, 7)
        assert result.resources.heat_current == 7

    def test_increment_heat(self, test_combatant: CombatantState):
        """Test increasing heat."""
        result = increment_heat(test_combatant, 2)
        assert result.resources.heat_current == 5

    def test_clear_heat(self, test_combatant: CombatantState):
        """Test clearing all heat."""
        result = clear_heat(test_combatant)
        assert result.resources.heat_current == 0


class TestStructureHelpers:
    """Tests for structure-related state helpers."""

    def test_set_structure(self, test_combatant: CombatantState):
        """Test setting structure to exact value."""
        result = set_structure(test_combatant, 2)
        assert result.resources.structure_current == 2

    def test_decrement_structure(self, test_combatant: CombatantState):
        """Test decreasing structure."""
        result = decrement_structure(test_combatant)
        assert result.resources.structure_current == 2


class TestStressHelpers:
    """Tests for stress-related state helpers."""

    def test_set_stress(self, test_combatant: CombatantState):
        """Test setting stress to exact value."""
        result = set_stress(test_combatant, 2)
        assert result.resources.stress_current == 2

    def test_decrement_stress(self, test_combatant: CombatantState):
        """Test decreasing stress."""
        result = decrement_stress(test_combatant)
        assert result.resources.stress_current == 0


class TestStatusHelpers:
    """Tests for status/condition helpers."""

    def test_add_status(self, test_combatant: CombatantState):
        """Test adding a status."""
        result = add_status(test_combatant, "impaired")
        assert "impaired" in result.statuses
        assert len(result.statuses) == 1

    def test_add_status_no_duplicate(self, test_combatant: CombatantState):
        """Test that adding existing status doesn't duplicate."""
        result = add_status(test_combatant, "impaired")
        result2 = add_status(result, "impaired")
        assert result2.statuses.count("impaired") == 1

    def test_add_statuses(self, test_combatant: CombatantState):
        """Test adding multiple statuses."""
        result = add_statuses(test_combatant, ["impaired", "shredded"])
        assert "impaired" in result.statuses
        assert "shredded" in result.statuses

    def test_remove_status(self, test_combatant: CombatantState):
        """Test removing a status."""
        result = add_status(test_combatant, "impaired")
        result2 = remove_status(result, "impaired")
        assert "impaired" not in result2.statuses

    def test_clear_statuses_specific(self, test_combatant: CombatantState):
        """Test clearing specific statuses."""
        result = add_statuses(test_combatant, ["impaired", "shredded", "exposed"])
        result2 = clear_statuses(result, ["impaired"])
        assert "impaired" not in result2.statuses
        assert "shredded" in result2.statuses
        assert "exposed" in result2.statuses

    def test_clear_statuses_all(self, test_combatant: CombatantState):
        """Test clearing all statuses."""
        result = add_statuses(test_combatant, ["impaired", "shredded"])
        result2 = clear_statuses(result)
        assert len(result2.statuses) == 0


class TestInventoryHelpers:
    """Tests for inventory destruction helpers."""

    def test_destroy_weapon(self, test_inventory: MechInventory):
        """Test destroying a specific weapon."""
        result = destroy_weapon(test_inventory.mounts[0], 0)
        assert result.weapons[0].destroyed is True
        assert result.weapons[1].destroyed is False

    def test_destroy_mount(self, test_inventory: MechInventory):
        """Test destroying all weapons on a mount."""
        result = destroy_mount(test_inventory, 0)
        assert result.mounts[0].weapons[0].destroyed is True
        assert result.mounts[0].weapons[1].destroyed is True
        assert result.mounts[1].weapons[0].destroyed is False

    def test_destroy_system(self, test_inventory: MechInventory):
        """Test destroying a specific system."""
        result = destroy_system(test_inventory, "scanner")
        assert result.systems[0].destroyed is True
        assert result.systems[1].destroyed is False

    def test_consume_limited_charge_weapon(self, test_inventory: MechInventory):
        """Test consuming limited charge from a weapon."""
        result, success = consume_limited_charge(test_inventory, "cannon")
        assert success is True
        # Find the cannon and check charges decreased
        for mount in result.mounts:
            for weapon in mount.weapons:
                if weapon.weapon_id == "cannon":
                    assert weapon.limited_charges_remaining == 2

    def test_consume_limited_charge_system(self, test_inventory: MechInventory):
        """Test consuming limited charge from a system."""
        result, success = consume_limited_charge(test_inventory, "shield")
        assert success is True
        for system in result.systems:
            if system.system_id == "shield":
                assert system.limited_charges_remaining == 1

    def test_consume_limited_charge_not_found(self, test_inventory: MechInventory):
        """Test consuming charge from non-existent item."""
        result, success = consume_limited_charge(test_inventory, "nonexistent")
        assert success is False


class TestReactionHelpers:
    """Tests for per-round reaction helpers."""

    def test_clear_per_round_reactions(self, test_combatant: CombatantState):
        """Test clearing per-round reactions."""
        result = clear_per_round_reactions(test_combatant)
        assert len(result.per_round_reactions) == 0

    def test_increment_reaction_use(self, test_combatant: CombatantState):
        """Test incrementing reaction use."""
        result = increment_reaction_use(test_combatant, "brace")
        assert result.per_round_reactions["brace"] == 1

    def test_increment_reaction_use_multiple(self, test_combatant: CombatantState):
        """Test incrementing reaction use multiple times."""
        result = increment_reaction_use(test_combatant, "brace")
        result = increment_reaction_use(result, "brace")
        assert result.per_round_reactions["brace"] == 2


class TestMeltdownHelpers:
    """Tests for meltdown state helpers."""

    def test_set_meltdown_state(self, test_combatant: CombatantState):
        """Test setting meltdown state."""
        meltdown = MeltdownState(turns_remaining=3, triggered_by_overheat=True)
        result = set_meltdown_state(test_combatant, meltdown)
        assert result.meltdown_state == meltdown
        assert result.meltdown_state.turns_remaining == 3

    def test_clear_meltdown_state(self, test_combatant: CombatantState):
        """Test clearing meltdown state."""
        meltdown = MeltdownState(turns_remaining=3)
        result = set_meltdown_state(test_combatant, meltdown)
        result2 = set_meltdown_state(result, None)
        assert result2.meltdown_state is None

    def test_decrement_meltdown_countdown_no_meltdown(
        self, test_combatant: CombatantState
    ):
        """Test decrementing when no meltdown active."""
        result, triggered = decrement_meltdown_countdown(test_combatant)
        assert triggered is False
        assert result.meltdown_state is None

    def test_decrement_meltdown_countdown_triggers(
        self, test_combatant: CombatantState
    ):
        """Test decrementing countdown that triggers meltdown."""
        meltdown = MeltdownState(turns_remaining=1, exposed_applied=True)
        result, triggered = decrement_meltdown_countdown(
            set_meltdown_state(test_combatant, meltdown)
        )
        assert triggered is True
        assert result.meltdown_state is None
        assert "exposed" not in result.statuses


class TestOverchargeHelpers:
    """Tests for overcharge state helpers."""

    def test_create_overcharge_state(self):
        """Test creating overcharge state."""
        state = create_overcharge_state(level=1, uses=0)
        assert state.current_level == 1
        assert state.uses_this_turn == 0

    def test_use_overcharge(self, test_combatant: CombatantState):
        """Test using overcharge."""
        result, new_state = use_overcharge(test_combatant)
        assert new_state.current_level == 1
        assert new_state.uses_this_turn == 1

    def test_use_overcharge_escalation(self, test_combatant: CombatantState):
        """Test overcharge escalation."""
        result = test_combatant
        for i in range(3):
            result, new_state = use_overcharge(result)
            assert new_state.current_level == i + 1
        # Should cap at level 3
        result, new_state = use_overcharge(result)
        assert new_state.current_level == 3  # Still 3, not 4

    def test_reset_overcharge_uses(self, test_combatant: CombatantState):
        """Test resetting overcharge uses."""
        result = test_combatant
        result, _ = use_overcharge(result)
        result = reset_overcharge_uses(result)
        assert result.overcharge_state.uses_this_turn == 0
        assert result.overcharge_state.current_level == 1


class TestTurnEndEffectHelpers:
    """Tests for turn-end effect helpers."""

    def test_set_effect_duration(self):
        """Test setting effect duration."""
        effect = TurnEndEffectState(
            effect_id="test",
            effect_type="buff",
            duration_type="end_of_next_turn",
            applied_by="actor1",
        )
        result = set_effect_duration(effect, "end_of_turn")
        assert result.duration_type == "end_of_turn"

    def test_advance_effect_to_next_turn(self):
        """Test advancing effect duration."""
        effect = TurnEndEffectState(
            effect_id="test",
            effect_type="buff",
            duration_type="end_of_next_turn",
            applied_by="actor1",
        )
        result = advance_effect_to_next_turn(effect)
        assert result.duration_type == "end_of_turn"

    def test_advance_effect_already_end_of_turn(self):
        """Test advancing effect that's already end_of_turn."""
        effect = TurnEndEffectState(
            effect_id="test",
            effect_type="buff",
            duration_type="end_of_turn",
            applied_by="actor1",
        )
        result = advance_effect_to_next_turn(effect)
        assert result.duration_type == "end_of_turn"


class TestDamageApplication:
    """Tests for damage application helpers."""

    def test_apply_damage_no_armor(self, test_combatant: CombatantState):
        """Test applying damage without armor (combatant has 2 armor, so 3 net damage)."""
        result = apply_damage(test_combatant, 5)
        # Armor is 2, so net damage is 5 - 2 = 3
        # HP 8 - 3 = 5
        assert result.updated_combatant.resources.hp_current == 5
        assert result.changes_summary["hp"] == (8, 5)

    def test_apply_damage_with_armor(self, test_combatant: CombatantState):
        """Test applying damage with armor piercing."""
        result = apply_damage(test_combatant, 5, armor_piercing=1)
        # Armor is 2, AP is 1, so effective armor is 1
        # Damage 5 - 1 = 4, HP 8 - 4 = 4
        assert result.updated_combatant.resources.hp_current == 4

    def test_apply_damage_armor_exceeds(self, test_combatant: CombatantState):
        """Test applying damage when armor exceeds damage."""
        result = apply_damage(test_combatant, 1, armor_piercing=0)
        # Armor 2 > Damage 1, so no damage (1 - 2 = 0)
        assert result.updated_combatant.resources.hp_current == 8

    def test_apply_damage_clamped_to_zero(self, test_combatant: CombatantState):
        """Test that damage is clamped to 0 HP."""
        result = apply_damage(test_combatant, 20)
        # Armor 2, net damage 18, HP 8 - 18 = 0 (clamped)
        assert result.updated_combatant.resources.hp_current == 0


class TestHeatDamageApplication:
    """Tests for heat damage application helpers."""

    def test_apply_heat_damage(self, test_combatant: CombatantState):
        """Test applying heat damage."""
        result = apply_heat_damage(test_combatant, 2)
        assert result.updated_combatant.resources.heat_current == 5
        assert result.changes_summary["heat"] == (3, 5)


class TestStructureDamageApplication:
    """Tests for structure damage application helpers."""

    def test_apply_structure_damage(self, test_combatant: CombatantState):
        """Test applying structure damage."""
        result = apply_structure_damage(test_combatant)
        assert result.updated_combatant.resources.structure_current == 2
        assert result.changes_summary["structure"] == (3, 2)


class TestOverheatResultApplication:
    """Tests for applying overheat resolution results."""

    def test_apply_overheat_result_basic(self, test_combatant: CombatantState):
        """Test applying basic overheat result."""
        result = apply_overheat_result(
            test_combatant,
            stress_after=2,
            heat_cleared=True,
        )
        assert result.updated_combatant.resources.stress_current == 2
        assert result.updated_combatant.resources.heat_current == 0

    def test_apply_overheat_result_with_statuses(self, test_combatant: CombatantState):
        """Test applying overheat result with statuses."""
        result = apply_overheat_result(
            test_combatant,
            stress_after=2,
            heat_cleared=True,
            statuses_to_add=["impaired", "exposed"],
        )
        assert "impaired" in result.updated_combatant.statuses
        assert "exposed" in result.updated_combatant.statuses

    def test_apply_overheat_result_with_meltdown(self, test_combatant: CombatantState):
        """Test applying overheat result with meltdown state."""
        meltdown = MeltdownState(turns_remaining=3)
        result = apply_overheat_result(
            test_combatant,
            stress_after=1,
            heat_cleared=True,
            meltdown_state=meltdown,
        )
        assert result.updated_combatant.meltdown_state == meltdown


class TestStateUpdateResult:
    """Tests for StateUpdateResult model."""

    def test_state_update_result_fields(self, test_combatant: CombatantState):
        """Test StateUpdateResult has correct fields."""
        result = apply_damage(test_combatant, 5)
        assert isinstance(result, StateUpdateResult)
        assert result.updated_combatant.id == "test_mech"
        assert "hp" in result.changes_summary

    def test_state_update_result_immutable(self, test_combatant: CombatantState):
        """Test that StateUpdateResult doesn't mutate original."""
        result = apply_damage(test_combatant, 5)
        assert test_combatant.resources.hp_current == 8


class TestImmutability:
    """Tests to verify immutability of original state."""

    def test_set_hp_immutable(self, test_combatant: CombatantState):
        """Test that set_hp doesn't mutate original."""
        set_hp(test_combatant, 5)
        assert test_combatant.resources.hp_current == 8

    def test_add_status_immutable(self, test_combatant: CombatantState):
        """Test that add_status doesn't mutate original."""
        add_status(test_combatant, "impaired")
        assert len(test_combatant.statuses) == 0

    def test_clear_statuses_immutable(self, test_combatant: CombatantState):
        """Test that clear_statuses doesn't mutate original."""
        result = add_status(test_combatant, "impaired")
        clear_statuses(result)
        assert len(result.statuses) == 1
