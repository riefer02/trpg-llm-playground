"""Tests for stabilize action resolution."""

import pytest
from core.shared.stabilize import (
    resolve_stabilize,
    apply_stabilize_result,
    StabilizeInput,
    StabilizeRule,
)
from core.mech.combat_state import (
    CombatantState,
    CombatStats,
    CombatResources,
    MechInventory,
    WeaponMountState,
    WeaponState,
)


@pytest.fixture
def test_combatant() -> CombatantState:
    """Create a test combatant for stabilize tests."""
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
        ),
        statuses=["exposed", "burn"],
        conditions=["impaired"],
    )


@pytest.fixture
def test_combatant_with_weapons(test_combatant: CombatantState) -> CombatantState:
    """Create a test combatant with Loading weapons."""
    inventory = MechInventory(
        mounts=[
            WeaponMountState(
                mount_index=0,
                slot_type="main",
                weapons=[
                    WeaponState(weapon_id="rifle", tags=["loading", "accurate"]),
                    WeaponState(weapon_id="aux_pistol", tags=[]),
                ],
            ),
            WeaponMountState(
                mount_index=1,
                slot_type="heavy",
                weapons=[
                    WeaponState(weapon_id="launcher", tags=["loading", "ordnance"]),
                ],
            ),
        ]
    )
    return test_combatant.model_copy(update={"inventory": inventory})


@pytest.fixture
def ally_combatant() -> CombatantState:
    """Create a test ally for condition clearing."""
    return CombatantState(
        id="ally_mech",
        name="Ally Mech",
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
            hp_current=10,
            heat_current=0,
            heat_cap=10,
        ),
        conditions=["jammed", "slowed"],
    )


class TestResolveStabilize:
    """Tests for stabilize resolution (pure logic)."""

    def test_cool_heat_and_clear_exposed(self):
        """Test cooling heat clears heat and exposed status."""
        result = resolve_stabilize(
            StabilizeInput(
                primary_choice="cool_heat",
                secondary_choice="reload_loading",
            )
        )
        assert result.primary_choice == "cool_heat"
        assert result.heat_cleared is True
        assert result.exposed_cleared is True

    def test_spend_repair_full_hp(self):
        """Test spending repair restores HP."""
        result = resolve_stabilize(
            StabilizeInput(
                primary_choice="spend_repair_full_hp",
                secondary_choice="reload_loading",
            )
        )
        assert result.primary_choice == "spend_repair_full_hp"
        assert result.hp_restored == 1  # repair_cost defaults to 1

    def test_reload_loading_weapons(self):
        """Test reloading Loading weapons."""
        result = resolve_stabilize(
            StabilizeInput(
                primary_choice="cool_heat",
                secondary_choice="reload_loading",
            )
        )
        assert result.secondary_choice == "reload_loading"
        assert result.weapons_reloaded == []

    def test_clear_burn(self):
        """Test clearing burn status."""
        result = resolve_stabilize(
            StabilizeInput(
                primary_choice="cool_heat",
                secondary_choice="clear_burn",
            )
        )
        assert result.secondary_choice == "clear_burn"
        assert result.burn_cleared is True

    def test_clear_condition(self):
        """Test clearing conditions."""
        result = resolve_stabilize(
            StabilizeInput(
                primary_choice="cool_heat",
                secondary_choice="clear_condition",
                condition_target_id="ally_mech",
            )
        )
        assert result.secondary_choice == "clear_condition"
        assert result.condition_target_id == "ally_mech"
        assert "impaired" in result.conditions_cleared
        assert "jammed" in result.conditions_cleared

    def test_clear_condition_requires_target(self):
        """Test that clear_condition requires condition_target_id."""
        result = resolve_stabilize(
            StabilizeInput(
                primary_choice="cool_heat",
                secondary_choice="clear_condition",
            )
        )
        assert len(result.validation_errors) == 1
        assert "condition_target_id is required" in result.validation_errors[0]

    def test_custom_rules(self):
        """Test using custom stabilize rules."""
        custom_rules = StabilizeRule(
            repair_cost=2,
            cool_heat_clears_exposed=False,
        )
        result = resolve_stabilize(
            StabilizeInput(
                primary_choice="spend_repair_full_hp",
                secondary_choice="reload_loading",
                rules=custom_rules,
            )
        )
        assert result.hp_restored == 2
        assert result.exposed_cleared is False


class TestApplyStabilizeResult:
    """Tests for applying stabilize results to combatant state."""

    def test_apply_cool_heat(self, test_combatant: CombatantState):
        """Test applying heat clearing."""
        result = resolve_stabilize(
            StabilizeInput(
                primary_choice="cool_heat",
                secondary_choice="reload_loading",
            )
        )
        applied = apply_stabilize_result(test_combatant, result)

        assert applied.heat_cleared is True
        assert applied.updated_combatant.resources.heat_current == 0

    def test_apply_clear_exposed(self, test_combatant: CombatantState):
        """Test applying exposed status clearing."""
        result = resolve_stabilize(
            StabilizeInput(
                primary_choice="cool_heat",
                secondary_choice="reload_loading",
            )
        )
        applied = apply_stabilize_result(test_combatant, result)

        assert "exposed" not in applied.updated_combatant.statuses
        assert "exposed" in applied.statuses_cleared

    def test_apply_repair_hp(self, test_combatant: CombatantState):
        """Test applying HP repair."""
        result = resolve_stabilize(
            StabilizeInput(
                primary_choice="spend_repair_full_hp",
                secondary_choice="reload_loading",
            )
        )
        applied = apply_stabilize_result(test_combatant, result)

        assert applied.hp_restored_amount == 1
        assert applied.hp_current_after == 6  # 5 + 1

    def test_apply_repair_caps_at_max(self, test_combatant: CombatantState):
        """Test that repair caps at max HP."""
        test_combatant_full = test_combatant.model_copy(
            update={
                "resources": test_combatant.resources.model_copy(
                    update={"hp_current": 9}
                )
            }
        )
        result = resolve_stabilize(
            StabilizeInput(
                primary_choice="spend_repair_full_hp",
                secondary_choice="reload_loading",
            )
        )
        applied = apply_stabilize_result(test_combatant_full, result)

        assert applied.hp_restored_amount == 1
        assert applied.hp_current_after == 10  # Capped at hp_max

    def test_apply_clear_burn(self, test_combatant: CombatantState):
        """Test applying burn clearing."""
        result = resolve_stabilize(
            StabilizeInput(
                primary_choice="cool_heat",
                secondary_choice="clear_burn",
            )
        )
        applied = apply_stabilize_result(test_combatant, result)

        assert "burn" not in applied.updated_combatant.statuses
        assert "burn" in applied.statuses_cleared

    def test_apply_reload_weapons(self, test_combatant_with_weapons: CombatantState):
        """Test reloading Loading-tagged weapons."""
        result = resolve_stabilize(
            StabilizeInput(
                primary_choice="cool_heat",
                secondary_choice="reload_loading",
            )
        )
        applied = apply_stabilize_result(test_combatant_with_weapons, result)

        assert "rifle" in applied.weapons_reloaded
        assert "launcher" in applied.weapons_reloaded
        assert "aux_pistol" not in applied.weapons_reloaded  # No Loading tag

    def test_apply_clear_condition_self(self, test_combatant: CombatantState):
        """Test clearing condition on self."""
        result = resolve_stabilize(
            StabilizeInput(
                primary_choice="cool_heat",
                secondary_choice="clear_condition",
                condition_target_id="test_mech",
            )
        )
        applied = apply_stabilize_result(test_combatant, result)

        assert "impaired" in applied.conditions_cleared
        assert "impaired" not in applied.updated_combatant.conditions

    def test_apply_clear_condition_ally(
        self, test_combatant: CombatantState, ally_combatant: CombatantState
    ):
        """Test clearing condition on ally."""
        result = resolve_stabilize(
            StabilizeInput(
                primary_choice="cool_heat",
                secondary_choice="clear_condition",
                condition_target_id="ally_mech",
            )
        )
        applied = apply_stabilize_result(
            test_combatant, result, target_combatant=ally_combatant
        )

        assert "jammed" in applied.conditions_cleared
        assert "slowed" in applied.conditions_cleared

    def test_apply_combined_cool_and_reload(
        self, test_combatant_with_weapons: CombatantState
    ):
        """Test combined cool heat and reload."""
        result = resolve_stabilize(
            StabilizeInput(
                primary_choice="cool_heat",
                secondary_choice="reload_loading",
            )
        )
        applied = apply_stabilize_result(test_combatant_with_weapons, result)

        assert applied.heat_cleared is True
        assert "rifle" in applied.weapons_reloaded
        assert "launcher" in applied.weapons_reloaded

    def test_apply_combined_repair_and_clear_burn(self, test_combatant: CombatantState):
        """Test combined repair and clear burn."""
        result = resolve_stabilize(
            StabilizeInput(
                primary_choice="spend_repair_full_hp",
                secondary_choice="clear_burn",
            )
        )
        applied = apply_stabilize_result(test_combatant, result)

        assert applied.hp_restored_amount == 1
        assert "burn" in applied.statuses_cleared


class TestStabilizeRules:
    """Tests for stabilize rule configuration."""

    def test_default_rules(self):
        """Test default stabilize rules have correct options."""
        from core.shared.stabilize import DEFAULT_STABILIZE_RULES

        assert "cool_heat" in DEFAULT_STABILIZE_RULES.primary_options
        assert "spend_repair_full_hp" in DEFAULT_STABILIZE_RULES.primary_options
        assert DEFAULT_STABILIZE_RULES.repair_cost == 1
        assert DEFAULT_STABILIZE_RULES.cool_heat_clears_exposed is True

    def test_custom_clearable_conditions(self):
        """Test custom clearable conditions."""
        rules = StabilizeRule(
            clearable_conditions=["impaired", "slowed"],
        )
        result = resolve_stabilize(
            StabilizeInput(
                primary_choice="cool_heat",
                secondary_choice="clear_condition",
                condition_target_id="ally",
                rules=rules,
            )
        )
        assert "impaired" in result.conditions_cleared
        assert "slowed" in result.conditions_cleared
        assert "jammed" not in result.conditions_cleared  # Not in custom list
