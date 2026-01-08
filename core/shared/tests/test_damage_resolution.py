"""Tests for damage resolution primitives."""

import pytest
from core.shared.damage import (
    DamageInput,
    DamageResolutionContext,
    DamageResolutionResult,
    resolve_damage_on_target,
    apply_damage_to_combatant,
    compute_damage_before_reductions,
    compute_armor_reduction,
    compute_resistance_reduction,
)
from core.mech.grid import HexCoord
from core.shared.enums import DamageType, StatusType
from core.mech.combat_state import (
    CombatantState,
    CombatResources,
    CombatStats,
)


def make_test_combatant(
    hp: int = 10,
    armor: int = 0,
    statuses: list[StatusType] | None = None,
    heat_cap: int = 10,
) -> CombatantState:
    """Create a test combatant for damage tests."""
    stats = CombatStats(
        size="size_2",
        hp_max=hp,
        evasion=10,
        e_defense=10,
        armor=armor,
        speed=4,
        sensor_range=5,
        tech_attack=0,
    )
    resources = CombatResources(
        hp_current=hp,
        heat_current=0,
        heat_cap=heat_cap,
        structure_current=3,
        stress_current=3,
    )
    return CombatantState(
        id="test_combatant",
        name="Test Mech",
        side="players",
        kind="mech",
        stats=stats,
        resources=resources,
        statuses=statuses or [],
    )


class TestDamageInput:
    """Tests for DamageInput model."""

    def test_basic_damage_input(self):
        dmg = DamageInput(damage=5, damage_type="kinetic")
        assert dmg.damage == 5
        assert dmg.damage_type == "kinetic"
        assert dmg.armor_piercing == 0
        assert dmg.bonus_damage == 0
        assert dmg.source is None

    def test_damage_input_with_ap(self):
        dmg = DamageInput(damage=5, damage_type="kinetic", armor_piercing=2)
        assert dmg.armor_piercing == 2

    def test_damage_input_with_bonus(self):
        dmg = DamageInput(damage=5, damage_type="kinetic", bonus_damage=2)
        assert dmg.bonus_damage == 2

    def test_damage_input_all_types(self):
        for dmg_type in ["kinetic", "explosive", "energy", "burn"]:  # type: ignore[assignment]
            dmg = DamageInput(damage=5, damage_type=dmg_type)  # type: ignore[arg-type]
            assert dmg.damage_type == dmg_type


class TestDamageResolutionContext:
    """Tests for DamageResolutionContext model."""

    def test_basic_context(self):
        combatant = make_test_combatant()
        ctx = DamageResolutionContext(
            attacker_id="attacker_1",
            target=combatant,
        )
        assert ctx.attacker_id == "attacker_1"
        assert ctx.target.id == "test_combatant"
        assert ctx.is_critical is False
        assert ctx.multi_target is False

    def test_context_with_position(self):
        combatant = make_test_combatant()
        ctx = DamageResolutionContext(
            attacker_id="attacker_1",
            target=combatant,
            target_position=HexCoord(q=3, r=4),
        )
        assert ctx.target_position == (3, 4)

    def test_context_critical_hit(self):
        combatant = make_test_combatant()
        ctx = DamageResolutionContext(
            attacker_id="attacker_1",
            target=combatant,
            is_critical=True,
        )
        assert ctx.is_critical is True

    def test_context_multi_target(self):
        combatant = make_test_combatant()
        ctx = DamageResolutionContext(
            attacker_id="attacker_1",
            target=combatant,
            multi_target=True,
        )
        assert ctx.multi_target is True


class TestResolveDamageOnTarget:
    """Tests for resolve_damage_on_target function."""

    def test_basic_kinetic_damage(self):
        """Basic kinetic damage without modifiers."""
        combatant = make_test_combatant(hp=10, armor=0)
        dmg = DamageInput(damage=5, damage_type="kinetic")
        ctx = DamageResolutionContext(attacker_id="attacker", target=combatant)

        result = resolve_damage_on_target(dmg, ctx)

        assert result.raw_damage == 5
        assert result.bonus_damage_applied == 0
        assert result.armor_reduction == 0
        assert result.resistance_reduction == 0
        assert result.net_damage == 5
        assert result.damage_to_hp == 5
        assert result.damage_type == "kinetic"
        assert result.is_shredded is False
        assert result.is_exposed is False

    def test_armor_reduction(self):
        """Armor reduces incoming damage."""
        combatant = make_test_combatant(hp=10, armor=3)
        dmg = DamageInput(damage=10, damage_type="kinetic")
        ctx = DamageResolutionContext(attacker_id="attacker", target=combatant)

        result = resolve_damage_on_target(dmg, ctx)

        assert result.armor_reduction == 3
        assert result.net_damage == 7
        assert result.damage_to_hp == 7

    def test_max_armor_is_4(self):
        """Mech armor is capped at 4 per PR2."""
        combatant = make_test_combatant(hp=10, armor=6)
        dmg = DamageInput(damage=10, damage_type="kinetic")
        ctx = DamageResolutionContext(attacker_id="attacker", target=combatant)

        result = resolve_damage_on_target(dmg, ctx)

        assert result.armor_reduction == 4
        assert result.net_damage == 6

    def test_armor_piercing_ignores_armor(self):
        """AP ignores armor completely."""
        combatant = make_test_combatant(hp=10, armor=4)
        dmg = DamageInput(damage=10, damage_type="kinetic", armor_piercing=2)
        ctx = DamageResolutionContext(attacker_id="attacker", target=combatant)

        result = resolve_damage_on_target(dmg, ctx)

        assert result.armor_reduction == 0
        assert result.armor_ignored is True
        assert result.net_damage == 10

    def test_burn_ignores_armor(self):
        """Burn damage ignores armor per PR2."""
        combatant = make_test_combatant(hp=10, armor=4)
        dmg = DamageInput(damage=5, damage_type="burn")
        ctx = DamageResolutionContext(attacker_id="attacker", target=combatant)

        result = resolve_damage_on_target(dmg, ctx)

        assert result.armor_reduction == 0
        assert result.burn_ignored_armor is True
        assert result.damage_to_hp == 0
        assert result.damage_to_heat == 5

    def test_exposed_doubles_damage(self):
        """Exposed condition doubles damage before reductions."""
        combatant = make_test_combatant(hp=10, armor=2)
        combatant = combatant.model_copy(update={"statuses": ["exposed"]})

        dmg = DamageInput(damage=5, damage_type="kinetic")
        ctx = DamageResolutionContext(attacker_id="attacker", target=combatant)

        result = resolve_damage_on_target(dmg, ctx)

        assert result.raw_damage == 10
        assert result.is_exposed is True
        assert result.net_damage == 8

    def test_shredded_ignores_armor_and_resistance(self):
        """Shredded condition prevents armor/resistance benefits."""
        combatant = make_test_combatant(hp=10, armor=4)
        combatant = combatant.model_copy(update={"statuses": ["shredded"]})

        dmg = DamageInput(damage=10, damage_type="kinetic")
        ctx = DamageResolutionContext(attacker_id="attacker", target=combatant)

        result = resolve_damage_on_target(dmg, ctx)

        assert result.armor_reduction == 0
        assert result.resistance_reduction == 0
        assert result.is_shredded is True
        assert result.net_damage == 10

    def test_shredded_and_exposed_combined(self):
        """Shredded and exposed together: 2x damage, no reductions."""
        combatant = make_test_combatant(hp=10, armor=4)
        combatant = combatant.model_copy(update={"statuses": ["shredded", "exposed"]})

        dmg = DamageInput(damage=5, damage_type="kinetic")
        ctx = DamageResolutionContext(attacker_id="attacker", target=combatant)

        result = resolve_damage_on_target(dmg, ctx)

        assert result.raw_damage == 10
        assert result.armor_reduction == 0
        assert result.resistance_reduction == 0
        assert result.net_damage == 10

    def test_bonus_damage_applied(self):
        """Bonus damage is added to base damage."""
        combatant = make_test_combatant(hp=10, armor=0)
        dmg = DamageInput(damage=5, damage_type="kinetic", bonus_damage=2)
        ctx = DamageResolutionContext(attacker_id="attacker", target=combatant)

        result = resolve_damage_on_target(dmg, ctx)

        assert result.bonus_damage_applied == 2
        assert result.net_damage == 7

    def test_multi_target_bonus_damage_halved(self):
        """Multi-target attacks have halved bonus damage."""
        combatant = make_test_combatant(hp=10, armor=0)
        dmg = DamageInput(damage=5, damage_type="kinetic", bonus_damage=4)
        ctx = DamageResolutionContext(
            attacker_id="attacker", target=combatant, multi_target=True
        )

        result = resolve_damage_on_target(dmg, ctx)

        assert result.bonus_damage_applied == 2
        assert result.net_damage == 7

    def test_all_damage_types(self):
        """All four damage types resolve correctly."""
        combatant = make_test_combatant(hp=10, armor=2)

        for dmg_type in ["kinetic", "explosive", "energy", "burn"]:  # type: ignore[assignment]
            dmg = DamageInput(damage=5, damage_type=dmg_type)  # type: ignore[arg-type]
            ctx = DamageResolutionContext(attacker_id="attacker", target=combatant)

            result = resolve_damage_on_target(dmg, ctx)

            assert result.damage_type == dmg_type

    def test_damage_cannot_be_negative(self):
        """Damage resolution doesn't produce negative values."""
        combatant = make_test_combatant(hp=10, armor=10)
        dmg = DamageInput(damage=3, damage_type="kinetic")
        ctx = DamageResolutionContext(attacker_id="attacker", target=combatant)

        result = resolve_damage_on_target(dmg, ctx)

        assert result.net_damage >= 0
        assert result.armor_reduction <= 3

    def test_zero_damage(self):
        """Zero damage is handled correctly."""
        combatant = make_test_combatant(hp=10, armor=0)
        dmg = DamageInput(damage=0, damage_type="kinetic")
        ctx = DamageResolutionContext(attacker_id="attacker", target=combatant)

        result = resolve_damage_on_target(dmg, ctx)

        assert result.raw_damage == 0
        assert result.net_damage == 0
        assert result.damage_to_hp == 0


class TestApplyDamageToCombatant:
    """Tests for apply_damage_to_combatant function."""

    def test_hp_reduction(self):
        """Damage reduces target HP."""
        combatant = make_test_combatant(hp=10, armor=0)
        dmg = DamageInput(damage=5, damage_type="kinetic")
        ctx = DamageResolutionContext(attacker_id="attacker", target=combatant)

        updated, result = apply_damage_to_combatant(dmg, ctx)

        assert updated.resources.hp_current == 5
        assert result.damage_to_hp == 5

    def test_hp_clamped_to_zero(self):
        """HP doesn't go below zero."""
        combatant = make_test_combatant(hp=5, armor=0)
        dmg = DamageInput(damage=10, damage_type="kinetic")
        ctx = DamageResolutionContext(attacker_id="attacker", target=combatant)

        updated, result = apply_damage_to_combatant(dmg, ctx)

        assert updated.resources.hp_current == 0

    def test_burn_adds_heat(self):
        """Burn damage adds to heat instead of HP."""
        combatant = make_test_combatant(hp=10, armor=0, heat_cap=10)
        dmg = DamageInput(damage=5, damage_type="burn")
        ctx = DamageResolutionContext(attacker_id="attacker", target=combatant)

        updated, result = apply_damage_to_combatant(dmg, ctx)

        assert updated.resources.hp_current == 10
        assert updated.resources.heat_current == 5

    def test_heat_accumulates(self):
        """Burn damage stacks with existing heat."""
        resources = CombatResources(
            hp_current=10,
            heat_current=3,
            heat_cap=10,
            structure_current=3,
            stress_current=3,
        )
        combatant = make_test_combatant(hp=10, armor=0, heat_cap=10)
        combatant = combatant.model_copy(update={"resources": resources})

        dmg = DamageInput(damage=5, damage_type="burn")
        ctx = DamageResolutionContext(attacker_id="attacker", target=combatant)

        updated, result = apply_damage_to_combatant(dmg, ctx)

        assert updated.resources.heat_current == 8


class TestComputeDamageBeforeReductions:
    """Tests for compute_damage_before_reductions helper."""

    def test_basic_damage(self):
        """Basic damage without multipliers."""
        result = compute_damage_before_reductions(5, "kinetic", is_exposed=False)
        assert result == 5

    def test_exposed_multiplier(self):
        """Exposed doubles damage."""
        result = compute_damage_before_reductions(5, "kinetic", is_exposed=True)
        assert result == 10

    def test_bonus_damage(self):
        """Bonus damage is added."""
        result = compute_damage_before_reductions(
            5, "kinetic", is_exposed=False, bonus_damage=3
        )
        assert result == 8

    def test_multi_target_halves_bonus(self):
        """Multi-target halves bonus damage."""
        result = compute_damage_before_reductions(
            5, "kinetic", is_exposed=False, bonus_damage=4, multi_target=True
        )
        assert result == 7

    def test_exposed_with_bonus(self):
        """Exposed multiplies base damage before adding bonus."""
        result = compute_damage_before_reductions(
            5, "kinetic", is_exposed=True, bonus_damage=2
        )
        assert result == 12


class TestComputeArmorReduction:
    """Tests for compute_armor_reduction helper."""

    def test_no_armor(self):
        """No armor means no reduction."""
        reduction, ignored = compute_armor_reduction(
            10, "kinetic", armor=0, armor_piercing=0, is_shredded=False
        )
        assert reduction == 0
        assert ignored is False

    def test_armor_reduces_damage(self):
        """Armor reduces incoming damage."""
        reduction, ignored = compute_armor_reduction(
            10, "kinetic", armor=3, armor_piercing=0, is_shredded=False
        )
        assert reduction == 3
        assert ignored is False

    def test_ap_ignores_armor(self):
        """AP ignores armor."""
        reduction, ignored = compute_armor_reduction(
            10, "kinetic", armor=3, armor_piercing=2, is_shredded=False
        )
        assert reduction == 0
        assert ignored is True

    def test_shredded_ignores_armor(self):
        """Shredded ignores armor."""
        reduction, ignored = compute_armor_reduction(
            10, "kinetic", armor=3, armor_piercing=0, is_shredded=True
        )
        assert reduction == 0
        assert ignored is False

    def test_burn_ignores_armor(self):
        """Burn ignores armor."""
        reduction, ignored = compute_armor_reduction(
            10, "burn", armor=3, armor_piercing=0, is_shredded=False
        )
        assert reduction == 0
        assert ignored is True

    def test_capped_at_damage_amount(self):
        """Armor reduction doesn't exceed damage."""
        reduction, ignored = compute_armor_reduction(
            3, "kinetic", armor=5, armor_piercing=0, is_shredded=False
        )
        assert reduction == 3


class TestComputeResistanceReduction:
    """Tests for compute_resistance_reduction helper."""

    def test_no_resistance(self):
        """No resistance means no reduction."""
        reduction = compute_resistance_reduction(10, "kinetic", is_shredded=False)
        assert reduction == 0

    def test_with_resistance(self):
        """Resistance halves damage."""
        reduction = compute_resistance_reduction(
            10, "kinetic", is_shredded=False, resistances=["kinetic"]
        )
        assert reduction == 5

    def test_shredded_ignores_resistance(self):
        """Shredded ignores resistance."""
        reduction = compute_resistance_reduction(
            10, "kinetic", is_shredded=True, resistances=["kinetic"]
        )
        assert reduction == 0

    def test_half_damage_rounded_up(self):
        """Resistance halves damage, rounded up."""
        reduction = compute_resistance_reduction(
            10, "kinetic", is_shredded=False, resistances=["kinetic"]
        )
        assert reduction == 5

    def test_odd_damage_rounds_up(self):
        """Odd damage is rounded up (e.g., 7 -> 4)."""
        reduction = compute_resistance_reduction(
            7, "kinetic", is_shredded=False, resistances=["kinetic"]
        )
        assert reduction == 4

    def test_burn_not_resistable(self):
        """Burn damage cannot be resisted."""
        reduction = compute_resistance_reduction(
            10, "burn", is_shredded=False, resistances=["burn"]
        )
        assert reduction == 0

    def test_wrong_resistance_type(self):
        """Resistance to wrong damage type doesn't apply."""
        reduction = compute_resistance_reduction(
            10, "kinetic", is_shredded=False, resistances=["energy"]
        )
        assert reduction == 0

    def test_multiple_resistances(self):
        """Multiple resistances work correctly."""
        reduction = compute_resistance_reduction(
            10, "kinetic", is_shredded=False, resistances=["kinetic", "energy"]
        )
        assert reduction == 5


class TestDamageResolutionIntegration:
    """Integration tests for damage resolution with complex scenarios."""

    def test_full_damage_resolution_chain(self):
        """Test complete damage resolution with all modifiers."""
        combatant = make_test_combatant(hp=20, armor=3)
        combatant = combatant.model_copy(update={"statuses": []})

        dmg = DamageInput(
            damage=8, damage_type="kinetic", armor_piercing=1, bonus_damage=2
        )
        ctx = DamageResolutionContext(attacker_id="attacker", target=combatant)

        result = resolve_damage_on_target(dmg, ctx)

        assert result.raw_damage == 8
        assert result.bonus_damage_applied == 2
        assert result.armor_reduction == 0
        assert result.armor_ignored is True
        assert result.net_damage == 10

    def test_exposed_with_armor(self):
        """Exposed damage still reduced by armor."""
        combatant = make_test_combatant(hp=20, armor=3)
        combatant = combatant.model_copy(update={"statuses": ["exposed"]})

        dmg = DamageInput(damage=5, damage_type="kinetic")
        ctx = DamageResolutionContext(attacker_id="attacker", target=combatant)

        result = resolve_damage_on_target(dmg, ctx)

        assert result.raw_damage == 10
        assert result.armor_reduction == 3
        assert result.net_damage == 7

    def test_exposed_with_ap(self):
        """Exposed damage with AP bypasses armor."""
        combatant = make_test_combatant(hp=20, armor=3)
        combatant = combatant.model_copy(update={"statuses": ["exposed"]})

        dmg = DamageInput(damage=5, damage_type="kinetic", armor_piercing=2)
        ctx = DamageResolutionContext(attacker_id="attacker", target=combatant)

        result = resolve_damage_on_target(dmg, ctx)

        assert result.raw_damage == 10
        assert result.armor_reduction == 0
        assert result.armor_ignored is True
        assert result.net_damage == 10

    def test_critical_hit_with_exposed(self):
        """Critical hits double damage before exposed multiplier."""
        combatant = make_test_combatant(hp=20, armor=2)
        combatant = combatant.model_copy(update={"statuses": ["exposed"]})

        dmg = DamageInput(damage=6, damage_type="kinetic")
        ctx = DamageResolutionContext(
            attacker_id="attacker", target=combatant, is_critical=True
        )

        result = resolve_damage_on_target(dmg, ctx)

        assert result.raw_damage == 12
        assert result.armor_reduction == 2
        assert result.net_damage == 10

    def test_multiple_conditions_with_damage(self):
        """Test damage with multiple conditions present."""
        combatant = make_test_combatant(hp=20, armor=3)
        combatant = combatant.model_copy(
            update={"statuses": ["exposed", "shredded", "impaired"]}
        )

        dmg = DamageInput(damage=5, damage_type="explosive")
        ctx = DamageResolutionContext(attacker_id="attacker", target=combatant)

        result = resolve_damage_on_target(dmg, ctx)

        assert result.raw_damage == 10
        assert result.armor_reduction == 0
        assert result.resistance_reduction == 0
        assert result.net_damage == 10

    def test_energy_damage_with_armor(self):
        """Energy damage is reduced by armor."""
        combatant = make_test_combatant(hp=20, armor=4)
        dmg = DamageInput(damage=8, damage_type="energy")
        ctx = DamageResolutionContext(attacker_id="attacker", target=combatant)

        result = resolve_damage_on_target(dmg, ctx)

        assert result.armor_reduction == 4
        assert result.net_damage == 4

    def test_explosive_damage_with_armor(self):
        """Explosive damage is reduced by armor."""
        combatant = make_test_combatant(hp=20, armor=2)
        dmg = DamageInput(damage=6, damage_type="explosive")
        ctx = DamageResolutionContext(attacker_id="attacker", target=combatant)

        result = resolve_damage_on_target(dmg, ctx)

        assert result.armor_reduction == 2
        assert result.net_damage == 4

    def test_pilot_damage_not_capped_at_4(self):
        """Pilots don't have the 4 armor cap."""
        from core.shared.enums import StatusType

        stats = CombatStats(
            size="size_1",
            hp_max=10,
            evasion=12,
            e_defense=10,
            armor=2,
            speed=6,
            sensor_range=0,
            tech_attack=0,
        )
        resources = CombatResources(
            hp_current=10,
            heat_current=0,
            heat_cap=0,
            structure_current=0,
            stress_current=0,
        )
        pilot = CombatantState(
            id="pilot_1",
            name="Test Pilot",
            side="players",
            kind="pilot",
            stats=stats,
            resources=resources,
            statuses=[],
        )

        dmg = DamageInput(damage=4, damage_type="kinetic")
        ctx = DamageResolutionContext(attacker_id="attacker", target=pilot)

        result = resolve_damage_on_target(dmg, ctx)

        assert result.armor_reduction == 2
        assert result.net_damage == 2
