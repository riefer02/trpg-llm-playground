"""Tests for Self Destruct action resolution."""

import pytest
from core.shared.self_destruct import (
    resolve_self_destruct_initiation,
    apply_self_destruct_initiation,
    resolve_self_destruct_explosion,
    apply_self_destruct_explosion,
    SelfDestructInput,
    SelfDestructRule,
    SelfDestructExplosionInput,
    SelfDestructResolutionResult,
    SelfDestructExplosionResult,
    DEFAULT_SELF_DESTRUCT_RULES,
)
from core.shared.heat import MeltdownState
from core.shared.dice import DiceExpression
from core.mech.combat_state import (
    CombatantState,
    CombatStats,
    CombatResources,
)
from core.mech.grid import HexPosition, HexCoord


@pytest.fixture
def test_mech() -> CombatantState:
    """Create a test mech for self destruct."""
    return CombatantState(
        id="test_mech",
        name="Test Mech",
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
        position=HexPosition(coord=HexCoord(q=0, r=0)),
    )


@pytest.fixture
def test_pilot() -> CombatantState:
    """Create a test pilot for self destruct."""
    return CombatantState(
        id="test_pilot",
        name="Test Pilot",
        side="players",
        kind="pilot",
        stats=CombatStats(
            size="size_half",
            hp_max=6,
            evasion=10,
            e_defense=10,
            armor=0,
            speed=4,
        ),
        resources=CombatResources(hp_current=6),
        position=HexPosition(coord=HexCoord(q=0, r=0)),
    )


@pytest.fixture
def enemy_mech() -> CombatantState:
    """Create an enemy mech in burst radius."""
    return CombatantState(
        id="enemy_mech",
        name="Enemy Mech",
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
        position=HexPosition(coord=HexCoord(q=1, r=0)),
    )


@pytest.fixture
def distant_mech() -> CombatantState:
    """Create a mech outside burst radius."""
    return CombatantState(
        id="distant_mech",
        name="Distant Mech",
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
        position=HexPosition(coord=HexCoord(q=5, r=0)),
    )


class TestResolveSelfDestructInitiation:
    """Tests for Self Destruct initiation resolution."""

    def test_resolve_self_destruct_initiation_one_turn(self, test_mech):
        """Test self destruct with 1 turn delay."""
        input_data = SelfDestructInput(
            actor_id="test_pilot",
            mech_id=test_mech.id,
            delay_turns=1,
        )
        result = resolve_self_destruct_initiation(input_data)

        assert result.countdown_started is True
        assert result.delay_turns == 1
        assert result.meltdown_state is not None
        assert result.meltdown_state.turns_remaining == 1
        assert result.meltdown_state.is_immediate is True
        assert len(result.validation_errors) == 0

    def test_resolve_self_destruct_initiation_two_turns(self, test_mech):
        """Test self destruct with 2 turn delay."""
        input_data = SelfDestructInput(
            actor_id="test_pilot",
            mech_id=test_mech.id,
            delay_turns=2,
        )
        result = resolve_self_destruct_initiation(input_data)

        assert result.countdown_started is True
        assert result.delay_turns == 2
        assert result.meltdown_state is not None
        assert result.meltdown_state.turns_remaining == 2

    def test_resolve_self_destruct_initiation_delay_too_long(self, test_mech):
        """Test self destruct fails with delay greater than 2."""
        input_data = SelfDestructInput(
            actor_id="test_pilot",
            mech_id=test_mech.id,
            delay_turns=3,
        )
        result = resolve_self_destruct_initiation(input_data)

        assert result.countdown_started is False
        assert result.meltdown_state is None
        assert len(result.validation_errors) > 0

    def test_resolve_self_destruct_initiation_with_custom_rules(self, test_mech):
        """Test self destruct with custom rules."""
        custom_rules = SelfDestructRule(
            min_delay_turns=1,
            max_delay_turns=3,
        )
        input_data = SelfDestructInput(
            actor_id="test_pilot",
            mech_id=test_mech.id,
            delay_turns=3,
            rules=custom_rules,
        )
        result = resolve_self_destruct_initiation(input_data, rules=custom_rules)

        assert result.countdown_started is True
        assert result.delay_turns == 3


class TestApplySelfDestructInitiation:
    """Tests for applying Self Destruct initiation."""

    def test_apply_self_destruct_initiation_success(self, test_mech):
        """Test applying successful self destruct initiation."""

        result = SelfDestructResolutionResult(
            actor_id="test_pilot",
            mech_id=test_mech.id,
            delay_turns=1,
            countdown_started=True,
            meltdown_state=MeltdownState(
                turns_remaining=1,
                triggered_by_overheat=False,
                exposed_applied=False,
                is_immediate=True,
            ),
        )

        updated = apply_self_destruct_initiation(test_mech, result)

        assert updated.meltdown_state is not None
        assert updated.meltdown_state.turns_remaining == 1

    def test_apply_self_destruct_initiation_failure(self, test_mech):
        """Test applying failed self destruct initiation."""

        result = SelfDestructResolutionResult(
            actor_id="test_pilot",
            mech_id=test_mech.id,
            delay_turns=0,
            countdown_started=False,
            meltdown_state=None,
            validation_errors=["Delay must be at least 1 turn(s)"],
        )

        updated = apply_self_destruct_initiation(test_mech, result)

        assert updated.meltdown_state is None
        assert updated.id == test_mech.id


class TestResolveSelfDestructExplosion:
    """Tests for Self Destruct explosion resolution."""

    def test_resolve_self_destruct_explosion(self, test_mech, enemy_mech):
        """Test self destruct explosion resolution."""
        input_data = SelfDestructExplosionInput(
            mech_id=test_mech.id,
            mech_position=test_mech.position,
        )
        all_combatants = [test_mech, enemy_mech]
        result = resolve_self_destruct_explosion(input_data, all_combatants)

        assert result.mech_id == test_mech.id
        assert result.burst_radius == 2
        assert result.mech_destroyed is True
        assert result.pilot_killed is True
        assert len(result.damage_rolls) == 4
        assert result.total_damage == sum(result.damage_rolls)

    def test_resolve_self_destruct_explosion_affects_adjacent(
        self, test_mech, enemy_mech
    ):
        """Test self destruct explosion affects adjacent targets."""
        input_data = SelfDestructExplosionInput(
            mech_id=test_mech.id,
            mech_position=test_mech.position,
        )
        all_combatants = [test_mech, enemy_mech]
        result = resolve_self_destruct_explosion(input_data, all_combatants)

        enemy_result = next(
            (t for t in result.target_results if t.target_id == enemy_mech.id), None
        )
        assert enemy_result is not None
        assert enemy_result.in_burst_radius is True
        assert enemy_result.distance == 1

    def test_resolve_self_destruct_explosion_ignores_distant(
        self, test_mech, distant_mech
    ):
        """Test self destruct explosion ignores distant targets."""
        input_data = SelfDestructExplosionInput(
            mech_id=test_mech.id,
            mech_position=test_mech.position,
        )
        all_combatants = [test_mech, distant_mech]
        result = resolve_self_destruct_explosion(input_data, all_combatants)

        distant_result = next(
            (t for t in result.target_results if t.target_id == distant_mech.id), None
        )
        assert distant_result is None or distant_result.in_burst_radius is False


class TestApplySelfDestructExplosion:
    """Tests for applying Self Destruct explosion."""

    def test_apply_self_destruct_explosion_mech_destroyed(self, test_mech):
        """Test applying explosion destroys the mech."""
        explosion_result = SelfDestructExplosionResult(
            mech_id=test_mech.id,
            mech_position=test_mech.position,
            burst_radius=2,
            damage_expression=DiceExpression.parse("4d6"),
            damage_rolls=[6, 6, 5, 4],
            total_damage=21,
            target_results=[],
            mech_destroyed=True,
            pilot_killed=True,
            wreckage=None,
        )

        application = apply_self_destruct_explosion(test_mech, None, explosion_result)

        assert application.updated_mech is not None
        assert "out" in application.updated_mech.statuses
        assert application.updated_mech.resources.hp_current == 0

    def test_apply_self_destruct_explosion_pilot_killed(self, test_mech, test_pilot):
        """Test applying explosion kills the pilot."""
        explosion_result = SelfDestructExplosionResult(
            mech_id=test_mech.id,
            mech_position=test_mech.position,
            burst_radius=2,
            damage_expression=DiceExpression.parse("4d6"),
            damage_rolls=[6, 6, 5, 4],
            total_damage=21,
            target_results=[],
            mech_destroyed=True,
            pilot_killed=True,
            wreckage=None,
        )

        application = apply_self_destruct_explosion(
            test_mech, test_pilot, explosion_result
        )

        assert application.updated_pilot is not None
        assert "out" in application.updated_pilot.statuses
        assert application.updated_pilot.resources.hp_current == 0


class TestSelfDestructRuleDefaults:
    """Tests for default Self Destruct Rule values."""

    def test_default_burst_radius(self):
        """Test default burst radius is 2."""

        assert DEFAULT_SELF_DESTRUCT_RULES.burst_radius == 2

    def test_default_damage(self):
        """Test default damage is 4d6."""

        assert DEFAULT_SELF_DESTRUCT_RULES.damage.count == 4
        assert DEFAULT_SELF_DESTRUCT_RULES.damage.size == 6

    def test_default_damage_type(self):
        """Test default damage type is explosive."""

        assert DEFAULT_SELF_DESTRUCT_RULES.damage_type == "explosive"

    def test_default_delay_range(self):
        """Test default delay is 1-2 turns."""

        assert DEFAULT_SELF_DESTRUCT_RULES.min_delay_turns == 1
        assert DEFAULT_SELF_DESTRUCT_RULES.max_delay_turns == 2

    def test_default_save_skill(self):
        """Test default save skill is agility."""

        assert DEFAULT_SELF_DESTRUCT_RULES.save_skill == "agility"
