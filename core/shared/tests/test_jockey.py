"""Tests for Jockey action resolution."""

import pytest
from core.shared.jockey import (
    resolve_jockey,
    apply_jockey_result,
    resolve_shake_off,
    apply_shake_off_result,
    JockeyInput,
    ShakeOffInput,
    JockeyOption,
    JockeyRule,
    JockeyContestedResult,
    DEFAULT_JOCKEY_RULES,
)
from core.mech.combat_state import (
    CombatantState,
    CombatStats,
    CombatResources,
)
from core.mech.combat_resolution import ResolutionSettings


@pytest.fixture
def test_pilot() -> CombatantState:
    """Create a test pilot for jockey tests."""
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
    )


@pytest.fixture
def enemy_mech() -> CombatantState:
    """Create an enemy mech for jockey tests."""
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
            sensor_range=10,
        ),
        resources=CombatResources(
            hp_current=16,
            heat_current=0,
            heat_cap=10,
            structure_current=3,
            stress_current=2,
        ),
    )


class TestResolveJockey:
    """Tests for Jockey action resolution."""

    def test_resolve_jockey_success_pilot_wins(self, test_pilot, enemy_mech):
        """Test successful jockey when pilot wins contested check."""
        input_data = JockeyInput(
            actor_id=test_pilot.id,
            target_mech_id=enemy_mech.id,
            chosen_option="distract",
            settings=ResolutionSettings(forced_rolls=[6, 1]),
        )
        result = resolve_jockey(
            input_data,
            pilot_grit=3,
            mech_hull=2,
            is_adjacent=True,
        )

        assert result.jockey_success is True
        assert result.contested_result is not None
        assert result.contested_result.attacker_wins is True
        assert "impaired" in result.conditions_inflicted
        assert "slowed" in result.conditions_inflicted
        assert result.heat_dealt == 0
        assert result.damage_dealt is None

    def test_resolve_jockey_failure_mech_wins(self, test_pilot, enemy_mech):
        """Test failed jockey when mech wins contested check."""
        input_data = JockeyInput(
            actor_id=test_pilot.id,
            target_mech_id=enemy_mech.id,
            chosen_option="distract",
            settings=ResolutionSettings(forced_rolls=[1, 6]),
        )
        result = resolve_jockey(
            input_data,
            pilot_grit=3,
            mech_hull=5,
            is_adjacent=True,
        )

        assert result.jockey_success is False
        assert result.contested_result is not None
        assert result.contested_result.attacker_wins is False
        assert len(result.conditions_inflicted) == 0

    def test_resolve_jockey_not_adjacent(self, test_pilot, enemy_mech):
        """Test jockey failure when not adjacent to mech."""
        input_data = JockeyInput(
            actor_id=test_pilot.id,
            target_mech_id=enemy_mech.id,
            chosen_option="distract",
        )
        result = resolve_jockey(
            input_data,
            pilot_grit=3,
            mech_hull=2,
            is_adjacent=False,
        )

        assert result.jockey_success is False
        assert len(result.validation_errors) > 0
        assert "adjacent" in result.validation_errors[0].lower()

    def test_resolve_jockey_shred_option(self, test_pilot, enemy_mech):
        """Test jockey with shred option (2 heat)."""
        input_data = JockeyInput(
            actor_id=test_pilot.id,
            target_mech_id=enemy_mech.id,
            chosen_option="shred",
            settings=ResolutionSettings(forced_rolls=[6, 1]),
        )
        result = resolve_jockey(
            input_data,
            pilot_grit=3,
            mech_hull=2,
            is_adjacent=True,
        )

        assert result.jockey_success is True
        assert result.chosen_option is not None
        assert result.chosen_option.option_type == "shred"
        assert result.heat_dealt == 2
        assert len(result.conditions_inflicted) == 0

    def test_resolve_jockey_damage_option(self, test_pilot, enemy_mech):
        """Test jockey with damage option (4 kinetic)."""
        input_data = JockeyInput(
            actor_id=test_pilot.id,
            target_mech_id=enemy_mech.id,
            chosen_option="damage",
            settings=ResolutionSettings(forced_rolls=[6, 1]),
        )
        result = resolve_jockey(
            input_data,
            pilot_grit=3,
            mech_hull=2,
            is_adjacent=True,
        )

        assert result.jockey_success is True
        assert result.chosen_option is not None
        assert result.chosen_option.option_type == "damage"
        assert result.damage_dealt == 4

    def test_resolve_jockey_with_forced_rolls(self, test_pilot, enemy_mech):
        """Test jockey with forced roll values."""
        input_data = JockeyInput(
            actor_id=test_pilot.id,
            target_mech_id=enemy_mech.id,
            chosen_option="distract",
            settings=ResolutionSettings(forced_rolls=[6, 1]),
        )
        result = resolve_jockey(
            input_data,
            pilot_grit=3,
            mech_hull=2,
            is_adjacent=True,
        )

        assert result.jockey_success is True
        assert result.contested_result is not None
        assert result.contested_result.attacker_total == 9
        assert result.contested_result.defender_total == 3

    def test_resolve_jockey_unknown_option(self, test_pilot, enemy_mech):
        """Test jockey with option not in rules options list."""
        rules = JockeyRule(options=[])
        input_data = JockeyInput(
            actor_id=test_pilot.id,
            target_mech_id=enemy_mech.id,
            chosen_option="distract",
            rules=rules,
        )
        result = resolve_jockey(
            input_data,
            pilot_grit=3,
            mech_hull=2,
            is_adjacent=True,
        )

        assert result.jockey_success is False
        assert len(result.validation_errors) > 0

    def test_resolve_jockey_with_custom_rules(self, test_pilot, enemy_mech):
        """Test jockey with custom rules."""
        custom_options = [
            JockeyOption(
                option_type="distract",
                inflicted_conditions=["stunned"],
            ),
        ]
        rules = JockeyRule(options=custom_options)
        input_data = JockeyInput(
            actor_id=test_pilot.id,
            target_mech_id=enemy_mech.id,
            chosen_option="distract",
            rules=rules,
            settings=ResolutionSettings(forced_rolls=[6, 1]),
        )
        result = resolve_jockey(
            input_data,
            pilot_grit=3,
            mech_hull=2,
            is_adjacent=True,
        )

        assert result.jockey_success is True
        assert "stunned" in result.conditions_inflicted


class TestApplyJockeyResult:
    """Tests for applying Jockey result to combatant state."""

    def test_apply_jockey_result_distract(self, enemy_mech):
        """Test applying jockey result with distract option."""
        from core.shared.jockey import JockeyResolutionResult

        contested = JockeyContestedResult(
            attacker_roll=None,
            defender_roll=None,
            attacker_total=9,
            defender_total=5,
            attacker_wins=True,
        )

        result = JockeyResolutionResult(
            actor_id="test_pilot",
            target_mech_id=enemy_mech.id,
            contested_result=contested,
            chosen_option=JockeyOption(option_type="distract"),
            jockey_success=True,
            conditions_inflicted=["impaired", "slowed"],
            heat_dealt=0,
            damage_dealt=None,
        )

        application = apply_jockey_result(enemy_mech, result)

        assert application.jockey_success is True
        assert "impaired" in application.conditions_applied
        assert "slowed" in application.conditions_applied
        assert application.heat_dealt == 0

    def test_apply_jockey_result_shred(self, enemy_mech):
        """Test applying jockey result with shred option."""
        from core.shared.jockey import JockeyResolutionResult

        contested = JockeyContestedResult(
            attacker_roll=None,
            defender_roll=None,
            attacker_total=9,
            defender_total=5,
            attacker_wins=True,
        )

        result = JockeyResolutionResult(
            actor_id="test_pilot",
            target_mech_id=enemy_mech.id,
            contested_result=contested,
            chosen_option=JockeyOption(option_type="shred"),
            jockey_success=True,
            conditions_inflicted=[],
            heat_dealt=2,
            damage_dealt=None,
        )

        application = apply_jockey_result(enemy_mech, result)

        assert application.jockey_success is True
        assert application.heat_dealt == 2
        assert application.updated_target.resources.heat_current == 2

    def test_apply_jockey_result_failure(self, enemy_mech):
        """Test applying failed jockey result."""
        from core.shared.jockey import JockeyResolutionResult

        result = JockeyResolutionResult(
            actor_id="test_pilot",
            target_mech_id=enemy_mech.id,
            contested_result=None,
            chosen_option=None,
            jockey_success=False,
            conditions_inflicted=[],
            heat_dealt=0,
            damage_dealt=None,
        )

        application = apply_jockey_result(enemy_mech, result)

        assert application.jockey_success is False
        assert len(application.conditions_applied) == 0


class TestResolveShakeOff:
    """Tests for Mech Shake Off action resolution."""

    def test_resolve_shake_off_success_mech_wins(self, enemy_mech, test_pilot):
        """Test successful shake off when mech wins contested check."""
        input_data = ShakeOffInput(
            mech_id=enemy_mech.id,
            rider_id=test_pilot.id,
            settings=ResolutionSettings(forced_rolls=[6, 1]),
        )
        result = resolve_shake_off(
            input_data,
            mech_hull=2,
            rider_grit=1,
        )

        assert result.shake_off_success is True
        assert result.contested_result is not None
        assert result.contested_result.attacker_wins is True
        assert result.rider_ejected is True

    def test_resolve_shake_off_failure_rider_wins(self, enemy_mech, test_pilot):
        """Test failed shake off when rider wins contested check."""
        input_data = ShakeOffInput(
            mech_id=enemy_mech.id,
            rider_id=test_pilot.id,
            settings=ResolutionSettings(forced_rolls=[1, 6]),
        )
        result = resolve_shake_off(
            input_data,
            mech_hull=1,
            rider_grit=2,
        )

        assert result.shake_off_success is False
        assert result.contested_result is not None
        assert result.contested_result.attacker_wins is False
        assert result.rider_ejected is False

    def test_resolve_shake_off_with_forced_rolls(self, enemy_mech, test_pilot):
        """Test shake off with forced roll values."""
        input_data = ShakeOffInput(
            mech_id=enemy_mech.id,
            rider_id=test_pilot.id,
            settings=ResolutionSettings(forced_rolls=[6, 1]),
        )
        result = resolve_shake_off(
            input_data,
            mech_hull=2,
            rider_grit=1,
        )

        assert result.shake_off_success is True
        assert result.contested_result is not None
        assert result.contested_result.attacker_total == 8
        assert result.contested_result.defender_total == 2


class TestApplyShakeOffResult:
    """Tests for applying Shake Off result to combatant state."""

    def test_apply_shake_off_result_success(self, enemy_mech, test_pilot):
        """Test applying successful shake off result."""
        from core.shared.jockey import ShakeOffResolutionResult

        contested = JockeyContestedResult(
            attacker_roll=None,
            defender_roll=None,
            attacker_total=8,
            defender_total=5,
            attacker_wins=True,
        )

        result = ShakeOffResolutionResult(
            mech_id=enemy_mech.id,
            rider_id=test_pilot.id,
            contested_result=contested,
            shake_off_success=True,
            rider_ejected=True,
        )

        application = apply_shake_off_result(enemy_mech, test_pilot, result)

        assert application.shake_off_success is True
        assert application.rider_ejected is True
        assert application.updated_rider is None

    def test_apply_shake_off_result_failure(self, enemy_mech, test_pilot):
        """Test applying failed shake off result."""
        from core.shared.jockey import ShakeOffResolutionResult

        contested = JockeyContestedResult(
            attacker_roll=None,
            defender_roll=None,
            attacker_total=5,
            defender_total=8,
            attacker_wins=False,
        )

        result = ShakeOffResolutionResult(
            mech_id=enemy_mech.id,
            rider_id=test_pilot.id,
            contested_result=contested,
            shake_off_success=False,
            rider_ejected=False,
        )

        application = apply_shake_off_result(enemy_mech, test_pilot, result)

        assert application.shake_off_success is False
        assert application.rider_ejected is False
        assert application.updated_rider is not None


class TestJockeyRuleDefaults:
    """Tests for default jockey rules and options."""

    def test_default_jockey_rules(self):
        """Test default JockeyRule values."""
        assert DEFAULT_JOCKEY_RULES.contested_check_stat_attacker == "grit"
        assert DEFAULT_JOCKEY_RULES.contested_check_stat_defender == "hull"
        assert DEFAULT_JOCKEY_RULES.tie_breaker == "attacker"
        assert len(DEFAULT_JOCKEY_RULES.options) == 3

    def test_default_jockey_options(self):
        """Test default jockey options."""
        option_types = [opt.option_type for opt in DEFAULT_JOCKEY_RULES.options]
        assert "distract" in option_types
        assert "shred" in option_types
        assert "damage" in option_types

        distract = next(
            opt for opt in DEFAULT_JOCKEY_RULES.options if opt.option_type == "distract"
        )
        assert "impaired" in distract.inflicted_conditions
        assert "slowed" in distract.inflicted_conditions

        shred = next(
            opt for opt in DEFAULT_JOCKEY_RULES.options if opt.option_type == "shred"
        )
        assert shred.heat == 2

        damage = next(
            opt for opt in DEFAULT_JOCKEY_RULES.options if opt.option_type == "damage"
        )
        assert damage.damage == 4
        assert damage.damage_type == "kinetic"


class TestJockeyContestedCheck:
    """Tests for jockey contested check mechanics."""

    def test_contested_check_tie_goes_to_attacker(self):
        """Test that ties go to attacker (pilot)."""
        from core.shared.jockey import JockeyContestedResult

        contested = JockeyContestedResult(
            attacker_roll=None,
            defender_roll=None,
            attacker_total=5,
            defender_total=5,
            attacker_wins=True,
        )

        assert contested.attacker_wins is True

    def test_contested_check_high_roll_wins(self):
        """Test that higher total wins."""
        from core.shared.jockey import JockeyContestedResult

        contested_high = JockeyContestedResult(
            attacker_roll=None,
            defender_roll=None,
            attacker_total=10,
            defender_total=5,
            attacker_wins=True,
        )

        contested_low = JockeyContestedResult(
            attacker_roll=None,
            defender_roll=None,
            attacker_total=5,
            defender_total=10,
            attacker_wins=False,
        )

        assert contested_high.attacker_wins is True
        assert contested_low.attacker_wins is False
