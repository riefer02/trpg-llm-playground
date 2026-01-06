"""Tests for AI/NHP control resolution primitives."""

import pytest
from core.shared.ai import (
    resolve_cede_control,
    resolve_cede_control_remote,
    resolve_unshackle_check,
    resolve_unshackled_behavior,
    resolve_remount,
    apply_cede_control,
    apply_cede_control_remote,
    apply_unshackle,
    apply_remount,
    CedeControlInput,
    CedeControlRemoteInput,
    UnshackleCheckInput,
    UnshackledBehaviorInput,
    RemountInput,
    AIRule,
    DEFAULT_AI_RULES,
    AIType,
    AIControlState,
    NHPBehaviorType,
)
from core.mech.combat_state import (
    CombatantState,
    CombatStats,
    CombatResources,
)


@pytest.fixture
def mech_combatant() -> CombatantState:
    """Create a test mech combatant."""
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
            hp_current=8,
            heat_current=0,
            heat_cap=10,
            structure_current=4,
            stress_current=0,
        ),
    )


@pytest.fixture
def nhp_mech_combatant() -> CombatantState:
    """Create a test mech combatant with NHP installed."""
    return CombatantState(
        id="nhp_mech",
        name="NHP Mech",
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
            hp_current=8,
            heat_current=0,
            heat_cap=10,
            structure_current=4,
            stress_current=0,
        ),
        ai_type="nhp",
    )


@pytest.fixture
def compcon_mech_combatant() -> CombatantState:
    """Create a test mech combatant with Comp/Con installed."""
    return CombatantState(
        id="compcon_mech",
        name="Comp/Con Mech",
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
            hp_current=8,
            heat_current=0,
            heat_cap=10,
            structure_current=4,
            stress_current=0,
        ),
        ai_type="compcon",
    )


@pytest.fixture
def pilot_combatant() -> CombatantState:
    """Create a test pilot combatant."""
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
        resources=CombatResources(
            hp_current=6,
        ),
    )


class TestResolveCedeControl:
    """Tests for resolve_cede_control function."""

    def test_cede_control_success(self, mech_combatant: CombatantState):
        """Test successful cede control when mech has AI property."""
        input_data = CedeControlInput(actor_id=mech_combatant.id)
        result = resolve_cede_control(input_data, has_ai_property=True)

        assert result.control_ceded is True
        assert result.cede_turns == 1
        assert result.validation_errors == []

    def test_cede_control_no_ai_property(self, mech_combatant: CombatantState):
        """Test cede control fails when mech lacks AI property."""
        input_data = CedeControlInput(actor_id=mech_combatant.id)
        result = resolve_cede_control(input_data, has_ai_property=False)

        assert result.control_ceded is False
        assert len(result.validation_errors) > 0

    def test_cede_control_with_custom_rules(self, mech_combatant: CombatantState):
        """Test cede control with custom rules."""
        custom_rules = AIRule(cede_turns_duration=2)
        input_data = CedeControlInput(actor_id=mech_combatant.id, rules=custom_rules)
        result = resolve_cede_control(input_data, has_ai_property=True)

        assert result.control_ceded is True
        assert result.cede_turns == 2


class TestResolveCedeControlRemote:
    """Tests for resolve_cede_control_remote function."""

    def test_remote_cede_success(self, mech_combatant: CombatantState):
        """Test successful remote cede control."""
        input_data = CedeControlRemoteInput(actor_id=mech_combatant.id)
        result = resolve_cede_control_remote(input_data, has_ai_property=True)

        assert result.control_ceded is True
        assert result.pilot_exited is True
        assert result.exit_position == "adjacent"
        assert result.validation_errors == []

    def test_remote_cede_no_ai_property(self, mech_combatant: CombatantState):
        """Test remote cede fails when mech lacks AI property."""
        input_data = CedeControlRemoteInput(actor_id=mech_combatant.id)
        result = resolve_cede_control_remote(input_data, has_ai_property=False)

        assert result.control_ceded is False
        assert result.pilot_exited is False


class TestResolveUnshackleCheck:
    """Tests for resolve_unshackle_check function."""

    def test_unshackle_on_structure_d20_1(self, nhp_mech_combatant: CombatantState):
        """Test unshackle occurs when d20 rolls 1 on structure check."""
        input_data = UnshackleCheckInput(
            actor_id=nhp_mech_combatant.id,
            check_type="structure",
            force_roll=1,
        )
        result = resolve_unshackle_check(input_data, has_nhp=True)

        assert result.d20_roll == 1
        assert result.unshackle_occurred is True
        assert result.nhps_affected == 1

    def test_unshackle_on_structure_d20_20(self, nhp_mech_combatant: CombatantState):
        """Test unshackle does NOT occur when d20 rolls 20 on structure check."""
        input_data = UnshackleCheckInput(
            actor_id=nhp_mech_combatant.id,
            check_type="structure",
            force_roll=20,
        )
        result = resolve_unshackle_check(input_data, has_nhp=True)

        assert result.d20_roll == 20
        assert result.unshackle_occurred is False
        assert result.nhps_affected == 0

    def test_unshackle_on_overheat_d20_1(self, nhp_mech_combatant: CombatantState):
        """Test unshackle occurs when d20 rolls 1 on overheat check."""
        input_data = UnshackleCheckInput(
            actor_id=nhp_mech_combatant.id,
            check_type="overheat",
            force_roll=1,
        )
        result = resolve_unshackle_check(input_data, has_nhp=True)

        assert result.unshackle_occurred is True
        assert result.check_type == "overheat"

    def test_no_nhp_no_unshackle(self, mech_combatant: CombatantState):
        """Test unshackle does NOT occur when mech has no NHP."""
        input_data = UnshackleCheckInput(
            actor_id=mech_combatant.id,
            check_type="structure",
            force_roll=1,
        )
        result = resolve_unshackle_check(input_data, has_nhp=False)

        assert result.unshackle_occurred is False
        assert result.nhps_affected == 0

    def test_compcon_no_unshackle(self, compcon_mech_combatant: CombatantState):
        """Test unshackle does NOT occur for Comp/Con (only NHPs unshackle)."""
        input_data = UnshackleCheckInput(
            actor_id=compcon_mech_combatant.id,
            check_type="structure",
            force_roll=1,
        )
        result = resolve_unshackle_check(input_data, has_nhp=False)

        assert result.unshackle_occurred is False

    @pytest.mark.parametrize(
        "d20_roll,should_unshackle",
        [
            (1, True),
            (2, False),
            (5, False),
            (10, False),
            (15, False),
            (20, False),
        ],
    )
    def test_unshackle_threshold(
        self,
        nhp_mech_combatant: CombatantState,
        d20_roll: int,
        should_unshackle: bool,
    ):
        """Test unshackle only occurs on d20 = 1."""
        input_data = UnshackleCheckInput(
            actor_id=nhp_mech_combatant.id,
            check_type="structure",
            force_roll=d20_roll,
        )
        result = resolve_unshackle_check(input_data, has_nhp=True)

        assert result.unshackle_occurred is should_unshackle


class TestResolveUnshackledBehavior:
    """Tests for resolve_unshackled_behavior function."""

    def test_behavior_selection(self, nhp_mech_combatant: CombatantState):
        """Test behavior is selected from valid options."""
        input_data = UnshackledBehaviorInput(actor_id=nhp_mech_combatant.id)
        result = resolve_unshackled_behavior(input_data)

        assert result.behavior in [
            "ignore_pilot",
            "overrule_pilot",
            "illogical",
            "remove_pilot",
        ]
        assert result.behavior_config is not None
        assert result.behavior_config.behavior == result.behavior

    def test_remove_pilot_has_mode(self, nhp_mech_combatant: CombatantState):
        """Test remove_pilot behavior has mode configured."""
        input_data = UnshackledBehaviorInput(actor_id=nhp_mech_combatant.id)
        result = resolve_unshackled_behavior(input_data)

        if result.behavior == "remove_pilot":
            assert result.remove_pilot_mode is not None


class TestResolveRemount:
    """Tests for resolve_remount function."""

    def test_remount_success_adjacent(self, pilot_combatant: CombatantState):
        """Test successful remount when pilot is adjacent."""
        input_data = RemountInput(
            pilot_id=pilot_combatant.id,
            mech_id="test_mech",
            is_adjacent=True,
        )
        result = resolve_remount(input_data, pilot_in_cede_remote=True)

        assert result.remount_success is True
        assert result.control_restored is True

    def test_remount_not_adjacent(self, pilot_combatant: CombatantState):
        """Test remount fails when pilot is not adjacent."""
        input_data = RemountInput(
            pilot_id=pilot_combatant.id,
            mech_id="test_mech",
            is_adjacent=False,
        )
        result = resolve_remount(input_data, pilot_in_cede_remote=True)

        assert result.remount_success is False
        assert result.control_restored is False

    def test_remount_not_in_cede_remote(self, pilot_combatant: CombatantState):
        """Test remount fails when pilot is not in cede_remote state."""
        input_data = RemountInput(
            pilot_id=pilot_combatant.id,
            mech_id="test_mech",
            is_adjacent=True,
        )
        result = resolve_remount(input_data, pilot_in_cede_remote=False)

        assert result.remount_success is False


class TestApplyCedeControl:
    """Tests for apply_cede_control function."""

    def test_apply_cede_control(self, mech_combatant: CombatantState):
        """Test applying cede control updates combatant state."""
        input_data = CedeControlInput(actor_id=mech_combatant.id)
        resolve_result = resolve_cede_control(input_data, has_ai_property=True)

        app_result = apply_cede_control(mech_combatant, resolve_result)

        assert app_result.control_state_changed is True
        assert app_result.updated_combatant.ai_control_state == "cede"
        assert app_result.cede_turns_remaining == 1

    def test_apply_cede_control_failure(self, mech_combatant: CombatantState):
        """Test applying failed cede control does not change state."""
        input_data = CedeControlInput(actor_id=mech_combatant.id)
        resolve_result = resolve_cede_control(input_data, has_ai_property=False)

        app_result = apply_cede_control(mech_combatant, resolve_result)

        assert app_result.control_state_changed is False


class TestApplyCedeControlRemote:
    """Tests for apply_cede_control_remote function."""

    def test_apply_remote_cede_control(
        self, mech_combatant: CombatantState, pilot_combatant: CombatantState
    ):
        """Test applying remote cede control updates both combatants."""
        input_data = CedeControlRemoteInput(actor_id=mech_combatant.id)
        resolve_result = resolve_cede_control_remote(input_data, has_ai_property=True)

        app_result = apply_cede_control_remote(
            mech_combatant, pilot_combatant, resolve_result
        )

        assert app_result.control_state_changed is True
        assert app_result.pilot_exited is True
        assert app_result.updated_combatant.ai_control_state == "cede_remote"


class TestApplyUnshackle:
    """Tests for apply_unshackle function."""

    def test_apply_unshackle_success(self, nhp_mech_combatant: CombatantState):
        """Test applying successful unshackle updates combatant state."""
        unshackle_input = UnshackleCheckInput(
            actor_id=nhp_mech_combatant.id,
            check_type="structure",
            force_roll=1,
        )
        unshackle_result = resolve_unshackle_check(unshackle_input, has_nhp=True)

        behavior_input = UnshackledBehaviorInput(actor_id=nhp_mech_combatant.id)
        behavior_result = resolve_unshackled_behavior(behavior_input)

        app_result = apply_unshackle(
            nhp_mech_combatant, unshackle_result, behavior_result
        )

        assert app_result.unshackled is True
        assert app_result.control_state == "unshackled"
        assert app_result.nhp_behavior in [
            "ignore_pilot",
            "overrule_pilot",
            "illogical",
            "remove_pilot",
        ]

    def test_apply_unshackle_failure(self, nhp_mech_combatant: CombatantState):
        """Test applying failed unshackle does not change state."""
        unshackle_input = UnshackleCheckInput(
            actor_id=nhp_mech_combatant.id,
            check_type="structure",
            force_roll=20,
        )
        unshackle_result = resolve_unshackle_check(unshackle_input, has_nhp=True)

        behavior_input = UnshackledBehaviorInput(actor_id=nhp_mech_combatant.id)
        behavior_result = resolve_unshackled_behavior(behavior_input)

        app_result = apply_unshackle(
            nhp_mech_combatant, unshackle_result, behavior_result
        )

        assert app_result.unshackled is False

    def test_unshackle_adds_status(self, nhp_mech_combatant: CombatantState):
        """Test unshackled combatant gets unshackled status."""
        unshackle_input = UnshackleCheckInput(
            actor_id=nhp_mech_combatant.id,
            check_type="structure",
            force_roll=1,
        )
        unshackle_result = resolve_unshackle_check(unshackle_input, has_nhp=True)

        behavior_input = UnshackledBehaviorInput(actor_id=nhp_mech_combatant.id)
        behavior_result = resolve_unshackled_behavior(behavior_input)

        app_result = apply_unshackle(
            nhp_mech_combatant, unshackle_result, behavior_result
        )

        assert "unshackled" in app_result.updated_combatant.statuses

    def test_remove_pilot_ejects_pilot(self, nhp_mech_combatant: CombatantState):
        """Test remove_pilot behavior ejects pilot."""
        unshackle_input = UnshackleCheckInput(
            actor_id=nhp_mech_combatant.id,
            check_type="structure",
            force_roll=1,
        )
        unshackle_result = resolve_unshackle_check(unshackle_input, has_nhp=True)

        while True:
            behavior_input = UnshackledBehaviorInput(actor_id=nhp_mech_combatant.id)
            behavior_result = resolve_unshackled_behavior(behavior_input)
            if behavior_result.behavior == "remove_pilot":
                break

        app_result = apply_unshackle(
            nhp_mech_combatant, unshackle_result, behavior_result
        )

        assert app_result.pilot_ejected is True


class TestApplyRemount:
    """Tests for apply_remount function."""

    def test_apply_remount_success(
        self, pilot_combatant: CombatantState, mech_combatant: CombatantState
    ):
        """Test applying successful remount restores control."""
        remount_input = RemountInput(
            pilot_id=pilot_combatant.id,
            mech_id=mech_combatant.id,
            is_adjacent=True,
        )
        remount_result = resolve_remount(remount_input, pilot_in_cede_remote=True)

        app_result = apply_remount(pilot_combatant, mech_combatant, remount_result)

        assert app_result.remount_success is True
        assert app_result.control_restored is True
        assert app_result.updated_mech.ai_control_state == "pilot"

    def test_apply_remount_failure(
        self, pilot_combatant: CombatantState, mech_combatant: CombatantState
    ):
        """Test applying failed remount does not change state."""
        remount_input = RemountInput(
            pilot_id=pilot_combatant.id,
            mech_id=mech_combatant.id,
            is_adjacent=False,
        )
        remount_result = resolve_remount(remount_input, pilot_in_cede_remote=True)

        app_result = apply_remount(pilot_combatant, mech_combatant, remount_result)

        assert app_result.remount_success is False


class TestAIRuleConfiguration:
    """Tests for AI rule configuration."""

    def test_default_rules(self):
        """Test default AI rules have expected values."""
        assert DEFAULT_AI_RULES.unshackle_threshold == 1
        assert DEFAULT_AI_RULES.unshackle_on_structure_check is True
        assert DEFAULT_AI_RULES.unshackle_on_overheat_check is True
        assert DEFAULT_AI_RULES.cede_turns_duration == 1

    def test_custom_rules(self):
        """Test custom AI rules can override defaults."""
        custom_rules = AIRule(
            unshackle_threshold=1,
            cede_turns_duration=2,
            behavior_weights={
                "ignore_pilot": 0.5,
                "overrule_pilot": 0.2,
                "illogical": 0.2,
                "remove_pilot": 0.1,
            },
        )

        assert custom_rules.cede_turns_duration == 2
        assert custom_rules.behavior_weights["ignore_pilot"] == 0.5
        assert custom_rules.behavior_weights["remove_pilot"] == 0.1


class TestAITypeAndControlState:
    """Tests for AI type and control state literals."""

    def test_ai_type_literal(self):
        """Test AIType accepts valid values."""
        valid_types: list[AIType] = ["compcon", "nhp"]
        assert all(t in valid_types for t in valid_types)

    def test_control_state_literal(self):
        """Test AIControlState accepts valid values."""
        valid_states: list[AIControlState] = [
            "pilot",
            "cede",
            "cede_remote",
            "unshackled",
        ]
        assert all(s in valid_states for s in valid_states)

    def test_behavior_type_literal(self):
        """Test NHPBehaviorType accepts valid values."""
        valid_behaviors: list[NHPBehaviorType] = [
            "ignore_pilot",
            "overrule_pilot",
            "illogical",
            "remove_pilot",
        ]
        assert all(b in valid_behaviors for b in valid_behaviors)
