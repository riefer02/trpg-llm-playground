"""Tests for mission cadence, downtime actions, and reserves."""


from core.pilot.mission import (
    validate_downtime_plan,
    DowntimePlan,
    DowntimeActionUse,
    ReserveEntry,
    MissionCadenceRules,
    DOWNTIME_ACTIONS_BY_ID,
    DOWNTIME_ACTION_DEFINITIONS,
    roll_for_reserve,
    NARRATIVE_RESERVES,
    MECH_RESERVES,
    TACTICAL_RESERVES,
    OrganizationState,
    _get_roll_tier,
)


def _assert_valid(validation) -> None:
    assert validation.valid, f"Expected valid, got issues: {validation.issues}"


def _assert_invalid(validation) -> None:
    assert not validation.valid, "Expected invalid"


class TestReserves:
    """Tests for reserve tables and lookup."""

    def test_narrative_reserves_count(self) -> None:
        assert len(NARRATIVE_RESERVES) == 10

    def test_mech_reserves_count(self) -> None:
        assert len(MECH_RESERVES) == 10

    def test_tactical_reserves_count(self) -> None:
        assert len(TACTICAL_RESERVES) == 10

    def test_narrative_reserve_lookup_roll_1(self) -> None:
        reserve = roll_for_reserve(1, "narrative")
        assert reserve is not None
        assert reserve.id == "narrative_access"

    def test_narrative_reserve_lookup_roll_3(self) -> None:
        reserve = roll_for_reserve(3, "narrative")
        assert reserve is not None
        assert reserve.id == "narrative_backing"

    def test_narrative_reserve_lookup_roll_20(self) -> None:
        reserve = roll_for_reserve(20, "narrative")
        assert reserve is not None
        assert reserve.id == "narrative_knowledge"

    def test_mech_reserve_lookup(self) -> None:
        reserve = roll_for_reserve(7, "mech")
        assert reserve is not None
        assert reserve.id == "mech_extra_repairs"

    def test_tactical_reserve_lookup(self) -> None:
        reserve = roll_for_reserve(15, "tactical")
        assert reserve is not None
        assert reserve.id == "tactical_ambush"

    def test_invalid_roll_returns_none(self) -> None:
        reserve = roll_for_reserve(99, "narrative")
        assert reserve is None


class TestDowntimeActions:
    """Tests for downtime action definitions."""

    def test_action_count(self) -> None:
        assert len(DOWNTIME_ACTION_DEFINITIONS) == 9

    def test_power_at_a_cost_exists(self) -> None:
        action = DOWNTIME_ACTIONS_BY_ID.get("power_at_a_cost")
        assert action is not None
        assert action.category == "tradeoff"

    def test_get_organized_exists(self) -> None:
        action = DOWNTIME_ACTIONS_BY_ID.get("get_organized")
        assert action is not None
        assert action.category == "project"
        assert action.grants_reserve is True

    def test_get_focused_exists(self) -> None:
        action = DOWNTIME_ACTIONS_BY_ID.get("get_focused")
        assert action is not None
        assert action.requires_skill_check is False

    def test_gather_information_exists(self) -> None:
        action = DOWNTIME_ACTIONS_BY_ID.get("gather_information")
        assert action is not None
        assert action.category == "intel"

    def test_get_connected_exists(self) -> None:
        action = DOWNTIME_ACTIONS_BY_ID.get("get_connected")
        assert action is not None
        assert action.category == "contact"

    def test_scrounge_and_barter_exists(self) -> None:
        action = DOWNTIME_ACTIONS_BY_ID.get("scrounge_and_barter")
        assert action is not None
        assert action.category == "resource"


class TestRollTier:
    """Tests for roll tier determination."""

    def test_failure_tier_9(self) -> None:
        assert _get_roll_tier(1) == "failure"
        assert _get_roll_tier(9) == "failure"

    def test_partial_tier_10_19(self) -> None:
        assert _get_roll_tier(10) == "partial"
        assert _get_roll_tier(15) == "partial"
        assert _get_roll_tier(19) == "partial"

    def test_success_tier_20(self) -> None:
        assert _get_roll_tier(20) == "success"


class TestMissionCadenceRules:
    """Tests for mission cadence defaults."""

    def test_default_downtime_actions(self) -> None:
        rules = MissionCadenceRules()
        assert rules.downtime_actions_per_pilot == 1
        assert rules.downtime_actions_long_session == 2

    def test_long_session_allows_more_actions(self) -> None:
        plan = DowntimePlan(pilot_id="test", actions=[])
        rules = MissionCadenceRules()
        result = validate_downtime_plan(plan, rules=rules, is_long_session=False)
        assert result.valid

        result_long = validate_downtime_plan(plan, rules=rules, is_long_session=True)
        assert result_long.valid


class TestDowntimeValidation:
    """Tests for downtime plan validation."""

    def test_empty_plan_valid(self) -> None:
        plan = DowntimePlan(pilot_id="test")
        result = validate_downtime_plan(plan)
        _assert_valid(result)

    def test_unknown_action_invalid(self) -> None:
        plan = DowntimePlan(
            pilot_id="test",
            actions=[DowntimeActionUse(action_id="fake_action", outcome="info")],
        )
        result = validate_downtime_plan(plan)
        _assert_invalid(result)
        error_codes = [i.code for i in result.issues]
        assert "unknown_downtime_action" in error_codes

    def test_too_many_actions_invalid(self) -> None:
        actions = [
            DowntimeActionUse(action_id="get_a_damn_drink", outcome="recovery")
            for _ in range(10)
        ]
        plan = DowntimePlan(pilot_id="test", actions=actions)
        result = validate_downtime_plan(plan)
        _assert_invalid(result)
        error_codes = [i.code for i in result.issues]
        assert "too_many_downtime_actions" in error_codes

    def test_action_with_roll_requires_roll_result(self) -> None:
        plan = DowntimePlan(
            pilot_id="test",
            actions=[
                DowntimeActionUse(
                    action_id="get_connected",
                    outcome="contact",
                )
            ],
        )
        result = validate_downtime_plan(plan)
        _assert_invalid(result)
        error_codes = [i.code for i in result.issues]
        assert "missing_roll_result" in error_codes

    def test_action_with_roll_valid_with_result(self) -> None:
        plan = DowntimePlan(
            pilot_id="test",
            actions=[
                DowntimeActionUse(
                    action_id="get_connected",
                    outcome="contact",
                    roll_result=15,
                    reserve=ReserveEntry(
                        id="test_reserve",
                        name="Test Reserve",
                        reserve_type="narrative",
                    ),
                ).with_roll_tier()
            ],
        )
        result = validate_downtime_plan(plan)
        _assert_valid(result)


class TestOrganizationState:
    """Tests for Get Organized organization tracking."""

    def test_initial_state(self) -> None:
        state = OrganizationState.initial_state()
        assert state.efficiency == 2
        assert state.influence == 0

    def test_gain_partial(self) -> None:
        state = OrganizationState.initial_state()
        new_state = state.gain(is_success=False)
        assert new_state.efficiency == 3
        assert new_state.influence == 1

    def test_gain_success(self) -> None:
        state = OrganizationState.initial_state()
        new_state = state.gain(is_success=True)
        assert new_state.efficiency == 4
        assert new_state.influence == 2

    def test_degrade(self) -> None:
        state = OrganizationState(efficiency=4, influence=4)
        new_state = state.degrade()
        assert new_state.efficiency == 2
        assert new_state.influence == 2

    def test_minimum_capped(self) -> None:
        state = OrganizationState(efficiency=1, influence=1)
        new_state = state.degrade()
        assert new_state.efficiency == 0
        assert new_state.influence == 0

    def test_maximum_capped(self) -> None:
        state = OrganizationState(efficiency=5, influence=5)
        new_state = state.gain(is_success=True)
        assert new_state.efficiency == 6
        assert new_state.influence == 6


class TestReserveEntry:
    """Tests for reserve entries."""

    def test_basic_reserve(self) -> None:
        reserve = ReserveEntry(
            id="test",
            name="Test Reserve",
            reserve_type="narrative",
        )
        assert reserve.id == "test"
        assert reserve.reserve_type == "narrative"
        assert reserve.uses_remaining == 1
        assert reserve.shared is True

    def test_reserve_with_effect(self) -> None:
        from core.shared.effects import MechanicalEffect, StatModifier

        reserve = ReserveEntry(
            id="test",
            name="Test Reserve",
            reserve_type="mech",
            mechanical_effect=MechanicalEffect(
                stat_mods=[StatModifier(stat="hp", value=2)]
            ),
        )
        assert reserve.mechanical_effect is not None


class TestDowntimeActionOutcomes:
    """Tests for downtime action roll-tier outcomes."""

    def test_get_connected_has_outcomes(self) -> None:
        action = DOWNTIME_ACTIONS_BY_ID.get("get_connected")
        assert action is not None
        assert len(action.roll_tier_outcomes) == 3

    def test_get_organized_has_outcomes(self) -> None:
        action = DOWNTIME_ACTIONS_BY_ID.get("get_organized")
        assert action is not None
        assert len(action.roll_tier_outcomes) == 3

    def test_outcome_tiers_complete(self) -> None:
        for action in DOWNTIME_ACTION_DEFINITIONS:
            if action.roll_tier_outcomes:
                tiers = {o.tier for o in action.roll_tier_outcomes}
                assert "success" in tiers, f"{action.id} missing success outcome"


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing code."""

    def test_old_downtime_action_still_imports(self) -> None:
        from core.pilot.mission import (
            DowntimeActionUse,
            DowntimePlan,
            ReserveEntry,
            validate_downtime_plan,
        )

        assert DowntimeActionUse is not None
        assert DowntimePlan is not None
        assert ReserveEntry is not None
        assert validate_downtime_plan is not None
