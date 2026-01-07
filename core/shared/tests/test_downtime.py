"""Tests for downtime actions and reserves system."""

import pytest
from core.shared.downtime import (
    ReserveType,
    NarrativeReserveType,
    MechReserveType,
    TacticalReserveType,
    Reserve,
    DowntimeOutcome,
    DowntimeAction,
    PowerAtACost,
    BuySomeTime,
    GetADamnDrink,
    GetCreative,
    GetFocused,
    GetOrganized,
    GatherInformation,
    GetConnected,
    ScroungeAndBarter,
    resolve_downtime_action,
    get_downtime_action,
    list_downtime_actions,
    DOWNTIME_ACTIONS,
)


class TestReserveTypeEnums:
    """Tests for reserve type enums."""

    def test_reserve_type_values(self):
        """Verify all reserve type values are available."""
        expected = {"narrative", "mech", "tactical"}
        available = set(ReserveType.__args__)
        assert expected == available

    def test_narrative_reserve_type_values(self):
        """Verify all narrative reserve types are available."""
        expected = {
            "access",
            "backing",
            "supplies",
            "disguise",
            "diversion",
            "blackmail",
            "reputation",
            "safe_harbor",
            "tracking",
            "knowledge",
        }
        available = set(NarrativeReserveType.__args__)
        assert expected == available

    def test_mech_reserve_type_values(self):
        """Verify all mech reserve types are available."""
        expected = {
            "ammo",
            "rented_gear",
            "extra_repairs",
            "core_battery",
            "deployable_shield",
            "redundant_repair",
            "systems_reinforcement",
            "smart_ammo",
            "boosted_servos",
            "jump_jets",
        }
        available = set(MechReserveType.__args__)
        assert expected == available

    def test_tactical_reserve_type_values(self):
        """Verify all tactical reserve types are available."""
        expected = {"scouting", "vehicle", "reinforcements"}
        available = set(TacticalReserveType.__args__)
        assert expected == available


class TestReserve:
    """Tests for Reserve model."""

    def test_create_basic_reserve(self):
        """Create a basic reserve."""
        reserve = Reserve(
            id="test_reserve",
            reserve_type="narrative",
            specific_type="supplies",
            description="Test supplies",
        )
        assert reserve.id == "test_reserve"
        assert reserve.reserve_type == "narrative"
        assert reserve.specific_type == "supplies"
        assert reserve.description == "Test supplies"
        assert reserve.quantity == 1
        assert reserve.mission_scoped is True

    def test_create_reserve_with_quantity(self):
        """Create a reserve with specific quantity."""
        reserve = Reserve(
            id="ammo_reserve",
            reserve_type="mech",
            specific_type="ammo",
            description="Extra ammo",
            quantity=3,
        )
        assert reserve.quantity == 3

    def test_reserve_not_mission_scoped(self):
        """Create a non-mission-scoped reserve."""
        reserve = Reserve(
            id="perm_rep",
            reserve_type="narrative",
            specific_type="reputation",
            description="Permanent reputation",
            mission_scoped=False,
        )
        assert reserve.mission_scoped is False


class TestDowntimeOutcome:
    """Tests for DowntimeOutcome model."""

    def test_create_failure_outcome(self):
        """Create a failure outcome."""
        outcome = DowntimeOutcome(
            tier="failure",
            consequences=["Action failed completely"],
        )
        assert outcome.tier == "failure"
        assert outcome.reserves_earned == []
        assert outcome.consequences == ["Action failed completely"]

    def test_create_success_outcome(self):
        """Create a success outcome with reserves."""
        reserve = Reserve(
            id="test",
            reserve_type="narrative",
            specific_type="supplies",
            description="Test",
        )
        outcome = DowntimeOutcome(
            tier="success",
            reserves_earned=[reserve],
            consequences=["Action succeeded"],
        )
        assert outcome.tier == "success"
        assert len(outcome.reserves_earned) == 1

    def test_create_outcome_with_state_changes(self):
        """Create an outcome with state changes."""
        outcome = DowntimeOutcome(
            tier="mixed",
            state_changes={"efficiency": 2, "influence": 0},
            consequences=["Organization is stable"],
        )
        assert outcome.state_changes["efficiency"] == 2

    def test_create_outcome_with_notes(self):
        """Create an outcome with notes."""
        outcome = DowntimeOutcome(
            tier="success",
            notes="Excellent result achieved",
        )
        assert outcome.notes == "Excellent result achieved"


class TestDowntimeAction:
    """Tests for base DowntimeAction class."""

    def test_action_outcome_thresholds(self):
        """Verify outcome tiers are correctly determined."""
        action = DowntimeAction(
            id="test_action",
            name="Test Action",
            description="A test action",
        )

        failure = action.get_outcome(roll_result=5)
        assert failure.tier == "failure"

        mixed = action.get_outcome(roll_result=15)
        assert mixed.tier == "mixed"

        success = action.get_outcome(roll_result=20)
        assert success.tier == "success"

        exceptional = action.get_outcome(roll_result=25)
        assert exceptional.tier == "success"

    def test_action_with_modifiers(self):
        """Verify modifiers affect outcome thresholds."""
        action = DowntimeAction(
            id="test_action",
            name="Test Action",
            description="A test action",
        )

        failure = action.get_outcome(roll_result=8, modifiers=0)
        assert failure.tier == "failure"

        mixed = action.get_outcome(roll_result=8, modifiers=5)
        assert mixed.tier == "mixed"

        success = action.get_outcome(roll_result=8, modifiers=15)
        assert success.tier == "success"

    def test_action_with_difficulty_modifier(self):
        """Verify difficulty modifiers affect outcome thresholds."""
        action = DowntimeAction(
            id="test_action",
            name="Test Action",
            description="A test action",
        )

        mixed = action.get_outcome(roll_result=15, modifiers=0, difficulty_modifier=0)
        assert mixed.tier == "mixed"

        failure = action.get_outcome(
            roll_result=15, modifiers=0, difficulty_modifier=10
        )
        assert failure.tier == "failure"

    def test_action_default_skill_context(self):
        """Verify default skill context is 'general'."""
        action = DowntimeAction(
            id="test_action",
            name="Test Action",
            description="A test action",
        )
        assert action.skill_context == "general"


class TestPowerAtACost:
    """Tests for Power at a Cost action."""

    def test_action_properties(self):
        """Verify action has correct properties."""
        action = PowerAtACost()
        assert action.id == "power_at_a_cost"
        assert action.name == "Power at a Cost"
        assert "resource" in action.description.lower()

    def test_mixed_outcome(self):
        """Test mixed outcome gives reserves."""
        action = PowerAtACost()
        outcome = action.get_outcome(roll_result=15)

        assert outcome.tier == "mixed"
        assert len(outcome.reserves_earned) == 1
        assert outcome.reserves_earned[0].reserve_type == "narrative"
        assert outcome.reserves_earned[0].specific_type == "supplies"

    def test_success_outcome(self):
        """Test success outcome gives reserves."""
        action = PowerAtACost()
        outcome = action.get_outcome(roll_result=20)

        assert outcome.tier == "success"
        assert len(outcome.reserves_earned) == 1


class TestBuySomeTime:
    """Tests for Buy Some Time action."""

    def test_action_properties(self):
        """Verify action has correct properties."""
        action = BuySomeTime()
        assert action.id == "buy_some_time"
        assert action.name == "Buy Some Time"
        assert action.skill_context == "systems"

    def test_failure_outcome(self):
        """Test failure outcome notes reckoning catches up."""
        action = BuySomeTime()
        outcome = action.get_outcome(roll_result=5)

        assert outcome.tier == "failure"
        assert (
            "reckoning" in outcome.notes.lower() or "drastic" in outcome.notes.lower()
        )

    def test_mixed_outcome_gives_tactical_reserve(self):
        """Test mixed outcome gives tactical reserve."""
        action = BuySomeTime()
        outcome = action.get_outcome(roll_result=15)

        assert outcome.tier == "mixed"
        assert len(outcome.reserves_earned) == 1
        assert outcome.reserves_earned[0].reserve_type == "tactical"

    def test_success_outcome_gives_scouting_reserve(self):
        """Test success outcome gives scouting reserve."""
        action = BuySomeTime()
        outcome = action.get_outcome(roll_result=20)

        assert outcome.tier == "success"
        assert len(outcome.reserves_earned) == 1
        assert outcome.reserves_earned[0].specific_type == "scouting"


class TestGetADamnDrink:
    """Tests for Get a Damn Drink action."""

    def test_action_properties(self):
        """Verify action has correct properties."""
        action = GetADamnDrink()
        assert action.id == "get_a_damn_drink"
        assert action.name == "Get a Damn Drink"
        assert "drink" in action.description.lower()

    def test_failure_outcome(self):
        """Test failure outcome has gutter choice."""
        action = GetADamnDrink()
        outcome = action.get_outcome(roll_result=5)

        assert outcome.tier == "failure"
        assert "gutter" in outcome.notes.lower() or "dignity" in outcome.notes.lower()

    def test_mixed_outcome(self):
        """Test mixed outcome has gain/lose pattern."""
        action = GetADamnDrink()
        outcome = action.get_outcome(roll_result=15)

        assert outcome.tier == "mixed"
        assert len(outcome.reserves_earned) == 1
        assert outcome.reserves_earned[0].reserve_type == "narrative"
        assert len(outcome.consequences) > 0

    def test_success_outcome_gains_two_reserves(self):
        """Test success outcome gains two reserves, loses nothing."""
        action = GetADamnDrink()
        outcome = action.get_outcome(roll_result=20)

        assert outcome.tier == "success"
        assert len(outcome.reserves_earned) == 2


class TestGetCreative:
    """Tests for Get Creative action."""

    def test_action_properties(self):
        """Verify action has correct properties."""
        action = GetCreative()
        assert action.id == "get_creative"
        assert action.name == "Get Creative"
        assert action.skill_context == "systems"

    def test_failure_outcome(self):
        """Test failure outcome has no progress."""
        action = GetCreative()
        outcome = action.get_outcome(roll_result=5)

        assert outcome.tier == "failure"
        assert "progress" in outcome.notes.lower() or "project" in outcome.notes.lower()

    def test_mixed_outcome(self):
        """Test mixed outcome gives partial progress."""
        action = GetCreative()
        outcome = action.get_outcome(roll_result=15)

        assert outcome.tier == "mixed"
        assert len(outcome.reserves_earned) == 1
        assert "progress" in outcome.consequences[0].lower()

    def test_success_outcome_completes_project(self):
        """Test success outcome completes project."""
        action = GetCreative()
        outcome = action.get_outcome(roll_result=20)

        assert outcome.tier == "success"
        assert "finish" in outcome.notes.lower() or "project" in outcome.notes.lower()


class TestGetFocused:
    """Tests for Get Focused action."""

    def test_action_properties(self):
        """Verify action has correct properties."""
        action = GetFocused()
        assert action.id == "get_focused"
        assert action.name == "Get Focused"
        assert "skill" in action.description.lower()

    def test_failure_outcome(self):
        """Test failure outcome gives no trigger."""
        action = GetFocused()
        outcome = action.get_outcome(roll_result=5)

        assert outcome.tier == "failure"
        assert outcome.state_changes.get("trigger_bonus") is None

    def test_mixed_and_success_outcome(self):
        """Test mixed and success give trigger bonus."""
        action = GetFocused()

        mixed = action.get_outcome(roll_result=15)
        assert mixed.tier == "mixed"
        assert mixed.state_changes.get("trigger_bonus") == 2

        success = action.get_outcome(roll_result=20)
        assert success.tier == "success"
        assert success.state_changes.get("trigger_bonus") == 2


class TestGetOrganized:
    """Tests for Get Organized action."""

    def test_action_properties(self):
        """Verify action has correct properties."""
        action = GetOrganized()
        assert action.id == "get_organized"
        assert action.name == "Get Organized"
        assert "organization" in action.description.lower()

    def test_failure_outcome(self):
        """Test failure outcome reduces org stats."""
        action = GetOrganized()
        outcome = action.get_outcome(roll_result=5)

        assert outcome.tier == "failure"
        assert outcome.state_changes.get("efficiency_change") == -2
        assert outcome.state_changes.get("influence_change") == -2

    def test_mixed_outcome(self):
        """Test mixed outcome increases one stat."""
        action = GetOrganized()
        outcome = action.get_outcome(roll_result=15)

        assert outcome.tier == "mixed"
        assert outcome.state_changes.get("efficiency_change") == 2
        assert outcome.state_changes.get("influence_change") == 0

    def test_success_outcome(self):
        """Test success outcome increases both stats."""
        action = GetOrganized()
        outcome = action.get_outcome(roll_result=20)

        assert outcome.tier == "success"
        assert outcome.state_changes.get("efficiency_change") == 2
        assert outcome.state_changes.get("influence_change") == 2


class TestGatherInformation:
    """Tests for Gather Information action."""

    def test_action_properties(self):
        """Verify action has correct properties."""
        action = GatherInformation()
        assert action.id == "gather_information"
        assert action.name == "Gather Information"
        assert action.skill_context == "systems"

    def test_failure_outcome(self):
        """Test failure outcome gets info but with trouble."""
        action = GatherInformation()
        outcome = action.get_outcome(roll_result=5)

        assert outcome.tier == "failure"
        assert len(outcome.reserves_earned) == 1
        assert "trouble" in outcome.notes.lower()

    def test_mixed_outcome(self):
        """Test mixed outcome has complication choice."""
        action = GatherInformation()
        outcome = action.get_outcome(roll_result=15)

        assert outcome.tier == "mixed"
        assert len(outcome.consequences) > 0

    def test_success_outcome(self):
        """Test success outcome is clean."""
        action = GatherInformation()
        outcome = action.get_outcome(roll_result=20)

        assert outcome.tier == "success"
        assert len(outcome.consequences) == 0


class TestGetConnected:
    """Tests for Get Connected action."""

    def test_action_properties(self):
        """Verify action has correct properties."""
        action = GetConnected()
        assert action.id == "get_connected"
        assert action.name == "Get Connected"
        assert action.skill_context == "charm"

    def test_failure_outcome(self):
        """Test failure outcome requires immediate favor."""
        action = GetConnected()
        outcome = action.get_outcome(roll_result=5)

        assert outcome.tier == "failure"
        assert (
            "favor" in outcome.consequences[0].lower()
            or "immediately" in outcome.notes.lower()
        )

    def test_mixed_outcome(self):
        """Test mixed outcome has debt."""
        action = GetConnected()
        outcome = action.get_outcome(roll_result=15)

        assert outcome.tier == "mixed"
        assert "after" in outcome.consequences[0].lower()

    def test_success_outcome(self):
        """Test success outcome has no strings."""
        action = GetConnected()
        outcome = action.get_outcome(roll_result=20)

        assert outcome.tier == "success"
        assert "no strings" in outcome.notes.lower()


class TestScroungeAndBarter:
    """Tests for Scrounge and Barter action."""

    def test_action_properties(self):
        """Verify action has correct properties."""
        action = ScroungeAndBarter()
        assert action.id == "scrounge_and_barter"
        assert action.name == "Scrounge and Barter"
        assert "gear" in action.description.lower()

    def test_failure_outcome(self):
        """Test failure outcome has problem choice."""
        action = ScroungeAndBarter()
        outcome = action.get_outcome(roll_result=5)

        assert outcome.tier == "failure"
        assert len(outcome.consequences) > 0

    def test_mixed_outcome(self):
        """Test mixed outcome has trade choice."""
        action = ScroungeAndBarter()
        outcome = action.get_outcome(roll_result=15)

        assert outcome.tier == "mixed"
        assert len(outcome.consequences) > 0

    def test_success_outcome(self):
        """Test success outcome is clean."""
        action = ScroungeAndBarter()
        outcome = action.get_outcome(roll_result=20)

        assert outcome.tier == "success"
        assert len(outcome.consequences) == 0


class TestResolveDowntimeAction:
    """Tests for resolve_downtime_action helper."""

    def test_resolve_action(self):
        """Test resolving a downtime action."""
        action = PowerAtACost()
        outcome = resolve_downtime_action(
            action=action,
            roll_result=15,
            modifiers=0,
        )
        assert outcome.tier == "mixed"

    def test_resolve_with_difficulty(self):
        """Test resolving with difficulty modifier."""
        action = BuySomeTime()
        outcome = resolve_downtime_action(
            action=action,
            roll_result=12,
            modifiers=0,
            difficulty_modifier=5,
        )
        assert outcome.tier == "failure"


class TestGetDowntimeAction:
    """Tests for get_downtime_action helper."""

    def test_get_existing_action(self):
        """Test getting an existing action."""
        action = get_downtime_action("power_at_a_cost")
        assert action is not None
        assert action.id == "power_at_a_cost"

    def test_get_nonexistent_action(self):
        """Test getting nonexistent action returns None."""
        action = get_downtime_action("nonexistent")
        assert action is None

    def test_get_all_actions(self):
        """Test all actions are retrievable."""
        for action_id in DOWNTIME_ACTIONS:
            action = get_downtime_action(action_id)
            assert action is not None
            assert action.id == action_id


class TestListDowntimeActions:
    """Tests for list_downtime_actions helper."""

    def test_list_returns_all_actions(self):
        """Test listing returns all 9 actions."""
        actions = list_downtime_actions()
        assert len(actions) == 9

    def test_list_format(self):
        """Test list format is correct."""
        actions = list_downtime_actions()
        for action_id, action in actions:
            assert isinstance(action_id, str)
            assert isinstance(action, DowntimeAction)

    def test_all_action_ids_match(self):
        """Test all action IDs match."""
        actions = list_downtime_actions()
        for action_id, action in actions:
            assert action_id == action.id


class TestDowntimeActionInheritance:
    """Tests for action inheritance patterns."""

    def test_all_actions_inherit_from_base(self):
        """Verify all actions inherit from DowntimeAction."""
        action_classes = DOWNTIME_ACTIONS.values()
        for cls in action_classes:
            action = cls()
            assert isinstance(action, DowntimeAction)

    def test_all_actions_implement_required_methods(self):
        """Verify all actions implement required methods."""
        for action_id, cls in DOWNTIME_ACTIONS.items():
            action = cls()
            assert hasattr(action, "get_outcome")
            assert hasattr(action, "_failure_outcome")
            assert hasattr(action, "_mixed_outcome")
            assert hasattr(action, "_success_outcome")

            outcome = action.get_outcome(roll_result=10)
            assert isinstance(outcome, DowntimeOutcome)


class TestIntegration:
    """Integration tests for downtime system."""

    def test_full_resolution_flow(self):
        """Test complete resolution flow."""
        action = GetADamnDrink()

        failure = action.get_outcome(roll_result=5)
        mixed = action.get_outcome(roll_result=10)
        success = action.get_outcome(roll_result=20)

        assert failure.tier == "failure"
        assert mixed.tier == "mixed"
        assert success.tier == "success"

    def test_modifier_effects(self):
        """Test various modifier combinations."""
        action = ScroungeAndBarter()

        no_mods = action.get_outcome(roll_result=10)
        positive_mods = action.get_outcome(roll_result=15, modifiers=5)
        difficult = action.get_outcome(roll_result=10, difficulty_modifier=5)
        combined = action.get_outcome(
            roll_result=10, modifiers=3, difficulty_modifier=2
        )

        assert no_mods.tier == "mixed"
        assert positive_mods.tier == "success"
        assert difficult.tier == "failure"
        assert combined.tier == "mixed"

    def test_reserve_accumulation(self):
        """Test reserves can be accumulated from multiple actions."""
        actions = list_downtime_actions()
        all_reserves = []

        for action_id, action in actions:
            outcome = action.get_outcome(roll_result=20)
            all_reserves.extend(outcome.reserves_earned)

        narrative_reserves = [r for r in all_reserves if r.reserve_type == "narrative"]
        mech_reserves = [r for r in all_reserves if r.reserve_type == "mech"]
        tactical_reserves = [r for r in all_reserves if r.reserve_type == "tactical"]

        assert len(narrative_reserves) > 0
        assert len(mech_reserves) > 0
        assert len(tactical_reserves) > 0
