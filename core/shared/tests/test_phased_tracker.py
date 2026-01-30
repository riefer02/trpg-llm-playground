"""Tests for PhasedTacticalTracker - phase-level action tracking for tactical combat."""

import pytest

from core.shared.combat.phased_tracker import (
    start_tactical_combat_with_phases,
    start_actor_turn,
    advance_phase,
    end_actor_turn,
    activate_protocol,
    prepare_action,
    drop_prepared_action,
    use_reaction,
    validate_action_timing,
    get_phase_state,
    get_eligible_actions,
    nominate_next_phase,
    get_turn_order_for_display,
)
from core.mech.timing import (
    ActionTimingValidationSettings,
)
from core.mech.combat_rules import TurnOrderRules


class TestPhasedTacticalTrackerCreation:
    """Tests for creating PhasedTacticalTracker instances."""

    def test_start_combat_with_phases_initializes_all_actors(self):
        """Test that all actors start at 'start' phase."""
        combatants = {
            "player1": "players",
            "player2": "players",
            "npc1": "hostiles",
        }
        tracker = start_tactical_combat_with_phases(combatants)

        assert tracker.current_actor_id == "player1"
        assert tracker.current_side == "players"
        assert tracker.round_index == 1

        assert tracker.actor_phases["player1"] == "start"
        assert tracker.actor_phases["player2"] == "start"
        assert tracker.actor_phases["npc1"] == "start"

        assert tracker.actor_prepared_actions["player1"] is None
        assert tracker.actor_protocols["player1"] is None
        assert tracker.per_round_reactions["player1"] == {}

    def test_start_combat_with_custom_rules(self):
        """Test initialization with custom turn order rules."""
        combatants = {"mech1": "hostiles"}
        rules = TurnOrderRules(players_act_first=False)
        tracker = start_tactical_combat_with_phases(combatants, turn_order_rules=rules)

        assert tracker.current_actor_id == "mech1"
        assert tracker.current_side == "hostiles"
        assert tracker.turn_order_rules.players_act_first is False

    def test_start_combat_with_priorities(self):
        """Test initialization with actor priorities."""
        combatants = {
            "player1": "players",
            "npc1": "hostiles",
            "npc2": "hostiles",
        }
        priorities = {"npc1": 10, "npc2": 5}
        tracker = start_tactical_combat_with_phases(
            combatants, actor_priorities=priorities
        )

        assert tracker.actor_priorities == priorities
        assert tracker.current_actor_id == "npc1"


class TestStartActorTurn:
    """Tests for starting an actor's turn."""

    def test_start_turn_resets_to_start_phase(self):
        """Test that starting a turn resets to 'start' phase."""
        combatants = {"mech1": "players", "mech2": "hostiles"}
        tracker = start_tactical_combat_with_phases(combatants)

        tracker = advance_phase(tracker, "mech1")
        assert tracker.actor_phases["mech1"] == "normal"

        tracker = start_actor_turn(tracker, "mech1")
        assert tracker.actor_phases["mech1"] == "start"

    def test_start_turn_clears_active_protocol(self):
        """Test that starting a turn clears the previous protocol."""
        combatants = {"mech1": "players"}
        tracker = start_tactical_combat_with_phases(combatants)

        tracker, _ = activate_protocol(tracker, "mech1", "scan_protocol")
        assert tracker.actor_protocols["mech1"] == "scan_protocol"

        tracker = start_actor_turn(tracker, "mech1")
        assert tracker.actor_protocols["mech1"] is None

    def test_start_turn_expires_prepared_actions(self):
        """Test that prepared actions expire at start of next turn."""
        combatants = {"mech1": "players"}
        tracker = start_tactical_combat_with_phases(combatants)

        tracker = prepare_action(
            tracker,
            "mech1",
            "skirmish",
            "quick",
            "enemy enters engagement",
            expires_on_turn=1,
        )
        assert tracker.actor_prepared_actions["mech1"] is not None

        tracker = start_actor_turn(tracker, "mech1")
        assert tracker.actor_prepared_actions["mech1"] is None

    def test_start_turn_updates_current_actor(self):
        """Test that starting a turn updates current actor tracking."""
        combatants = {"mech1": "players", "mech2": "hostiles"}
        tracker = start_tactical_combat_with_phases(combatants)

        tracker = start_actor_turn(tracker, "mech2")
        assert tracker.current_actor_id == "mech2"
        assert tracker.current_side == "hostiles"


class TestPhaseTransitions:
    """Tests for phase advancement."""

    def test_advance_from_start_to_normal(self):
        """Test advancing from start phase to normal phase."""
        combatants = {"mech1": "players"}
        tracker = start_tactical_combat_with_phases(combatants)

        tracker = advance_phase(tracker, "mech1")
        assert tracker.actor_phases["mech1"] == "normal"

    def test_advance_from_normal_to_end(self):
        """Test advancing from normal phase to end phase."""
        combatants = {"mech1": "players"}
        tracker = start_tactical_combat_with_phases(combatants)

        tracker = advance_phase(tracker, "mech1")
        tracker = advance_phase(tracker, "mech1")
        assert tracker.actor_phases["mech1"] == "end"

    def test_cannot_advance_from_end_phase(self):
        """Test that advancing from end phase raises an error."""
        combatants = {"mech1": "players"}
        tracker = start_tactical_combat_with_phases(combatants)

        tracker = advance_phase(tracker, "mech1")
        tracker = advance_phase(tracker, "mech1")

        with pytest.raises(ValueError, match="Cannot advance from end phase"):
            advance_phase(tracker, "mech1")


class TestEndActorTurn:
    """Tests for ending an actor's turn."""

    def test_end_turn_marks_actor_as_acted(self):
        """Test that ending a turn marks the actor as having acted."""
        combatants = {"mech1": "players", "mech2": "hostiles"}
        tracker = start_tactical_combat_with_phases(combatants)

        tracker, is_new_round = end_actor_turn(tracker, "mech1")
        assert tracker.has_acted_this_round("mech1") is True
        assert is_new_round is False

    def test_end_turn_when_all_actors_acted_starts_new_round(self):
        """Test that ending the last actor's turn starts a new round."""
        combatants = {"mech1": "players", "mech2": "hostiles"}
        tracker = start_tactical_combat_with_phases(combatants)

        tracker, _ = end_actor_turn(tracker, "mech1")
        assert tracker.round_index == 1

        tracker, is_new_round = end_actor_turn(tracker, "mech2")
        assert is_new_round is True
        assert tracker.round_index == 2

    def test_new_round_resets_reaction_tracking(self):
        """Test that a new round resets per-round reaction tracking."""
        combatants = {"mech1": "players", "mech2": "hostiles"}
        tracker = start_tactical_combat_with_phases(combatants)

        tracker, _ = use_reaction(tracker, "mech1", "brace")
        assert tracker.get_reaction_count("mech1", "brace") == 1

        tracker, _ = end_actor_turn(tracker, "mech1")
        tracker, is_new_round = end_actor_turn(tracker, "mech2")

        assert is_new_round is True
        assert tracker.get_reaction_count("mech1", "brace") == 0


class TestProtocolActivation:
    """Tests for protocol timing and activation."""

    def test_protocol_at_start_phase_valid(self):
        """Test that protocol at start phase is valid."""
        combatants = {"mech1": "players"}
        tracker = start_tactical_combat_with_phases(combatants)

        tracker, result = activate_protocol(tracker, "mech1", "scan_protocol")
        assert result.valid is True
        assert tracker.actor_protocols["mech1"] == "scan_protocol"

    def test_protocol_at_normal_phase_invalid(self):
        """Test that protocol at normal phase is invalid."""
        combatants = {"mech1": "players"}
        tracker = start_tactical_combat_with_phases(combatants)

        tracker = advance_phase(tracker, "mech1")

        tracker, result = activate_protocol(tracker, "mech1", "scan_protocol")
        assert result.valid is False
        assert "start of your turn" in result.errors[0]

    def test_protocol_at_end_phase_invalid(self):
        """Test that protocol at end phase is invalid."""
        combatants = {"mech1": "players"}
        tracker = start_tactical_combat_with_phases(combatants)

        tracker = advance_phase(tracker, "mech1")
        tracker = advance_phase(tracker, "mech1")

        tracker, result = activate_protocol(tracker, "mech1", "scan_protocol")
        assert result.valid is False

    def test_protocol_with_narrative_settings_allowed(self):
        """Test that narrative settings allow protocols outside start."""
        combatants = {"mech1": "players"}
        tracker = start_tactical_combat_with_phases(combatants)

        tracker = advance_phase(tracker, "mech1")
        settings = ActionTimingValidationSettings(allow_protocol_outside_start=True)

        tracker, result = activate_protocol(
            tracker, "mech1", "scan_protocol", settings=settings
        )
        assert result.valid is True


class TestPreparedActions:
    """Tests for prepared action mechanics."""

    def test_prepare_action_creates_state(self):
        """Test that prepare_action creates a PreparedActionState."""
        combatants = {"mech1": "players"}
        tracker = start_tactical_combat_with_phases(combatants)

        tracker = prepare_action(
            tracker,
            "mech1",
            "skirmish",
            "quick",
            "enemy enters engagement",
            expires_on_turn=2,
        )

        prepared = tracker.actor_prepared_actions["mech1"]
        assert prepared is not None
        assert prepared.held_action_id == "skirmish"
        assert prepared.held_action_type == "quick"
        assert prepared.blocks_actions is True
        assert prepared.blocks_reactions is True
        assert prepared.blocks_movement is True

    def test_drop_prepared_action_clears_lockout(self):
        """Test that dropping a prepared action clears the lockout."""
        combatants = {"mech1": "players"}
        tracker = start_tactical_combat_with_phases(combatants)

        tracker = prepare_action(
            tracker, "mech1", "skirmish", "quick", "trigger", expires_on_turn=2
        )
        assert tracker.actor_prepared_actions["mech1"] is not None

        tracker = drop_prepared_action(tracker, "mech1")
        assert tracker.actor_prepared_actions["mech1"] is None


class TestReactionTracking:
    """Tests for per-round reaction tracking."""

    def test_use_reaction_tracks_usage(self):
        """Test that using a reaction increments its count."""
        combatants = {"mech1": "players"}
        tracker = start_tactical_combat_with_phases(combatants)

        tracker, result = use_reaction(tracker, "mech1", "brace")
        assert result.valid is True
        assert tracker.get_reaction_count("mech1", "brace") == 1

    def test_reaction_at_limit_invalid(self):
        """Test that reaction at limit returns invalid result."""
        combatants = {"mech1": "players"}
        tracker = start_tactical_combat_with_phases(combatants)

        tracker, _ = use_reaction(tracker, "mech1", "brace", max_per_round=1)

        tracker, result = use_reaction(tracker, "mech1", "brace", max_per_round=1)
        assert result.valid is False
        assert "already used" in result.errors[0]

    def test_different_reactions_track_separately(self):
        """Test that different reactions track separately."""
        combatants = {"mech1": "players"}
        tracker = start_tactical_combat_with_phases(combatants)

        tracker, _ = use_reaction(tracker, "mech1", "brace")
        tracker, _ = use_reaction(tracker, "mech1", "overwatch")

        assert tracker.get_reaction_count("mech1", "brace") == 1
        assert tracker.get_reaction_count("mech1", "overwatch") == 1


class TestActionTimingValidation:
    """Tests for combined action timing validation."""

    def test_non_protocol_action_at_normal_phase_valid(self):
        """Test that non-protocol actions are valid at normal phase."""
        combatants = {"mech1": "players"}
        tracker = start_tactical_combat_with_phases(combatants)

        tracker = advance_phase(tracker, "mech1")

        result = validate_action_timing(
            tracker, "mech1", "skirmish", "quick", is_protocol=False
        )
        assert result.valid is True

    def test_protocol_at_normal_phase_invalid(self):
        """Test that protocol is invalid at normal phase."""
        combatants = {"mech1": "players"}
        tracker = start_tactical_combat_with_phases(combatants)

        tracker = advance_phase(tracker, "mech1")

        result = validate_action_timing(
            tracker, "mech1", "scan_protocol", "free", is_protocol=True
        )
        assert result.valid is False

    def test_action_blocked_while_prepared(self):
        """Test that actions are blocked while prepared."""
        combatants = {"mech1": "players"}
        tracker = start_tactical_combat_with_phases(combatants)

        tracker = prepare_action(
            tracker, "mech1", "skirmish", "quick", "trigger", expires_on_turn=2
        )

        result = validate_action_timing(
            tracker, "mech1", "boost", "quick", is_protocol=False
        )
        assert result.valid is False

    def test_reaction_allowed_while_prepared(self):
        """Test that reactions are allowed while prepared (if not blocked)."""
        combatants = {"mech1": "players"}
        tracker = start_tactical_combat_with_phases(combatants)

        tracker = prepare_action(
            tracker, "mech1", "skirmish", "quick", "trigger", expires_on_turn=2
        )

        result = validate_action_timing(
            tracker, "mech1", "brace", "reaction", is_protocol=False
        )
        assert result.valid is False

    def test_validation_with_custom_settings(self):
        """Test validation with custom timing settings."""
        combatants = {"mech1": "players"}
        tracker = start_tactical_combat_with_phases(combatants)

        tracker = advance_phase(tracker, "mech1")
        settings = ActionTimingValidationSettings(allow_protocol_outside_start=True)

        result = validate_action_timing(
            tracker,
            "mech1",
            "scan_protocol",
            "free",
            is_protocol=True,
            settings=settings,
        )
        assert result.valid is True


class TestPhaseState:
    """Tests for phase state integration."""

    def test_get_phase_state_returns_turnphasestate(self):
        """Test that get_phase_state returns a TurnPhaseState."""
        combatants = {"mech1": "players"}
        tracker = start_tactical_combat_with_phases(combatants)

        state = get_phase_state(tracker, "mech1")
        assert state is not None
        assert state.current_phase == "start"
        assert state.protocol_activated is False
        assert state.protocol_id is None

    def test_phase_state_with_active_protocol(self):
        """Test phase state reflects active protocol."""
        combatants = {"mech1": "players"}
        tracker = start_tactical_combat_with_phases(combatants)

        tracker, _ = activate_protocol(tracker, "mech1", "scan_protocol")

        state = get_phase_state(tracker, "mech1")
        assert state is not None
        assert state.protocol_activated is True
        assert state.protocol_id == "scan_protocol"


class TestEligibleActions:
    """Tests for action eligibility filtering."""

    def test_filter_protocols_outside_start_phase(self):
        """Test that protocols are filtered out outside start phase."""
        combatants = {"mech1": "players"}
        tracker = start_tactical_combat_with_phases(combatants)

        tracker = advance_phase(tracker, "mech1")

        actions = [
            ("scan_protocol", "free", True),
            ("skirmish", "quick", False),
            ("boost", "quick", False),
        ]

        eligible = get_eligible_actions(tracker, "mech1", actions)
        assert len(eligible) == 2
        assert ("scan_protocol", "free", True) not in eligible

    def test_filter_actions_while_prepared(self):
        """Test that actions are filtered while prepared."""
        combatants = {"mech1": "players"}
        tracker = start_tactical_combat_with_phases(combatants)

        tracker = prepare_action(
            tracker, "mech1", "skirmish", "quick", "trigger", expires_on_turn=2
        )

        actions = [
            ("skirmish", "quick", False),
            ("boost", "quick", False),
            ("brace", "reaction", False),
        ]

        eligible = get_eligible_actions(tracker, "mech1", actions)
        assert len(eligible) == 0

        tracker = drop_prepared_action(tracker, "mech1")

        eligible = get_eligible_actions(tracker, "mech1", actions)
        assert len(eligible) == 3


class TestNominateNextPhase:
    """Tests for nomination with phase reset."""

    def test_nominate_next_resets_to_start_phase(self):
        """Test that nominating an actor resets them to start phase."""
        combatants = {
            "player1": "players",
            "player2": "players",
            "npc1": "hostiles",
        }
        tracker = start_tactical_combat_with_phases(combatants)

        tracker = advance_phase(tracker, "player1")
        tracker = advance_phase(tracker, "player1")

        tracker, is_valid, error = nominate_next_phase(tracker, "player1", "player2")

        assert is_valid is True
        assert error is None
        assert tracker.actor_phases["player2"] == "start"

    def test_nominate_next_clears_prepared_action_if_expired(self):
        """Test that prepared action expires on nomination."""
        combatants = {
            "player1": "players",
            "player2": "players",
        }
        tracker = start_tactical_combat_with_phases(combatants)

        tracker = prepare_action(
            tracker, "player1", "skirmish", "quick", "trigger", expires_on_turn=1
        )

        tracker, is_valid, error = nominate_next_phase(tracker, "player1", "player2")

        assert is_valid is True
        assert tracker.actor_prepared_actions["player2"] is None


class TestTurnOrderDisplay:
    """Tests for turn order display."""

    def test_get_turn_order_for_display_includes_phase(self):
        """Test that display includes phase information."""
        combatants = {"mech1": "players"}
        tracker = start_tactical_combat_with_phases(combatants)

        display = get_turn_order_for_display(tracker)
        assert len(display) == 1
        actor_id, round_idx, side, phase = display[0]
        assert actor_id == "mech1"
        assert round_idx == 1
        assert side == "players"
        assert phase == "start"


class TestIntegration:
    """Integration tests for full turn flow."""

    def test_full_player_turn_flow(self):
        """Test a complete player turn: start -> normal -> end."""
        combatants = {
            "player1": "players",
            "npc1": "hostiles",
        }
        tracker = start_tactical_combat_with_phases(combatants)

        assert tracker.actor_phases["player1"] == "start"
        assert tracker.current_actor_id == "player1"

        tracker, _ = activate_protocol(tracker, "player1", "scan_protocol")
        assert tracker.actor_protocols["player1"] == "scan_protocol"

        tracker = advance_phase(tracker, "player1")
        assert tracker.actor_phases["player1"] == "normal"

        tracker = advance_phase(tracker, "player1")
        assert tracker.actor_phases["player1"] == "end"

        tracker, is_new_round = end_actor_turn(tracker, "player1")
        assert is_new_round is False
        assert tracker.has_acted_this_round("player1") is True

    def test_two_actor_combat_with_phases(self):
        """Test combat with two actors going through full rounds."""
        combatants = {
            "player1": "players",
            "npc1": "hostiles",
        }
        tracker = start_tactical_combat_with_phases(combatants)

        assert tracker.current_actor_id == "player1"
        assert tracker.actor_phases["player1"] == "start"

        tracker = advance_phase(tracker, "player1")
        tracker = advance_phase(tracker, "player1")

        tracker, _ = end_actor_turn(tracker, "player1")
        assert tracker.has_acted_this_round("player1") is True

        tracker, is_valid, _ = nominate_next_phase(tracker, "player1", "npc1")
        assert is_valid is True
        assert tracker.current_actor_id == "npc1"
        assert tracker.actor_phases["npc1"] == "start"

        tracker, is_new_round = end_actor_turn(tracker, "npc1")
        assert is_new_round is True
        assert tracker.round_index == 2

    def test_reaction_tracking_across_turns(self):
        """Test that reactions track correctly across turns within a round."""
        combatants = {
            "player1": "players",
            "npc1": "hostiles",
        }
        tracker = start_tactical_combat_with_phases(combatants)

        tracker, _ = use_reaction(tracker, "player1", "brace")
        assert tracker.get_reaction_count("player1", "brace") == 1

        tracker, is_valid, _ = nominate_next_phase(tracker, "player1", "npc1")

        tracker, _ = end_actor_turn(tracker, "npc1")
        tracker, _ = end_actor_turn(tracker, "player1")

        assert tracker.get_reaction_count("player1", "brace") == 0

    def test_prepared_action_blocks_then_expires(self):
        """Test that prepared action blocks actions then expires."""
        combatants = {
            "player1": "players",
            "player2": "players",
        }
        tracker = start_tactical_combat_with_phases(combatants)

        tracker = prepare_action(
            tracker,
            "player1",
            "skirmish",
            "quick",
            "enemy moves",
            expires_on_turn=1,
        )

        result = validate_action_timing(
            tracker, "player1", "boost", "quick", is_protocol=False
        )
        assert result.valid is False

        tracker, is_valid, _ = nominate_next_phase(tracker, "player1", "player2")

        tracker = start_actor_turn(tracker, "player1")
        assert tracker.actor_prepared_actions["player1"] is None

        result = validate_action_timing(
            tracker, "player1", "boost", "quick", is_protocol=False
        )
        assert result.valid is True


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_nonexistent_actor_prepare_action(self):
        """Test that preparing action for nonexistent actor works."""
        combatants = {"mech1": "players"}
        tracker = start_tactical_combat_with_phases(combatants)

        tracker = prepare_action(
            tracker,
            "mech2",
            "skirmish",
            "quick",
            "trigger",
            expires_on_turn=2,
        )
        assert tracker.actor_prepared_actions.get("mech2") is not None

    def test_nonexistent_actor_phase_advance(self):
        """Test that advancing phase for nonexistent actor raises error."""
        combatants = {"mech1": "players"}
        tracker = start_tactical_combat_with_phases(combatants)

        with pytest.raises(ValueError):
            advance_phase(tracker, "mech2")

    def test_empty_combatants(self):
        """Test initialization with no combatants."""
        tracker = start_tactical_combat_with_phases({})
        assert tracker.current_actor_id is None
        assert tracker.all_combatants == {}
        assert tracker.actor_phases == {}
