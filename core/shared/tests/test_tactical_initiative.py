"""Tests for tactical initiative tracking per PR2 3703-3725."""

import pytest
from core.shared.combat.tactical_initiative import (
    ActorPriority,
    complete_turn,
    get_eligible_nominees,
    get_remaining_actors_on_side,
    get_turn_order_for_display,
    is_valid_nomination,
    nominate_next,
    start_tactical_combat,
)
from core.mech.combat_rules import TurnOrderRules


class TestStartTacticalCombat:
    """Tests for start_tactical_combat function."""

    def test_basic_players_first(self):
        """Players should act first by default."""
        combatants = {
            "p1": "players",
            "p2": "players",
            "npc1": "hostiles",
            "npc2": "hostiles",
        }
        tracker = start_tactical_combat(combatants)

        assert tracker.current_actor_id == "p1"
        assert tracker.current_side == "players"
        assert tracker.round_index == 1

    def test_hostiles_first_when_configured(self):
        """Hostiles can be configured to act first."""
        combatants = {
            "p1": "players",
            "npc1": "hostiles",
        }
        rules = TurnOrderRules(players_act_first=False)
        tracker = start_tactical_combat(combatants, turn_order_rules=rules)

        assert tracker.current_actor_id == "npc1"
        assert tracker.current_side == "hostiles"

    def test_with_neutral_combatants(self):
        """Neutral combatants should work alongside players/hostiles."""
        combatants = {
            "p1": "players",
            "neutral1": "neutral",
            "npc1": "hostiles",
        }
        tracker = start_tactical_combat(combatants)

        assert tracker.current_actor_id == "p1"
        assert tracker.all_combatants == combatants

    def test_priority_override_veteran(self):
        """Veteran NPCs with priority should go first."""
        combatants = {
            "p1": "players",
            "veteran": "hostiles",
        }
        priorities = {"veteran": 10}
        tracker = start_tactical_combat(combatants, actor_priorities=priorities)

        assert tracker.current_actor_id == "veteran"
        assert tracker.current_side == "hostiles"

    def test_empty_combatants(self):
        """Empty combatants should create tracker with None current actor."""
        tracker = start_tactical_combat({})
        assert tracker.current_actor_id is None
        assert tracker.current_side is None

    def test_players_only(self):
        """Combat with only players should work."""
        combatants = {
            "p1": "players",
            "p2": "players",
        }
        tracker = start_tactical_combat(combatants)

        assert tracker.current_actor_id == "p1"
        assert tracker.current_side == "players"


class TestNominateNext:
    """Tests for nominate_next function."""

    def test_player_nominates_player(self):
        """Players can nominate other players."""
        combatants = {"p1": "players", "p2": "players"}
        tracker = start_tactical_combat(combatants)

        updated = nominate_next(tracker, "p1", "p2")

        assert updated.current_actor_id == "p2"
        assert updated.last_actor_id == "p1"
        assert updated.current_side == "players"

    def test_hostile_nominates_hostile(self):
        """Hostiles can nominate other hostiles."""
        combatants = {"npc1": "hostiles", "npc2": "hostiles"}
        rules = TurnOrderRules(players_act_first=False)
        tracker = start_tactical_combat(combatants, turn_order_rules=rules)

        updated = nominate_next(tracker, "npc1", "npc2")

        assert updated.current_actor_id == "npc2"
        assert updated.last_actor_id == "npc1"

    def test_cross_side_nomination_fails(self):
        """Cannot nominate across sides when alternation required."""
        combatants = {"p1": "players", "npc1": "hostiles"}
        tracker = start_tactical_combat(combatants)

        with pytest.raises(ValueError, match="Cannot nominate across sides"):
            nominate_next(tracker, "p1", "npc1")

    def test_cross_side_allowed_when_disabled(self):
        """Can nominate across sides when nomination_required is False."""
        combatants = {"p1": "players", "npc1": "hostiles"}
        rules = TurnOrderRules(nomination_required=False)
        tracker = start_tactical_combat(combatants, turn_order_rules=rules)

        updated = nominate_next(tracker, "p1", "npc1")

        assert updated.current_actor_id == "npc1"
        assert updated.current_side == "hostiles"

    def test_nominate_already_acted_fails(self):
        """Cannot nominate someone who has already acted."""
        combatants = {"p1": "players", "p2": "players"}
        tracker = start_tactical_combat(combatants)
        tracker, _ = complete_turn(tracker, "p2")

        with pytest.raises(ValueError, match="already acted this round"):
            nominate_next(tracker, "p1", "p2")

    def test_nominator_not_in_combat_fails(self):
        """Nominator must be in combat."""
        combatants = {"p1": "players"}
        tracker = start_tactical_combat(combatants)

        with pytest.raises(ValueError, match="not in combat"):
            nominate_next(tracker, "p1", "nonexistent")

    def test_nominee_not_in_combat_fails(self):
        """Nominee must be in combat."""
        combatants = {"p1": "players"}
        tracker = start_tactical_combat(combatants)

        with pytest.raises(ValueError, match="not in combat"):
            nominate_next(tracker, "nonexistent", "p1")


class TestCompleteTurn:
    """Tests for complete_turn function."""

    def test_complete_turn_marks_actor(self):
        """Completing a turn should mark the actor as having acted."""
        combatants = {"p1": "players", "npc1": "hostiles"}
        tracker = start_tactical_combat(combatants)

        updated, is_new_round = complete_turn(tracker, "p1")

        assert updated.has_acted_this_round("p1") is True
        assert is_new_round is False

    def test_new_round_when_all_acted(self):
        """New round should start when all actors have acted."""
        combatants = {"p1": "players", "npc1": "hostiles"}
        tracker = start_tactical_combat(combatants)

        updated, _ = complete_turn(tracker, "p1")
        updated, is_new_round = complete_turn(updated, "npc1")

        assert is_new_round is True
        assert updated.round_index == 2
        assert updated.has_acted_this_round("p1") is False
        assert updated.has_acted_this_round("npc1") is False

    def test_three_actor_round_completion(self):
        """Three actors should complete round after all three act."""
        combatants = {"p1": "players", "p2": "players", "npc1": "hostiles"}
        tracker = start_tactical_combat(combatants)

        updated, r1 = complete_turn(tracker, "p1")
        assert r1 is False
        assert updated.has_acted_this_round("p1")

        updated, r2 = complete_turn(updated, "p2")
        assert r2 is False
        assert updated.has_acted_this_round("p2")

        updated, r3 = complete_turn(updated, "npc1")
        assert r3 is True
        assert updated.round_index == 2
        assert updated.has_acted_this_round("p1") is False
        assert updated.has_acted_this_round("p2") is False
        assert updated.has_acted_this_round("npc1") is False


class TestGetEligibleNominees:
    """Tests for get_eligible_nominees function."""

    def test_unacted_players_eligible(self):
        """Players who haven't acted should be eligible (excluding nominator)."""
        combatants = {"p1": "players", "p2": "players", "npc1": "hostiles"}
        tracker = start_tactical_combat(combatants)

        eligible = get_eligible_nominees(tracker, "p1")

        assert "p2" in eligible
        assert "p1" not in eligible
        assert "npc1" not in eligible

    def test_after_acting_not_eligible(self):
        """Actors who have acted should not be eligible."""
        combatants = {"p1": "players", "p2": "players"}
        tracker = start_tactical_combat(combatants)
        tracker, _ = complete_turn(tracker, "p2")

        eligible = get_eligible_nominees(tracker, "p1")

        assert "p2" not in eligible

    def test_nominator_not_in_combat_returns_empty(self):
        """Nominator not in combat should return empty list."""
        tracker = start_tactical_combat({"p1": "players"})

        eligible = get_eligible_nominees(tracker, "nonexistent")

        assert eligible == []


class TestGetRemainingActorsOnSide:
    """Tests for get_remaining_actors_on_side function."""

    def test_remaining_players(self):
        """Should return players who haven't acted."""
        combatants = {"p1": "players", "p2": "players", "npc1": "hostiles"}
        tracker = start_tactical_combat(combatants)
        tracker, _ = complete_turn(tracker, "p1")

        remaining = get_remaining_actors_on_side(tracker, "players")

        assert "p2" in remaining
        assert "p1" not in remaining

    def test_remaining_hostiles(self):
        """Should return hostiles who haven't acted."""
        combatants = {"p1": "players", "npc1": "hostiles", "npc2": "hostiles"}
        tracker = start_tactical_combat(combatants)

        remaining = get_remaining_actors_on_side(tracker, "hostiles")

        assert "npc1" in remaining
        assert "npc2" in remaining


class TestIsValidNomination:
    """Tests for is_valid_nomination function."""

    def test_valid_nomination(self):
        """Valid nomination should return True."""
        combatants = {"p1": "players", "p2": "players"}
        tracker = start_tactical_combat(combatants)

        valid, error = is_valid_nomination(tracker, "p1", "p2")

        assert valid is True
        assert error is None

    def test_invalid_cross_side(self):
        """Cross-side nomination should return False."""
        combatants = {"p1": "players", "npc1": "hostiles"}
        tracker = start_tactical_combat(combatants)

        valid, error = is_valid_nomination(tracker, "p1", "npc1")

        assert valid is False
        assert "Cannot nominate across sides" in error

    def test_already_acted_nominee(self):
        """Already acted nominee should return False."""
        combatants = {"p1": "players", "p2": "players"}
        tracker = start_tactical_combat(combatants)
        tracker, _ = complete_turn(tracker, "p2")

        valid, error = is_valid_nomination(tracker, "p1", "p2")

        assert valid is False
        assert "already acted" in error

    def test_nomination_disabled(self):
        """When nomination_required is False, all nominations valid."""
        combatants = {"p1": "players", "npc1": "hostiles"}
        rules = TurnOrderRules(nomination_required=False)
        tracker = start_tactical_combat(combatants, turn_order_rules=rules)

        valid, error = is_valid_nomination(tracker, "p1", "npc1")

        assert valid is True
        assert error is None


class TestTurnOrderForDisplay:
    """Tests for get_turn_order_for_display function."""

    def test_basic_turn_order(self):
        """Should return correct turn order for display."""
        combatants = {"p1": "players", "p2": "players", "npc1": "hostiles"}
        tracker = start_tactical_combat(combatants)

        turn_order = get_turn_order_for_display(tracker)

        assert len(turn_order) == 3
        assert turn_order[0] == ("p1", 1, "players")
        assert turn_order[2] == ("npc1", 1, "hostiles")

    def test_with_priority_in_turn_order(self):
        """Priority actors should appear first within their side."""
        combatants = {"p1": "players", "veteran": "hostiles", "grunt": "hostiles"}
        priorities = {"veteran": 10}
        tracker = start_tactical_combat(combatants, actor_priorities=priorities)

        order = tracker.get_turn_order()

        assert order[1] == "veteran"
        assert order[2] == "grunt"


class TestAlternationRules:
    """Tests for side alternation rules."""

    def test_strict_alternation(self):
        """Sides should strictly alternate."""
        combatants = {
            "p1": "players",
            "p2": "players",
            "npc1": "hostiles",
            "npc2": "hostiles",
        }
        tracker = start_tactical_combat(combatants)

        order = tracker.get_turn_order()
        side_order = [tracker.get_side(a) for a in order]

        expected = ["players", "hostiles", "players", "hostiles"]
        assert side_order == expected

    def test_remaining_side_any_order(self):
        """Remaining actors on exhausted side should go in any order."""
        combatants = {"p1": "players", "npc1": "hostiles"}
        tracker = start_tactical_combat(combatants)

        tracker, _ = complete_turn(tracker, "p1")
        tracker, _ = complete_turn(tracker, "npc1")

        assert tracker.round_index == 2


class TestPriorityOverrides:
    """Tests for actor priority override behavior."""

    def test_higher_priority_jumps_queue(self):
        """Higher priority actor should go first, breaking alternation."""
        combatants = {"p1": "players", "veteran": "hostiles", "grunt": "hostiles"}
        priorities = {"veteran": 10}

        tracker = start_tactical_combat(combatants, actor_priorities=priorities)

        assert tracker.current_actor_id == "veteran"

    def test_priority_preserved_across_rounds(self):
        """Priority should persist across rounds."""
        combatants = {"p1": "players", "veteran": "hostiles"}
        priorities = {"veteran": 10}
        tracker = start_tactical_combat(combatants, actor_priorities=priorities)

        tracker, _ = complete_turn(tracker, "veteran")
        tracker, _ = complete_turn(tracker, "p1")

        assert tracker.round_index == 2
        assert tracker.current_actor_id == "veteran"

    def test_multiple_priorities_sorted(self):
        """Multiple actors with priorities should be sorted correctly within side."""
        combatants = {"p1": "players", "elite": "hostiles", "grunt": "hostiles"}
        priorities = {"elite": 20, "grunt": 5}

        tracker = start_tactical_combat(combatants, actor_priorities=priorities)
        order = tracker.get_turn_order()

        assert order[0] == "p1"
        assert order[1] == "elite"
        assert order[2] == "grunt"


class TestGetTurnOrder:
    """Tests for get_turn_order method."""

    def test_returns_all_actors(self):
        """Should return all actors in turn order."""
        combatants = {"p1": "players", "npc1": "hostiles"}
        tracker = start_tactical_combat(combatants)

        order = tracker.get_turn_order()

        assert len(order) == 2
        assert set(order) == {"p1", "npc1"}

    def test_actors_by_side_with_priority(self):
        """Actors should be sorted by priority within sides."""
        combatants = {"slow": "players", "fast": "players", "npc1": "hostiles"}
        priorities = {"slow": 0, "fast": 10}

        tracker = start_tactical_combat(combatants, actor_priorities=priorities)
        by_side = tracker.get_actors_by_side()

        assert by_side["players"][0] == "fast"


class TestActorPriority:
    """Tests for ActorPriority model."""

    def test_basic_actor_priority(self):
        """ActorPriority should store priority data."""
        priority = ActorPriority(actor_id="vet", priority=10, reason="Viper's Speed")

        assert priority.actor_id == "vet"
        assert priority.priority == 10
        assert priority.reason == "Viper's Speed"

    def test_actor_priority_no_reason(self):
        """ActorPriority can omit reason."""
        priority = ActorPriority(actor_id="vet", priority=5)

        assert priority.reason is None


class TestIntegration:
    """Integration tests for complete nomination flows."""

    def test_full_combat_flow(self):
        """Test complete combat flow: nominate, act, nominate, act."""
        combatants = {
            "p1": "players",
            "p2": "players",
            "npc1": "hostiles",
        }
        tracker = start_tactical_combat(combatants)

        assert tracker.current_actor_id == "p1"

        tracker, _ = complete_turn(tracker, "p1")
        eligible = get_eligible_nominees(tracker, "p1")
        assert "p2" in eligible

        tracker = nominate_next(tracker, "p1", "p2")
        assert tracker.current_actor_id == "p2"

        tracker, _ = complete_turn(tracker, "p2")
        eligible = get_eligible_nominees(tracker, "p2")
        assert len(eligible) == 0

        tracker = nominate_next(tracker, "p2", "npc1")
        assert tracker.current_actor_id == "npc1"

    def test_round_transition(self):
        """Test round transition after all actors have acted."""
        combatants = {"p1": "players", "npc1": "hostiles"}
        tracker = start_tactical_combat(combatants)

        tracker, r1 = complete_turn(tracker, "p1")
        assert r1 is False
        assert tracker.round_index == 1

        tracker, r2 = complete_turn(tracker, "npc1")
        assert r2 is True
        assert tracker.round_index == 2

        assert tracker.current_actor_id == "p1"

    def test_neutral_in_actors_by_side(self):
        """Neutral combatants should appear in actors_by_side."""
        combatants = {
            "p1": "players",
            "neutral": "neutral",
            "npc1": "hostiles",
        }
        tracker = start_tactical_combat(combatants)
        by_side = tracker.get_actors_by_side()

        assert "neutral" in by_side
        assert len(by_side["neutral"]) == 1


class TestEdgeCases:
    """Edge case tests."""

    def test_single_combatant(self):
        """Single combatant should work."""
        combatants = {"p1": "players"}
        tracker = start_tactical_combat(combatants)

        assert tracker.current_actor_id == "p1"

        tracker, is_new = complete_turn(tracker, "p1")
        assert is_new is True
        assert tracker.round_index == 2

    def test_equal_sized_sides(self):
        """Equal sized sides should alternate cleanly."""
        combatants = {
            "p1": "players",
            "p2": "players",
            "npc1": "hostiles",
            "npc2": "hostiles",
        }
        tracker = start_tactical_combat(combatants)

        order = tracker.get_turn_order()

        assert order == ["p1", "npc1", "p2", "npc2"]

    def test_many_actors(self):
        """Many actors should be handled correctly."""
        combatants = {}
        for i in range(5):
            combatants[f"p{i}"] = "players"
        for i in range(5):
            combatants[f"npc{i}"] = "hostiles"

        tracker = start_tactical_combat(combatants)
        order = tracker.get_turn_order()

        assert len(order) == 10
        sides = [tracker.get_side(a) for a in order]
        assert sides == ["players", "hostiles"] * 5

    def test_invalid_state_rejection(self):
        """Should reject invalid states gracefully."""
        combatants = {"p1": "players", "npc1": "hostiles"}
        tracker = start_tactical_combat(combatants)

        valid, _ = is_valid_nomination(tracker, "p1", "p2")
        assert valid is False

    def test_cross_side_allowed_when_exhausted(self):
        """Cross-side nomination should be allowed when side is exhausted."""
        combatants = {"p1": "players", "npc1": "hostiles"}
        tracker = start_tactical_combat(combatants)

        tracker = nominate_next(tracker, "p1", "p1")
        tracker, _ = complete_turn(tracker, "p1")

        valid, _ = is_valid_nomination(tracker, "p1", "npc1")
        assert valid is True
