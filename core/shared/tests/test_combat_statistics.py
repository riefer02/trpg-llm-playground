"""Tests for combat statistics tracking system."""

import pytest
from core.shared.combat.statistics import (
    ActionTypeCount,
    CombatStatistics,
    CombatStatisticsTracker,
    CombatantStatistics,
)
from core.shared.combat.statistics_integration import (
    initialize_statistics_for_scenario,
    update_statistics_for_turn_end,
    update_statistics_from_action,
)
from core.mech.combat_state import (
    MechCombatScenario,
    CombatantState,
    CombatStats,
    CombatResources,
)
from core.mech.combat_models import ActionExecutionResult, ResourceChange
from core.mech.grid import HexPosition, HexCoord


class TestActionTypeCount:
    """Tests for ActionTypeCount model."""

    def test_default_values(self):
        """Test that ActionTypeCount initializes with zeros."""
        count = ActionTypeCount()
        assert count.attacks == 0
        assert count.moves == 0
        assert count.techs == 0
        assert count.full_actions == 0
        assert count.quick_actions == 0
        assert count.reactions == 0
        assert count.overcharges == 0

    def test_custom_values(self):
        """Test that ActionTypeCount can be initialized with custom values."""
        count = ActionTypeCount(attacks=5, moves=3, techs=2)
        assert count.attacks == 5
        assert count.moves == 3
        assert count.techs == 2


class TestCombatStatistics:
    """Tests for CombatStatistics model."""

    def test_default_values(self):
        """Test that CombatStatistics initializes with defaults."""
        stats = CombatStatistics()
        assert stats.rounds_completed == 0
        assert stats.total_turns == 0
        assert stats.total_damage_dealt_by_players == 0
        assert stats.total_damage_received_by_players == 0
        assert stats.total_enemies_destroyed == 0
        assert stats.closest_call_hp == 0
        assert stats.closest_call_combatant == ""
        assert stats.max_overkill == 0

    def test_get_player_stats(self):
        """Test filtering player stats."""
        player_stats = CombatantStatistics(
            combatant_id="p1",
            combatant_name="Player 1",
            side="players",
        )
        enemy_stats = CombatantStatistics(
            combatant_id="e1",
            combatant_name="Enemy 1",
            side="hostiles",
        )
        stats = CombatStatistics(
            combatant_stats={"p1": player_stats, "e1": enemy_stats}
        )

        players = stats.get_player_stats()
        assert len(players) == 1
        assert players[0].combatant_id == "p1"

    def test_get_hostile_stats(self):
        """Test filtering hostile stats."""
        player_stats = CombatantStatistics(
            combatant_id="p1",
            combatant_name="Player 1",
            side="players",
        )
        enemy_stats = CombatantStatistics(
            combatant_id="e1",
            combatant_name="Enemy 1",
            side="hostiles",
        )
        stats = CombatStatistics(
            combatant_stats={"p1": player_stats, "e1": enemy_stats}
        )

        hostiles = stats.get_hostile_stats()
        assert len(hostiles) == 1
        assert hostiles[0].combatant_id == "e1"

    def test_get_total_actions(self):
        """Test calculating total actions."""
        action_totals = ActionTypeCount(attacks=5, moves=3, techs=2)
        stats = CombatStatistics(action_totals=action_totals)

        assert stats.get_total_actions() == 10


class TestCombatStatisticsTracker:
    """Tests for CombatStatisticsTracker."""

    def test_initialize_combatant(self):
        """Test initializing a combatant in the tracker."""
        tracker = CombatStatisticsTracker()
        tracker.initialize_combatant("c1", "Combatant 1", "players", 100)

        assert "c1" in tracker.combatant_stats
        stats = tracker.combatant_stats["c1"]
        assert stats.combatant_name == "Combatant 1"
        assert stats.side == "players"
        assert stats.hp_at_start == 100
        assert stats.lowest_hp_reached == 100

    def test_record_damage_dealt(self):
        """Test recording damage dealt."""
        tracker = CombatStatisticsTracker()
        tracker.initialize_combatant("attacker", "Attacker", "players", 100)
        tracker.initialize_combatant("target", "Target", "hostiles", 50)

        tracker.record_damage_dealt("attacker", "target", 20, 50, 30, False)

        attacker_stats = tracker.combatant_stats["attacker"]
        assert attacker_stats.damage_dealt == 20
        assert attacker_stats.overkill_dealt == 0

        target_stats = tracker.combatant_stats["target"]
        assert target_stats.damage_received == 20
        assert target_stats.lowest_hp_reached == 30

    def test_record_damage_dealt_with_overkill(self):
        """Test recording damage with overkill."""
        tracker = CombatStatisticsTracker()
        tracker.initialize_combatant("attacker", "Attacker", "players", 100)
        tracker.initialize_combatant("target", "Target", "hostiles", 50)

        tracker.record_damage_dealt("attacker", "target", 60, 50, 0, True)

        attacker_stats = tracker.combatant_stats["attacker"]
        assert attacker_stats.damage_dealt == 60
        assert attacker_stats.overkill_dealt == 10  # 60 - 50
        assert attacker_stats.enemies_destroyed == 1
        assert "target" in attacker_stats.destroyed_enemy_ids

        assert tracker.max_overkill == 10
        assert tracker.total_enemies_destroyed == 1

    def test_record_action(self):
        """Test recording an action."""
        tracker = CombatStatisticsTracker()
        tracker.initialize_combatant("c1", "Combatant 1", "players", 100)

        tracker.record_action("c1", "attack")
        tracker.record_action("c1", "move")
        tracker.record_action("c1", "full")

        stats = tracker.combatant_stats["c1"]
        assert stats.actions_taken.attacks == 1
        assert stats.actions_taken.moves == 1
        assert stats.actions_taken.full_actions == 1

        assert tracker.action_totals.attacks == 1
        assert tracker.action_totals.moves == 1
        assert tracker.action_totals.full_actions == 1

    def test_record_turn_taken(self):
        """Test recording a turn."""
        tracker = CombatStatisticsTracker()
        tracker.initialize_combatant("c1", "Combatant 1", "players", 100)

        tracker.record_turn_taken("c1")
        tracker.record_turn_taken("c1")

        stats = tracker.combatant_stats["c1"]
        assert stats.turns_taken == 2
        assert tracker.total_turns == 2

    def test_record_round_completed(self):
        """Test recording a round."""
        tracker = CombatStatisticsTracker()

        tracker.record_round_completed()
        tracker.record_round_completed()

        assert tracker.rounds_completed == 2

    def test_closest_call_tracking(self):
        """Test tracking closest call for players."""
        tracker = CombatStatisticsTracker()
        tracker.initialize_combatant("player", "Player", "players", 100)

        # Simulate taking damage
        tracker.record_damage_dealt("enemy", "player", 30, 100, 70, False)
        tracker.record_damage_dealt("enemy", "player", 40, 70, 30, False)
        tracker.record_damage_dealt("enemy", "player", 10, 30, 20, False)

        # HP never went below 20, so closest call is 20
        assert tracker.closest_call_hp == 20
        assert tracker.closest_call_combatant == "Player"

    def test_to_combat_statistics(self):
        """Test converting tracker to immutable CombatStatistics."""
        tracker = CombatStatisticsTracker()
        tracker.initialize_combatant("c1", "Combatant 1", "players", 100)
        tracker.record_action("c1", "attack")
        tracker.record_turn_taken("c1")
        tracker.record_round_completed()

        stats = tracker.to_combat_statistics()

        assert stats.rounds_completed == 1
        assert stats.total_turns == 1
        assert "c1" in stats.combatant_stats
        assert stats.combatant_stats["c1"].actions_taken.attacks == 1


class TestStatisticsIntegration:
    """Tests for statistics integration helpers."""

    def test_initialize_statistics_for_scenario(self):
        """Test initializing statistics for a new scenario."""
        combatant = CombatantState(
            id="c1",
            name="Test Combatant",
            side="players",
            kind="mech",
            stats=CombatStats(
                size="size_1",
                hp_max=100,
                evasion=8,
                e_defense=8,
                armor=0,
                speed=4,
                sensor_range=10,
            ),
            resources=CombatResources(hp_current=100, structure_current=4),
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        scenario = MechCombatScenario(combatants=[combatant])

        stats = initialize_statistics_for_scenario(scenario)

        assert "c1" in stats.combatant_stats
        assert stats.combatant_stats["c1"].hp_at_start == 100
        assert stats.combatant_stats["c1"].side == "players"

    def test_update_statistics_for_turn_end(self):
        """Test updating statistics when a turn ends."""
        initial_stats = CombatStatistics()

        updated = update_statistics_for_turn_end(
            initial_stats, "c1", is_new_round=False
        )

        assert updated.total_turns == 1

        updated = update_statistics_for_turn_end(updated, "c1", is_new_round=True)

        assert updated.total_turns == 2
        assert updated.rounds_completed == 1

    def test_update_statistics_from_action_with_damage(self):
        """Test updating statistics from an action with damage."""
        # Create a simple scenario
        attacker = CombatantState(
            id="attacker",
            name="Attacker",
            side="players",
            kind="mech",
            stats=CombatStats(
                size="size_1",
                hp_max=100,
                evasion=8,
                e_defense=8,
                armor=0,
                speed=4,
                sensor_range=10,
            ),
            resources=CombatResources(hp_current=100, structure_current=4),
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        target = CombatantState(
            id="target",
            name="Target",
            side="hostiles",
            kind="npc",
            stats=CombatStats(
                size="size_1",
                hp_max=50,
                evasion=8,
                e_defense=8,
                armor=0,
                speed=4,
                sensor_range=10,
            ),
            resources=CombatResources(hp_current=50, structure_current=4),
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
        )
        scenario = MechCombatScenario(combatants=[attacker, target])
        initial_stats = initialize_statistics_for_scenario(scenario)

        # Create an action result with damage
        action_result = ActionExecutionResult(
            success=True,
            damage_dealt=25,
            resource_changes=[ResourceChange(combatant_id="target", hp_change=-25)],
        )

        updated = update_statistics_from_action(
            initial_stats,
            scenario,
            attacker,
            action_result,
            action_type="full",
        )

        # Check damage was tracked
        assert updated.total_damage_dealt_by_players == 25
        attacker_stats = updated.combatant_stats["attacker"]
        assert attacker_stats.damage_dealt == 25

        target_stats = updated.combatant_stats["target"]
        assert target_stats.damage_received == 25
        assert target_stats.lowest_hp_reached == 25  # 50 - 25

    def test_update_statistics_from_action_tracks_closest_call(self):
        """Test that damage to players tracks closest call."""
        player = CombatantState(
            id="player",
            name="Player",
            side="players",
            kind="mech",
            stats=CombatStats(
                size="size_1",
                hp_max=100,
                evasion=8,
                e_defense=8,
                armor=0,
                speed=4,
                sensor_range=10,
            ),
            resources=CombatResources(hp_current=100, structure_current=4),
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        enemy = CombatantState(
            id="enemy",
            name="Enemy",
            side="hostiles",
            kind="npc",
            stats=CombatStats(
                size="size_1",
                hp_max=50,
                evasion=8,
                e_defense=8,
                armor=0,
                speed=4,
                sensor_range=10,
            ),
            resources=CombatResources(hp_current=50, structure_current=4),
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
        )
        scenario = MechCombatScenario(combatants=[player, enemy])
        initial_stats = initialize_statistics_for_scenario(scenario)

        # Enemy attacks player
        action_result = ActionExecutionResult(
            success=True,
            damage_dealt=70,
            resource_changes=[ResourceChange(combatant_id="player", hp_change=-70)],
        )

        updated = update_statistics_from_action(
            initial_stats,
            scenario,
            enemy,
            action_result,
            action_type="full",
        )

        # Player HP went from 100 to 30
        assert updated.closest_call_hp == 30
        assert updated.closest_call_combatant == "Player"
        assert updated.total_damage_received_by_players == 70

    def test_update_statistics_action_counting(self):
        """Test that action types are counted correctly."""
        combatant = CombatantState(
            id="c1",
            name="Test",
            side="players",
            kind="mech",
            stats=CombatStats(
                size="size_1",
                hp_max=100,
                evasion=8,
                e_defense=8,
                armor=0,
                speed=4,
                sensor_range=10,
            ),
            resources=CombatResources(hp_current=100, structure_current=4),
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        scenario = MechCombatScenario(combatants=[combatant])
        initial_stats = initialize_statistics_for_scenario(scenario)

        # Movement action
        move_result = ActionExecutionResult(
            success=True,
            position_updates={"c1": {"q": 1, "r": 0}},
        )
        updated = update_statistics_from_action(
            initial_stats, scenario, combatant, move_result, action_type="quick"
        )
        assert updated.action_totals.moves == 1

        # Attack action
        attack_result = ActionExecutionResult(
            success=True,
            damage_dealt=10,
        )
        updated = update_statistics_from_action(
            updated, scenario, combatant, attack_result, action_type="full"
        )
        assert updated.action_totals.attacks == 1
        assert updated.action_totals.full_actions == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
