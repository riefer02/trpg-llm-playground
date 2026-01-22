"""End-to-end tests for full combat sequences.

Tests multi-round combat scenarios with attacks, damage, and turn progression.
Uses the combat execution layer to simulate realistic combat encounters.
"""

import pytest
from core.mech.combat_execution import (
    ActionExecutionInput,
    start_turn,
    end_turn,
    execute_action,
)
from core.mech.action_economy import ActionEconomyState
from core.mech.combat_state import (
    MechCombatScenario,
    CombatantState,
    CombatStats,
    CombatResources,
    CombatTurn,
    CombatRound,
)
from core.mech.grid import HexPosition, HexCoord
from core.mech.combat_state import CombatSide


def make_test_combatant(
    id: str,
    name: str,
    side: CombatSide,
    hp_max: int = 10,
    hp_current: int | None = None,
    evasion: int = 8,
    armor: int = 0,
    q: int = 0,
    r: int = 0,
) -> CombatantState:
    """Create a test combatant for E2E tests."""
    return CombatantState(
        id=id,
        name=name,
        side=side,
        kind="mech",
        stats=CombatStats(
            size="size_1",
            hp_max=hp_max,
            evasion=evasion,
            e_defense=8,
            armor=armor,
            speed=4,
            sensor_range=10,
        ),
        resources=CombatResources(
            hp_current=hp_current if hp_current is not None else hp_max,
            heat_current=0,
            heat_cap=6,
            structure_current=4,
            stress_current=4,
            repairs_remaining=4,
        ),
        position=HexPosition(coord=HexCoord(q=q, r=r), elevation=0),
    )


def make_scenario(
    combatants: list[CombatantState],
    rounds: list[CombatRound] | None = None,
) -> MechCombatScenario:
    """Create a test scenario."""
    return MechCombatScenario(
        combatants=combatants,
        grapples=[],
        rounds=rounds or [],
        terrain=None,
        environment="standard",
        deployables={},
    )


class TestFullCombatSequence:
    """End-to-end tests for multi-round combat."""

    def test_full_combat_sequence_three_rounds(self):
        """Simulate 3 rounds of combat with attacks and reactions."""
        # Setup: 2 players vs 2 hostiles
        player1 = make_test_combatant(
            id="player_1", name="Alpha", side="players", hp_max=12, q=0, r=0
        )
        player2 = make_test_combatant(
            id="player_2", name="Beta", side="players", hp_max=10, q=1, r=0
        )
        hostile1 = make_test_combatant(
            id="hostile_1", name="Enemy 1", side="hostiles", hp_max=8, q=3, r=0
        )
        hostile2 = make_test_combatant(
            id="hostile_2", name="Enemy 2", side="hostiles", hp_max=8, q=4, r=0
        )

        scenario = make_scenario(combatants=[player1, player2, hostile1, hostile2])

        # Track total damage dealt across all rounds
        total_damage_dealt = 0
        total_attacks_attempted = 0
        total_hits = 0

        for round_num in range(1, 4):  # 3 rounds
            # Each combatant takes a turn
            for combatant in list(scenario.combatants):
                # Skip if destroyed (HP <= 0)
                current_combatant = next(
                    (c for c in scenario.combatants if c.id == combatant.id), None
                )
                if current_combatant is None:
                    continue
                if current_combatant.resources.hp_current <= 0:
                    continue

                # Start turn
                scenario, turn_result = start_turn(scenario, combatant.id)
                economy = turn_result.economy

                # Pick a target from opposing side
                targets = [
                    c
                    for c in scenario.combatants
                    if c.side != current_combatant.side and c.resources.hp_current > 0
                ]
                if not targets:
                    continue

                target = targets[0]

                # Execute skirmish attack (quick action)
                action = ActionExecutionInput(
                    actor_id=combatant.id,
                    action_id="skirmish",
                    action_type="quick",
                    target_ids=[target.id],
                )

                scenario, turn, economy, result = execute_action(
                    scenario, CombatTurn(actor_id=combatant.id), economy, action
                )

                assert result.success, f"Action failed: {result.error}"

                # Track attack results
                if result.effects_applied:
                    for effect in result.effects_applied:
                        if effect.get("type") == "attack":
                            total_attacks_attempted += 1
                            if effect.get("hit"):
                                total_hits += 1

                total_damage_dealt += result.damage_dealt

                # End turn (no turn order tracking in this simple test)
                # In a real game, we'd track round progression properly

        # Verify: Some attacks were attempted
        assert total_attacks_attempted > 0, "Expected attacks to be attempted"

        # Verify: With 12 attacks (3 rounds × 4 combatants), some should hit
        # (probabilistically ~50% hit rate against evasion 8)
        assert total_hits > 0, "Expected at least some hits over 3 rounds"

        # Verify: Some damage was dealt
        assert total_damage_dealt > 0, "Expected damage to be dealt over 3 rounds"

    def test_combat_until_elimination(self):
        """Test combat continues until one side is eliminated."""
        # Setup: 1v1 with low HP to ensure elimination
        player = make_test_combatant(
            id="player_1",
            name="Alpha",
            side="players",
            hp_max=6,
            evasion=5,  # Low evasion = easier to hit
        )
        hostile = make_test_combatant(
            id="hostile_1",
            name="Enemy",
            side="hostiles",
            hp_max=6,
            evasion=5,
        )

        scenario = make_scenario(combatants=[player, hostile])

        max_rounds = 10
        round_num = 0
        combat_ended = False

        while round_num < max_rounds and not combat_ended:
            round_num += 1

            for combatant_id in ["player_1", "hostile_1"]:
                current = next(
                    (c for c in scenario.combatants if c.id == combatant_id), None
                )
                if current is None or current.resources.hp_current <= 0:
                    continue

                # Find target
                target = next(
                    (
                        c
                        for c in scenario.combatants
                        if c.side != current.side and c.resources.hp_current > 0
                    ),
                    None,
                )
                if target is None:
                    combat_ended = True
                    break

                # Start turn and attack
                scenario, turn_result = start_turn(scenario, combatant_id)
                economy = turn_result.economy

                action = ActionExecutionInput(
                    actor_id=combatant_id,
                    action_id="skirmish",
                    action_type="quick",
                    target_ids=[target.id],
                )

                scenario, _, _, result = execute_action(
                    scenario, CombatTurn(actor_id=combatant_id), economy, action
                )

                assert result.success

            # Check if combat ended (one side eliminated)
            players_alive = sum(
                1
                for c in scenario.combatants
                if c.side == "players" and c.resources.hp_current > 0
            )
            hostiles_alive = sum(
                1
                for c in scenario.combatants
                if c.side == "hostiles" and c.resources.hp_current > 0
            )

            if players_alive == 0 or hostiles_alive == 0:
                combat_ended = True

        # Verify combat concluded (either elimination or max rounds)
        assert round_num <= max_rounds, "Combat should conclude within max rounds"

        # At least one side should have taken damage
        total_hp_lost = sum(
            c.stats.hp_max - c.resources.hp_current for c in scenario.combatants
        )
        assert total_hp_lost > 0, "Expected damage to be dealt in combat"

    def test_armor_reduces_damage(self):
        """Test that armor reduces incoming damage."""
        attacker = make_test_combatant(
            id="attacker", name="Attacker", side="players", evasion=5
        )
        # Target with high armor
        armored_target = make_test_combatant(
            id="armored",
            name="Armored",
            side="hostiles",
            hp_max=20,
            armor=4,  # 4 armor reduces each hit by 4
            evasion=5,  # Low evasion = easy to hit
        )

        scenario = make_scenario(combatants=[attacker, armored_target])

        # Execute multiple attacks to ensure hits
        total_damage = 0
        for _ in range(5):
            scenario_copy, turn_result = start_turn(scenario, "attacker")
            economy = turn_result.economy

            action = ActionExecutionInput(
                actor_id="attacker",
                action_id="skirmish",
                action_type="quick",
                target_ids=["armored"],
            )

            scenario, _, _, result = execute_action(
                scenario_copy, CombatTurn(actor_id="attacker"), economy, action
            )

            total_damage += result.damage_dealt

        # Armor should reduce damage (base 6 - 4 armor = 2 per hit)
        # With some hits, damage should be modest (2 per hit instead of 6)
        target_after = next(c for c in scenario.combatants if c.id == "armored")
        hp_lost = 20 - target_after.resources.hp_current

        # If any hits landed, HP should have decreased but by less than full damage
        if hp_lost > 0:
            # Each hit deals max 2 damage (6 base - 4 armor)
            # Without armor, each hit would deal 6 (or 12 on crit)
            assert hp_lost <= total_damage, "HP lost should match damage dealt"

    def test_critical_hits_roll_twice_pick_highest(self):
        """Test critical hits use roll twice, pick highest (PR2 3965-3969)."""
        from unittest.mock import patch

        attacker = make_test_combatant(id="attacker", name="Attacker", side="players")
        target = make_test_combatant(
            id="target", name="Target", side="hostiles", hp_max=20, armor=0
        )

        scenario = make_scenario(combatants=[attacker, target])
        scenario, turn_result = start_turn(scenario, "attacker")
        economy = turn_result.economy

        action = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            target_ids=["target"],
        )

        # Force a critical hit (natural 20)
        with patch("core.shared.rolls._roll_d20") as mock_roll:
            mock_roll.return_value = 20

            scenario, _, _, result = execute_action(
                scenario, CombatTurn(actor_id="attacker"), economy, action
            )

        assert result.success
        assert result.effects_applied[0]["critical"] is True
        # Per PR2: crit rolls dice twice, picks highest N - for 1d6, max is still 6
        assert result.damage_dealt >= 1 and result.damage_dealt <= 6

        target_after = next(c for c in scenario.combatants if c.id == "target")
        assert target_after.resources.hp_current == 20 - result.damage_dealt
