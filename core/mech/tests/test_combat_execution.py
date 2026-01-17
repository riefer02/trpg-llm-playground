"""Tests for combat turn execution and action resolution."""

import pytest
from core.mech.combat_execution import (
    ActionExecutionInput,
    ActionExecutionResult,
    TurnStartResult,
    TurnEndResult,
    ReactionInput,
    ReactionResult,
    AvailableAction,
    AvailableActionsResult,
    ResourceChange,
    start_turn,
    end_turn,
    execute_action,
    execute_reaction,
    get_available_actions,
    get_current_actor,
    apply_damage,
    apply_heat,
    clear_heat,
    lookup_weapon_damage_and_ap,
    check_structure_cascade,
    check_overheat_cascade,
)
from core.mech.action_economy import (
    ActionEconomyState,
    use_full_action,
    use_quick_action,
)
from core.mech.combat_state import (
    MechCombatScenario,
    CombatantState,
    CombatStats,
    CombatResources,
    CombatTurn,
    CombatRound,
    OverchargeState,
)
from core.mech.grid import HexPosition, HexCoord
from core.shared.effects import CooldownState


# =============================================================================
# Test Fixtures
# =============================================================================


def make_combatant(
    id: str = "mech_1",
    name: str = "Test Mech",
    side: str = "players",
    hp_max: int = 10,
    hp_current: int = 10,
    heat_current: int = 0,
    heat_cap: int = 6,
    grit: int = 0,
    **kwargs,
) -> CombatantState:
    """Create a test combatant."""
    return CombatantState(
        id=id,
        name=name,
        side=side,
        kind="mech",
        stats=CombatStats(
            size="size_1",
            hp_max=hp_max,
            evasion=8,
            e_defense=8,
            armor=0,
            speed=4,
            sensor_range=10,
            grit=grit,
        ),
        resources=CombatResources(
            hp_current=hp_current,
            heat_current=heat_current,
            heat_cap=heat_cap,
            structure_current=4,
            stress_current=4,
            repairs_remaining=4,
        ),
        position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        **kwargs,
    )


def make_scenario(
    combatants: list[CombatantState] | None = None,
    rounds: list[CombatRound] | None = None,
) -> MechCombatScenario:
    """Create a test scenario."""
    return MechCombatScenario(
        combatants=combatants or [],
        grapples=[],
        rounds=rounds or [],
        terrain=None,
        environment="standard",
        deployables={},
    )


def make_turn(actor_id: str = "mech_1") -> CombatTurn:
    """Create a test combat turn."""
    return CombatTurn(
        actor_id=actor_id,
        move_used=False,
        movement_mode="ground",
        movement_path=[],
        actions=[],
    )


def make_round(round_index: int = 1, turns: list[CombatTurn] | None = None) -> CombatRound:
    """Create a test combat round."""
    return CombatRound(
        round_index=round_index,
        turns=turns or [],
        reaction_counts_by_actor={},
    )


# =============================================================================
# Turn Start Tests
# =============================================================================


class TestStartTurn:
    """Tests for start_turn function."""

    def test_start_turn_basic(self):
        """Test basic turn start returns fresh economy."""
        combatant = make_combatant(id="actor_1", name="Test Actor")
        scenario = make_scenario(combatants=[combatant])

        updated_scenario, result = start_turn(scenario, "actor_1")

        assert result.actor_id == "actor_1"
        assert result.actor_name == "Test Actor"
        assert result.economy.full_actions_used == 0
        assert result.economy.quick_actions_used == 0
        assert result.economy.overcharge_used is False
        assert result.prepared_action_expired is False

    def test_start_turn_unknown_actor(self):
        """Test start_turn with unknown actor returns default result."""
        scenario = make_scenario(combatants=[])

        updated_scenario, result = start_turn(scenario, "unknown")

        assert result.actor_id == "unknown"
        assert result.actor_name == "Unknown"
        assert result.economy.full_actions_used == 0

    def test_start_turn_expires_prepared_action(self):
        """Test that prepared actions expire at turn start."""
        from core.mech.timing import PreparedActionState

        prepared = PreparedActionState(
            held_action_id="skirmish",
            held_action_type="quick",
            trigger_condition="enemy enters range",
            created_on_turn=1,
            expires_on_turn=2,
        )
        combatant = make_combatant(id="actor_1", prepared_action=prepared)
        scenario = make_scenario(combatants=[combatant])

        updated_scenario, result = start_turn(scenario, "actor_1")

        assert result.prepared_action_expired is True
        # Check the actor's prepared action was cleared
        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        assert updated_actor.prepared_action is None

    def test_start_turn_decrements_cooldowns(self):
        """Test that turn-start cooldowns are decremented."""
        cooldowns = {
            "effect_1": CooldownState(
                effect_id="effect_1",
                turns_remaining=2,
                duration=2,
                reset_on="turn_start",
            ),
        }
        combatant = make_combatant(id="actor_1", cooldown_states=cooldowns)
        scenario = make_scenario(combatants=[combatant])

        updated_scenario, result = start_turn(scenario, "actor_1")

        assert "effect_1" in result.cooldowns_decremented
        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        assert updated_actor.cooldown_states["effect_1"].turns_remaining == 1

    def test_start_turn_resets_overcharge_uses(self):
        """Test that overcharge uses are reset at turn start."""
        overcharge = OverchargeState(current_level=1, uses_this_turn=1)
        combatant = make_combatant(id="actor_1", overcharge_state=overcharge)
        scenario = make_scenario(combatants=[combatant])

        updated_scenario, result = start_turn(scenario, "actor_1")

        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        assert updated_actor.overcharge_state.uses_this_turn == 0
        assert updated_actor.overcharge_state.current_level == 1  # Level preserved

    def test_start_turn_returns_available_actions(self):
        """Test that turn start returns list of available actions."""
        combatant = make_combatant(id="actor_1")
        scenario = make_scenario(combatants=[combatant])

        updated_scenario, result = start_turn(scenario, "actor_1")

        assert len(result.available_actions) > 0
        assert "skirmish" in result.available_actions
        assert "barrage" in result.available_actions

    def test_start_turn_stunned_actor_limited_actions(self):
        """Test that stunned actors have limited actions."""
        combatant = make_combatant(id="actor_1", statuses=["stunned"])
        scenario = make_scenario(combatants=[combatant])

        updated_scenario, result = start_turn(scenario, "actor_1")

        # Stunned can only mount/dismount
        assert "mount_dismount" in result.available_actions
        assert "skirmish" not in result.available_actions


# =============================================================================
# Turn End Tests
# =============================================================================


class TestEndTurn:
    """Tests for end_turn function."""

    def test_end_turn_advances_to_next_actor(self):
        """Test that end_turn advances to next actor in round."""
        turn1 = make_turn(actor_id="actor_1")
        turn2 = make_turn(actor_id="actor_2")
        round1 = make_round(round_index=1, turns=[turn1, turn2])

        c1 = make_combatant(id="actor_1", name="Actor 1")
        c2 = make_combatant(id="actor_2", name="Actor 2")
        scenario = make_scenario(combatants=[c1, c2], rounds=[round1])

        updated_scenario, result, new_round, new_turn_idx = end_turn(
            scenario, current_round=1, current_turn_index=0, current_turn=turn1
        )

        assert result.actor_id == "actor_1"
        assert result.next_actor_id == "actor_2"
        assert result.next_actor_name == "Actor 2"
        assert result.round_advanced is False
        assert new_round == 1
        assert new_turn_idx == 1

    def test_end_turn_advances_round(self):
        """Test that end_turn advances to next round when current round ends."""
        turn1 = make_turn(actor_id="actor_1")
        turn2 = make_turn(actor_id="actor_2")
        round1 = make_round(round_index=1, turns=[turn1])
        round2 = make_round(round_index=2, turns=[turn2])

        c1 = make_combatant(id="actor_1", name="Actor 1")
        c2 = make_combatant(id="actor_2", name="Actor 2")
        scenario = make_scenario(combatants=[c1, c2], rounds=[round1, round2])

        updated_scenario, result, new_round, new_turn_idx = end_turn(
            scenario, current_round=1, current_turn_index=0, current_turn=turn1
        )

        assert result.round_advanced is True
        assert result.new_round_number == 2
        assert result.next_actor_id == "actor_2"
        assert new_round == 2
        assert new_turn_idx == 0

    def test_end_turn_resets_per_round_reactions_on_round_advance(self):
        """Test that per-round reactions reset when round advances."""
        turn1 = make_turn(actor_id="actor_1")
        round1 = make_round(round_index=1, turns=[turn1])

        c1 = make_combatant(id="actor_1", per_round_reactions={"brace": 1})
        scenario = make_scenario(combatants=[c1], rounds=[round1])

        updated_scenario, result, new_round, new_turn_idx = end_turn(
            scenario, current_round=1, current_turn_index=0, current_turn=turn1
        )

        assert result.round_advanced is True
        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        assert updated_actor.per_round_reactions == {}

    def test_end_turn_decrements_turn_end_cooldowns(self):
        """Test that turn-end cooldowns are decremented."""
        cooldowns = {
            "effect_1": CooldownState(
                effect_id="effect_1",
                turns_remaining=2,
                duration=2,
                reset_on="turn_end",
            ),
        }
        combatant = make_combatant(id="actor_1", cooldown_states=cooldowns)
        turn1 = make_turn(actor_id="actor_1")
        round1 = make_round(round_index=1, turns=[turn1])
        scenario = make_scenario(combatants=[combatant], rounds=[round1])

        updated_scenario, result, _, _ = end_turn(
            scenario, current_round=1, current_turn_index=0, current_turn=turn1
        )

        assert "effect_1" in result.cooldowns_decremented
        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        assert updated_actor.cooldown_states["effect_1"].turns_remaining == 1


# =============================================================================
# Action Execution Tests
# =============================================================================


class TestExecuteAction:
    """Tests for execute_action function."""

    def test_execute_quick_action_success(self):
        """Test executing a valid quick action."""
        combatant = make_combatant(id="actor_1")
        scenario = make_scenario(combatants=[combatant])
        turn = make_turn(actor_id="actor_1")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="skirmish",
            action_type="quick",
            target_ids=["target_1"],
        )

        updated_scenario, updated_turn, updated_economy, result = execute_action(
            scenario, turn, economy, action_input
        )

        assert result.success is True
        assert result.action_use is not None
        assert result.action_use.action_id == "skirmish"
        assert result.action_use.action_type == "quick"
        assert updated_economy.quick_actions_used == 1

    def test_execute_full_action_success(self):
        """Test executing a valid full action."""
        combatant = make_combatant(id="actor_1")
        scenario = make_scenario(combatants=[combatant])
        turn = make_turn(actor_id="actor_1")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="barrage",
            action_type="full",
            target_ids=["target_1"],
        )

        updated_scenario, updated_turn, updated_economy, result = execute_action(
            scenario, turn, economy, action_input
        )

        assert result.success is True
        assert updated_economy.full_actions_used == 1

    def test_execute_action_economy_exhausted(self):
        """Test that actions fail when economy is exhausted."""
        combatant = make_combatant(id="actor_1")
        scenario = make_scenario(combatants=[combatant])
        turn = make_turn(actor_id="actor_1")
        economy = ActionEconomyState(full_actions_used=1)  # Full action already used

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="barrage",
            action_type="full",
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is False
        assert "already used" in result.error.lower()

    def test_execute_action_unknown_actor(self):
        """Test that actions fail for unknown actors."""
        scenario = make_scenario(combatants=[])
        turn = make_turn(actor_id="unknown")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="unknown",
            action_id="skirmish",
            action_type="quick",
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_execute_overcharge_generates_heat(self):
        """Test that overcharge generates heat."""
        overcharge = OverchargeState(current_level=0, uses_this_turn=0)
        combatant = make_combatant(id="actor_1", heat_current=0, overcharge_state=overcharge)
        scenario = make_scenario(combatants=[combatant])
        turn = make_turn(actor_id="actor_1")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="overcharge",
            action_type="free",
            is_overcharge=True,
        )

        updated_scenario, _, updated_economy, result = execute_action(
            scenario, turn, economy, action_input
        )

        assert result.success is True
        assert result.heat_generated == 1  # Level 0 = 1 heat
        assert updated_economy.overcharge_used is True

        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        assert updated_actor.resources.heat_current == 1
        assert updated_actor.overcharge_state.current_level == 1

    def test_execute_action_records_in_turn(self):
        """Test that executed actions are recorded in the turn."""
        combatant = make_combatant(id="actor_1")
        scenario = make_scenario(combatants=[combatant])
        turn = make_turn(actor_id="actor_1")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="skirmish",
            action_type="quick",
        )

        _, updated_turn, _, _ = execute_action(scenario, turn, economy, action_input)

        assert len(updated_turn.actions) == 1
        assert updated_turn.actions[0].action_id == "skirmish"

    def test_execute_attack_deals_damage(self):
        """Attack action should resolve hit/miss and apply damage."""
        attacker = make_combatant(id="attacker", name="Attacker", side="players")
        defender = make_combatant(id="defender", name="Defender", side="hostiles", hp_current=10)
        scenario = make_scenario(combatants=[attacker, defender])
        turn = make_turn(actor_id="attacker")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            target_ids=["defender"],
            weapon_id="assault_rifle",
        )

        updated_scenario, updated_turn, updated_economy, result = execute_action(
            scenario, turn, economy, action_input
        )

        assert result.success is True
        assert len(result.effects_applied) == 1
        assert result.effects_applied[0]["type"] == "attack"
        assert result.effects_applied[0]["target_id"] == "defender"
        assert "roll" in result.effects_applied[0]
        assert "hit" in result.effects_applied[0]

        # If hit, damage should be applied
        if result.effects_applied[0]["hit"]:
            assert result.damage_dealt > 0
            defender_after = next(
                c for c in updated_scenario.combatants if c.id == "defender"
            )
            assert defender_after.resources.hp_current < 10  # Started with 10 HP

    def test_execute_attack_with_forced_hit(self):
        """Attack with natural 20 should always hit and deal double damage."""
        from unittest.mock import patch

        attacker = make_combatant(id="attacker", name="Attacker", side="players")
        # Defender with enough HP to take full crit damage
        defender = make_combatant(id="defender", name="Defender", side="hostiles", hp_max=20, hp_current=20)
        scenario = make_scenario(combatants=[attacker, defender])
        turn = make_turn(actor_id="attacker")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            target_ids=["defender"],
        )

        # Force a natural 20 (critical hit)
        with patch("core.shared.rolls._roll_d20") as mock_roll:
            mock_roll.return_value = 20  # Force roll of 20

            updated_scenario, _, _, result = execute_action(
                scenario, turn, economy, action_input
            )

        assert result.success is True
        assert result.effects_applied[0]["hit"] is True
        assert result.effects_applied[0]["critical"] is True
        # Crit damage = base_damage * 2 = 6 * 2 = 12
        assert result.damage_dealt == 12

    def test_execute_attack_missing_target(self):
        """Attack with non-existent target should not error."""
        attacker = make_combatant(id="attacker", name="Attacker", side="players")
        scenario = make_scenario(combatants=[attacker])
        turn = make_turn(actor_id="attacker")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            target_ids=["nonexistent"],
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        # Should succeed (action executes) but no attack effects (target not found)
        assert result.success is True
        assert len(result.effects_applied) == 0


# =============================================================================
# Reaction Tests
# =============================================================================


class TestExecuteReaction:
    """Tests for execute_reaction function."""

    def test_execute_brace_reaction(self):
        """Test executing a brace reaction."""
        combatant = make_combatant(id="reactor_1")
        scenario = make_scenario(combatants=[combatant])
        economy = ActionEconomyState()

        reaction_input = ReactionInput(
            reactor_id="reactor_1",
            reaction_type="brace",
            trigger_action_id="enemy_attack",
        )

        updated_scenario, updated_economy, result = execute_reaction(
            scenario, economy, reaction_input
        )

        assert result.success is True
        assert result.reaction_used == "brace"
        assert updated_economy.reactions_used_this_turn == 1

    def test_execute_overwatch_reaction(self):
        """Test executing an overwatch reaction."""
        combatant = make_combatant(id="reactor_1")
        scenario = make_scenario(combatants=[combatant])
        economy = ActionEconomyState()

        reaction_input = ReactionInput(
            reactor_id="reactor_1",
            reaction_type="overwatch",
            target_ids=["enemy_1"],
        )

        _, _, result = execute_reaction(scenario, economy, reaction_input)

        assert result.success is True
        assert result.reaction_used == "overwatch"

    def test_reaction_fails_when_used_this_round(self):
        """Test that reactions fail when already used this round."""
        combatant = make_combatant(id="reactor_1", per_round_reactions={"brace": 1})
        scenario = make_scenario(combatants=[combatant])
        economy = ActionEconomyState()

        reaction_input = ReactionInput(
            reactor_id="reactor_1",
            reaction_type="brace",
        )

        _, _, result = execute_reaction(scenario, economy, reaction_input)

        assert result.success is False
        assert "already used" in result.error.lower()

    def test_reaction_unknown_reactor(self):
        """Test that reactions fail for unknown reactors."""
        scenario = make_scenario(combatants=[])
        economy = ActionEconomyState()

        reaction_input = ReactionInput(
            reactor_id="unknown",
            reaction_type="brace",
        )

        _, _, result = execute_reaction(scenario, economy, reaction_input)

        assert result.success is False
        assert "not found" in result.error.lower()


# =============================================================================
# Available Actions Tests
# =============================================================================


class TestGetAvailableActions:
    """Tests for get_available_actions function."""

    def test_fresh_economy_all_actions_available(self):
        """Test that fresh economy has all standard actions available."""
        combatant = make_combatant(id="actor_1")
        scenario = make_scenario(combatants=[combatant])
        economy = ActionEconomyState()

        result = get_available_actions(scenario, "actor_1", economy)

        assert result.actor_id == "actor_1"
        assert result.can_overcharge is True

        # Check full actions
        full_action_ids = [a.action_id for a in result.full_actions]
        assert "barrage" in full_action_ids

        # Check quick actions
        quick_action_ids = [a.action_id for a in result.quick_actions]
        assert "skirmish" in quick_action_ids

    def test_exhausted_full_action_not_available(self):
        """Test that full actions show unavailable when exhausted."""
        combatant = make_combatant(id="actor_1")
        scenario = make_scenario(combatants=[combatant])
        economy = ActionEconomyState(full_actions_used=1)

        result = get_available_actions(scenario, "actor_1", economy)

        barrage = next((a for a in result.full_actions if a.action_id == "barrage"), None)
        assert barrage is not None
        assert barrage.is_available is False
        assert barrage.unavailable_reason is not None

    def test_exhausted_quick_actions_not_available(self):
        """Test that quick actions show unavailable when exhausted."""
        combatant = make_combatant(id="actor_1")
        scenario = make_scenario(combatants=[combatant])
        economy = ActionEconomyState(quick_actions_used=2)

        result = get_available_actions(scenario, "actor_1", economy)

        skirmish = next((a for a in result.quick_actions if a.action_id == "skirmish"), None)
        assert skirmish is not None
        assert skirmish.is_available is False

    def test_overcharge_unavailable_when_used(self):
        """Test that overcharge shows unavailable when already used."""
        combatant = make_combatant(id="actor_1")
        scenario = make_scenario(combatants=[combatant])
        economy = ActionEconomyState(overcharge_used=True)

        result = get_available_actions(scenario, "actor_1", economy)

        assert result.can_overcharge is False
        overcharge = next((a for a in result.free_actions if a.action_id == "overcharge"), None)
        assert overcharge is not None
        assert overcharge.is_available is False

    def test_unknown_actor_returns_empty_result(self):
        """Test that unknown actor returns minimal result."""
        scenario = make_scenario(combatants=[])
        economy = ActionEconomyState()

        result = get_available_actions(scenario, "unknown", economy)

        assert result.actor_id == "unknown"
        assert result.can_overcharge is False


# =============================================================================
# Resource Mutation Tests
# =============================================================================


class TestApplyDamage:
    """Tests for apply_damage function."""

    def test_apply_damage_reduces_hp(self):
        """Test that damage reduces HP."""
        combatant = make_combatant(id="target_1", hp_current=10)
        scenario = make_scenario(combatants=[combatant])

        updated_scenario, change, structure_result = apply_damage(scenario, "target_1", damage=3)

        updated_target = next(c for c in updated_scenario.combatants if c.id == "target_1")
        assert updated_target.resources.hp_current == 7
        assert change.hp_change == -3
        assert structure_result is None  # No structure check triggered

    def test_apply_damage_respects_armor(self):
        """Test that armor reduces damage."""
        combatant = CombatantState(
            id="target_1",
            name="Armored",
            side="players",
            kind="mech",
            stats=CombatStats(
                size="size_1",
                hp_max=10,
                evasion=8,
                e_defense=8,
                armor=2,
                speed=4,
                sensor_range=10,
            ),
            resources=CombatResources(
                hp_current=10,
                heat_current=0,
                heat_cap=6,
                structure_current=4,
                stress_current=4,
                repairs_remaining=4,
            ),
        )
        scenario = make_scenario(combatants=[combatant])

        updated_scenario, change, structure_result = apply_damage(scenario, "target_1", damage=5)

        updated_target = next(c for c in updated_scenario.combatants if c.id == "target_1")
        assert updated_target.resources.hp_current == 7  # 10 - (5 - 2 armor) = 7
        assert change.hp_change == -3
        assert structure_result is None

    def test_apply_damage_armor_piercing(self):
        """Test that AP bypasses armor."""
        combatant = CombatantState(
            id="target_1",
            name="Armored",
            side="players",
            kind="mech",
            stats=CombatStats(
                size="size_1",
                hp_max=10,
                evasion=8,
                e_defense=8,
                armor=2,
                speed=4,
                sensor_range=10,
            ),
            resources=CombatResources(
                hp_current=10,
                heat_current=0,
                heat_cap=6,
                structure_current=4,
                stress_current=4,
                repairs_remaining=4,
            ),
        )
        scenario = make_scenario(combatants=[combatant])

        updated_scenario, change, _ = apply_damage(scenario, "target_1", damage=5, armor_piercing=2)

        updated_target = next(c for c in updated_scenario.combatants if c.id == "target_1")
        assert updated_target.resources.hp_current == 5  # Full 5 damage, armor bypassed

    def test_apply_damage_minimum_zero_triggers_cascade(self):
        """Test that damage to 0 HP triggers structure cascade."""
        combatant = make_combatant(id="target_1", hp_current=5)
        scenario = make_scenario(combatants=[combatant])

        updated_scenario, change, structure_result = apply_damage(scenario, "target_1", damage=10)

        # Structure check should trigger when HP reaches 0
        assert structure_result is not None
        assert change.hp_change == -5  # Original HP change before cascade
        assert change.structure_change == -1  # Structure decremented

        updated_target = next(c for c in updated_scenario.combatants if c.id == "target_1")
        # If not destroyed, HP resets to max; structure decremented
        if not structure_result.mech_destroyed:
            assert updated_target.resources.hp_current == 10  # Reset to max
            assert updated_target.resources.structure_current == 3  # 4 - 1


class TestApplyHeat:
    """Tests for apply_heat function."""

    def test_apply_heat_increases_heat(self):
        """Test that heat is added to combatant."""
        combatant = make_combatant(id="target_1", heat_current=2)
        scenario = make_scenario(combatants=[combatant])

        updated_scenario, change, overheat_result = apply_heat(scenario, "target_1", heat=3)

        updated_target = next(c for c in updated_scenario.combatants if c.id == "target_1")
        assert updated_target.resources.heat_current == 5
        assert change.heat_change == 3
        assert overheat_result is None  # No overheat triggered

    def test_apply_heat_triggers_overheat_cascade(self):
        """Test that heat exceeding cap triggers overheat cascade."""
        combatant = make_combatant(id="target_1", heat_current=5, heat_cap=6)
        scenario = make_scenario(combatants=[combatant])

        updated_scenario, change, overheat_result = apply_heat(scenario, "target_1", heat=3)

        updated_target = next(c for c in updated_scenario.combatants if c.id == "target_1")
        # Heat was cleared by overheat cascade
        assert updated_target.resources.heat_current == 0
        # Stress was decremented
        assert updated_target.resources.stress_current == 3  # Started at 4, lost 1
        # Overheat result should be present
        assert overheat_result is not None
        assert overheat_result.outcome in (
            "emergency_shunt", "power_plant_destabilize", "meltdown", "irreversible_meltdown"
        )


class TestClearHeat:
    """Tests for clear_heat function."""

    def test_clear_heat_reduces_heat(self):
        """Test that heat is cleared from combatant."""
        combatant = make_combatant(id="target_1", heat_current=5)
        scenario = make_scenario(combatants=[combatant])

        updated_scenario, change = clear_heat(scenario, "target_1", amount=3)

        updated_target = next(c for c in updated_scenario.combatants if c.id == "target_1")
        assert updated_target.resources.heat_current == 2
        assert change.heat_change == -3

    def test_clear_heat_minimum_zero(self):
        """Test that heat cannot go below zero."""
        combatant = make_combatant(id="target_1", heat_current=2)
        scenario = make_scenario(combatants=[combatant])

        updated_scenario, change = clear_heat(scenario, "target_1", amount=5)

        updated_target = next(c for c in updated_scenario.combatants if c.id == "target_1")
        assert updated_target.resources.heat_current == 0
        assert change.heat_change == -2  # Only cleared 2


# =============================================================================
# Get Current Actor Tests
# =============================================================================


class TestGetCurrentActor:
    """Tests for get_current_actor function."""

    def test_get_current_actor_success(self):
        """Test getting current actor from scenario."""
        c1 = make_combatant(id="actor_1", name="First")
        c2 = make_combatant(id="actor_2", name="Second")
        turn1 = make_turn(actor_id="actor_1")
        turn2 = make_turn(actor_id="actor_2")
        round1 = make_round(round_index=1, turns=[turn1, turn2])
        scenario = make_scenario(combatants=[c1, c2], rounds=[round1])

        actor = get_current_actor(scenario, current_round=1, current_turn_index=0)

        assert actor is not None
        assert actor.id == "actor_1"
        assert actor.name == "First"

    def test_get_current_actor_second_turn(self):
        """Test getting second actor in round."""
        c1 = make_combatant(id="actor_1", name="First")
        c2 = make_combatant(id="actor_2", name="Second")
        turn1 = make_turn(actor_id="actor_1")
        turn2 = make_turn(actor_id="actor_2")
        round1 = make_round(round_index=1, turns=[turn1, turn2])
        scenario = make_scenario(combatants=[c1, c2], rounds=[round1])

        actor = get_current_actor(scenario, current_round=1, current_turn_index=1)

        assert actor is not None
        assert actor.id == "actor_2"

    def test_get_current_actor_no_rounds(self):
        """Test that no rounds returns None."""
        scenario = make_scenario(combatants=[], rounds=[])

        actor = get_current_actor(scenario, current_round=1, current_turn_index=0)

        assert actor is None

    def test_get_current_actor_invalid_round(self):
        """Test that invalid round returns None."""
        round1 = make_round(round_index=1, turns=[])
        scenario = make_scenario(combatants=[], rounds=[round1])

        actor = get_current_actor(scenario, current_round=5, current_turn_index=0)

        assert actor is None

    def test_get_current_actor_invalid_turn_index(self):
        """Test that invalid turn index returns None."""
        turn1 = make_turn(actor_id="actor_1")
        round1 = make_round(round_index=1, turns=[turn1])
        scenario = make_scenario(combatants=[], rounds=[round1])

        actor = get_current_actor(scenario, current_round=1, current_turn_index=5)

        assert actor is None


# =============================================================================
# Weapon Damage Lookup Tests
# =============================================================================


class TestWeaponDamageLookup:
    """Tests for weapon damage lookup in combat execution."""

    def test_lookup_returns_default_for_none_weapon(self):
        """Lookup with None weapon_id should return default damage."""
        damage, ap = lookup_weapon_damage_and_ap(None)

        assert damage == 6
        assert ap == 0

    def test_lookup_returns_default_for_unknown_weapon(self):
        """Lookup with unknown weapon should fall back to default damage."""
        damage, ap = lookup_weapon_damage_and_ap("nonexistent_weapon_xyz")

        assert damage == 6
        assert ap == 0

    def test_lookup_with_known_weapon_returns_damage(self):
        """Lookup with compendium weapon should return rolled damage."""
        # assault_rifle has 1d6 kinetic damage
        damage, ap = lookup_weapon_damage_and_ap("assault_rifle")

        # Should be in valid d6 range (1-6)
        assert 1 <= damage <= 6
        assert ap == 0  # Assault rifle has no AP

    def test_attack_uses_actor_grit_for_attack_bonus(self):
        """Attack bonus should come from actor's grit stat."""
        from unittest.mock import patch

        # Create attacker with grit=3
        attacker = make_combatant(id="attacker", name="Attacker", side="players", grit=3)
        defender = make_combatant(id="defender", name="Defender", side="hostiles", hp_max=20, hp_current=20)
        scenario = make_scenario(combatants=[attacker, defender])
        turn = make_turn(actor_id="attacker")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            target_ids=["defender"],
        )

        # Force a roll of 10, with grit 3 the total should be 13
        with patch("core.shared.rolls._roll_d20") as mock_roll:
            mock_roll.return_value = 10

            updated_scenario, _, _, result = execute_action(
                scenario, turn, economy, action_input
            )

        assert result.success is True
        assert len(result.effects_applied) == 1
        # With grit 3 and roll 10, total should be 13
        # Defender evasion is 8, so 13 >= 8 should hit
        assert result.effects_applied[0]["total"] == 13
        assert result.effects_applied[0]["hit"] is True

    def test_attack_with_compendium_weapon_uses_weapon_damage(self):
        """Attack with compendium weapon should use its damage."""
        from unittest.mock import patch

        attacker = make_combatant(id="attacker", name="Attacker", side="players")
        defender = make_combatant(id="defender", name="Defender", side="hostiles", hp_max=30, hp_current=30)
        scenario = make_scenario(combatants=[attacker, defender])
        turn = make_turn(actor_id="attacker")
        economy = ActionEconomyState()

        # Use assault_rifle (1d6 kinetic)
        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            target_ids=["defender"],
            weapon_id="assault_rifle",
        )

        # Force hit with a natural 20 (crit)
        with patch("core.shared.rolls._roll_d20") as mock_d20, \
             patch("core.shared.dice.random.randint") as mock_dice:
            mock_d20.return_value = 20  # Crit
            mock_dice.return_value = 4  # Roll 4 on the d6

            updated_scenario, _, _, result = execute_action(
                scenario, turn, economy, action_input
            )

        assert result.success is True
        assert result.effects_applied[0]["hit"] is True
        assert result.effects_applied[0]["critical"] is True
        # Crit doubles damage: 4 * 2 = 8
        assert result.damage_dealt == 8

        defender_after = next(c for c in updated_scenario.combatants if c.id == "defender")
        assert defender_after.resources.hp_current == 22  # 30 - 8

    def test_attack_with_unknown_weapon_uses_default_damage(self):
        """Attack with unknown weapon should fall back to default damage."""
        from unittest.mock import patch

        attacker = make_combatant(id="attacker", name="Attacker", side="players")
        defender = make_combatant(id="defender", name="Defender", side="hostiles", hp_max=20, hp_current=20)
        scenario = make_scenario(combatants=[attacker, defender])
        turn = make_turn(actor_id="attacker")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            target_ids=["defender"],
            weapon_id="nonexistent_weapon",  # Unknown weapon
        )

        # Force a natural 20 (crit) to ensure hit
        with patch("core.shared.rolls._roll_d20") as mock_roll:
            mock_roll.return_value = 20

            updated_scenario, _, _, result = execute_action(
                scenario, turn, economy, action_input
            )

        assert result.success is True
        assert result.effects_applied[0]["critical"] is True
        # Default damage is 6, doubled for crit = 12
        assert result.damage_dealt == 12

        defender_after = next(c for c in updated_scenario.combatants if c.id == "defender")
        assert defender_after.resources.hp_current == 8  # 20 - 12


# =============================================================================
# Structure Cascade Tests
# =============================================================================


class TestStructureCascade:
    """Tests for structure damage cascade on HP=0."""

    def test_damage_to_zero_hp_triggers_structure_check(self):
        """When HP reaches 0, structure check should trigger."""
        combatant = make_combatant(id="target_1", hp_current=5, hp_max=10)
        scenario = make_scenario(combatants=[combatant])

        updated_scenario, change, structure_result = apply_damage(
            scenario, "target_1", damage=10
        )

        # Structure check should have triggered
        assert structure_result is not None
        assert structure_result.outcome in (
            "glancing_blow", "system_trauma", "direct_hit", "crushing_hit"
        )

    def test_structure_check_reduces_structure(self):
        """Structure check should decrement structure_current."""
        combatant = make_combatant(id="target_1", hp_current=5, hp_max=10)
        scenario = make_scenario(combatants=[combatant])

        updated_scenario, change, structure_result = apply_damage(
            scenario, "target_1", damage=10
        )

        updated_target = next(
            c for c in updated_scenario.combatants if c.id == "target_1"
        )
        # Structure should have decremented from 4 to 3
        assert updated_target.resources.structure_current == 3
        assert change.structure_change == -1

    def test_structure_check_resets_hp_to_max(self):
        """After structure loss (non-destruction), HP should reset to max."""
        from unittest.mock import patch

        combatant = make_combatant(id="target_1", hp_current=5, hp_max=10)
        scenario = make_scenario(combatants=[combatant])

        # Force a roll of 5-6 (glancing blow) to avoid destruction
        with patch("core.shared.dice.DiceExpression.roll") as mock_roll:
            mock_roll.return_value = [5]  # Glancing blow

            updated_scenario, change, structure_result = apply_damage(
                scenario, "target_1", damage=10
            )

        updated_target = next(
            c for c in updated_scenario.combatants if c.id == "target_1"
        )
        # HP should reset to max after non-destroying structure check
        assert updated_target.resources.hp_current == 10
        assert structure_result.mech_destroyed is False

    def test_structure_check_applies_statuses(self):
        """Structure outcomes should apply appropriate statuses."""
        from unittest.mock import patch

        combatant = make_combatant(id="target_1", hp_current=5, hp_max=10)
        scenario = make_scenario(combatants=[combatant])

        # Force a roll of 5-6 (glancing blow) which applies impaired
        with patch("core.shared.dice.DiceExpression.roll") as mock_roll:
            mock_roll.return_value = [5]

            updated_scenario, _, structure_result = apply_damage(
                scenario, "target_1", damage=10
            )

        assert structure_result.outcome == "glancing_blow"
        assert "impaired" in structure_result.statuses_to_apply

        updated_target = next(
            c for c in updated_scenario.combatants if c.id == "target_1"
        )
        assert "impaired" in updated_target.statuses

    def test_zero_structure_means_destruction(self):
        """When structure reaches 0 via direct hit, mech is destroyed."""
        from unittest.mock import patch

        # Start with 1 structure
        combatant = CombatantState(
            id="target_1",
            name="Almost Dead",
            side="hostiles",
            kind="mech",
            stats=CombatStats(
                size="size_1",
                hp_max=10,
                evasion=8,
                e_defense=8,
            ),
            resources=CombatResources(
                hp_current=5,
                structure_current=1,
                stress_current=4,
            ),
        )
        scenario = make_scenario(combatants=[combatant])

        # Force a roll of 1 (direct hit at 1 structure = destroyed)
        with patch("core.shared.dice.DiceExpression.roll") as mock_roll:
            mock_roll.return_value = [1]

            updated_scenario, _, structure_result = apply_damage(
                scenario, "target_1", damage=10
            )

        assert structure_result.mech_destroyed is True
        assert structure_result.outcome == "direct_hit"

        updated_target = next(
            c for c in updated_scenario.combatants if c.id == "target_1"
        )
        assert updated_target.resources.structure_current == 0
        assert updated_target.resources.hp_current == 0

    def test_no_cascade_when_hp_above_zero(self):
        """Structure check should not trigger when HP stays above 0."""
        combatant = make_combatant(id="target_1", hp_current=10, hp_max=10)
        scenario = make_scenario(combatants=[combatant])

        updated_scenario, change, structure_result = apply_damage(
            scenario, "target_1", damage=5
        )

        # No structure check because HP didn't reach 0
        assert structure_result is None
        updated_target = next(
            c for c in updated_scenario.combatants if c.id == "target_1"
        )
        assert updated_target.resources.hp_current == 5
        assert updated_target.resources.structure_current == 4  # Unchanged


# =============================================================================
# Overheat Cascade Tests
# =============================================================================


class TestOverheatCascade:
    """Tests for stress cascade on heat overflow."""

    def test_heat_over_cap_triggers_stress_check(self):
        """When heat exceeds cap, stress check should trigger."""
        combatant = make_combatant(id="target_1", heat_current=5, heat_cap=6)
        scenario = make_scenario(combatants=[combatant])

        updated_scenario, change, overheat_result = apply_heat(
            scenario, "target_1", heat=3
        )

        # Overheat check should have triggered
        assert overheat_result is not None
        assert overheat_result.outcome in (
            "emergency_shunt", "power_plant_destabilize",
            "meltdown", "irreversible_meltdown"
        )

    def test_stress_check_clears_heat(self):
        """Overheat should clear all heat."""
        combatant = make_combatant(id="target_1", heat_current=5, heat_cap=6)
        scenario = make_scenario(combatants=[combatant])

        updated_scenario, _, overheat_result = apply_heat(
            scenario, "target_1", heat=3
        )

        updated_target = next(
            c for c in updated_scenario.combatants if c.id == "target_1"
        )
        # Heat should be cleared
        assert updated_target.resources.heat_current == 0
        assert overheat_result.heat_cleared is True

    def test_stress_check_decrements_stress(self):
        """Overheat should decrement stress."""
        combatant = make_combatant(id="target_1", heat_current=5, heat_cap=6)
        scenario = make_scenario(combatants=[combatant])

        updated_scenario, change, overheat_result = apply_heat(
            scenario, "target_1", heat=3
        )

        updated_target = next(
            c for c in updated_scenario.combatants if c.id == "target_1"
        )
        # Stress should have decremented from 4 to 3
        assert updated_target.resources.stress_current == 3
        assert change.stress_change == -1

    def test_meltdown_sets_meltdown_state(self):
        """Meltdown outcome at low stress should set meltdown_state on combatant."""
        from unittest.mock import patch

        # Start with 1 stress (meltdown at 1 stress = immediate meltdown)
        combatant = CombatantState(
            id="target_1",
            name="Hot Stuff",
            side="hostiles",
            kind="mech",
            stats=CombatStats(
                size="size_1",
                hp_max=10,
                evasion=8,
                e_defense=8,
            ),
            resources=CombatResources(
                hp_current=10,
                heat_current=5,
                heat_cap=6,
                structure_current=4,
                stress_current=1,
            ),
        )
        scenario = make_scenario(combatants=[combatant])

        # Force a roll of 1 (meltdown) at 1 stress = immediate
        with patch("core.shared.dice.DiceExpression.roll") as mock_roll:
            mock_roll.return_value = [1]

            updated_scenario, _, overheat_result = apply_heat(
                scenario, "target_1", heat=3
            )

        # Should have meltdown state
        assert overheat_result.outcome == "meltdown"
        assert overheat_result.meltdown_state is not None
        assert overheat_result.meltdown_state.is_immediate is True

        updated_target = next(
            c for c in updated_scenario.combatants if c.id == "target_1"
        )
        assert updated_target.meltdown_state is not None

    def test_no_cascade_when_below_cap(self):
        """Overheat check should not trigger when heat stays below cap."""
        combatant = make_combatant(id="target_1", heat_current=2, heat_cap=6)
        scenario = make_scenario(combatants=[combatant])

        updated_scenario, change, overheat_result = apply_heat(
            scenario, "target_1", heat=2
        )

        # No overheat check because heat didn't meet cap
        assert overheat_result is None
        updated_target = next(
            c for c in updated_scenario.combatants if c.id == "target_1"
        )
        assert updated_target.resources.heat_current == 4
        assert updated_target.resources.stress_current == 4  # Unchanged

    def test_overheat_applies_statuses(self):
        """Overheat outcomes should apply appropriate statuses."""
        from unittest.mock import patch

        combatant = make_combatant(id="target_1", heat_current=5, heat_cap=6)
        scenario = make_scenario(combatants=[combatant])

        # Force a roll of 5-6 (emergency shunt) which applies impaired
        with patch("core.shared.dice.DiceExpression.roll") as mock_roll:
            mock_roll.return_value = [5]

            updated_scenario, _, overheat_result = apply_heat(
                scenario, "target_1", heat=3
            )

        assert overheat_result.outcome == "emergency_shunt"
        assert "impaired" in overheat_result.statuses_to_apply

        updated_target = next(
            c for c in updated_scenario.combatants if c.id == "target_1"
        )
        assert "impaired" in updated_target.statuses
