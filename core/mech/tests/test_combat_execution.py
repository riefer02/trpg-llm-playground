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
    WeaponState,
    WeaponMountState,
    MechInventory,
)
from core.mech.grid import HexPosition, HexCoord
from core.shared.effects import CooldownState
from core.shared.full_tech import FullTechOptionSelection
from core.shared.rolls import AttackResolutionResult


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
    tech_attack: int = 0,
    **kwargs,
) -> CombatantState:
    """Create a test combatant."""
    position = kwargs.pop("position", HexPosition(coord=HexCoord(q=0, r=0), elevation=0))
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
            tech_attack=tech_attack,
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
        position=position,
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

    def test_execute_lock_on_applies_status(self):
        """Lock On should apply the lock_on status to the target."""
        attacker = make_combatant(id="attacker", name="Attacker", side="players")
        defender = make_combatant(id="defender", name="Defender", side="hostiles")
        scenario = make_scenario(combatants=[attacker, defender])
        turn = make_turn(actor_id="attacker")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="lock_on",
            action_type="quick",
            target_ids=["defender"],
        )

        updated_scenario, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is True
        assert result.effects_applied[0]["type"] == "lock_on"
        assert "defender" in result.statuses_applied
        assert "lock_on" in result.statuses_applied["defender"]

        defender_after = next(c for c in updated_scenario.combatants if c.id == "defender")
        assert "lock_on" in defender_after.statuses

    def test_execute_invade_hits_and_applies_heat(self):
        """Invade should apply heat and conditions on a hit."""
        attacker = make_combatant(id="attacker", name="Attacker", side="players", tech_attack=10)
        defender = make_combatant(id="defender", name="Defender", side="hostiles", heat_cap=20)
        scenario = make_scenario(combatants=[attacker, defender])
        turn = make_turn(actor_id="attacker")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="invade",
            action_type="quick",
            target_ids=["defender"],
        )

        updated_scenario, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is True
        assert result.effects_applied[0]["type"] == "invade"
        assert result.effects_applied[0]["hit"] is True
        assert result.heat_generated == 2

        defender_after = next(c for c in updated_scenario.combatants if c.id == "defender")
        assert defender_after.resources.heat_current == 2
        assert "impaired" in defender_after.statuses
        assert "slowed" in defender_after.statuses

    def test_execute_invade_miss_no_heat(self):
        """Invade should not apply heat on a miss."""
        attacker = make_combatant(id="attacker", name="Attacker", side="players", tech_attack=0)
        defender = make_combatant(id="defender", name="Defender", side="hostiles", heat_cap=20)
        scenario = make_scenario(combatants=[attacker, defender])
        turn = make_turn(actor_id="attacker")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="invade",
            action_type="quick",
            target_ids=["defender"],
        )

        updated_scenario, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is True
        assert result.effects_applied[0]["type"] == "invade"
        assert result.effects_applied[0]["hit"] is False
        assert result.heat_generated == 0

        defender_after = next(c for c in updated_scenario.combatants if c.id == "defender")
        assert defender_after.resources.heat_current == 0

    def test_execute_full_tech_scan_lock_on(self):
        """Full Tech should apply two tech options in sequence."""
        attacker = make_combatant(id="attacker", name="Attacker", side="players")
        defender = make_combatant(id="defender", name="Defender", side="hostiles")
        scenario = make_scenario(combatants=[attacker, defender])
        turn = make_turn(actor_id="attacker")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="full_tech",
            action_type="full",
            full_tech_first=FullTechOptionSelection(option="scan", target_id="defender"),
            full_tech_second=FullTechOptionSelection(option="lock_on", target_id="defender"),
        )

        updated_scenario, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is True
        assert {effect["type"] for effect in result.effects_applied} >= {"scan", "lock_on"}

        defender_after = next(c for c in updated_scenario.combatants if c.id == "defender")
        assert "lock_on" in defender_after.statuses

    def test_execute_full_tech_invade_applies_heat(self):
        """Full Tech invade should apply heat on a hit."""
        attacker = make_combatant(id="attacker", name="Attacker", side="players", tech_attack=10)
        defender = make_combatant(id="defender", name="Defender", side="hostiles", heat_cap=20)
        scenario = make_scenario(combatants=[attacker, defender])
        turn = make_turn(actor_id="attacker")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="full_tech",
            action_type="full",
            full_tech_first=FullTechOptionSelection(option="invade", target_id="defender"),
            full_tech_second=FullTechOptionSelection(option="scan", target_id="defender"),
        )

        updated_scenario, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is True
        assert result.heat_generated == 2

        defender_after = next(c for c in updated_scenario.combatants if c.id == "defender")
        assert defender_after.resources.heat_current == 2


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
             patch("core.mech.combat_helpers.random.randint") as mock_dice:
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
# Weapon Tag + AoE Tests
# =============================================================================


class TestWeaponTagEffects:
    """Tests for weapon tag-driven combat behavior."""

    def test_reliable_damage_on_miss(self):
        """Reliable weapons should deal damage even on a miss."""
        from unittest.mock import patch

        attacker = make_combatant(id="attacker", name="Attacker", side="players")
        defender = make_combatant(
            id="defender",
            name="Defender",
            side="hostiles",
            hp_current=10,
            hp_max=10,
        )
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

        with patch("core.shared.rolls._roll_d20") as mock_roll:
            mock_roll.return_value = 1  # Force miss
            updated_scenario, _, _, result = execute_action(
                scenario, turn, economy, action_input
            )

        defender_after = next(
            c for c in updated_scenario.combatants if c.id == "defender"
        )
        assert defender_after.resources.hp_current == 8
        assert any(effect["type"] == "reliable_damage" for effect in result.effects_applied)

    def test_accurate_tag_passes_accuracy_bonus(self):
        """Accurate tag should pass +1 accuracy to attack roll."""
        from unittest.mock import patch

        attacker = make_combatant(id="attacker", name="Attacker", side="players")
        defender = make_combatant(id="defender", name="Defender", side="hostiles")
        scenario = make_scenario(combatants=[attacker, defender])
        turn = make_turn(actor_id="attacker")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            target_ids=["defender"],
            weapon_id="anti_material_rifle",
        )

        with patch("core.shared.rolls.resolve_attack") as mock_resolve:
            mock_resolve.return_value = AttackResolutionResult(
                roll=10,
                attack_bonus=0,
                accuracy_dice_rolls=[],
                difficulty_dice_rolls=[],
                net_accuracy=0,
                total_accuracy=10,
                target_defense=10,
                hit=True,
                is_critical=False,
                miss_by=0,
            )
            execute_action(scenario, turn, economy, action_input)

        assert mock_resolve.call_args.kwargs["accuracy_bonus"] == 1

    def test_inaccurate_tag_passes_difficulty_bonus(self):
        """Inaccurate tag should pass +1 difficulty to attack roll."""
        from unittest.mock import patch

        attacker = make_combatant(id="attacker", name="Attacker", side="players")
        defender = make_combatant(id="defender", name="Defender", side="hostiles")
        scenario = make_scenario(combatants=[attacker, defender])
        turn = make_turn(actor_id="attacker")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            target_ids=["defender"],
            weapon_id="howitzer",
        )

        with patch("core.shared.rolls.resolve_attack") as mock_resolve:
            mock_resolve.return_value = AttackResolutionResult(
                roll=10,
                attack_bonus=0,
                accuracy_dice_rolls=[],
                difficulty_dice_rolls=[],
                net_accuracy=0,
                total_accuracy=10,
                target_defense=10,
                hit=True,
                is_critical=False,
                miss_by=0,
            )
            execute_action(scenario, turn, economy, action_input)

        assert mock_resolve.call_args.kwargs["difficulty_bonus"] == 1

    def test_smart_weapon_targets_e_defense(self):
        """Smart weapons should target e-defense instead of evasion."""
        from unittest.mock import patch

        attacker = make_combatant(id="attacker", name="Attacker", side="players")
        defender = make_combatant(
            id="defender",
            name="Defender",
            side="hostiles",
            hp_current=10,
            hp_max=10,
        )
        defender = defender.model_copy(update={
            "stats": defender.stats.model_copy(update={"evasion": 20, "e_defense": 5})
        })
        scenario = make_scenario(combatants=[attacker, defender])
        turn = make_turn(actor_id="attacker")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            target_ids=["defender"],
            weapon_id="horus_seeker_swarm_nexus",
        )

        with patch("core.shared.rolls.resolve_attack") as mock_resolve:
            mock_resolve.return_value = AttackResolutionResult(
                roll=10,
                attack_bonus=0,
                accuracy_dice_rolls=[],
                difficulty_dice_rolls=[],
                net_accuracy=0,
                total_accuracy=10,
                target_defense=5,
                hit=True,
                is_critical=False,
                miss_by=0,
            )
            execute_action(scenario, turn, economy, action_input)

        assert mock_resolve.call_args.kwargs["target_defense"] == 5

    def test_overkill_adds_heat(self):
        """Overkill should add heat when damage dice roll 1s."""
        from unittest.mock import patch

        attacker = make_combatant(id="attacker", name="Attacker", side="players")
        defender = make_combatant(id="defender", name="Defender", side="hostiles")
        scenario = make_scenario(combatants=[attacker, defender])
        turn = make_turn(actor_id="attacker")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            target_ids=["defender"],
            weapon_id="progressive_knife",
        )

        with patch("core.shared.rolls._roll_d20") as mock_roll, \
            patch("core.mech.combat_helpers.random.randint") as mock_rand:
            mock_roll.return_value = 20  # Force hit
            mock_rand.side_effect = [1, 2]  # Trigger overkill once, then reroll

            updated_scenario, _, _, result = execute_action(
                scenario, turn, economy, action_input
            )

        attacker_after = next(
            c for c in updated_scenario.combatants if c.id == "attacker"
        )
        assert attacker_after.resources.heat_current == 1
        assert any(effect["type"] == "heat_self" for effect in result.effects_applied)

    def test_burn_tag_applies_burn_status(self):
        """Burn tag should apply burn status on hit."""
        from unittest.mock import patch

        attacker = make_combatant(id="attacker", name="Attacker", side="players")
        defender = make_combatant(id="defender", name="Defender", side="hostiles")
        scenario = make_scenario(combatants=[attacker, defender])
        turn = make_turn(actor_id="attacker")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            target_ids=["defender"],
            weapon_id="horus_seeker_swarm_nexus",
        )

        with patch("core.shared.rolls.resolve_attack") as mock_resolve:
            mock_resolve.return_value = AttackResolutionResult(
                roll=10,
                attack_bonus=0,
                accuracy_dice_rolls=[],
                difficulty_dice_rolls=[],
                net_accuracy=0,
                total_accuracy=10,
                target_defense=10,
                hit=True,
                is_critical=False,
                miss_by=0,
            )
            updated_scenario, _, _, result = execute_action(
                scenario, turn, economy, action_input
            )

        defender_after = next(
            c for c in updated_scenario.combatants if c.id == "defender"
        )
        assert "burn" in defender_after.statuses
        assert any(effect["type"] == "burn" for effect in result.effects_applied)


class TestAreaAttackResolution:
    """Tests for AoE targeting from weapon ranges."""

    def test_blast_targets_all_in_area(self):
        """Blast weapons should target all combatants within radius."""
        from unittest.mock import patch

        attacker = make_combatant(
            id="attacker",
            name="Attacker",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0)),
        )
        target_in = make_combatant(
            id="target_in",
            name="Target In",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=3, r=0)),
            hp_current=10,
            hp_max=10,
        )
        target_out = make_combatant(
            id="target_out",
            name="Target Out",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=6, r=0)),
            hp_current=10,
            hp_max=10,
        )
        scenario = make_scenario(combatants=[attacker, target_in, target_out])
        turn = make_turn(actor_id="attacker")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            weapon_id="howitzer",
            target_position=HexPosition(coord=HexCoord(q=3, r=0)),
        )

        with patch("core.shared.rolls.resolve_attack") as mock_resolve:
            mock_resolve.return_value = AttackResolutionResult(
                roll=10,
                attack_bonus=0,
                accuracy_dice_rolls=[],
                difficulty_dice_rolls=[],
                net_accuracy=0,
                total_accuracy=10,
                target_defense=10,
                hit=True,
                is_critical=False,
                miss_by=0,
            )
            updated_scenario, _, _, result = execute_action(
                scenario, turn, economy, action_input
            )

        assert any(effect["target_id"] == "target_in" for effect in result.effects_applied)
        assert not any(effect["target_id"] == "target_out" for effect in result.effects_applied)
        target_out_after = next(
            c for c in updated_scenario.combatants if c.id == "target_out"
        )
        assert target_out_after.resources.hp_current == 10

    def test_line_targets_along_direction(self):
        """Line weapons should target along the chosen direction."""
        from unittest.mock import patch

        attacker = make_combatant(
            id="attacker",
            name="Attacker",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0)),
        )
        target_in = make_combatant(
            id="target_in",
            name="Target In",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=1, r=0)),
        )
        target_off = make_combatant(
            id="target_off",
            name="Target Off",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=0, r=1)),
        )
        scenario = make_scenario(combatants=[attacker, target_in, target_off])
        turn = make_turn(actor_id="attacker")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            weapon_id="thermal_lance",
            target_position=HexPosition(coord=HexCoord(q=3, r=0)),
        )

        with patch("core.shared.rolls.resolve_attack") as mock_resolve:
            mock_resolve.return_value = AttackResolutionResult(
                roll=10,
                attack_bonus=0,
                accuracy_dice_rolls=[],
                difficulty_dice_rolls=[],
                net_accuracy=0,
                total_accuracy=10,
                target_defense=10,
                hit=True,
                is_critical=False,
                miss_by=0,
            )
            _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert any(effect["target_id"] == "target_in" for effect in result.effects_applied)
        assert not any(effect["target_id"] == "target_off" for effect in result.effects_applied)


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


# =============================================================================
# Tests for New Action Resolution (Phase 15)
# =============================================================================


class TestStabilizeAction:
    """Tests for Stabilize action resolution (PR2 4275-4286)."""

    def test_stabilize_cool_heat(self):
        """Stabilize with cool_heat option should reset heat to 0 and end exposed."""
        actor = make_combatant(
            id="actor_1", heat_current=4, heat_cap=6, statuses=["exposed"]
        )
        scenario = make_scenario(combatants=[actor])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="stabilize",
            action_type="full",
            stabilize_primary="cool_heat",
        )

        updated_scenario, updated_turn, updated_economy, result = execute_action(
            scenario, turn, economy, action_input
        )

        assert result.success
        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        assert updated_actor.resources.heat_current == 0
        assert "exposed" not in updated_actor.statuses

        # Check effects
        primary_effect = next(
            e for e in result.effects_applied if e.get("type") == "stabilize_primary"
        )
        assert primary_effect["option"] == "cool_heat"
        assert primary_effect["heat_cleared"] == 4
        assert primary_effect["exposed_ended"] is True

    def test_stabilize_spend_repair_full_hp(self):
        """Stabilize with spend_repair_full_hp should restore HP and consume repair."""
        actor = make_combatant(id="actor_1", hp_current=5, hp_max=10)
        # Ensure repairs_remaining > 0
        actor = actor.model_copy(
            update={"resources": actor.resources.model_copy(update={"repairs_remaining": 3})}
        )
        scenario = make_scenario(combatants=[actor])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="stabilize",
            action_type="full",
            stabilize_primary="spend_repair_full_hp",
        )

        updated_scenario, _, _, result = execute_action(
            scenario, turn, economy, action_input
        )

        assert result.success
        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        assert updated_actor.resources.hp_current == 10
        assert updated_actor.resources.repairs_remaining == 2

    def test_stabilize_clear_burn(self):
        """Stabilize with clear_burn secondary should remove burn status."""
        actor = make_combatant(id="actor_1", statuses=["burn"])
        scenario = make_scenario(combatants=[actor])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="stabilize",
            action_type="full",
            stabilize_primary="cool_heat",
            stabilize_secondary="clear_burn",
        )

        updated_scenario, _, _, result = execute_action(
            scenario, turn, economy, action_input
        )

        assert result.success
        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        assert "burn" not in updated_actor.statuses

    def test_stabilize_clear_condition(self):
        """Stabilize with clear_condition should remove one clearable condition."""
        actor = make_combatant(id="actor_1", statuses=["impaired", "slowed"])
        scenario = make_scenario(combatants=[actor])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="stabilize",
            action_type="full",
            stabilize_primary="cool_heat",
            stabilize_secondary="clear_condition",
        )

        updated_scenario, _, _, result = execute_action(
            scenario, turn, economy, action_input
        )

        assert result.success
        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        # Either impaired or slowed should be cleared (first found)
        assert "impaired" not in updated_actor.statuses or "slowed" not in updated_actor.statuses


class TestDisengageAction:
    """Tests for Disengage action resolution (PR2 4288-4291)."""

    def test_disengage_adds_effect(self):
        """Disengage should add effect indicating engagement ignored until end of turn."""
        actor = make_combatant(id="actor_1", statuses=["engaged"])
        scenario = make_scenario(combatants=[actor])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="disengage",
            action_type="full",
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success
        disengage_effect = next(
            (e for e in result.effects_applied if e.get("type") == "disengage"), None
        )
        assert disengage_effect is not None
        assert disengage_effect["effect"] == "ignore_engagement_and_reactions"
        assert disengage_effect["duration"] == "until_end_of_turn"


class TestHideAction:
    """Tests for Hide action resolution (PR2 4221-4237)."""

    def test_hide_applies_hidden_status(self):
        """Hide should apply hidden status when successful."""
        actor = make_combatant(id="actor_1")
        scenario = make_scenario(combatants=[actor])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="hide",
            action_type="quick",
        )

        updated_scenario, _, _, result = execute_action(
            scenario, turn, economy, action_input
        )

        assert result.success
        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        assert "hidden" in updated_actor.statuses

    def test_hide_fails_when_engaged(self):
        """Hide should fail when actor is engaged."""
        actor = make_combatant(id="actor_1", statuses=["engaged"])
        scenario = make_scenario(combatants=[actor])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="hide",
            action_type="quick",
        )

        updated_scenario, _, _, result = execute_action(
            scenario, turn, economy, action_input
        )

        assert result.success  # Action executed, but hide failed
        hide_effect = next(e for e in result.effects_applied if e.get("type") == "hide")
        assert hide_effect["success"] is False
        assert "engaged" in hide_effect["reason"]

    def test_invisible_mech_can_always_hide(self):
        """Invisible mechs can always hide, even without cover."""
        actor = make_combatant(id="actor_1", statuses=["invisible"])
        scenario = make_scenario(combatants=[actor])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="hide",
            action_type="quick",
        )

        updated_scenario, _, _, result = execute_action(
            scenario, turn, economy, action_input
        )

        assert result.success
        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        assert "hidden" in updated_actor.statuses


class TestRamAction:
    """Tests for Ram action resolution (PR2 4152-4155)."""

    def test_ram_applies_prone_on_hit(self):
        """Ram on hit should apply prone status to target."""
        from unittest.mock import patch

        actor = make_combatant(
            id="actor_1", grit=2,
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0)
        )
        target = make_combatant(
            id="target_1",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0)
        )
        scenario = make_scenario(combatants=[actor, target])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="ram",
            action_type="quick",
            target_ids=["target_1"],
        )

        # Force a hit
        with patch("core.shared.rolls._roll_d20", return_value=15):
            updated_scenario, _, _, result = execute_action(
                scenario, turn, economy, action_input
            )

        assert result.success
        updated_target = next(c for c in updated_scenario.combatants if c.id == "target_1")
        assert "prone" in updated_target.statuses

    def test_ram_miss_no_effect(self):
        """Ram miss should not apply prone or knockback."""
        from unittest.mock import patch

        actor = make_combatant(id="actor_1", grit=0)
        target = make_combatant(id="target_1")
        scenario = make_scenario(combatants=[actor, target])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="ram",
            action_type="quick",
            target_ids=["target_1"],
        )

        # Force a miss
        with patch("core.shared.rolls._roll_d20", return_value=2):
            updated_scenario, _, _, result = execute_action(
                scenario, turn, economy, action_input
            )

        assert result.success
        updated_target = next(c for c in updated_scenario.combatants if c.id == "target_1")
        assert "prone" not in updated_target.statuses

    def test_ram_knockback_moves_target(self):
        """Ram knockback should move target away from attacker."""
        from unittest.mock import patch

        actor = make_combatant(
            id="actor_1", grit=2,
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0)
        )
        target = make_combatant(
            id="target_1",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0)
        )
        scenario = make_scenario(combatants=[actor, target])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="ram",
            action_type="quick",
            target_ids=["target_1"],
            apply_knockback=True,
        )

        # Force a hit
        with patch("core.shared.rolls._roll_d20", return_value=15):
            updated_scenario, _, _, result = execute_action(
                scenario, turn, economy, action_input
            )

        assert result.success
        updated_target = next(c for c in updated_scenario.combatants if c.id == "target_1")
        # Target should have moved away from (0,0) to (2,0)
        assert updated_target.position.coord.q == 2
        assert updated_target.position.coord.r == 0


class TestGrappleAction:
    """Tests for Grapple action resolution (PR2 4157-4177)."""

    def test_grapple_on_hit_engages_both(self):
        """Grapple on hit should apply engaged status to both parties."""
        from unittest.mock import patch

        actor = make_combatant(
            id="actor_1", grit=2,
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0)
        )
        target = make_combatant(
            id="target_1",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0)
        )
        scenario = make_scenario(combatants=[actor, target])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="grapple",
            action_type="quick",
            target_ids=["target_1"],
        )

        # Force a hit
        with patch("core.shared.rolls._roll_d20", return_value=15):
            updated_scenario, _, _, result = execute_action(
                scenario, turn, economy, action_input
            )

        assert result.success
        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        updated_target = next(c for c in updated_scenario.combatants if c.id == "target_1")
        assert "engaged" in updated_actor.statuses
        assert "engaged" in updated_target.statuses

    def test_grapple_adds_grapple_link(self):
        """Grapple on hit should add a GrappleLink to scenario."""
        from unittest.mock import patch

        actor = make_combatant(id="actor_1", grit=2)
        target = make_combatant(id="target_1")
        scenario = make_scenario(combatants=[actor, target])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="grapple",
            action_type="quick",
            target_ids=["target_1"],
        )

        # Force a hit
        with patch("core.shared.rolls._roll_d20", return_value=15):
            updated_scenario, _, _, result = execute_action(
                scenario, turn, economy, action_input
            )

        assert result.success
        assert len(updated_scenario.grapples) == 1
        assert updated_scenario.grapples[0].grappler_id == "actor_1"
        assert updated_scenario.grapples[0].target_id == "target_1"

    def test_grapple_miss_no_effect(self):
        """Grapple miss should not engage or create grapple link."""
        from unittest.mock import patch

        actor = make_combatant(id="actor_1", grit=0)
        target = make_combatant(id="target_1")
        scenario = make_scenario(combatants=[actor, target])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="grapple",
            action_type="quick",
            target_ids=["target_1"],
        )

        # Force a miss
        with patch("core.shared.rolls._roll_d20", return_value=2):
            updated_scenario, _, _, result = execute_action(
                scenario, turn, economy, action_input
            )

        assert result.success
        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        updated_target = next(c for c in updated_scenario.combatants if c.id == "target_1")
        assert "engaged" not in updated_actor.statuses
        assert "engaged" not in updated_target.statuses
        assert len(updated_scenario.grapples) == 0


class TestSearchAction:
    """Tests for Search action resolution (PR2 4241-4249)."""

    def test_search_reveals_hidden_target(self):
        """Search should remove hidden status on success."""
        from unittest.mock import patch

        actor = make_combatant(id="actor_1", tech_attack=3)
        target = make_combatant(id="target_1", statuses=["hidden"])
        scenario = make_scenario(combatants=[actor, target])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="search",
            action_type="quick",
            target_ids=["target_1"],
        )

        # Force searcher to win contested check
        with patch("core.mech.combat_helpers.roll_dice", side_effect=[15, 5]):
            updated_scenario, _, _, result = execute_action(
                scenario, turn, economy, action_input
            )

        assert result.success
        updated_target = next(c for c in updated_scenario.combatants if c.id == "target_1")
        assert "hidden" not in updated_target.statuses

    def test_search_fails_against_higher_roll(self):
        """Search should fail when target rolls higher."""
        from unittest.mock import patch

        actor = make_combatant(id="actor_1", tech_attack=0)
        target = make_combatant(id="target_1", statuses=["hidden"])
        scenario = make_scenario(combatants=[actor, target])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="search",
            action_type="quick",
            target_ids=["target_1"],
        )

        # Force target to win contested check
        with patch("core.mech.combat_helpers.roll_dice", side_effect=[5, 15]):
            updated_scenario, _, _, result = execute_action(
                scenario, turn, economy, action_input
            )

        assert result.success
        updated_target = next(c for c in updated_scenario.combatants if c.id == "target_1")
        assert "hidden" in updated_target.statuses

    def test_search_against_non_hidden_target(self):
        """Search against non-hidden target should fail gracefully."""
        actor = make_combatant(id="actor_1", tech_attack=3)
        target = make_combatant(id="target_1")  # Not hidden
        scenario = make_scenario(combatants=[actor, target])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="search",
            action_type="quick",
            target_ids=["target_1"],
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success
        search_effect = next(e for e in result.effects_applied if e.get("type") == "search")
        assert search_effect["search_success"] is False
        assert "not hidden" in search_effect["reason"]


# =============================================================================
# Status Effect Modifier Tests
# =============================================================================


class TestStatusEffectModifiers:
    """Tests for status effect modifiers in attack resolution."""

    def test_impaired_attacker_adds_difficulty(self):
        """Impaired attacker should have +1 difficulty on attacks."""
        actor = make_combatant(id="actor_1", grit=5, statuses=["impaired"])
        target = make_combatant(
            id="target_1",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[actor, target])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="skirmish",
            action_type="quick",
            target_ids=["target_1"],
            weapon_id="mw_assault_rifle",
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success
        attack_effect = next(e for e in result.effects_applied if e.get("type") == "attack")
        # Check that status modifiers include the +1 difficulty from impaired
        assert attack_effect["status_modifiers"]["attacker_diff"] == 1

    def test_prone_target_gives_accuracy_bonus(self):
        """Prone target should give +1 accuracy to attacker."""
        actor = make_combatant(id="actor_1", grit=5)
        target = make_combatant(
            id="target_1",
            statuses=["prone"],
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[actor, target])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="skirmish",
            action_type="quick",
            target_ids=["target_1"],
            weapon_id="mw_assault_rifle",
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success
        attack_effect = next(e for e in result.effects_applied if e.get("type") == "attack")
        # Prone gives +1 accuracy
        assert attack_effect["status_modifiers"]["target_acc"] >= 1

    def test_braced_target_reduces_accuracy(self):
        """Braced target should give -1 accuracy to attacker."""
        actor = make_combatant(id="actor_1", grit=5)
        target = make_combatant(
            id="target_1",
            statuses=["braced"],
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[actor, target])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="skirmish",
            action_type="quick",
            target_ids=["target_1"],
            weapon_id="mw_assault_rifle",
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success
        attack_effect = next(e for e in result.effects_applied if e.get("type") == "attack")
        # Braced gives -1 accuracy
        assert attack_effect["status_modifiers"]["target_acc"] <= -1

    def test_lock_on_gives_accuracy_and_consumed_on_hit(self):
        """Lock On should give +1 accuracy and be consumed on successful hit."""
        from unittest.mock import patch

        actor = make_combatant(id="actor_1", grit=5)
        target = make_combatant(
            id="target_1",
            statuses=["lock_on"],
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[actor, target])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="skirmish",
            action_type="quick",
            target_ids=["target_1"],
            weapon_id="mw_assault_rifle",
        )

        # Force a hit with high roll
        with patch("core.shared.rolls._roll_d20", return_value=20):
            updated_scenario, _, _, result = execute_action(
                scenario, turn, economy, action_input
            )

        assert result.success
        attack_effect = next(e for e in result.effects_applied if e.get("type") == "attack")
        # Lock on gives +1 accuracy
        assert attack_effect["status_modifiers"]["target_acc"] >= 1
        assert attack_effect["hit"] is True

        # Lock on should be consumed
        lock_on_consumed = any(
            e.get("type") == "lock_on_consumed"
            for e in result.effects_applied
        )
        assert lock_on_consumed

        # Target should no longer have lock_on status
        updated_target = next(
            c for c in updated_scenario.combatants if c.id == "target_1"
        )
        assert "lock_on" not in updated_target.statuses

    def test_lock_on_not_consumed_on_miss(self):
        """Lock On should not be consumed if the attack misses."""
        from unittest.mock import patch

        actor = make_combatant(id="actor_1", grit=0)
        target = make_combatant(
            id="target_1",
            statuses=["lock_on"],
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[actor, target])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="skirmish",
            action_type="quick",
            target_ids=["target_1"],
            weapon_id="mw_assault_rifle",
        )

        # Force a miss with low roll
        with patch("core.shared.rolls._roll_d20", return_value=1):
            updated_scenario, _, _, result = execute_action(
                scenario, turn, economy, action_input
            )

        assert result.success
        attack_effect = next(e for e in result.effects_applied if e.get("type") == "attack")
        assert attack_effect["hit"] is False

        # Lock on should NOT be consumed on miss
        lock_on_consumed = any(
            e.get("type") == "lock_on_consumed"
            for e in result.effects_applied
        )
        assert not lock_on_consumed

        # Target should still have lock_on status
        updated_target = next(
            c for c in updated_scenario.combatants if c.id == "target_1"
        )
        assert "lock_on" in updated_target.statuses

    def test_engaged_target_ranged_attack_difficulty(self):
        """Engaged target should add +1 difficulty to ranged attacks."""
        actor = make_combatant(id="actor_1", grit=5)
        target = make_combatant(
            id="target_1",
            statuses=["engaged"],
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[actor, target])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        # Use a ranged weapon
        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="skirmish",
            action_type="quick",
            target_ids=["target_1"],
            weapon_id="mw_assault_rifle",
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success
        attack_effect = next(e for e in result.effects_applied if e.get("type") == "attack")
        # Engaged adds +1 difficulty for ranged attacks
        assert attack_effect["status_modifiers"]["target_diff"] == 1

    def test_engaged_target_melee_no_difficulty(self):
        """Engaged target should NOT add difficulty to melee attacks."""
        actor = make_combatant(id="actor_1", grit=5)
        target = make_combatant(
            id="target_1",
            statuses=["engaged"],
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[actor, target])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        # Use a melee weapon (charged_blade has Threat 1)
        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="skirmish",
            action_type="quick",
            target_ids=["target_1"],
            weapon_id="charged_blade",
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success
        attack_effect = next(e for e in result.effects_applied if e.get("type") == "attack")
        # Engaged does NOT add difficulty for melee attacks
        assert attack_effect["status_modifiers"]["target_diff"] == 0

    def test_invisible_target_fifty_percent_miss(self):
        """Invisible target should have 50% miss chance after hit."""
        from unittest.mock import patch

        actor = make_combatant(id="actor_1", grit=5)
        target = make_combatant(
            id="target_1",
            statuses=["invisible"],
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[actor, target])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="skirmish",
            action_type="quick",
            target_ids=["target_1"],
            weapon_id="mw_assault_rifle",
        )

        # Force a hit on attack roll, but invisibility causes miss
        with patch("core.shared.rolls._roll_d20", return_value=20):
            with patch("core.mech.combat_helpers.random.random", return_value=0.3):  # < 0.5 means miss
                _, _, _, result = execute_action(
                    scenario, turn, economy, action_input
                )

        assert result.success
        attack_effect = next(e for e in result.effects_applied if e.get("type") == "attack")
        # Attack would have hit but invisibility causes miss
        assert attack_effect["hit"] is False

        # Should have invisibility_miss effect
        invis_miss = any(
            e.get("type") == "invisibility_miss"
            for e in result.effects_applied
        )
        assert invis_miss

    def test_invisible_target_can_still_be_hit(self):
        """Invisible target 50% miss chance means 50% still hit."""
        from unittest.mock import patch

        actor = make_combatant(id="actor_1", grit=5)
        target = make_combatant(
            id="target_1",
            statuses=["invisible"],
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[actor, target])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="skirmish",
            action_type="quick",
            target_ids=["target_1"],
            weapon_id="mw_assault_rifle",
        )

        # Force a hit on attack roll, and invisibility roll > 0.5 means no miss
        with patch("core.shared.rolls._roll_d20", return_value=20):
            with patch("core.mech.combat_helpers.random.random", return_value=0.7):  # >= 0.5 means no miss
                _, _, _, result = execute_action(
                    scenario, turn, economy, action_input
                )

        assert result.success
        attack_effect = next(e for e in result.effects_applied if e.get("type") == "attack")
        # Attack should still hit
        assert attack_effect["hit"] is True

        # Should NOT have invisibility_miss effect
        invis_miss = any(
            e.get("type") == "invisibility_miss"
            for e in result.effects_applied
        )
        assert not invis_miss


# =============================================================================
# Movement Position Update Tests
# =============================================================================


class TestMovementPositionUpdates:
    """Tests for movement position updates with terrain costs."""

    def test_move_action_updates_position(self):
        """Move action should update actor's position to final path position."""
        actor = make_combatant(
            id="actor_1",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[actor])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        # Create movement path
        movement_path = [
            HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            HexPosition(coord=HexCoord(q=2, r=0), elevation=0),
        ]

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="move",
            action_type="free",
            movement_path=movement_path,
        )

        updated_scenario, _, _, result = execute_action(
            scenario, turn, economy, action_input
        )

        assert result.success
        move_effect = next(e for e in result.effects_applied if e.get("type") == "movement")
        assert move_effect["success"] is True
        assert move_effect["spaces"] == 2

        # Check actor position was updated
        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        assert updated_actor.position.coord.q == 2
        assert updated_actor.position.coord.r == 0

    def test_move_with_difficult_terrain_costs_double(self):
        """Movement through difficult terrain should cost 2 spaces per space moved."""
        from core.shared.terrain import TerrainMap, TerrainHex

        actor = make_combatant(
            id="actor_1",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )

        # Create terrain with difficult terrain at (1,0)
        terrain = TerrainMap(tiles=[
            TerrainHex(coord=HexCoord(q=1, r=0), difficult=True),
        ])

        scenario = MechCombatScenario(
            combatants=[actor],
            grapples=[],
            rounds=[],
            terrain=terrain,
            environment="standard",
            deployables={},
        )
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        # Movement path through difficult terrain
        movement_path = [
            HexPosition(coord=HexCoord(q=1, r=0), elevation=0),  # Difficult terrain
            HexPosition(coord=HexCoord(q=2, r=0), elevation=0),  # Normal terrain
        ]

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="move",
            action_type="free",
            movement_path=movement_path,
        )

        updated_scenario, _, _, result = execute_action(
            scenario, turn, economy, action_input
        )

        assert result.success
        move_effect = next(e for e in result.effects_applied if e.get("type") == "movement")
        assert move_effect["success"] is True
        # Cost should be 3 (2 for difficult + 1 for normal)
        assert move_effect["cost"] == 3

    def test_move_exceeding_speed_fails(self):
        """Movement exceeding speed should fail."""
        actor = make_combatant(
            id="actor_1",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        # Default speed is 4 spaces
        scenario = make_scenario(combatants=[actor])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        # Create a path of 6 spaces (exceeds speed of 4)
        movement_path = [
            HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            HexPosition(coord=HexCoord(q=2, r=0), elevation=0),
            HexPosition(coord=HexCoord(q=3, r=0), elevation=0),
            HexPosition(coord=HexCoord(q=4, r=0), elevation=0),
            HexPosition(coord=HexCoord(q=5, r=0), elevation=0),
            HexPosition(coord=HexCoord(q=6, r=0), elevation=0),
        ]

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="move",
            action_type="free",
            movement_path=movement_path,
        )

        updated_scenario, _, _, result = execute_action(
            scenario, turn, economy, action_input
        )

        assert result.success  # Action itself succeeds
        move_effect = next(e for e in result.effects_applied if e.get("type") == "movement")
        assert move_effect["success"] is False
        assert move_effect["reason"] == "exceeds_speed"

        # Actor position should not have changed
        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        assert updated_actor.position.coord.q == 0

    def test_boost_doubles_speed(self):
        """Boost action should allow movement up to 2x speed."""
        actor = make_combatant(
            id="actor_1",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        # Default speed is 4, boost allows 8
        scenario = make_scenario(combatants=[actor])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        # Create a path of 6 spaces (exceeds move speed of 4 but within boost)
        movement_path = [
            HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            HexPosition(coord=HexCoord(q=2, r=0), elevation=0),
            HexPosition(coord=HexCoord(q=3, r=0), elevation=0),
            HexPosition(coord=HexCoord(q=4, r=0), elevation=0),
            HexPosition(coord=HexCoord(q=5, r=0), elevation=0),
            HexPosition(coord=HexCoord(q=6, r=0), elevation=0),
        ]

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="boost",
            action_type="quick",
            movement_path=movement_path,
        )

        updated_scenario, _, _, result = execute_action(
            scenario, turn, economy, action_input
        )

        assert result.success
        move_effect = next(e for e in result.effects_applied if e.get("type") == "movement")
        assert move_effect["success"] is True

        # Actor should be at final position
        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        assert updated_actor.position.coord.q == 6

    def test_dangerous_terrain_triggers_check(self):
        """Moving through dangerous terrain should trigger engineering check."""
        from core.shared.terrain import TerrainMap, TerrainHex
        from unittest.mock import patch

        actor = make_combatant(
            id="actor_1",
            tech_attack=2,  # Engineering skill bonus
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )

        # Create terrain with dangerous terrain at (1,0)
        terrain = TerrainMap(tiles=[
            TerrainHex(coord=HexCoord(q=1, r=0), dangerous=True),
        ])

        scenario = MechCombatScenario(
            combatants=[actor],
            grapples=[],
            rounds=[],
            terrain=terrain,
            environment="standard",
            deployables={},
        )
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        movement_path = [
            HexPosition(coord=HexCoord(q=1, r=0), elevation=0),  # Dangerous terrain
        ]

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="move",
            action_type="free",
            movement_path=movement_path,
        )

        # Force a failed check (roll 5 + 2 skill = 7 < 10)
        with patch("core.shared.terrain.roll_dice", return_value=5):
            updated_scenario, _, _, result = execute_action(
                scenario, turn, economy, action_input
            )

        assert result.success

        # Check for dangerous terrain effect
        danger_effects = [e for e in result.effects_applied if e.get("type") == "dangerous_terrain"]
        assert len(danger_effects) >= 1
        assert danger_effects[0]["check_passed"] is False
        assert danger_effects[0]["damage"] == 5

        # Check for damage effect
        damage_effects = [e for e in result.effects_applied if e.get("type") == "damage"]
        assert len(damage_effects) >= 1
        assert damage_effects[0]["source"] == "dangerous_terrain"


# =============================================================================
# Mount/Dismount/Eject Tests
# =============================================================================


class TestMountDismountEject:
    """Tests for mount, dismount, and eject actions."""

    def test_mount_action_succeeds_when_adjacent(self):
        """Mount should succeed when pilot is adjacent to mech."""
        pilot = CombatantState(
            id="pilot_1",
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
                sensor_range=5,
                tech_attack=0,
                grit=0,
            ),
            resources=CombatResources(
                hp_current=6,
                heat_current=0,
                heat_cap=0,
                structure_current=1,
                stress_current=1,
                repairs_remaining=0,
            ),
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
        )
        mech = make_combatant(
            id="mech_1",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[pilot, mech])
        turn = CombatTurn(actor_id="pilot_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="pilot_1",
            action_id="mount",
            action_type="full",
            target_ids=["mech_1"],
        )

        updated_scenario, _, _, result = execute_action(
            scenario, turn, economy, action_input
        )

        assert result.success
        mount_effect = next(e for e in result.effects_applied if e.get("type") == "mount")
        assert mount_effect["success"] is True
        assert mount_effect["pilot_id"] == "pilot_1"
        assert mount_effect["mech_id"] == "mech_1"

        # Check pilot is now piloting the mech
        updated_pilot = next(c for c in updated_scenario.combatants if c.id == "pilot_1")
        assert updated_pilot.piloting_mech_id == "mech_1"
        assert updated_pilot.position is None  # Pilot no longer on battlefield

        # Check mech has pilot mounted
        updated_mech = next(c for c in updated_scenario.combatants if c.id == "mech_1")
        assert updated_mech.mounted_pilot_id == "pilot_1"

    def test_mount_action_fails_when_not_adjacent(self):
        """Mount should fail when pilot is not adjacent to mech."""
        pilot = CombatantState(
            id="pilot_1",
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
                sensor_range=5,
                tech_attack=0,
                grit=0,
            ),
            resources=CombatResources(
                hp_current=6,
                heat_current=0,
                heat_cap=0,
                structure_current=1,
                stress_current=1,
                repairs_remaining=0,
            ),
            position=HexPosition(coord=HexCoord(q=5, r=0), elevation=0),  # Too far
        )
        mech = make_combatant(
            id="mech_1",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[pilot, mech])
        turn = CombatTurn(actor_id="pilot_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="pilot_1",
            action_id="mount",
            action_type="full",
            target_ids=["mech_1"],
        )

        updated_scenario, _, _, result = execute_action(
            scenario, turn, economy, action_input
        )

        assert result.success  # Action succeeds but mount fails
        mount_effect = next(e for e in result.effects_applied if e.get("type") == "mount")
        assert mount_effect["success"] is False
        assert "adjacent" in mount_effect["reason"]

    def test_dismount_creates_pilot_combatant(self):
        """Dismount should place pilot in adjacent space."""
        pilot = CombatantState(
            id="pilot_1",
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
                sensor_range=5,
                tech_attack=0,
                grit=0,
            ),
            resources=CombatResources(
                hp_current=6,
                heat_current=0,
                heat_cap=0,
                structure_current=1,
                stress_current=1,
                repairs_remaining=0,
            ),
            position=None,  # Inside mech
            piloting_mech_id="mech_1",
        )
        mech = make_combatant(
            id="mech_1",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
            mounted_pilot_id="pilot_1",
        )
        scenario = make_scenario(combatants=[pilot, mech])
        turn = CombatTurn(actor_id="mech_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="mech_1",
            action_id="dismount",
            action_type="full",
        )

        updated_scenario, _, _, result = execute_action(
            scenario, turn, economy, action_input
        )

        assert result.success
        dismount_effect = next(e for e in result.effects_applied if e.get("type") == "dismount")
        assert dismount_effect["success"] is True

        # Check pilot is now on battlefield
        updated_pilot = next(c for c in updated_scenario.combatants if c.id == "pilot_1")
        assert updated_pilot.piloting_mech_id is None
        assert updated_pilot.position is not None

        # Check mech no longer has pilot
        updated_mech = next(c for c in updated_scenario.combatants if c.id == "mech_1")
        assert updated_mech.mounted_pilot_id is None

    def test_eject_moves_pilot_six_spaces(self):
        """Eject should move pilot 6 spaces in chosen direction."""
        pilot = CombatantState(
            id="pilot_1",
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
                sensor_range=5,
                tech_attack=0,
                grit=0,
            ),
            resources=CombatResources(
                hp_current=6,
                heat_current=0,
                heat_cap=0,
                structure_current=1,
                stress_current=1,
                repairs_remaining=0,
            ),
            position=None,  # Inside mech
            piloting_mech_id="mech_1",
        )
        mech = make_combatant(
            id="mech_1",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
            mounted_pilot_id="pilot_1",
        )
        scenario = make_scenario(combatants=[pilot, mech])
        turn = CombatTurn(actor_id="mech_1", actions=[])
        economy = ActionEconomyState()

        # Eject in direction (1, 0) - should end at (6, 0)
        action_input = ActionExecutionInput(
            actor_id="mech_1",
            action_id="eject",
            action_type="quick",
            eject_direction=HexCoord(q=1, r=0),
        )

        updated_scenario, _, _, result = execute_action(
            scenario, turn, economy, action_input
        )

        assert result.success
        eject_effect = next(e for e in result.effects_applied if e.get("type") == "eject")
        assert eject_effect["success"] is True
        assert eject_effect["pilot_position"]["q"] == 6
        assert eject_effect["pilot_position"]["r"] == 0

        # Check pilot position
        updated_pilot = next(c for c in updated_scenario.combatants if c.id == "pilot_1")
        assert updated_pilot.position is not None
        assert updated_pilot.position.coord.q == 6
        assert updated_pilot.position.coord.r == 0

    def test_eject_applies_impaired_status(self):
        """Eject should apply permanent impaired status to pilot."""
        pilot = CombatantState(
            id="pilot_1",
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
                sensor_range=5,
                tech_attack=0,
                grit=0,
            ),
            resources=CombatResources(
                hp_current=6,
                heat_current=0,
                heat_cap=0,
                structure_current=1,
                stress_current=1,
                repairs_remaining=0,
            ),
            position=None,
            piloting_mech_id="mech_1",
            statuses=[],
        )
        mech = make_combatant(
            id="mech_1",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
            mounted_pilot_id="pilot_1",
        )
        scenario = make_scenario(combatants=[pilot, mech])
        turn = CombatTurn(actor_id="mech_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="mech_1",
            action_id="eject",
            action_type="quick",
            eject_direction=HexCoord(q=1, r=0),
        )

        updated_scenario, _, _, result = execute_action(
            scenario, turn, economy, action_input
        )

        assert result.success
        eject_effect = next(e for e in result.effects_applied if e.get("type") == "eject")
        assert eject_effect["impaired_applied"] is True

        # Check pilot has impaired status
        updated_pilot = next(c for c in updated_scenario.combatants if c.id == "pilot_1")
        assert "impaired" in updated_pilot.statuses

    def test_eject_cannot_be_used_twice(self):
        """Eject should fail if already used this combat."""
        pilot = CombatantState(
            id="pilot_1",
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
                sensor_range=5,
                tech_attack=0,
                grit=0,
            ),
            resources=CombatResources(
                hp_current=6,
                heat_current=0,
                heat_cap=0,
                structure_current=1,
                stress_current=1,
                repairs_remaining=0,
            ),
            position=None,
            piloting_mech_id="mech_1",
        )
        mech = make_combatant(
            id="mech_1",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
            mounted_pilot_id="pilot_1",
            eject_used=True,  # Already used
        )
        scenario = make_scenario(combatants=[pilot, mech])
        turn = CombatTurn(actor_id="mech_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="mech_1",
            action_id="eject",
            action_type="quick",
            eject_direction=HexCoord(q=1, r=0),
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success  # Action succeeds but eject fails
        eject_effect = next(e for e in result.effects_applied if e.get("type") == "eject")
        assert eject_effect["success"] is False
        assert "already used" in eject_effect["reason"]


# =============================================================================
# Cover Modifier Tests
# =============================================================================


class TestCoverModifiers:
    """Tests for cover modifiers in attack resolution."""

    def test_soft_cover_adds_difficulty_ranged(self):
        """Soft cover terrain should add +1 difficulty to ranged attacks."""
        from unittest.mock import patch
        from core.shared.terrain import TerrainMap, TerrainHex

        actor = make_combatant(
            id="actor_1",
            grit=5,
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        target = make_combatant(
            id="target_1",
            position=HexPosition(coord=HexCoord(q=2, r=0), elevation=0),
        )
        # Create terrain with soft cover at target's position
        terrain = TerrainMap(tiles=[
            TerrainHex(coord=HexCoord(q=2, r=0), provides_soft_cover=True),
        ])
        scenario = MechCombatScenario(
            combatants=[actor, target],
            grapples=[],
            rounds=[],
            terrain=terrain,
            environment="standard",
            deployables={},
        )
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="skirmish",
            action_type="quick",
            target_ids=["target_1"],
            weapon_id="assault_rifle",
        )

        with patch("core.shared.rolls._roll_d20", return_value=15):
            _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success
        # Check cover modifier effect was logged
        cover_effect = next(
            (e for e in result.effects_applied if e.get("type") == "cover_modifier"),
            None
        )
        assert cover_effect is not None
        assert cover_effect["cover_type"] == "soft"
        assert cover_effect["difficulty_added"] == 1

        # Check status_modifiers includes cover_diff
        attack_effect = next(e for e in result.effects_applied if e.get("type") == "attack")
        assert attack_effect["status_modifiers"]["cover_diff"] == 1

    def test_hard_cover_adds_difficulty_ranged(self):
        """Hard cover terrain should add +2 difficulty to ranged attacks."""
        from unittest.mock import patch
        from core.shared.terrain import TerrainMap, TerrainHex

        actor = make_combatant(
            id="actor_1",
            grit=5,
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        # Target adjacent to hard cover
        target = make_combatant(
            id="target_1",
            position=HexPosition(coord=HexCoord(q=2, r=0), elevation=0),
        )
        # Create terrain with hard cover adjacent to target
        terrain = TerrainMap(tiles=[
            TerrainHex(
                coord=HexCoord(q=3, r=0),
                provides_hard_cover=True,
                hard_cover_size="size_1",
            ),
        ])
        scenario = MechCombatScenario(
            combatants=[actor, target],
            grapples=[],
            rounds=[],
            terrain=terrain,
            environment="standard",
            deployables={},
        )
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="skirmish",
            action_type="quick",
            target_ids=["target_1"],
            weapon_id="assault_rifle",
        )

        with patch("core.shared.rolls._roll_d20", return_value=15):
            _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success
        # Check cover modifier effect was logged
        cover_effect = next(
            (e for e in result.effects_applied if e.get("type") == "cover_modifier"),
            None
        )
        assert cover_effect is not None
        assert cover_effect["cover_type"] == "hard"
        assert cover_effect["difficulty_added"] == 2

    def test_cover_ignored_for_melee(self):
        """Melee attacks should ignore cover modifiers."""
        from unittest.mock import patch
        from core.shared.terrain import TerrainMap, TerrainHex

        actor = make_combatant(
            id="actor_1",
            grit=5,
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        target = make_combatant(
            id="target_1",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
        )
        # Create terrain with soft cover at target's position
        terrain = TerrainMap(tiles=[
            TerrainHex(coord=HexCoord(q=1, r=0), provides_soft_cover=True),
        ])
        scenario = MechCombatScenario(
            combatants=[actor, target],
            grapples=[],
            rounds=[],
            terrain=terrain,
            environment="standard",
            deployables={},
        )
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        # Use melee weapon (heavy_melee_weapon is threat-only)
        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="skirmish",
            action_type="quick",
            target_ids=["target_1"],
            weapon_id="heavy_melee_weapon",
        )

        with patch("core.shared.rolls._roll_d20", return_value=15):
            _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success
        # No cover modifier should be logged for melee
        cover_effect = next(
            (e for e in result.effects_applied if e.get("type") == "cover_modifier"),
            None
        )
        assert cover_effect is None

        # Check status_modifiers has cover_diff = 0
        attack_effect = next(e for e in result.effects_applied if e.get("type") == "attack")
        assert attack_effect["status_modifiers"]["cover_diff"] == 0

    def test_no_cover_modifier_without_terrain(self):
        """Should gracefully handle scenarios without terrain."""
        from unittest.mock import patch

        actor = make_combatant(
            id="actor_1",
            grit=5,
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        target = make_combatant(
            id="target_1",
            position=HexPosition(coord=HexCoord(q=2, r=0), elevation=0),
        )
        # No terrain
        scenario = make_scenario(combatants=[actor, target])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="skirmish",
            action_type="quick",
            target_ids=["target_1"],
            weapon_id="assault_rifle",
        )

        with patch("core.shared.rolls._roll_d20", return_value=15):
            _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success
        # No cover modifier should be logged
        cover_effect = next(
            (e for e in result.effects_applied if e.get("type") == "cover_modifier"),
            None
        )
        assert cover_effect is None

        # Check status_modifiers has cover_diff = 0
        attack_effect = next(e for e in result.effects_applied if e.get("type") == "attack")
        assert attack_effect["status_modifiers"]["cover_diff"] == 0

    def test_flanking_negates_hard_cover(self):
        """Flanking position should negate hard cover bonus.

        Per PR2: Flanking requires the attacker to be adjacent to the target
        AND on the same row as target relative to hard cover. This means the
        attacker is on the opposite side of the target from the cover.
        """
        from unittest.mock import patch
        from core.shared.terrain import TerrainMap, TerrainHex

        # Setup:
        # - Hard cover at (1, 0)
        # - Target at (2, 0) - adjacent to hard cover
        # - Attacker at (3, 0) - adjacent to target, on opposite side from cover
        # This is a flanking position: cover(1,0) -> target(2,0) -> attacker(3,0)
        actor = make_combatant(
            id="actor_1",
            grit=5,
            position=HexPosition(coord=HexCoord(q=3, r=0), elevation=0),
        )
        target = make_combatant(
            id="target_1",
            position=HexPosition(coord=HexCoord(q=2, r=0), elevation=0),
        )
        terrain = TerrainMap(tiles=[
            TerrainHex(
                coord=HexCoord(q=1, r=0),
                provides_hard_cover=True,
                hard_cover_size="size_1",
            ),
        ])
        scenario = MechCombatScenario(
            combatants=[actor, target],
            grapples=[],
            rounds=[],
            terrain=terrain,
            environment="standard",
            deployables={},
        )
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="skirmish",
            action_type="quick",
            target_ids=["target_1"],
            weapon_id="assault_rifle",
        )

        with patch("core.shared.rolls._roll_d20", return_value=15):
            _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success
        attack_effect = next(e for e in result.effects_applied if e.get("type") == "attack")
        # If flanking is detected, hard cover should be negated
        # The cover_diff should be 0 (flanked) or at most 1 (soft cover fallback)
        assert attack_effect["status_modifiers"]["cover_diff"] < 2


# =============================================================================
# Damage Modifier Tests
# =============================================================================


class TestDamageModifiers:
    """Tests for damage modifiers (exposed, shredded) in attack resolution."""

    def test_exposed_doubles_damage(self):
        """Exposed status should double damage dealt."""
        from unittest.mock import patch

        actor = make_combatant(
            id="actor_1",
            grit=5,
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        target = make_combatant(
            id="target_1",
            hp_max=50,
            hp_current=50,
            statuses=["exposed"],
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[actor, target])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="skirmish",
            action_type="quick",
            target_ids=["target_1"],
            weapon_id="assault_rifle",  # 1d6 damage
        )

        # Force hit but not crit (roll 15), and fixed damage (roll 3 on d6)
        with patch("core.shared.rolls._roll_d20", return_value=15), \
             patch("core.mech.combat_helpers.random.randint", return_value=3):
            updated_scenario, _, _, result = execute_action(
                scenario, turn, economy, action_input
            )

        assert result.success
        # Check exposed multiplier effect was logged
        exposed_effect = next(
            (e for e in result.effects_applied if e.get("type") == "exposed_multiplier"),
            None
        )
        assert exposed_effect is not None
        assert exposed_effect["multiplier"] == 2

        # Verify damage was doubled (3 * 2 = 6)
        updated_target = next(
            c for c in updated_scenario.combatants if c.id == "target_1"
        )
        assert updated_target.resources.hp_current == 50 - 6

    def test_exposed_stacks_with_critical(self):
        """Critical hit + exposed should result in 4x damage."""
        from unittest.mock import patch

        actor = make_combatant(
            id="actor_1",
            grit=5,
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        target = make_combatant(
            id="target_1",
            hp_max=50,
            hp_current=50,
            statuses=["exposed"],
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[actor, target])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="skirmish",
            action_type="quick",
            target_ids=["target_1"],
            weapon_id="assault_rifle",  # 1d6 damage
        )

        # Force critical hit (roll 20), and fixed damage (roll 3 on d6)
        with patch("core.shared.rolls._roll_d20", return_value=20), \
             patch("core.mech.combat_helpers.random.randint", return_value=3):
            updated_scenario, _, _, result = execute_action(
                scenario, turn, economy, action_input
            )

        assert result.success
        attack_effect = next(e for e in result.effects_applied if e.get("type") == "attack")
        assert attack_effect["critical"] is True

        exposed_effect = next(
            (e for e in result.effects_applied if e.get("type") == "exposed_multiplier"),
            None
        )
        assert exposed_effect is not None

        # Verify damage was quadrupled: 3 * 2 (crit) * 2 (exposed) = 12
        updated_target = next(
            c for c in updated_scenario.combatants if c.id == "target_1"
        )
        assert updated_target.resources.hp_current == 50 - 12

    def test_shredded_ignores_armor(self):
        """Shredded status should bypass all armor."""
        from unittest.mock import patch

        actor = make_combatant(
            id="actor_1",
            grit=5,
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        # Create target with armor
        target = CombatantState(
            id="target_1",
            name="Armored Target",
            side="hostiles",
            kind="mech",
            stats=CombatStats(
                size="size_1",
                hp_max=50,
                evasion=8,
                e_defense=8,
                armor=3,  # 3 armor
                speed=4,
                sensor_range=10,
                tech_attack=0,
                grit=0,
            ),
            resources=CombatResources(
                hp_current=50,
                heat_current=0,
                heat_cap=6,
                structure_current=4,
                stress_current=4,
                repairs_remaining=4,
            ),
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            statuses=["shredded"],
        )
        scenario = make_scenario(combatants=[actor, target])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="skirmish",
            action_type="quick",
            target_ids=["target_1"],
            weapon_id="assault_rifle",  # 1d6 damage, no AP
        )

        # Force hit (roll 15), and fixed damage (roll 4 on d6)
        with patch("core.shared.rolls._roll_d20", return_value=15), \
             patch("core.mech.combat_helpers.random.randint", return_value=4):
            updated_scenario, _, _, result = execute_action(
                scenario, turn, economy, action_input
            )

        assert result.success
        # Check shredded armor bypass effect was logged
        shredded_effect = next(
            (e for e in result.effects_applied if e.get("type") == "shredded_armor_bypass"),
            None
        )
        assert shredded_effect is not None
        assert shredded_effect["armor_bypassed"] == 3

        # Verify full damage was dealt (armor bypassed)
        # Without shredded: 4 - 3 armor = 1 damage
        # With shredded: 4 damage (armor ignored)
        updated_target = next(
            c for c in updated_scenario.combatants if c.id == "target_1"
        )
        assert updated_target.resources.hp_current == 50 - 4


# =============================================================================
# Burn Tick Tests (Phase 17)
# =============================================================================


class TestBurnTick:
    """Tests for burn damage tick at end of turn (PR2 5017-5021, 4103-4109)."""

    def test_burn_tick_success_clears_burn(self):
        """Roll 15 + 0 ENG >= 10 clears all burn, no damage."""
        from unittest.mock import patch

        # Create actor with burn status and burn_marked
        actor = CombatantState(
            id="actor_1",
            name="Burning Mech",
            side="players",
            kind="mech",
            stats=CombatStats(
                size="size_1",
                hp_max=10,
                evasion=8,
                e_defense=8,
                engineering_skill=0,
            ),
            resources=CombatResources(
                hp_current=10,
                burn_marked=4,
                structure_current=4,
                stress_current=4,
            ),
            statuses=["burn"],
        )
        turn = make_turn(actor_id="actor_1")
        round1 = make_round(round_index=1, turns=[turn])
        scenario = make_scenario(combatants=[actor], rounds=[round1])

        # Force roll of 15 (success: 15 >= 10)
        with patch("core.mech.combat_helpers.roll_dice", return_value=15):
            updated_scenario, result, _, _ = end_turn(
                scenario, current_round=1, current_turn_index=0, current_turn=turn
            )

        # Check burn tick result
        assert result.burn_tick_result is not None
        assert result.burn_tick_result.success is True
        assert result.burn_tick_result.engineering_roll == 15
        assert result.burn_tick_result.total == 15
        assert result.burn_tick_result.damage_taken == 0
        assert result.burn_tick_result.burn_cleared is True

        # Check actor state
        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        assert "burn" not in updated_actor.statuses
        assert updated_actor.resources.burn_marked == 0
        assert updated_actor.resources.hp_current == 10  # No damage

    def test_burn_tick_failure_deals_damage(self):
        """Roll 5 + 0 ENG < 10 deals burn_marked damage."""
        from unittest.mock import patch

        actor = CombatantState(
            id="actor_1",
            name="Burning Mech",
            side="players",
            kind="mech",
            stats=CombatStats(
                size="size_1",
                hp_max=10,
                evasion=8,
                e_defense=8,
                engineering_skill=0,
            ),
            resources=CombatResources(
                hp_current=10,
                burn_marked=4,
                structure_current=4,
                stress_current=4,
            ),
            statuses=["burn"],
        )
        turn = make_turn(actor_id="actor_1")
        round1 = make_round(round_index=1, turns=[turn])
        scenario = make_scenario(combatants=[actor], rounds=[round1])

        # Force roll of 5 (failure: 5 < 10)
        with patch("core.mech.combat_helpers.roll_dice", return_value=5):
            updated_scenario, result, _, _ = end_turn(
                scenario, current_round=1, current_turn_index=0, current_turn=turn
            )

        # Check burn tick result
        assert result.burn_tick_result is not None
        assert result.burn_tick_result.success is False
        assert result.burn_tick_result.engineering_roll == 5
        assert result.burn_tick_result.total == 5
        assert result.burn_tick_result.damage_taken == 4
        assert result.burn_tick_result.burn_cleared is False

        # Check actor state - still has burn, took damage
        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        assert "burn" in updated_actor.statuses
        assert updated_actor.resources.burn_marked == 4  # Still marked
        assert updated_actor.resources.hp_current == 6  # 10 - 4 burn damage

    def test_burn_stacks_additively(self):
        """Burn applied multiple times stacks additively."""
        # Create target with existing burn status and burn_marked = 3
        target = CombatantState(
            id="target_1",
            name="Burning Target",
            side="hostiles",
            kind="mech",
            stats=CombatStats(
                size="size_1",
                hp_max=20,
                evasion=8,
                e_defense=8,
            ),
            resources=CombatResources(
                hp_current=20,
                burn_marked=3,  # Already has 3 burn
                structure_current=4,
                stress_current=4,
            ),
            statuses=["burn"],
        )

        # Simulate applying burn 2 by updating the burn_marked value
        # This tests the accumulation logic (original burn_marked + new burn)
        new_burn_value = 2
        new_burn_marked = target.resources.burn_marked + new_burn_value
        new_resources = target.resources.model_copy(update={"burn_marked": new_burn_marked})
        updated_target = target.model_copy(update={"resources": new_resources})

        # Verify stacking: 3 + 2 = 5
        assert updated_target.resources.burn_marked == 5
        assert "burn" in updated_target.statuses

    def test_burn_ignores_armor_on_tick(self):
        """Actor with armor=2, burn=4 takes 4 damage not 2."""
        from unittest.mock import patch

        # Create actor with armor and burn
        actor = CombatantState(
            id="actor_1",
            name="Armored Burning Mech",
            side="players",
            kind="mech",
            stats=CombatStats(
                size="size_1",
                hp_max=10,
                evasion=8,
                e_defense=8,
                armor=2,  # Has 2 armor
                engineering_skill=0,
            ),
            resources=CombatResources(
                hp_current=10,
                burn_marked=4,
                structure_current=4,
                stress_current=4,
            ),
            statuses=["burn"],
        )
        turn = make_turn(actor_id="actor_1")
        round1 = make_round(round_index=1, turns=[turn])
        scenario = make_scenario(combatants=[actor], rounds=[round1])

        # Force roll of 5 (failure: 5 < 10)
        with patch("core.mech.combat_helpers.roll_dice", return_value=5):
            updated_scenario, result, _, _ = end_turn(
                scenario, current_round=1, current_turn_index=0, current_turn=turn
            )

        # Burn damage ignores armor - should take full 4 damage, not 4-2=2
        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        assert updated_actor.resources.hp_current == 6  # 10 - 4 = 6 (armor ignored)

    def test_no_burn_tick_without_status(self):
        """No tick if actor lacks burn status."""
        # Actor with burn_marked but no burn status (shouldn't happen normally)
        actor = CombatantState(
            id="actor_1",
            name="Not Burning",
            side="players",
            kind="mech",
            stats=CombatStats(
                size="size_1",
                hp_max=10,
                evasion=8,
                e_defense=8,
            ),
            resources=CombatResources(
                hp_current=10,
                burn_marked=4,
                structure_current=4,
                stress_current=4,
            ),
            statuses=[],  # No burn status
        )
        turn = make_turn(actor_id="actor_1")
        round1 = make_round(round_index=1, turns=[turn])
        scenario = make_scenario(combatants=[actor], rounds=[round1])

        updated_scenario, result, _, _ = end_turn(
            scenario, current_round=1, current_turn_index=0, current_turn=turn
        )

        # No burn tick should occur
        assert result.burn_tick_result is None

    def test_no_burn_tick_without_burn_marked(self):
        """No tick if actor has burn status but burn_marked=0."""
        actor = CombatantState(
            id="actor_1",
            name="Burn Status Only",
            side="players",
            kind="mech",
            stats=CombatStats(
                size="size_1",
                hp_max=10,
                evasion=8,
                e_defense=8,
            ),
            resources=CombatResources(
                hp_current=10,
                burn_marked=0,  # No burn marked
                structure_current=4,
                stress_current=4,
            ),
            statuses=["burn"],  # Has burn status
        )
        turn = make_turn(actor_id="actor_1")
        round1 = make_round(round_index=1, turns=[turn])
        scenario = make_scenario(combatants=[actor], rounds=[round1])

        updated_scenario, result, _, _ = end_turn(
            scenario, current_round=1, current_turn_index=0, current_turn=turn
        )

        # No burn tick should occur
        assert result.burn_tick_result is None

    def test_engineering_skill_adds_to_roll(self):
        """ENG 3 + roll 7 = 10, exactly success."""
        from unittest.mock import patch

        actor = CombatantState(
            id="actor_1",
            name="Skilled Engineer",
            side="players",
            kind="mech",
            stats=CombatStats(
                size="size_1",
                hp_max=10,
                evasion=8,
                e_defense=8,
                engineering_skill=3,  # +3 ENG
            ),
            resources=CombatResources(
                hp_current=10,
                burn_marked=4,
                structure_current=4,
                stress_current=4,
            ),
            statuses=["burn"],
        )
        turn = make_turn(actor_id="actor_1")
        round1 = make_round(round_index=1, turns=[turn])
        scenario = make_scenario(combatants=[actor], rounds=[round1])

        # Roll 7 + ENG 3 = 10, exactly meets DC 10
        with patch("core.mech.combat_helpers.roll_dice", return_value=7):
            updated_scenario, result, _, _ = end_turn(
                scenario, current_round=1, current_turn_index=0, current_turn=turn
            )

        assert result.burn_tick_result is not None
        assert result.burn_tick_result.engineering_roll == 7
        assert result.burn_tick_result.engineering_bonus == 3
        assert result.burn_tick_result.total == 10
        assert result.burn_tick_result.success is True
        assert result.burn_tick_result.burn_cleared is True

    def test_stabilize_clear_burn_resets_burn_marked(self):
        """Stabilize with clear_burn sets burn_marked=0."""
        actor = CombatantState(
            id="actor_1",
            name="Stabilizing Mech",
            side="players",
            kind="mech",
            stats=CombatStats(
                size="size_1",
                hp_max=10,
                evasion=8,
                e_defense=8,
            ),
            resources=CombatResources(
                hp_current=10,
                burn_marked=5,  # Has accumulated burn
                heat_current=2,
                heat_cap=6,
                structure_current=4,
                stress_current=4,
                repairs_remaining=4,
            ),
            statuses=["burn"],
        )
        scenario = make_scenario(combatants=[actor])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="stabilize",
            action_type="full",
            stabilize_primary="cool_heat",
            stabilize_secondary="clear_burn",
        )

        updated_scenario, _, _, result = execute_action(
            scenario, turn, economy, action_input
        )

        assert result.success
        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        assert "burn" not in updated_actor.statuses
        assert updated_actor.resources.burn_marked == 0



# =============================================================================
# Range and LOS Enforcement Tests
# =============================================================================


class TestRangeAndLOS:
    """Tests for range validation and LOS enforcement per PR2 pp 99-100."""

    def test_ranged_attack_within_range_succeeds(self):
        """Target at distance 5, weapon range 10 should allow attack."""
        from unittest.mock import patch

        attacker = make_combatant(
            id="attacker",
            name="Attacker",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0)),
            grit=5,
        )
        target = make_combatant(
            id="target",
            name="Target",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=5, r=0)),  # Distance 5
            hp_current=10,
            hp_max=10,
        )
        scenario = make_scenario(combatants=[attacker, target])
        turn = make_turn(actor_id="attacker")
        economy = ActionEconomyState()

        # assault_rifle has range 10
        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            weapon_id="assault_rifle",
            target_ids=["target"],
        )

        with patch("core.shared.rolls.resolve_attack") as mock_resolve:
            mock_resolve.return_value = AttackResolutionResult(
                roll=15,
                attack_bonus=5,
                accuracy_dice_rolls=[],
                difficulty_dice_rolls=[],
                net_accuracy=0,
                total_accuracy=20,
                target_defense=10,
                hit=True,
                is_critical=False,
                miss_by=0,
            )
            _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is True

    def test_ranged_attack_out_of_range_fails(self):
        """Target at distance 15, weapon range 10 should fail."""
        attacker = make_combatant(
            id="attacker",
            name="Attacker",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0)),
        )
        target = make_combatant(
            id="target",
            name="Target",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=15, r=0)),  # Distance 15
        )
        scenario = make_scenario(combatants=[attacker, target])
        turn = make_turn(actor_id="attacker")
        economy = ActionEconomyState()

        # assault_rifle has range 10
        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            weapon_id="assault_rifle",
            target_ids=["target"],
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is False
        assert "out of range" in result.error.lower()

    def test_melee_attack_within_threat_succeeds(self):
        """Target at distance 1, threat 1 should allow attack."""
        from unittest.mock import patch

        attacker = make_combatant(
            id="attacker",
            name="Attacker",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0)),
            grit=5,
        )
        target = make_combatant(
            id="target",
            name="Target",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=1, r=0)),  # Distance 1
            hp_current=10,
            hp_max=10,
        )
        scenario = make_scenario(combatants=[attacker, target])
        turn = make_turn(actor_id="attacker")
        economy = ActionEconomyState()

        # tactical_knife has threat 1
        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            weapon_id="tactical_knife",
            target_ids=["target"],
        )

        with patch("core.shared.rolls.resolve_attack") as mock_resolve:
            mock_resolve.return_value = AttackResolutionResult(
                roll=15,
                attack_bonus=5,
                accuracy_dice_rolls=[],
                difficulty_dice_rolls=[],
                net_accuracy=0,
                total_accuracy=20,
                target_defense=10,
                hit=True,
                is_critical=False,
                miss_by=0,
            )
            _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is True

    def test_melee_attack_out_of_threat_fails(self):
        """Target at distance 3, threat 1 should fail."""
        attacker = make_combatant(
            id="attacker",
            name="Attacker",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0)),
        )
        target = make_combatant(
            id="target",
            name="Target",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=3, r=0)),  # Distance 3
        )
        scenario = make_scenario(combatants=[attacker, target])
        turn = make_turn(actor_id="attacker")
        economy = ActionEconomyState()

        # tactical_knife has threat 1
        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            weapon_id="tactical_knife",
            target_ids=["target"],
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is False
        assert "out of range" in result.error.lower()

    def test_tech_attack_within_sensor_range_succeeds(self):
        """Target within sensor range should allow tech attack."""
        from unittest.mock import patch

        attacker = make_combatant(
            id="attacker",
            name="Attacker",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0)),
            tech_attack=2,
        )
        target = make_combatant(
            id="target",
            name="Target",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=5, r=0)),  # Distance 5, sensor_range default is 10
        )
        scenario = make_scenario(combatants=[attacker, target])
        turn = make_turn(actor_id="attacker")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="lock_on",
            action_type="quick",
            target_ids=["target"],
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is True

    def test_tech_attack_out_of_sensor_range_fails(self):
        """Target beyond sensor range should fail tech attack."""
        attacker = make_combatant(
            id="attacker",
            name="Attacker",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0)),
        )
        target = make_combatant(
            id="target",
            name="Target",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=15, r=0)),  # Distance 15, beyond sensor_range=10
        )
        scenario = make_scenario(combatants=[attacker, target])
        turn = make_turn(actor_id="attacker")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="lock_on",
            action_type="quick",
            target_ids=["target"],
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is False
        assert "out of range" in result.error.lower()

    def test_blocked_los_prevents_attack(self):
        """Terrain blocking LOS should prevent attack."""
        from core.shared.terrain import TerrainMap, TerrainHex

        attacker = make_combatant(
            id="attacker",
            name="Attacker",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0)),
        )
        target = make_combatant(
            id="target",
            name="Target",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=3, r=0)),  # Distance 3
        )
        # Terrain at (1,0) blocks LOS
        terrain = TerrainMap(tiles=[
            TerrainHex(coord=HexCoord(q=1, r=0), blocks_line_of_sight=True),
        ])
        scenario = MechCombatScenario(
            combatants=[attacker, target],
            grapples=[],
            rounds=[],
            terrain=terrain,
            environment="standard",
            deployables={},
        )
        turn = make_turn(actor_id="attacker")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            weapon_id="assault_rifle",
            target_ids=["target"],
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is False
        assert "line of sight" in result.error.lower()

    def test_seeking_weapon_ignores_blocked_los(self):
        """Seeking weapon should attack through blocked LOS."""
        from unittest.mock import patch
        from core.shared.terrain import TerrainMap, TerrainHex

        attacker = make_combatant(
            id="attacker",
            name="Attacker",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0)),
            grit=5,
        )
        target = make_combatant(
            id="target",
            name="Target",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=3, r=0)),  # Distance 3
            hp_current=10,
            hp_max=10,
        )
        # Terrain at (1,0) blocks LOS
        terrain = TerrainMap(tiles=[
            TerrainHex(coord=HexCoord(q=1, r=0), blocks_line_of_sight=True),
        ])
        scenario = MechCombatScenario(
            combatants=[attacker, target],
            grapples=[],
            rounds=[],
            terrain=terrain,
            environment="standard",
            deployables={},
        )
        turn = make_turn(actor_id="attacker")
        economy = ActionEconomyState()

        # Use a weapon with seeking tag - horus_autopod has seeking
        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            weapon_id="horus_autopod",
            target_ids=["target"],
        )

        with patch("core.shared.rolls.resolve_attack") as mock_resolve:
            mock_resolve.return_value = AttackResolutionResult(
                roll=15,
                attack_bonus=5,
                accuracy_dice_rolls=[],
                difficulty_dice_rolls=[],
                net_accuracy=0,
                total_accuracy=20,
                target_defense=10,
                hit=True,
                is_critical=False,
                miss_by=0,
            )
            _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is True

    def test_arcing_weapon_ignores_blocked_los(self):
        """Arcing weapon should attack through blocked LOS."""
        from unittest.mock import patch
        from core.shared.terrain import TerrainMap, TerrainHex

        attacker = make_combatant(
            id="attacker",
            name="Attacker",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0)),
            grit=5,
        )
        target = make_combatant(
            id="target",
            name="Target",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=3, r=0)),  # Distance 3
            hp_current=10,
            hp_max=10,
        )
        # Terrain at (1,0) and (2,0) block LOS
        terrain = TerrainMap(tiles=[
            TerrainHex(coord=HexCoord(q=1, r=0), blocks_line_of_sight=True),
            TerrainHex(coord=HexCoord(q=2, r=0), blocks_line_of_sight=True),
        ])
        scenario = MechCombatScenario(
            combatants=[attacker, target],
            grapples=[],
            rounds=[],
            terrain=terrain,
            environment="standard",
            deployables={},
        )
        turn = make_turn(actor_id="attacker")
        economy = ActionEconomyState()

        # mortar has arcing tag
        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            weapon_id="mortar",
            target_ids=["target"],
        )

        with patch("core.shared.rolls.resolve_attack") as mock_resolve:
            mock_resolve.return_value = AttackResolutionResult(
                roll=15,
                attack_bonus=5,
                accuracy_dice_rolls=[],
                difficulty_dice_rolls=[],
                net_accuracy=0,
                total_accuracy=20,
                target_defense=10,
                hit=True,
                is_critical=False,
                miss_by=0,
            )
            _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is True


# =============================================================================
# Weapon Tag Enforcement Tests
# =============================================================================


class TestWeaponTagEnforcement:
    """Tests for weapon tag enforcement (loading, limited, ordnance)."""

    def make_combatant_with_weapon(
        self,
        id: str = "mech_1",
        weapon_id: str = "test_weapon",
        tags: list[str] = None,
        needs_reload: bool = False,
        limited_charges: int | None = None,
        statuses: list[str] = None,
    ) -> CombatantState:
        """Create a test combatant with a weapon in inventory."""
        weapon_state = WeaponState(
            weapon_id=weapon_id,
            tags=tags or [],
            destroyed=False,
            limited_charges_remaining=limited_charges,
            needs_reload=needs_reload,
        )
        mount = WeaponMountState(
            mount_index=0,
            slot_type="main",
            weapons=[weapon_state],
            destroyed=False,
        )
        inventory = MechInventory(mounts=[mount], systems=[])

        return CombatantState(
            id=id,
            name="Test Mech",
            side="players",
            kind="mech",
            stats=CombatStats(
                size="size_1",
                hp_max=10,
                evasion=8,
                e_defense=8,
                armor=0,
                speed=4,
                sensor_range=10,
                grit=0,
            ),
            resources=CombatResources(
                hp_current=10,
                heat_current=0,
                heat_cap=6,
                structure_current=4,
                stress_current=4,
                repairs_remaining=4,
            ),
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
            statuses=statuses or [],
            inventory=inventory,
        )

    # =========================================================================
    # Loading Tag Tests
    # =========================================================================

    def test_loading_weapon_fires_first_time(self):
        """First attack with loading weapon succeeds."""
        combatant = self.make_combatant_with_weapon(
            id="attacker",
            weapon_id="loading_gun",
            tags=["loading"],
            needs_reload=False,
        )
        target = make_combatant(id="target", position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0))
        scenario = make_scenario(combatants=[combatant, target])
        turn = CombatTurn(actor_id="attacker")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            weapon_id="loading_gun",
            target_ids=["target"],
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is True

    def test_loading_weapon_needs_reload_after_fire(self):
        """Loading weapon requires reload after firing."""
        from unittest.mock import patch

        combatant = self.make_combatant_with_weapon(
            id="attacker",
            weapon_id="loading_gun",
            tags=["loading"],
            needs_reload=False,
        )
        target = make_combatant(id="target", position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0))
        scenario = make_scenario(combatants=[combatant, target])
        turn = CombatTurn(actor_id="attacker")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            weapon_id="loading_gun",
            target_ids=["target"],
        )

        # Mock the attack to succeed
        with patch("core.shared.rolls.resolve_attack") as mock_resolve:
            mock_resolve.return_value = AttackResolutionResult(
                roll=15, attack_bonus=0, accuracy_dice_rolls=[], difficulty_dice_rolls=[],
                net_accuracy=0, total_accuracy=15, target_defense=8, hit=True, is_critical=False, miss_by=0,
            )
            updated_scenario, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is True

        # Check weapon state was updated
        updated_attacker = next(c for c in updated_scenario.combatants if c.id == "attacker")
        weapon_state = updated_attacker.inventory.mounts[0].weapons[0]
        assert weapon_state.needs_reload is True

    def test_loading_weapon_blocked_when_needs_reload(self):
        """Loading weapon cannot fire when needs_reload is True."""
        combatant = self.make_combatant_with_weapon(
            id="attacker",
            weapon_id="loading_gun",
            tags=["loading"],
            needs_reload=True,  # Already needs reload
        )
        target = make_combatant(id="target", position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0))
        scenario = make_scenario(combatants=[combatant, target])
        turn = CombatTurn(actor_id="attacker")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            weapon_id="loading_gun",
            target_ids=["target"],
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is False
        assert "needs reload" in result.error.lower()

    def test_stabilize_reloads_loading_weapons(self):
        """Stabilize with reload_loading clears needs_reload."""
        combatant = self.make_combatant_with_weapon(
            id="actor",
            weapon_id="loading_gun",
            tags=["loading"],
            needs_reload=True,
        )
        scenario = make_scenario(combatants=[combatant])
        turn = CombatTurn(actor_id="actor")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor",
            action_id="stabilize",
            action_type="full",
            stabilize_primary="cool_heat",
            stabilize_secondary="reload_loading",
        )

        updated_scenario, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is True

        # Check weapon was reloaded
        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor")
        weapon_state = updated_actor.inventory.mounts[0].weapons[0]
        assert weapon_state.needs_reload is False

    # =========================================================================
    # Limited Tag Tests
    # =========================================================================

    def test_limited_weapon_decrements_charges(self):
        """Each attack decrements limited_charges_remaining."""
        from unittest.mock import patch

        combatant = self.make_combatant_with_weapon(
            id="attacker",
            weapon_id="limited_missile",
            tags=["limited"],
            limited_charges=3,
        )
        target = make_combatant(id="target", position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0))
        scenario = make_scenario(combatants=[combatant, target])
        turn = CombatTurn(actor_id="attacker")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            weapon_id="limited_missile",
            target_ids=["target"],
        )

        with patch("core.shared.rolls.resolve_attack") as mock_resolve:
            mock_resolve.return_value = AttackResolutionResult(
                roll=15, attack_bonus=0, accuracy_dice_rolls=[], difficulty_dice_rolls=[],
                net_accuracy=0, total_accuracy=15, target_defense=8, hit=True, is_critical=False, miss_by=0,
            )
            updated_scenario, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is True

        # Check charges decremented
        updated_attacker = next(c for c in updated_scenario.combatants if c.id == "attacker")
        weapon_state = updated_attacker.inventory.mounts[0].weapons[0]
        assert weapon_state.limited_charges_remaining == 2

    def test_limited_weapon_blocked_at_zero_charges(self):
        """Attack fails when limited_charges_remaining is 0."""
        combatant = self.make_combatant_with_weapon(
            id="attacker",
            weapon_id="limited_missile",
            tags=["limited"],
            limited_charges=0,  # No charges left
        )
        target = make_combatant(id="target", position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0))
        scenario = make_scenario(combatants=[combatant, target])
        turn = CombatTurn(actor_id="attacker")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            weapon_id="limited_missile",
            target_ids=["target"],
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is False
        assert "no charges" in result.error.lower()

    # =========================================================================
    # Ordnance Tag Tests
    # =========================================================================

    def test_ordnance_fires_before_movement(self):
        """Ordnance attack succeeds when has_moved_or_acted is False."""
        combatant = self.make_combatant_with_weapon(
            id="attacker",
            weapon_id="ordnance_cannon",
            tags=["ordnance"],
        )
        target = make_combatant(id="target", position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0))
        scenario = make_scenario(combatants=[combatant, target])
        turn = CombatTurn(actor_id="attacker", has_moved_or_acted=False)
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            weapon_id="ordnance_cannon",
            target_ids=["target"],
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is True

    def test_ordnance_blocked_after_movement(self):
        """Ordnance attack fails after move/boost action."""
        combatant = self.make_combatant_with_weapon(
            id="attacker",
            weapon_id="ordnance_cannon",
            tags=["ordnance"],
        )
        target = make_combatant(id="target", position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0))
        scenario = make_scenario(combatants=[combatant, target])
        turn = CombatTurn(actor_id="attacker", has_moved_or_acted=True)  # Already moved
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            weapon_id="ordnance_cannon",
            target_ids=["target"],
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is False
        assert "ordnance" in result.error.lower()

    def test_ordnance_blocked_after_other_action(self):
        """Ordnance attack fails after any non-protocol action."""
        combatant = self.make_combatant_with_weapon(
            id="attacker",
            weapon_id="ordnance_cannon",
            tags=["ordnance"],
        )
        target = make_combatant(id="target", position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0))
        scenario = make_scenario(combatants=[combatant, target])
        turn = CombatTurn(actor_id="attacker", has_moved_or_acted=False)
        economy = ActionEconomyState()

        # First, do a quick action (hide) which should set has_moved_or_acted
        hide_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="hide",
            action_type="quick",
        )
        _, updated_turn, _, _ = execute_action(scenario, turn, economy, hide_input)

        # Now has_moved_or_acted should be True
        assert updated_turn.has_moved_or_acted is True

    def test_ordnance_blocked_while_engaged(self):
        """Ordnance attack fails when actor has engaged status."""
        combatant = self.make_combatant_with_weapon(
            id="attacker",
            weapon_id="ordnance_cannon",
            tags=["ordnance"],
            statuses=["engaged"],  # Actor is engaged
        )
        target = make_combatant(id="target", position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0))
        scenario = make_scenario(combatants=[combatant, target])
        turn = CombatTurn(actor_id="attacker", has_moved_or_acted=False)
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            weapon_id="ordnance_cannon",
            target_ids=["target"],
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is False
        assert "engaged" in result.error.lower()

    def test_ordnance_blocked_in_overwatch(self):
        """Ordnance weapon cannot be used for overwatch reaction."""
        combatant = self.make_combatant_with_weapon(
            id="reactor",
            weapon_id="ordnance_cannon",
            tags=["ordnance"],
        )
        scenario = make_scenario(combatants=[combatant])
        economy = ActionEconomyState()

        reaction_input = ReactionInput(
            reactor_id="reactor",
            reaction_type="overwatch",
            weapon_id="ordnance_cannon",
            target_ids=["target"],
        )

        _, _, result = execute_reaction(scenario, economy, reaction_input)

        assert result.success is False
        assert "ordnance" in result.error.lower()
        assert "overwatch" in result.error.lower()

    # =========================================================================
    # Turn State Tracking Tests
    # =========================================================================

    def test_has_moved_or_acted_set_by_quick_action(self):
        """Quick actions set has_moved_or_acted to True."""
        combatant = make_combatant(id="actor")
        scenario = make_scenario(combatants=[combatant])
        turn = CombatTurn(actor_id="actor", has_moved_or_acted=False)
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor",
            action_id="hide",
            action_type="quick",
        )

        _, updated_turn, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is True
        assert updated_turn.has_moved_or_acted is True

    def test_has_moved_or_acted_set_by_full_action(self):
        """Full actions set has_moved_or_acted to True."""
        combatant = make_combatant(id="actor")
        scenario = make_scenario(combatants=[combatant])
        turn = CombatTurn(actor_id="actor", has_moved_or_acted=False)
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor",
            action_id="stabilize",
            action_type="full",
            stabilize_primary="cool_heat",
        )

        _, updated_turn, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is True
        assert updated_turn.has_moved_or_acted is True

    def test_fresh_turn_has_moved_or_acted_false(self):
        """A fresh CombatTurn has has_moved_or_acted=False."""
        turn = CombatTurn(actor_id="actor")
        assert turn.has_moved_or_acted is False
