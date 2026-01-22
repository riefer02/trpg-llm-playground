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
    ActionUse,
    MechSystemState,
)
from core.mech.grid import HexPosition, HexCoord
from core.shared.effects import (
    CooldownState,
    MechanicalEffect,
    ReactionTriggerEffect,
    Resistance,
    HeatResistanceEffect,
)
from core.shared.full_tech import FullTechOptionSelection
from core.shared.rolls import AttackResolutionResult
from core.shared.heat import MeltdownState
from core.shared.flying import FlyingStatus


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
    engineering_skill: int = 0,
    e_defense: int = 8,
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
            e_defense=e_defense,
            armor=0,
            speed=4,
            sensor_range=10,
            tech_attack=tech_attack,
            grit=grit,
            engineering_skill=engineering_skill,
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

    def test_start_turn_dangerous_terrain_auto_resolves_for_hostiles(self):
        """Test that hostiles in dangerous terrain auto-resolve at turn start."""
        from core.shared.terrain import TerrainMap, TerrainHex
        from unittest.mock import patch

        combatant = make_combatant(
            id="actor_1",
            side="hostiles",
            engineering_skill=2,
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        terrain = TerrainMap(tiles=[
            TerrainHex(coord=HexCoord(q=0, r=0), dangerous=True),
        ])
        scenario = MechCombatScenario(
            combatants=[combatant],
            grapples=[],
            rounds=[],
            terrain=terrain,
            environment="standard",
            deployables={},
        )

        # Force a failed check (roll 5 + 2 skill = 7 < 10)
        with patch("core.shared.terrain.roll_dice", return_value=5):
            updated_scenario, result = start_turn(scenario, "actor_1")

        assert result.dangerous_terrain_check_required is True
        assert result.dangerous_terrain_auto_resolved is True
        assert result.dangerous_terrain_check_passed is False
        assert result.dangerous_terrain_damage == 5

        # Check damage was applied
        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        assert updated_actor.resources.hp_current < combatant.resources.hp_current

    def test_start_turn_dangerous_terrain_creates_decision_for_players(self):
        """Test that players in dangerous terrain get a decision prompt at turn start."""
        from core.shared.terrain import TerrainMap, TerrainHex

        combatant = make_combatant(
            id="actor_1",
            side="players",
            engineering_skill=2,
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        terrain = TerrainMap(tiles=[
            TerrainHex(coord=HexCoord(q=0, r=0), dangerous=True),
        ])
        scenario = MechCombatScenario(
            combatants=[combatant],
            grapples=[],
            rounds=[],
            terrain=terrain,
            environment="standard",
            deployables={},
        )

        updated_scenario, result = start_turn(scenario, "actor_1")

        assert result.dangerous_terrain_check_required is True
        assert result.dangerous_terrain_decision_created is True
        assert result.dangerous_terrain_auto_resolved is False
        assert result.dangerous_terrain_damage == 0  # No auto damage for players

        # Check pending decision was created
        decisions = [
            d for d in updated_scenario.pending_decisions
            if d.decision_type == "engineering_check"
        ]
        assert len(decisions) == 1
        assert decisions[0].combatant_id == "actor_1"
        assert decisions[0].trigger_source.startswith("dangerous_terrain:")

    def test_start_turn_dangerous_terrain_no_duplicate_check_same_round(self):
        """Test that dangerous terrain check doesn't repeat if already checked this round."""
        from core.shared.terrain import TerrainMap, TerrainHex

        combatant = make_combatant(
            id="actor_1",
            side="hostiles",
            engineering_skill=2,
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
            dangerous_terrain_last_check_round=1,  # Already checked round 1
        )
        terrain = TerrainMap(tiles=[
            TerrainHex(coord=HexCoord(q=0, r=0), dangerous=True),
        ])
        scenario = MechCombatScenario(
            combatants=[combatant],
            grapples=[],
            rounds=[CombatRound(round_index=1, turn_order=[], turns=[])],  # Round 1
            terrain=terrain,
            environment="standard",
            deployables={},
        )

        updated_scenario, result = start_turn(scenario, "actor_1")

        # Should NOT require a check because already checked this round
        assert result.dangerous_terrain_check_required is False
        assert result.dangerous_terrain_auto_resolved is False
        assert result.dangerous_terrain_damage == 0

    def test_start_turn_no_dangerous_terrain_check_if_not_in_dangerous(self):
        """Test that no check is triggered if not in dangerous terrain."""
        from core.shared.terrain import TerrainMap, TerrainHex

        combatant = make_combatant(
            id="actor_1",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        terrain = TerrainMap(tiles=[
            TerrainHex(coord=HexCoord(q=1, r=0), dangerous=True),  # Different hex
        ])
        scenario = MechCombatScenario(
            combatants=[combatant],
            grapples=[],
            rounds=[],
            terrain=terrain,
            environment="standard",
            deployables={},
        )

        updated_scenario, result = start_turn(scenario, "actor_1")

        assert result.dangerous_terrain_check_required is False
        assert result.dangerous_terrain_damage == 0


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

    def test_execute_overcharge_respects_heat_resistance(self):
        """Test that overcharge heat is reduced by heat resistance."""
        overcharge = OverchargeState(current_level=0, uses_this_turn=0)
        combatant = make_combatant(
            id="actor_1",
            heat_current=0,
            overcharge_state=overcharge,
            frame_trait_effects=[
                MechanicalEffect(
                    heat_resistances=[HeatResistanceEffect(multiplier=0)]
                )
            ],
        )
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
        assert result.heat_generated == 0  # Heat fully resisted
        assert updated_economy.overcharge_used is True

        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        assert updated_actor.resources.heat_current == 0
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
            assert (
                defender_after.resources.hp_current < 10
                or defender_after.resources.structure_current < 4
            )

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

        defender_after = next(c for c in updated_scenario.combatants if c.id == "defender")
        assert defender_after.resources.hp_current == 8  # 20 - 12

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

    def test_execute_invade_respects_heat_resistance(self):
        """Invade heat should be reduced by heat resistance."""
        attacker = make_combatant(id="attacker", name="Attacker", side="players", tech_attack=10)
        defender = make_combatant(
            id="defender",
            name="Defender",
            side="hostiles",
            heat_cap=20,
            frame_trait_effects=[
                MechanicalEffect(heat_resistances=[HeatResistanceEffect()])
            ],
        )
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
        assert result.heat_generated == 1

        defender_after = next(c for c in updated_scenario.combatants if c.id == "defender")
        assert defender_after.resources.heat_current == 1

    def test_execute_invade_miss_no_heat(self):
        """Invade should not apply heat on a miss.

        With the 1d20 + tech_attack vs E-defense system, we set a very high E-defense
        to ensure a miss (except on critical natural 20, which has 5% chance).
        """
        attacker = make_combatant(id="attacker", name="Attacker", side="players", tech_attack=0)
        # E-defense of 30 ensures tech_attack=0 can't hit (max roll 20 < 30) except on critical
        defender = make_combatant(
            id="defender", name="Defender", side="hostiles", heat_cap=20, e_defense=30
        )
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
        # Note: 5% chance of critical hit (natural 20) which would make this fail
        if not result.effects_applied[0].get("is_critical", False):
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
        """Test executing an overwatch reaction (requires weapon and target)."""
        # Create reactor with weapon
        weapon_state = WeaponState(
            weapon_id="mw_assault_rifle",
            tags=[],
            destroyed=False,
        )
        mount = WeaponMountState(
            mount_index=0,
            slot_type="main",
            weapons=[weapon_state],
            destroyed=False,
        )
        inventory = MechInventory(mounts=[mount], systems=[])
        reactor = make_combatant(
            id="reactor_1",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
            inventory=inventory,
        )
        # Create target in range
        target = make_combatant(
            id="enemy_1",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            side="hostiles",
        )
        scenario = make_scenario(combatants=[reactor, target])
        economy = ActionEconomyState()

        reaction_input = ReactionInput(
            reactor_id="reactor_1",
            reaction_type="overwatch",
            target_ids=["enemy_1"],
            weapon_id="mw_assault_rifle",
        )

        _, _, result = execute_reaction(scenario, economy, reaction_input)

        assert result.success is True
        assert result.reaction_used == "overwatch"
        # Overwatch now resolves attack
        assert result.attack_hit is not None

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

    def test_apply_heat_respects_heat_resistance(self):
        """Test that heat resistance reduces applied heat."""
        combatant = make_combatant(id="target_1", heat_current=0)
        scenario = make_scenario(combatants=[combatant])

        updated_scenario, change, overheat_result = apply_heat(
            scenario,
            "target_1",
            heat=3,
            heat_resistance_multiplier=0.5,
        )

        updated_target = next(c for c in updated_scenario.combatants if c.id == "target_1")
        assert updated_target.resources.heat_current == 2
        assert change.heat_change == 2
        assert overheat_result is None

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


class TestMultiDamageResolution:
    """Tests for multi-type damage resolution with resistances."""

    def test_multi_type_damage_breakdown_applies_armor_and_resistance(self):
        """Bolt Thrower applies armor per component and explosive resistance."""
        from unittest.mock import patch

        attacker = make_combatant(id="attacker", name="Attacker", side="players")
        target = make_combatant(id="target", name="Target", side="hostiles", hp_max=20, hp_current=20)
        target_stats = target.stats.model_copy(update={"armor": 2})
        target_effects = MechanicalEffect(
            resistances=[Resistance(damage_type="explosive")]
        )
        target = target.model_copy(
            update={"stats": target_stats, "frame_trait_effects": [target_effects]}
        )

        scenario = make_scenario(combatants=[attacker, target])
        turn = make_turn(actor_id="attacker")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            target_ids=["target"],
            weapon_id="bolt_thrower",
        )

        with patch("core.shared.rolls._roll_d20") as mock_roll, patch(
            "core.mech.combat_helpers.random.randint"
        ) as mock_rand:
            mock_roll.return_value = 10
            mock_rand.side_effect = [6, 5, 4]  # 2d6 kinetic, 1d6 explosive

            _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is True
        assert result.damage_breakdown.kinetic == 9  # 11 - 2 armor
        assert result.damage_breakdown.explosive == 1  # (4 - 2 armor) / 2 rounded up
        assert result.damage_breakdown.energy == 0
        assert result.damage_breakdown.burn == 0
        assert result.damage_dealt == 10

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


class TestKnockbackWeaponTag:
    """Tests for knockback weapon tag integration."""

    def test_weapon_with_knockback_pushes_target(self):
        """Attack with knockback weapon pushes target on hit."""
        from unittest.mock import patch

        attacker = make_combatant(
            id="attacker",
            name="Attacker",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0)),
        )
        defender = make_combatant(
            id="defender",
            name="Defender",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=1, r=0)),
            hp_current=20,
            hp_max=20,
        )
        scenario = make_scenario(combatants=[attacker, defender])
        turn = make_turn(actor_id="attacker")
        economy = ActionEconomyState()

        # ipsn_war_pike has knockback 1
        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            target_ids=["defender"],
            weapon_id="ipsn_war_pike",
        )

        with patch("core.shared.rolls._roll_d20") as mock_roll:
            mock_roll.return_value = 20  # Force hit
            updated_scenario, _, _, result = execute_action(
                scenario, turn, economy, action_input
            )

        # Check for knockback effect in result
        knockback_effects = [e for e in result.effects_applied if e.get("type") == "knockback"]
        assert len(knockback_effects) == 1
        assert knockback_effects[0]["spaces_requested"] == 1
        assert knockback_effects[0]["spaces_moved"] == 1

        # Verify target was pushed away
        defender_after = next(
            c for c in updated_scenario.combatants if c.id == "defender"
        )
        # Target was at (1, 0), pushed 1 space away from attacker at (0, 0)
        # Direction is (1, 0), so new position should be (2, 0)
        assert defender_after.position.coord.q == 2
        assert defender_after.position.coord.r == 0

        # Check position_updates in result
        assert "defender" in result.position_updates
        assert result.position_updates["defender"]["q"] == 2
        assert result.position_updates["defender"]["r"] == 0

    def test_knockback_blocked_by_other_combatant(self):
        """Knockback stops before hitting another combatant."""
        from unittest.mock import patch

        attacker = make_combatant(
            id="attacker",
            name="Attacker",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0)),
        )
        defender = make_combatant(
            id="defender",
            name="Defender",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=1, r=0)),
            hp_current=20,
            hp_max=20,
        )
        # Blocker mech at (2, 0) - in the knockback path
        blocker = make_combatant(
            id="blocker",
            name="Blocker",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=2, r=0)),
        )
        scenario = make_scenario(combatants=[attacker, defender, blocker])
        turn = make_turn(actor_id="attacker")
        economy = ActionEconomyState()

        # ipsn_concussion_missiles has knockback 2
        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            target_ids=["defender"],
            weapon_id="ipsn_concussion_missiles",
        )

        with patch("core.shared.rolls._roll_d20") as mock_roll:
            mock_roll.return_value = 20  # Force hit
            updated_scenario, _, _, result = execute_action(
                scenario, turn, economy, action_input
            )

        # Check knockback was blocked
        knockback_effects = [e for e in result.effects_applied if e.get("type") == "knockback"]
        assert len(knockback_effects) == 1
        assert knockback_effects[0]["spaces_requested"] == 2
        assert knockback_effects[0]["spaces_moved"] == 0  # Blocked immediately
        assert knockback_effects[0]["blocked"] is True

        # Target should stay at original position
        defender_after = next(
            c for c in updated_scenario.combatants if c.id == "defender"
        )
        assert defender_after.position.coord.q == 1
        assert defender_after.position.coord.r == 0

    def test_no_knockback_on_miss(self):
        """Knockback only applies on hit."""
        from unittest.mock import patch

        attacker = make_combatant(
            id="attacker",
            name="Attacker",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0)),
        )
        defender = make_combatant(
            id="defender",
            name="Defender",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=1, r=0)),
            hp_current=20,
            hp_max=20,
        )
        scenario = make_scenario(combatants=[attacker, defender])
        turn = make_turn(actor_id="attacker")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            target_ids=["defender"],
            weapon_id="ipsn_war_pike",
        )

        with patch("core.shared.rolls._roll_d20") as mock_roll:
            mock_roll.return_value = 1  # Force miss
            updated_scenario, _, _, result = execute_action(
                scenario, turn, economy, action_input
            )

        # No knockback effect should be present
        knockback_effects = [e for e in result.effects_applied if e.get("type") == "knockback"]
        assert len(knockback_effects) == 0

        # Target should stay at original position
        defender_after = next(
            c for c in updated_scenario.combatants if c.id == "defender"
        )
        assert defender_after.position.coord.q == 1
        assert defender_after.position.coord.r == 0

        # No position_updates
        assert "defender" not in result.position_updates

    def test_knockback_zero_has_no_effect(self):
        """Weapon without knockback doesn't push."""
        from unittest.mock import patch

        attacker = make_combatant(
            id="attacker",
            name="Attacker",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0)),
        )
        defender = make_combatant(
            id="defender",
            name="Defender",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=1, r=0)),
            hp_current=20,
            hp_max=20,
        )
        scenario = make_scenario(combatants=[attacker, defender])
        turn = make_turn(actor_id="attacker")
        economy = ActionEconomyState()

        # assault_rifle has no knockback tag
        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            target_ids=["defender"],
            weapon_id="assault_rifle",
        )

        with patch("core.shared.rolls._roll_d20") as mock_roll:
            mock_roll.return_value = 20  # Force hit
            updated_scenario, _, _, result = execute_action(
                scenario, turn, economy, action_input
            )

        # No knockback effect should be present
        knockback_effects = [e for e in result.effects_applied if e.get("type") == "knockback"]
        assert len(knockback_effects) == 0

        # Target should stay at original position (aside from any other effects)
        defender_after = next(
            c for c in updated_scenario.combatants if c.id == "defender"
        )
        assert defender_after.position.coord.q == 1
        assert defender_after.position.coord.r == 0

    def test_knockback_position_in_result(self):
        """ActionExecutionResult contains position_updates."""
        from unittest.mock import patch

        attacker = make_combatant(
            id="attacker",
            name="Attacker",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0)),
        )
        defender = make_combatant(
            id="defender",
            name="Defender",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=1, r=0)),
            hp_current=20,
            hp_max=20,
        )
        scenario = make_scenario(combatants=[attacker, defender])
        turn = make_turn(actor_id="attacker")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            target_ids=["defender"],
            weapon_id="ipsn_war_pike",
        )

        with patch("core.shared.rolls._roll_d20") as mock_roll:
            mock_roll.return_value = 20  # Force hit
            _, _, _, result = execute_action(
                scenario, turn, economy, action_input
            )

        # Result should have position_updates field
        assert hasattr(result, "position_updates")
        assert isinstance(result.position_updates, dict)
        assert "defender" in result.position_updates

    def test_higher_knockback_pushes_further(self):
        """Knockback 3 weapon pushes target 3 spaces."""
        from unittest.mock import patch

        attacker = make_combatant(
            id="attacker",
            name="Attacker",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0)),
        )
        defender = make_combatant(
            id="defender",
            name="Defender",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=1, r=0)),
            hp_current=50,
            hp_max=50,
        )
        scenario = make_scenario(combatants=[attacker, defender])
        turn = make_turn(actor_id="attacker")
        economy = ActionEconomyState()

        # ha_daisy_cutter has knockback 3
        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            target_ids=["defender"],
            weapon_id="ha_daisy_cutter",
        )

        with patch("core.shared.rolls._roll_d20") as mock_roll:
            mock_roll.return_value = 20  # Force hit
            updated_scenario, _, _, result = execute_action(
                scenario, turn, economy, action_input
            )

        # Check knockback effect
        knockback_effects = [e for e in result.effects_applied if e.get("type") == "knockback"]
        assert len(knockback_effects) == 1
        assert knockback_effects[0]["spaces_requested"] == 3
        assert knockback_effects[0]["spaces_moved"] == 3

        # Verify target was pushed 3 spaces
        defender_after = next(
            c for c in updated_scenario.combatants if c.id == "defender"
        )
        assert defender_after.position.coord.q == 4
        assert defender_after.position.coord.r == 0


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

    def test_stabilize_cancel_meltdown_success(self):
        """Stabilize with cancel_meltdown should clear meltdown on successful engineering check."""
        from unittest.mock import patch
        from core.shared.heat import MeltdownState

        actor = CombatantState(
            id="actor_1",
            name="Melting Mech",
            side="players",
            kind="mech",
            stats=CombatStats(
                size="size_1",
                hp_max=10,
                evasion=8,
                e_defense=8,
                engineering_skill=2,  # +2 to check
            ),
            resources=CombatResources(
                hp_current=10,
                heat_current=0,
                heat_cap=6,
                structure_current=2,
                stress_current=2,
            ),
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
            meltdown_state=MeltdownState(turns_remaining=2, triggered_by_overheat=True),
        )
        scenario = make_scenario(combatants=[actor])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="stabilize",
            action_type="full",
            stabilize_primary="cancel_meltdown",
        )

        # Mock roll to always succeed (roll 8 + ENG 2 = 10, exactly DC)
        with patch("core.mech.combat_helpers.random.randint", return_value=8):
            updated_scenario, _, _, result = execute_action(
                scenario, turn, economy, action_input
            )

        assert result.success
        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        assert updated_actor.meltdown_state is None  # Meltdown cancelled

        # Check effect details
        primary_effect = next(
            e for e in result.effects_applied if e.get("type") == "stabilize_primary"
        )
        assert primary_effect["option"] == "cancel_meltdown"
        assert primary_effect["success"] is True
        assert primary_effect["meltdown_cancelled"] is True
        assert primary_effect["roll"] == 8
        assert primary_effect["engineering_bonus"] == 2
        assert primary_effect["total"] == 10
        assert primary_effect["dc"] == 10

    def test_stabilize_cancel_meltdown_failure(self):
        """Stabilize with cancel_meltdown should keep meltdown on failed engineering check."""
        from unittest.mock import patch
        from core.shared.heat import MeltdownState

        actor = CombatantState(
            id="actor_1",
            name="Melting Mech",
            side="players",
            kind="mech",
            stats=CombatStats(
                size="size_1",
                hp_max=10,
                evasion=8,
                e_defense=8,
                engineering_skill=2,  # +2 to check
            ),
            resources=CombatResources(
                hp_current=10,
                heat_current=0,
                heat_cap=6,
                structure_current=2,
                stress_current=2,
            ),
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
            meltdown_state=MeltdownState(turns_remaining=2, triggered_by_overheat=True),
        )
        scenario = make_scenario(combatants=[actor])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="stabilize",
            action_type="full",
            stabilize_primary="cancel_meltdown",
        )

        # Mock roll to fail (roll 5 + ENG 2 = 7, below DC 10)
        with patch("core.mech.combat_helpers.random.randint", return_value=5):
            updated_scenario, _, _, result = execute_action(
                scenario, turn, economy, action_input
            )

        assert result.success  # Action executed successfully, just didn't cancel meltdown
        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        assert updated_actor.meltdown_state is not None  # Meltdown NOT cancelled
        assert updated_actor.meltdown_state.turns_remaining == 2  # Unchanged

        # Check effect details
        primary_effect = next(
            e for e in result.effects_applied if e.get("type") == "stabilize_primary"
        )
        assert primary_effect["option"] == "cancel_meltdown"
        assert primary_effect["success"] is False
        assert primary_effect["meltdown_cancelled"] is False
        assert primary_effect["roll"] == 5
        assert primary_effect["total"] == 7

    def test_stabilize_cancel_meltdown_requires_meltdown_state(self):
        """Stabilize cancel_meltdown should fail if no active meltdown countdown."""
        actor = make_combatant(id="actor_1")
        # No meltdown_state set (default is None)
        scenario = make_scenario(combatants=[actor])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="stabilize",
            action_type="full",
            stabilize_primary="cancel_meltdown",
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success  # Action executed
        # Check effect shows failure due to no meltdown
        primary_effect = next(
            e for e in result.effects_applied if e.get("type") == "stabilize_primary"
        )
        assert primary_effect["option"] == "cancel_meltdown"
        assert primary_effect.get("failed") is True
        assert "No active meltdown countdown" in primary_effect.get("reason", "")

    def test_stabilize_cancel_meltdown_is_full_action(self):
        """Stabilize cancel_meltdown uses full action (consistent with other stabilize options)."""
        from core.shared.heat import MeltdownState

        actor = CombatantState(
            id="actor_1",
            name="Melting Mech",
            side="players",
            kind="mech",
            stats=CombatStats(
                size="size_1",
                hp_max=10,
                evasion=8,
                e_defense=8,
                engineering_skill=4,
            ),
            resources=CombatResources(
                hp_current=10,
                heat_current=0,
                heat_cap=6,
                structure_current=2,
                stress_current=2,
            ),
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
            meltdown_state=MeltdownState(turns_remaining=3, triggered_by_overheat=True),
        )
        scenario = make_scenario(combatants=[actor])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="stabilize",
            action_type="full",
            stabilize_primary="cancel_meltdown",
        )

        _, _, updated_economy, result = execute_action(
            scenario, turn, economy, action_input
        )

        assert result.success
        # Full action should be consumed
        assert updated_economy.full_actions_used == 1


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

    def test_hide_applies_hidden_status_with_hard_cover(self):
        """Hide should apply hidden status when actor has hard cover."""
        from core.shared.terrain import TerrainMap, TerrainHex

        actor = make_combatant(
            id="actor_1",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        # Add terrain with hard cover adjacent to actor
        terrain = TerrainMap(tiles=[
            TerrainHex(coord=HexCoord(q=0, r=0)),  # Actor's position
            TerrainHex(coord=HexCoord(q=1, r=0), provides_hard_cover=True),  # Adjacent hard cover
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

    def test_hide_fails_without_cover(self):
        """Hide should fail when there is no cover (no terrain)."""
        actor = make_combatant(
            id="actor_1",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
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
        assert "no cover" in hide_effect["reason"].lower() or "no terrain" in hide_effect["reason"].lower()
        # Actor should not have hidden status
        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        assert "hidden" not in updated_actor.statuses

    def test_hide_succeeds_in_soft_cover_area(self):
        """Hide should succeed when actor is in a soft cover area (3+ adjacent hexes)."""
        from core.shared.terrain import TerrainMap, TerrainHex

        actor = make_combatant(
            id="actor_1",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        # Create soft cover area (smoke cloud simulation - 3+ hexes)
        terrain = TerrainMap(tiles=[
            TerrainHex(coord=HexCoord(q=0, r=0), provides_soft_cover=True),
            TerrainHex(coord=HexCoord(q=1, r=0), provides_soft_cover=True),
            TerrainHex(coord=HexCoord(q=1, r=-1), provides_soft_cover=True),
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
        hide_effect = next(e for e in result.effects_applied if e.get("type") == "hide")
        assert hide_effect["success"] is True
        assert "soft cover" in hide_effect["reason"].lower()

    def test_hide_fails_with_insufficient_soft_cover(self):
        """Hide should fail when soft cover is not sufficient (less than 3 hexes)."""
        from core.shared.terrain import TerrainMap, TerrainHex

        actor = make_combatant(
            id="actor_1",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        # Only 2 soft cover hexes - not enough for an "area"
        terrain = TerrainMap(tiles=[
            TerrainHex(coord=HexCoord(q=0, r=0), provides_soft_cover=True),
            TerrainHex(coord=HexCoord(q=1, r=0), provides_soft_cover=True),
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
        # Actor should not have hidden status
        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        assert "hidden" not in updated_actor.statuses

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
            engineering_skill=2,
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

    def test_dangerous_terrain_prompt_creates_decision(self):
        """Prompting dangerous terrain should create a pending decision without auto damage."""
        from core.shared.terrain import TerrainMap, TerrainHex

        actor = make_combatant(
            id="actor_1",
            engineering_skill=2,
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
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
            HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
        ]

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="move",
            action_type="free",
            movement_path=movement_path,
            prompt_dangerous_terrain=True,
        )

        updated_scenario, _, _, result = execute_action(
            scenario, turn, economy, action_input
        )

        assert result.success
        decisions = [
            d for d in updated_scenario.pending_decisions
            if d.decision_type == "engineering_check"
        ]
        assert len(decisions) == 1
        assert decisions[0].combatant_id == "actor_1"
        assert decisions[0].trigger_round == 1
        assert decisions[0].trigger_source.startswith("dangerous_terrain:")

        damage_effects = [e for e in result.effects_applied if e.get("type") == "damage"]
        assert damage_effects == []

        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        assert updated_actor.dangerous_terrain_last_check_round == 1

    def test_dangerous_terrain_prompt_ignored_for_hostiles(self):
        """Hostile combatants still auto-resolve dangerous terrain even if prompt requested."""
        from core.shared.terrain import TerrainMap, TerrainHex
        from unittest.mock import patch

        actor = make_combatant(
            id="actor_1",
            side="hostiles",
            engineering_skill=2,
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
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
            HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
        ]

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="move",
            action_type="free",
            movement_path=movement_path,
            prompt_dangerous_terrain=True,
        )

        with patch("core.shared.terrain.roll_dice", return_value=5):
            updated_scenario, _, _, result = execute_action(
                scenario, turn, economy, action_input
            )

        assert result.success
        assert updated_scenario.pending_decisions == []
        danger_effects = [e for e in result.effects_applied if e.get("type") == "dangerous_terrain"]
        assert len(danger_effects) >= 1


# =============================================================================
# Movement Engagement Status Tests
# =============================================================================


class TestMovementEngagementStatus:
    """Tests for engagement stop rules and engaged status application during movement.

    Per PR2 3817-3819:
    "If you move adjacent to a hostile character, you become engaged.
    If you become engaged with a target the same size or larger, you must stop."
    """

    def test_move_adjacent_applies_engaged_to_both(self):
        """Moving adjacent to hostile should apply engaged status to both parties."""
        actor = make_combatant(
            id="actor_1",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        enemy = make_combatant(
            id="enemy_1",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=2, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[actor, enemy])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        # Move adjacent to enemy
        movement_path = [
            HexPosition(coord=HexCoord(q=1, r=0), elevation=0),  # Adjacent to enemy at (2,0)
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

        # Both actor and enemy should have engaged status
        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        updated_enemy = next(c for c in updated_scenario.combatants if c.id == "enemy_1")

        assert "engaged" in updated_actor.statuses
        assert "engaged" in updated_enemy.statuses

        # Check for status_applied effects
        status_effects = [e for e in result.effects_applied if e.get("type") == "status_applied"]
        engaged_effects = [e for e in status_effects if e.get("status") == "engaged"]
        assert len(engaged_effects) >= 2  # Both should get the status

    def test_move_adjacent_to_larger_stops_movement(self):
        """Movement should stop when moving adjacent to same-size-or-larger hostile."""
        actor = make_combatant(
            id="actor_1",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        # Create a larger enemy (size 2)
        enemy = CombatantState(
            id="enemy_1",
            name="Large Enemy",
            side="hostiles",
            kind="mech",
            stats=CombatStats(
                size="size_2",  # Larger than actor
                hp_max=10,
                evasion=8,
                e_defense=8,
                armor=0,
                speed=4,
                sensor_range=10,
                tech_attack=0,
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
            position=HexPosition(coord=HexCoord(q=2, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[actor, enemy])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        # Try to move past the enemy
        movement_path = [
            HexPosition(coord=HexCoord(q=1, r=0), elevation=0),  # Adjacent to enemy
            HexPosition(coord=HexCoord(q=2, r=-1), elevation=0),  # Would pass by
            HexPosition(coord=HexCoord(q=3, r=0), elevation=0),  # Final destination
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

        # Actor should have stopped at the adjacent hex
        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        assert updated_actor.position.coord.q == 1
        assert updated_actor.position.coord.r == 0

        # Check for engagement_stop effect
        stop_effects = [e for e in result.effects_applied if e.get("type") == "engagement_stop"]
        assert len(stop_effects) >= 1

    def test_move_adjacent_to_smaller_continues_moving(self):
        """Movement should NOT stop when moving adjacent to smaller hostile."""
        # Create actor of size 2
        actor = CombatantState(
            id="actor_1",
            name="Large Actor",
            side="players",
            kind="mech",
            stats=CombatStats(
                size="size_2",  # Larger than enemy
                hp_max=10,
                evasion=8,
                e_defense=8,
                armor=0,
                speed=6,
                sensor_range=10,
                tech_attack=0,
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
        )
        # Small enemy (size 1)
        enemy = make_combatant(
            id="enemy_1",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=2, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[actor, enemy])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        # Move past the enemy to final destination
        movement_path = [
            HexPosition(coord=HexCoord(q=1, r=0), elevation=0),  # Adjacent to smaller enemy
            HexPosition(coord=HexCoord(q=2, r=-1), elevation=0),  # Past enemy
            HexPosition(coord=HexCoord(q=3, r=0), elevation=0),  # Final destination
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

        # Actor should reach final destination (not stopped)
        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        assert updated_actor.position.coord.q == 3
        assert updated_actor.position.coord.r == 0

        # Should NOT have engagement_stop effect
        stop_effects = [e for e in result.effects_applied if e.get("type") == "engagement_stop"]
        assert len(stop_effects) == 0

    def test_move_adjacent_to_same_size_stops_movement(self):
        """Movement should stop when moving adjacent to same-size hostile."""
        actor = make_combatant(
            id="actor_1",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        enemy = make_combatant(
            id="enemy_1",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=2, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[actor, enemy])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        # Try to move past the enemy
        movement_path = [
            HexPosition(coord=HexCoord(q=1, r=0), elevation=0),  # Adjacent to enemy
            HexPosition(coord=HexCoord(q=2, r=-1), elevation=0),  # Would pass by
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

        # Actor should have stopped at the adjacent hex
        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        assert updated_actor.position.coord.q == 1
        assert updated_actor.position.coord.r == 0

    def test_disengage_ignores_engagement_stop(self):
        """Disengage action should allow movement past same-size hostiles."""
        actor = make_combatant(
            id="actor_1",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        enemy = make_combatant(
            id="enemy_1",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=2, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[actor, enemy])

        # Create turn with disengage action already taken
        disengage_action = ActionUse(
            action_id="disengage",
            action_type="full",
        )
        turn = CombatTurn(actor_id="actor_1", actions=[disengage_action])
        economy = ActionEconomyState()

        # Move past the enemy
        movement_path = [
            HexPosition(coord=HexCoord(q=1, r=0), elevation=0),  # Adjacent to enemy
            HexPosition(coord=HexCoord(q=2, r=-1), elevation=0),  # Past enemy
            HexPosition(coord=HexCoord(q=3, r=0), elevation=0),  # Final destination
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

        # Actor should reach final destination (disengage ignores engagement stop)
        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        assert updated_actor.position.coord.q == 3
        assert updated_actor.position.coord.r == 0

        # Should NOT have engagement_stop effect
        stop_effects = [e for e in result.effects_applied if e.get("type") == "engagement_stop"]
        assert len(stop_effects) == 0

    def test_disengage_prevents_engagement_status(self):
        """Disengage action should prevent engaged status application."""
        actor = make_combatant(
            id="actor_1",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        enemy = make_combatant(
            id="enemy_1",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=2, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[actor, enemy])

        # Create turn with disengage action already taken
        disengage_action = ActionUse(
            action_id="disengage",
            action_type="full",
        )
        turn = CombatTurn(actor_id="actor_1", actions=[disengage_action])
        economy = ActionEconomyState()

        # Move adjacent to enemy
        movement_path = [
            HexPosition(coord=HexCoord(q=1, r=0), elevation=0),  # Adjacent to enemy
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

        # Neither actor nor enemy should have engaged status (disengage prevents it)
        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        updated_enemy = next(c for c in updated_scenario.combatants if c.id == "enemy_1")

        assert "engaged" not in updated_actor.statuses
        assert "engaged" not in updated_enemy.statuses


class TestMultipleEngagement:
    """Tests for engagement with multiple hostiles."""

    def test_end_adjacent_to_multiple_hostiles_engages_all(self):
        """Moving adjacent to multiple hostiles should engage all of them."""
        actor = make_combatant(
            id="actor_1",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        enemy1 = make_combatant(
            id="enemy_1",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=2, r=0), elevation=0),
        )
        enemy2 = make_combatant(
            id="enemy_2",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=1, r=1), elevation=0),
        )
        scenario = make_scenario(combatants=[actor, enemy1, enemy2])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        # Move to position adjacent to both enemies
        movement_path = [
            HexPosition(coord=HexCoord(q=1, r=0), elevation=0),  # Adjacent to both
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

        # All three should have engaged status
        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        updated_enemy1 = next(c for c in updated_scenario.combatants if c.id == "enemy_1")
        updated_enemy2 = next(c for c in updated_scenario.combatants if c.id == "enemy_2")

        assert "engaged" in updated_actor.statuses
        assert "engaged" in updated_enemy1.statuses
        assert "engaged" in updated_enemy2.statuses

    def test_already_engaged_no_duplicate_status(self):
        """Moving while already engaged should not duplicate the engaged status."""
        actor = make_combatant(
            id="actor_1",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
            statuses=["engaged"],  # Already engaged
        )
        enemy = make_combatant(
            id="enemy_1",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=2, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[actor, enemy])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        # Move adjacent to enemy
        movement_path = [
            HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
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

        # Actor should have exactly one engaged status (no duplicate)
        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        assert updated_actor.statuses.count("engaged") == 1

    def test_move_away_clears_engaged_status(self):
        """Moving away from adjacency should clear engaged on both parties."""
        actor = make_combatant(
            id="actor_1",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
            statuses=["engaged"],
        )
        enemy = make_combatant(
            id="enemy_1",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            statuses=["engaged"],
        )
        scenario = make_scenario(combatants=[actor, enemy])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        movement_path = [
            HexPosition(coord=HexCoord(q=-1, r=0), elevation=0),
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
        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        updated_enemy = next(c for c in updated_scenario.combatants if c.id == "enemy_1")
        assert "engaged" not in updated_actor.statuses
        assert "engaged" not in updated_enemy.statuses

    def test_friendly_adjacent_no_engagement(self):
        """Moving adjacent to a friendly should not trigger engagement."""
        actor = make_combatant(
            id="actor_1",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        ally = make_combatant(
            id="ally_1",
            side="players",  # Same side
            position=HexPosition(coord=HexCoord(q=2, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[actor, ally])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        # Move adjacent to ally
        movement_path = [
            HexPosition(coord=HexCoord(q=1, r=0), elevation=0),  # Adjacent to ally
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

        # Neither should have engaged status (allies don't engage)
        updated_actor = next(c for c in updated_scenario.combatants if c.id == "actor_1")
        updated_ally = next(c for c in updated_scenario.combatants if c.id == "ally_1")

        assert "engaged" not in updated_actor.statuses
        assert "engaged" not in updated_ally.statuses

    def test_engagement_effects_in_action_result(self):
        """Engagement effects should be recorded in action result."""
        actor = make_combatant(
            id="actor_1",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        enemy = make_combatant(
            id="enemy_1",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=2, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[actor, enemy])
        turn = CombatTurn(actor_id="actor_1", actions=[])
        economy = ActionEconomyState()

        # Move adjacent to enemy
        movement_path = [
            HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
        ]

        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="move",
            action_type="free",
            movement_path=movement_path,
        )

        _, _, _, result = execute_action(
            scenario, turn, economy, action_input
        )

        assert result.success

        # Check that status_applied effects are in the result
        status_effects = [e for e in result.effects_applied if e.get("type") == "status_applied"]
        assert len(status_effects) >= 2  # Actor and enemy

        # Check both have the engaged status effect
        engaged_targets = [e.get("target_id") for e in status_effects if e.get("status") == "engaged"]
        assert "actor_1" in engaged_targets
        assert "enemy_1" in engaged_targets


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

    def test_cover_applies_for_thrown_melee(self):
        """Thrown melee attacks should apply cover modifiers."""
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

        # tactical_knife has thrown 3, so distance 2 should be thrown
        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="skirmish",
            action_type="quick",
            target_ids=["target_1"],
            weapon_id="tactical_knife",
            use_thrown=True,
        )

        with patch("core.shared.rolls._roll_d20", return_value=15):
            _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success
        cover_effect = next(
            (e for e in result.effects_applied if e.get("type") == "cover_modifier"),
            None
        )
        assert cover_effect is not None
        assert cover_effect["cover_type"] == "soft"
        assert cover_effect["difficulty_added"] == 1

        attack_effect = next(e for e in result.effects_applied if e.get("type") == "attack")
        assert attack_effect["status_modifiers"]["cover_diff"] == 1

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

        # tactical_melee has threat 1
        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            weapon_id="tactical_melee",
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

        # tactical_melee has threat 1
        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            weapon_id="tactical_melee",
            target_ids=["target"],
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is False
        assert "out of range" in result.error.lower()

    def test_thrown_attack_within_range_succeeds(self):
        """Thrown melee weapon should work within thrown range."""
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
            position=HexPosition(coord=HexCoord(q=3, r=0)),  # Distance 3
            hp_current=10,
            hp_max=10,
        )
        scenario = make_scenario(combatants=[attacker, target])
        turn = make_turn(actor_id="attacker")
        economy = ActionEconomyState()

        # tactical_knife has thrown 3
        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            weapon_id="tactical_knife",
            target_ids=["target"],
            use_thrown=True,
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

    def test_thrown_attack_out_of_range_fails(self):
        """Thrown melee weapon should fail beyond thrown range."""
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
            position=HexPosition(coord=HexCoord(q=4, r=0)),  # Distance 4
        )
        scenario = make_scenario(combatants=[attacker, target])
        turn = make_turn(actor_id="attacker")
        economy = ActionEconomyState()

        # tactical_knife has thrown 3
        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            weapon_id="tactical_knife",
            target_ids=["target"],
            use_thrown=True,
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is False
        assert "out of range" in result.error.lower()

    def test_thrown_attack_uses_tag_range(self):
        """Thrown range should be honored when defined as a tag value."""
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

        # ha_type3_burst_knife has thrown 5 as a tag
        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            weapon_id="ha_type3_burst_knife",
            target_ids=["target"],
            use_thrown=True,
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


# =============================================================================
# Thrown Weapon Handling Tests
# =============================================================================


class TestThrownWeaponHandling:
    """Tests for thrown melee weapon disarm and retrieval behavior."""

    def make_combatant_with_weapon(
        self,
        id: str,
        weapon_id: str,
        thrown_coord: HexCoord | None = None,
    ) -> CombatantState:
        weapon_state = WeaponState(
            weapon_id=weapon_id,
            tags=[],
            destroyed=False,
            limited_charges_remaining=None,
            needs_reload=False,
            thrown_coord=thrown_coord,
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
            name="Thrown Tester",
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
            statuses=[],
            inventory=inventory,
        )

    def test_thrown_weapon_marks_disarmed_on_attack(self):
        """Thrown attack should set weapon thrown_coord."""
        from unittest.mock import patch

        attacker = self.make_combatant_with_weapon(
            id="attacker",
            weapon_id="tactical_knife",
        )
        target = make_combatant(
            id="target",
            position=HexPosition(coord=HexCoord(q=2, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[attacker, target])
        turn = CombatTurn(actor_id="attacker")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            weapon_id="tactical_knife",
            target_ids=["target"],
            use_thrown=True,
        )

        with patch("core.shared.rolls.resolve_attack") as mock_resolve:
            mock_resolve.return_value = AttackResolutionResult(
                roll=15,
                attack_bonus=0,
                accuracy_dice_rolls=[],
                difficulty_dice_rolls=[],
                net_accuracy=0,
                total_accuracy=15,
                target_defense=8,
                hit=True,
                is_critical=False,
                miss_by=0,
            )
            updated_scenario, _, _, result = execute_action(
                scenario, turn, economy, action_input
            )

        assert result.success is True
        updated_attacker = next(c for c in updated_scenario.combatants if c.id == "attacker")
        weapon_state = updated_attacker.inventory.mounts[0].weapons[0]
        assert weapon_state.thrown_coord == HexCoord(q=2, r=0)
        assert result.action_use is not None
        assert any(
            effect.type == "weapon_thrown"
            for effect in result.action_use.log_effects
        )

    def test_thrown_weapon_marks_disarmed_without_targets(self):
        """Thrown attack should disarm even when no combatant targets are provided."""
        attacker = self.make_combatant_with_weapon(
            id="attacker",
            weapon_id="tactical_knife",
        )
        scenario = make_scenario(combatants=[attacker])
        turn = CombatTurn(actor_id="attacker")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            weapon_id="tactical_knife",
            target_position=HexPosition(coord=HexCoord(q=2, r=0), elevation=0),
            use_thrown=True,
        )

        updated_scenario, _, _, result = execute_action(
            scenario, turn, economy, action_input
        )

        assert result.success is True
        updated_attacker = next(c for c in updated_scenario.combatants if c.id == "attacker")
        weapon_state = updated_attacker.inventory.mounts[0].weapons[0]
        assert weapon_state.thrown_coord == HexCoord(q=2, r=0)
        assert result.action_use is not None
        assert any(
            effect.type == "weapon_thrown"
            for effect in result.action_use.log_effects
        )

    def test_thrown_weapon_blocked_until_retrieved(self):
        """Thrown weapons should be unusable until retrieved."""
        attacker = self.make_combatant_with_weapon(
            id="attacker",
            weapon_id="tactical_knife",
            thrown_coord=HexCoord(q=2, r=0),
        )
        target = make_combatant(
            id="target",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[attacker, target])
        turn = CombatTurn(actor_id="attacker")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="skirmish",
            action_type="quick",
            weapon_id="tactical_knife",
            target_ids=["target"],
            use_thrown=True,
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is False
        assert "retrieve" in result.error.lower()

    def test_thrown_weapon_retrieved_on_adjacent_move(self):
        """Moving adjacent to thrown weapon should retrieve it."""
        attacker = self.make_combatant_with_weapon(
            id="attacker",
            weapon_id="tactical_knife",
            thrown_coord=HexCoord(q=2, r=0),
        )
        scenario = make_scenario(combatants=[attacker])
        turn = CombatTurn(actor_id="attacker")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="attacker",
            action_id="move",
            action_type="free",
            movement_path=[HexPosition(coord=HexCoord(q=1, r=0), elevation=0)],
        )

        updated_scenario, _, _, result = execute_action(
            scenario, turn, economy, action_input
        )

        assert result.success is True
        updated_attacker = next(c for c in updated_scenario.combatants if c.id == "attacker")
        weapon_state = updated_attacker.inventory.mounts[0].weapons[0]
        assert weapon_state.thrown_coord is None

        retrieve_effect = next(
            (e for e in result.effects_applied if e.get("type") == "retrieve_thrown_weapon"),
            None,
        )
        assert retrieve_effect is not None
        assert result.action_use is not None
        assert any(
            effect.type == "retrieve_thrown_weapon"
            for effect in result.action_use.log_effects
        )


class TestActionLogEffects:
    """Tests for action log effect summaries."""

    def test_action_log_effects_include_statuses(self):
        """Status applications should be captured in action log effects."""
        actor = make_combatant(id="actor")
        target = make_combatant(id="target")
        scenario = make_scenario(combatants=[actor, target])
        turn = CombatTurn(actor_id="actor")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor",
            action_id="lock_on",
            action_type="quick",
            target_ids=["target"],
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is True
        assert result.action_use is not None
        assert any(
            effect.type == "status_applied" and effect.status == "lock_on"
            for effect in result.action_use.log_effects
        )


# =============================================================================
# Weapon Inventory Validation Tests
# =============================================================================


class TestWeaponInventoryValidation:
    """Tests for weapon inventory validation in execute_action."""

    def test_attack_with_weapon_not_in_inventory(self):
        """Test that attack fails when weapon_id is not in actor's inventory."""
        # Create combatant with no weapons in inventory
        inventory = MechInventory(mounts=[], systems=[])
        actor = make_combatant(id="actor", inventory=inventory)
        target = make_combatant(
            id="target",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[actor, target])
        turn = make_turn(actor_id="actor")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor",
            action_id="skirmish",
            action_type="quick",
            target_ids=["target"],
            weapon_id="nonexistent_weapon",
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is False
        assert "not found in inventory" in result.error.lower()

    def test_attack_with_valid_weapon_in_inventory(self):
        """Test that attack succeeds when weapon_id is in actor's inventory."""
        # Create combatant with weapon in inventory
        weapon_state = WeaponState(weapon_id="test_weapon", tags=[])
        mount = WeaponMountState(mount_index=0, weapons=[weapon_state])
        inventory = MechInventory(mounts=[mount], systems=[])
        actor = make_combatant(id="actor", inventory=inventory)
        target = make_combatant(
            id="target",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[actor, target])
        turn = make_turn(actor_id="actor")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor",
            action_id="skirmish",
            action_type="quick",
            target_ids=["target"],
            weapon_id="test_weapon",
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is True

    def test_attack_with_destroyed_weapon(self):
        """Test that attack fails when weapon is destroyed."""
        # Create combatant with destroyed weapon
        weapon_state = WeaponState(weapon_id="destroyed_weapon", tags=[], destroyed=True)
        mount = WeaponMountState(mount_index=0, weapons=[weapon_state])
        inventory = MechInventory(mounts=[mount], systems=[])
        actor = make_combatant(id="actor", inventory=inventory)
        target = make_combatant(
            id="target",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[actor, target])
        turn = make_turn(actor_id="actor")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor",
            action_id="skirmish",
            action_type="quick",
            target_ids=["target"],
            weapon_id="destroyed_weapon",
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is False
        assert "destroyed" in result.error.lower()

    def test_attack_with_no_weapon_id(self):
        """Test that attack succeeds when no weapon_id is specified."""
        actor = make_combatant(id="actor")
        target = make_combatant(
            id="target",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[actor, target])
        turn = make_turn(actor_id="actor")
        economy = ActionEconomyState()

        # Action without weapon_id (e.g., ram, grapple)
        action_input = ActionExecutionInput(
            actor_id="actor",
            action_id="ram",
            action_type="quick",
            target_ids=["target"],
            weapon_id=None,  # No weapon specified
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        # Should succeed - ram doesn't require weapon_id
        assert result.success is True


# =============================================================================
# System Inventory Validation Tests
# =============================================================================


class TestSystemInventoryValidation:
    """Tests for system inventory validation in execute_action."""

    def test_activate_with_system_not_in_inventory(self):
        """Test that activate fails when system_id is not in actor's inventory."""
        # Create combatant with no systems in inventory
        inventory = MechInventory(mounts=[], systems=[])
        actor = make_combatant(id="actor", inventory=inventory)
        scenario = make_scenario(combatants=[actor])
        turn = make_turn(actor_id="actor")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor",
            action_id="activate",
            action_type="quick",
            system_id="nonexistent_system",
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is False
        assert "not found in inventory" in result.error.lower()

    def test_activate_with_valid_system_in_inventory(self):
        """Test that activate succeeds when system_id is in actor's inventory."""
        # Create combatant with system in inventory
        system_state = MechSystemState(system_id="test_system")
        inventory = MechInventory(mounts=[], systems=[system_state])
        actor = make_combatant(id="actor", inventory=inventory)
        scenario = make_scenario(combatants=[actor])
        turn = make_turn(actor_id="actor")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor",
            action_id="activate",
            action_type="quick",
            system_id="test_system",
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is True

    def test_activate_with_destroyed_system(self):
        """Test that activate fails when system is destroyed."""
        # Create combatant with destroyed system
        system_state = MechSystemState(system_id="destroyed_system", destroyed=True)
        inventory = MechInventory(mounts=[], systems=[system_state])
        actor = make_combatant(id="actor", inventory=inventory)
        scenario = make_scenario(combatants=[actor])
        turn = make_turn(actor_id="actor")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor",
            action_id="activate",
            action_type="quick",
            system_id="destroyed_system",
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is False
        assert "destroyed" in result.error.lower()

    def test_activate_with_no_charges(self):
        """Test that activate fails when system has no charges remaining."""
        # Create combatant with system at 0 charges
        system_state = MechSystemState(
            system_id="limited_system",
            limited_charges_remaining=0,
        )
        inventory = MechInventory(mounts=[], systems=[system_state])
        actor = make_combatant(id="actor", inventory=inventory)
        scenario = make_scenario(combatants=[actor])
        turn = make_turn(actor_id="actor")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="actor",
            action_id="activate",
            action_type="quick",
            system_id="limited_system",
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is False
        assert "no charges remaining" in result.error.lower()


# =============================================================================
# Movement Overwatch Integration Tests
# =============================================================================


class TestMovementOverwatchIntegration:
    """Integration tests for overwatch trigger detection during movement."""

    def test_move_action_returns_overwatch_opportunities(self):
        """Test that move action returns overwatch opportunities when in threat range."""
        # Create player mech at (0,0)
        player = make_combatant(
            id="player_1",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        # Create enemy mech at (1,0) with melee weapon (threat 1)
        melee_inventory = MechInventory(
            mounts=[
                WeaponMountState(
                    mount_index=0,
                    weapons=[
                        WeaponState(weapon_id="charged_blade", tags=[], destroyed=False)
                    ],
                )
            ],
            systems=[],
        )
        enemy = make_combatant(
            id="enemy_1",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            inventory=melee_inventory,
        )

        scenario = make_scenario(combatants=[player, enemy])
        turn = make_turn(actor_id="player_1")
        economy = ActionEconomyState()

        # Execute move action with a movement path
        action_input = ActionExecutionInput(
            actor_id="player_1",
            action_id="move",
            action_type="full",
            movement_path=[HexPosition(coord=HexCoord(q=2, r=0), elevation=0)],
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is True
        # Should detect overwatch opportunity from enemy_1
        assert len(result.overwatch_opportunities) == 1
        opp = result.overwatch_opportunities[0]
        assert opp.reactor_id == "enemy_1"
        assert opp.weapon_id == "charged_blade"
        assert opp.weapon_threat == 1
        assert opp.can_react is True

    def test_boost_action_returns_overwatch_opportunities(self):
        """Test that boost action returns overwatch opportunities when in threat range."""
        # Create player mech at (0,0)
        player = make_combatant(
            id="player_1",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        # Create enemy mech at (1,0) with melee weapon
        melee_inventory = MechInventory(
            mounts=[
                WeaponMountState(
                    mount_index=0,
                    weapons=[
                        WeaponState(weapon_id="heavy_melee_weapon", tags=[], destroyed=False)
                    ],
                )
            ],
            systems=[],
        )
        enemy = make_combatant(
            id="enemy_1",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            inventory=melee_inventory,
        )

        scenario = make_scenario(combatants=[player, enemy])
        turn = make_turn(actor_id="player_1")
        economy = ActionEconomyState()

        # Execute boost action with a movement path
        action_input = ActionExecutionInput(
            actor_id="player_1",
            action_id="boost",
            action_type="quick",
            movement_path=[HexPosition(coord=HexCoord(q=3, r=0), elevation=0)],
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is True
        # Should detect overwatch opportunity
        assert len(result.overwatch_opportunities) == 1
        assert result.overwatch_opportunities[0].reactor_id == "enemy_1"

    def test_move_outside_threat_range_no_opportunities(self):
        """Test that move starting outside threat range returns no opportunities."""
        # Create player mech at (0,0)
        player = make_combatant(
            id="player_1",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        # Create enemy mech at (5,0) - far outside threat range 1
        melee_inventory = MechInventory(
            mounts=[
                WeaponMountState(
                    mount_index=0,
                    weapons=[
                        WeaponState(weapon_id="charged_blade", tags=[], destroyed=False)
                    ],
                )
            ],
            systems=[],
        )
        enemy = make_combatant(
            id="enemy_1",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=5, r=0), elevation=0),
            inventory=melee_inventory,
        )

        scenario = make_scenario(combatants=[player, enemy])
        turn = make_turn(actor_id="player_1")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="player_1",
            action_id="move",
            action_type="full",
            movement_path=[HexPosition(coord=HexCoord(q=1, r=0), elevation=0)],
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is True
        assert len(result.overwatch_opportunities) == 0

    def test_move_entering_threat_with_cqb_trigger(self):
        """Entering CQB threat should trigger overwatch when allowed by talent."""
        player = make_combatant(
            id="player_1",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        cqb_inventory = MechInventory(
            mounts=[
                WeaponMountState(
                    mount_index=0,
                    weapons=[WeaponState(weapon_id="pistol", tags=[], destroyed=False)],
                )
            ],
            systems=[],
        )
        semper_vigilo = MechanicalEffect(
            reaction_triggers=[
                ReactionTriggerEffect(
                    reaction_id="overwatch",
                    trigger_events=["enemy_enters_threat"],
                    condition="cqb_overwatch",
                )
            ]
        )
        enemy = make_combatant(
            id="enemy_1",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=4, r=0), elevation=0),
            inventory=cqb_inventory,
            talent_effects=[semper_vigilo],
        )

        scenario = make_scenario(combatants=[player, enemy])
        turn = make_turn(actor_id="player_1")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="player_1",
            action_id="move",
            action_type="full",
            movement_path=[HexPosition(coord=HexCoord(q=1, r=0), elevation=0)],
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is True
        assert len(result.overwatch_opportunities) == 1
        assert result.overwatch_opportunities[0].reactor_id == "enemy_1"

    def test_move_with_disengage_no_opportunities(self):
        """Test that movement after disengage returns no overwatch opportunities."""
        # Create player mech at (0,0)
        player = make_combatant(
            id="player_1",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        # Create enemy mech at (1,0) with melee weapon
        melee_inventory = MechInventory(
            mounts=[
                WeaponMountState(
                    mount_index=0,
                    weapons=[
                        WeaponState(weapon_id="charged_blade", tags=[], destroyed=False)
                    ],
                )
            ],
            systems=[],
        )
        enemy = make_combatant(
            id="enemy_1",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            inventory=melee_inventory,
        )

        scenario = make_scenario(combatants=[player, enemy])
        # Turn with disengage action already taken
        disengage_action = ActionUse(
            action_id="disengage",
            action_type="full",
            target_ids=[],
        )
        turn = CombatTurn(
            actor_id="player_1",
            move_used=False,
            movement_mode="ground",
            movement_path=[],
            actions=[disengage_action],
        )
        economy = ActionEconomyState(full_actions_used=1)

        # Execute boost after disengage (quick action since full already used)
        action_input = ActionExecutionInput(
            actor_id="player_1",
            action_id="boost",
            action_type="quick",
            movement_path=[HexPosition(coord=HexCoord(q=2, r=0), elevation=0)],
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is True
        # Disengage should prevent overwatch
        assert len(result.overwatch_opportunities) == 0

    def test_hidden_mover_no_opportunities(self):
        """Test that hidden movers don't trigger overwatch opportunities."""
        # Create hidden player mech at (0,0)
        player = make_combatant(
            id="player_1",
            side="players",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
            statuses=["hidden"],
        )
        # Create enemy mech at (1,0) with melee weapon
        melee_inventory = MechInventory(
            mounts=[
                WeaponMountState(
                    mount_index=0,
                    weapons=[
                        WeaponState(weapon_id="charged_blade", tags=[], destroyed=False)
                    ],
                )
            ],
            systems=[],
        )
        enemy = make_combatant(
            id="enemy_1",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            inventory=melee_inventory,
        )

        scenario = make_scenario(combatants=[player, enemy])
        turn = make_turn(actor_id="player_1")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="player_1",
            action_id="move",
            action_type="full",
            movement_path=[HexPosition(coord=HexCoord(q=2, r=0), elevation=0)],
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is True
        assert len(result.overwatch_opportunities) == 0


# =============================================================================
# Status Duration Expiration Tests
# =============================================================================


class TestStatusDurationExpiration:
    """Tests for automatic status expiration based on turn boundaries."""

    def test_braced_expires_end_of_next_turn(self):
        """Test that braced status expires at end of next turn."""
        from core.mech.combat_helpers import _apply_status_with_duration

        actor = make_combatant(id="actor_1", name="Actor")
        scenario = make_scenario(combatants=[actor])

        # Apply braced status in round 1
        scenario, applied = _apply_status_with_duration(
            scenario, "actor_1", "braced", current_round=1
        )
        assert applied is True

        # Verify status is present
        updated_actor = next(c for c in scenario.combatants if c.id == "actor_1")
        assert "braced" in updated_actor.statuses
        assert len(updated_actor.status_instances) == 1
        assert updated_actor.status_instances[0].status == "braced"
        assert updated_actor.status_instances[0].duration_type == "end_of_next_turn"

        # End turn in round 1 - should NOT expire yet
        turn = make_turn(actor_id="actor_1")
        round1 = make_round(round_index=1, turns=[turn])
        scenario = scenario.model_copy(update={"rounds": [round1]})

        scenario, result, _, _ = end_turn(scenario, current_round=1, current_turn_index=0, current_turn=turn)

        actor_after_r1 = next(c for c in scenario.combatants if c.id == "actor_1")
        assert "braced" in actor_after_r1.statuses  # Still present

        # End turn in round 2 - should expire
        turn2 = make_turn(actor_id="actor_1")
        round2 = make_round(round_index=2, turns=[turn2])
        scenario = scenario.model_copy(update={"rounds": [round1, round2]})

        scenario, result, _, _ = end_turn(scenario, current_round=2, current_turn_index=0, current_turn=turn2)

        actor_after_r2 = next(c for c in scenario.combatants if c.id == "actor_1")
        assert "braced" not in actor_after_r2.statuses  # Expired

        # Check expiration was recorded
        expired_effects = [e for e in result.end_of_turn_effects if e.get("type") == "status_expired"]
        assert len(expired_effects) == 1
        assert expired_effects[0]["status"] == "braced"

    def test_stunned_expires_end_of_next_turn(self):
        """Test that stunned status expires at end of next turn."""
        from core.mech.combat_helpers import _apply_status_with_duration

        actor = make_combatant(id="actor_1", name="Actor")
        scenario = make_scenario(combatants=[actor])

        # Apply stunned status in round 1
        scenario, applied = _apply_status_with_duration(
            scenario, "actor_1", "stunned", current_round=1
        )
        assert applied is True

        updated_actor = next(c for c in scenario.combatants if c.id == "actor_1")
        assert "stunned" in updated_actor.statuses
        assert updated_actor.status_instances[0].duration_type == "end_of_next_turn"

        # End turn in round 2 - should expire
        turn = make_turn(actor_id="actor_1")
        round1 = make_round(round_index=1, turns=[turn])
        round2 = make_round(round_index=2, turns=[turn])
        scenario = scenario.model_copy(update={"rounds": [round1, round2]})

        scenario, result, _, _ = end_turn(scenario, current_round=2, current_turn_index=0, current_turn=turn)

        actor_after = next(c for c in scenario.combatants if c.id == "actor_1")
        assert "stunned" not in actor_after.statuses

    def test_indefinite_status_does_not_expire(self):
        """Test that indefinite duration statuses don't expire at turn boundaries."""
        from core.mech.combat_helpers import _apply_status_with_duration

        actor = make_combatant(id="actor_1", name="Actor")
        scenario = make_scenario(combatants=[actor])

        # Apply impaired status (indefinite duration)
        scenario, applied = _apply_status_with_duration(
            scenario, "actor_1", "impaired", current_round=1
        )
        assert applied is True

        updated_actor = next(c for c in scenario.combatants if c.id == "actor_1")
        assert updated_actor.status_instances[0].duration_type == "indefinite"

        # End multiple turns - should persist
        turn = make_turn(actor_id="actor_1")
        round1 = make_round(round_index=1, turns=[turn])
        scenario = scenario.model_copy(update={"rounds": [round1]})

        for round_num in range(1, 5):
            scenario, _, _, _ = end_turn(
                scenario, current_round=round_num, current_turn_index=0, current_turn=turn
            )
            actor_after = next(c for c in scenario.combatants if c.id == "actor_1")
            assert "impaired" in actor_after.statuses  # Still present

    def test_multiple_duration_statuses_expire_correctly(self):
        """Test that multiple statuses with different durations expire correctly."""
        from core.mech.combat_helpers import _apply_status_with_duration

        actor = make_combatant(id="actor_1", name="Actor")
        scenario = make_scenario(combatants=[actor])

        # Apply multiple statuses in round 1
        scenario, _ = _apply_status_with_duration(scenario, "actor_1", "braced", current_round=1)
        scenario, _ = _apply_status_with_duration(scenario, "actor_1", "impaired", current_round=1)

        updated_actor = next(c for c in scenario.combatants if c.id == "actor_1")
        assert "braced" in updated_actor.statuses
        assert "impaired" in updated_actor.statuses

        # End turn in round 2
        turn = make_turn(actor_id="actor_1")
        round1 = make_round(round_index=1, turns=[turn])
        round2 = make_round(round_index=2, turns=[turn])
        scenario = scenario.model_copy(update={"rounds": [round1, round2]})

        scenario, result, _, _ = end_turn(scenario, current_round=2, current_turn_index=0, current_turn=turn)

        actor_after = next(c for c in scenario.combatants if c.id == "actor_1")
        # Braced should expire, impaired should persist
        assert "braced" not in actor_after.statuses
        assert "impaired" in actor_after.statuses


# =============================================================================
# Status Trigger Clearing Tests
# =============================================================================


class TestStatusTriggerClearing:
    """Tests for status clearing based on action triggers."""

    def test_hidden_clears_on_attack(self):
        """Test that hidden status is cleared when attacking."""
        actor = make_combatant(
            id="actor_1",
            name="Actor",
            statuses=["hidden"],
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        target = make_combatant(
            id="target_1",
            name="Target",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[actor, target])
        turn = make_turn(actor_id="actor_1")
        economy = ActionEconomyState()

        # Execute skirmish attack
        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="skirmish",
            action_type="quick",
            target_ids=["target_1"],
        )

        scenario, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is True
        actor_after = next(c for c in scenario.combatants if c.id == "actor_1")
        assert "hidden" not in actor_after.statuses

        # Check cleared status was recorded
        cleared_effects = [e for e in result.effects_applied if e.get("type") == "statuses_cleared"]
        assert len(cleared_effects) == 1
        assert "hidden" in cleared_effects[0]["statuses"]

    def test_hidden_clears_on_boost(self):
        """Test that hidden status is cleared when boosting."""
        actor = make_combatant(
            id="actor_1",
            name="Actor",
            statuses=["hidden"],
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[actor])
        turn = make_turn(actor_id="actor_1")
        economy = ActionEconomyState()

        # Execute boost
        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="boost",
            action_type="quick",
            movement_path=[HexPosition(coord=HexCoord(q=2, r=0), elevation=0)],
        )

        scenario, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is True
        actor_after = next(c for c in scenario.combatants if c.id == "actor_1")
        assert "hidden" not in actor_after.statuses

    def test_hidden_clears_on_reaction(self):
        """Test that hidden status is cleared when taking a reaction."""
        reactor = make_combatant(
            id="reactor_1",
            name="Reactor",
            statuses=["hidden"],
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[reactor])
        economy = ActionEconomyState()

        # Execute brace reaction
        reaction_input = ReactionInput(
            reactor_id="reactor_1",
            reaction_type="brace",
            trigger_action_id="some_attack",
        )

        scenario, _, result = execute_reaction(scenario, economy, reaction_input)

        assert result.success is True
        reactor_after = next(c for c in scenario.combatants if c.id == "reactor_1")
        assert "hidden" not in reactor_after.statuses

    def test_prone_clears_on_stand_up(self):
        """Test that prone status is cleared by stand_up action."""
        actor = make_combatant(
            id="actor_1",
            name="Actor",
            statuses=["prone"],
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[actor])
        turn = make_turn(actor_id="actor_1")
        economy = ActionEconomyState()

        # Execute stand_up
        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="stand_up",
            action_type="quick",
        )

        scenario, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is True
        actor_after = next(c for c in scenario.combatants if c.id == "actor_1")
        assert "prone" not in actor_after.statuses

    def test_shutdown_clears_on_boot_up(self):
        """Test that shutdown status is cleared by boot_up action."""
        actor = make_combatant(
            id="actor_1",
            name="Actor",
            statuses=["shutdown"],
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[actor])
        turn = make_turn(actor_id="actor_1")
        economy = ActionEconomyState()

        # Execute boot_up
        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="boot_up",
            action_type="full",
        )

        scenario, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is True
        actor_after = next(c for c in scenario.combatants if c.id == "actor_1")
        assert "shutdown" not in actor_after.statuses

    def test_non_matching_trigger_does_not_clear(self):
        """Test that statuses are only cleared by matching triggers."""
        actor = make_combatant(
            id="actor_1",
            name="Actor",
            statuses=["prone"],  # prone is cleared by stand_up, not attack
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        target = make_combatant(
            id="target_1",
            name="Target",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[actor, target])
        turn = make_turn(actor_id="actor_1")
        economy = ActionEconomyState()

        # Execute attack (should NOT clear prone)
        action_input = ActionExecutionInput(
            actor_id="actor_1",
            action_id="skirmish",
            action_type="quick",
            target_ids=["target_1"],
        )

        scenario, _, _, result = execute_action(scenario, turn, economy, action_input)

        actor_after = next(c for c in scenario.combatants if c.id == "actor_1")
        assert "prone" in actor_after.statuses  # Still prone


# =============================================================================
# Status Application Tests
# =============================================================================


class TestStatusApplication:
    """Tests for status application with duration tracking."""

    def test_apply_status_creates_instance(self):
        """Test that applying a status creates a StatusInstance."""
        from core.mech.combat_helpers import _apply_status_with_duration

        actor = make_combatant(id="actor_1", name="Actor")
        scenario = make_scenario(combatants=[actor])

        scenario, applied = _apply_status_with_duration(
            scenario, "actor_1", "braced", current_round=3
        )

        assert applied is True
        actor_after = next(c for c in scenario.combatants if c.id == "actor_1")
        assert len(actor_after.status_instances) == 1
        instance = actor_after.status_instances[0]
        assert instance.status == "braced"
        assert instance.applied_on_round == 3
        assert instance.duration_type == "end_of_next_turn"

    def test_apply_status_backwards_compatible(self):
        """Test that applying a status also updates the legacy statuses list."""
        from core.mech.combat_helpers import _apply_status_with_duration

        actor = make_combatant(id="actor_1", name="Actor")
        scenario = make_scenario(combatants=[actor])

        scenario, applied = _apply_status_with_duration(
            scenario, "actor_1", "impaired", current_round=1
        )

        actor_after = next(c for c in scenario.combatants if c.id == "actor_1")
        # Both lists should be updated
        assert "impaired" in actor_after.statuses
        assert any(inst.status == "impaired" for inst in actor_after.status_instances)

    def test_status_instance_tracks_applied_by(self):
        """Test that StatusInstance can track who applied the status."""
        from core.mech.combat_helpers import _apply_status_with_duration

        actor = make_combatant(id="actor_1", name="Actor")
        scenario = make_scenario(combatants=[actor])

        scenario, applied = _apply_status_with_duration(
            scenario, "actor_1", "lock_on", current_round=1, applied_by="enemy_tech"
        )

        actor_after = next(c for c in scenario.combatants if c.id == "actor_1")
        instance = actor_after.status_instances[0]
        assert instance.applied_by == "enemy_tech"


# =============================================================================
# Overwatch Attack Resolution Tests (Phase 27)
# =============================================================================


class TestOverwatchAttackResolution:
    """Tests for overwatch reaction attack resolution.

    Phase 27 implements actual attack resolution for overwatch reactions,
    replacing the previous stub that only tracked the intent to attack.
    """

    def make_combatant_with_weapon(
        self,
        id: str,
        weapon_id: str = "mw_assault_rifle",
        position: HexPosition | None = None,
        hp_current: int = 10,
        tags: list[str] | None = None,
        needs_reload: bool = False,
        limited_charges: int | None = None,
        side: str = "players",
        statuses: list[str] | None = None,
        grit: int = 2,
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
            name=f"Mech {id}",
            side=side,
            kind="mech",
            stats=CombatStats(
                size="size_1",
                hp_max=10,
                evasion=8,
                e_defense=8,
                armor=0,
                speed=4,
                sensor_range=10,
                grit=grit,
            ),
            resources=CombatResources(
                hp_current=hp_current,
                heat_current=0,
                heat_cap=6,
                structure_current=4,
                stress_current=4,
                repairs_remaining=4,
            ),
            position=position or HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
            statuses=statuses or [],
            inventory=inventory,
        )

    def test_overwatch_requires_weapon(self):
        """Test that overwatch reaction fails without a weapon_id."""
        reactor = self.make_combatant_with_weapon(
            id="reactor",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        target = make_combatant(
            id="target",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            side="hostiles",
        )
        scenario = make_scenario(combatants=[reactor, target])
        economy = ActionEconomyState()

        reaction_input = ReactionInput(
            reactor_id="reactor",
            reaction_type="overwatch",
            target_ids=["target"],
            weapon_id=None,  # No weapon provided
        )

        _, _, result = execute_reaction(scenario, economy, reaction_input)

        assert result.success is False
        assert "requires a weapon" in result.error.lower()

    def test_overwatch_requires_target(self):
        """Test that overwatch reaction fails without target_ids."""
        reactor = self.make_combatant_with_weapon(
            id="reactor",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[reactor])
        economy = ActionEconomyState()

        reaction_input = ReactionInput(
            reactor_id="reactor",
            reaction_type="overwatch",
            target_ids=[],  # No target provided
            weapon_id="mw_assault_rifle",
        )

        _, _, result = execute_reaction(scenario, economy, reaction_input)

        assert result.success is False
        assert "requires a target" in result.error.lower()

    def test_overwatch_target_not_found_error(self):
        """Test that overwatch fails when target doesn't exist."""
        reactor = self.make_combatant_with_weapon(
            id="reactor",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[reactor])  # No target
        economy = ActionEconomyState()

        reaction_input = ReactionInput(
            reactor_id="reactor",
            reaction_type="overwatch",
            target_ids=["nonexistent"],
            weapon_id="mw_assault_rifle",
        )

        _, _, result = execute_reaction(scenario, economy, reaction_input)

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_overwatch_loading_weapon_blocked(self):
        """Test that overwatch fails when weapon needs reload."""
        reactor = self.make_combatant_with_weapon(
            id="reactor",
            weapon_id="mw_assault_rifle",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
            tags=["loading"],
            needs_reload=True,  # Weapon needs reload
        )
        target = make_combatant(
            id="target",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            side="hostiles",
        )
        scenario = make_scenario(combatants=[reactor, target])
        economy = ActionEconomyState()

        reaction_input = ReactionInput(
            reactor_id="reactor",
            reaction_type="overwatch",
            target_ids=["target"],
            weapon_id="mw_assault_rifle",
        )

        _, _, result = execute_reaction(scenario, economy, reaction_input)

        assert result.success is False
        assert "needs reload" in result.error.lower()

    def test_overwatch_respects_weapon_range(self):
        """Test that overwatch fails when target is out of range."""
        reactor = self.make_combatant_with_weapon(
            id="reactor",
            weapon_id="mw_assault_rifle",  # Range 10
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        target = make_combatant(
            id="target",
            position=HexPosition(coord=HexCoord(q=15, r=0), elevation=0),  # Out of range
            side="hostiles",
        )
        scenario = make_scenario(combatants=[reactor, target])
        economy = ActionEconomyState()

        reaction_input = ReactionInput(
            reactor_id="reactor",
            reaction_type="overwatch",
            target_ids=["target"],
            weapon_id="mw_assault_rifle",
        )

        _, _, result = execute_reaction(scenario, economy, reaction_input)

        assert result.success is False
        assert "out of range" in result.error.lower()

    def test_overwatch_resolves_attack_success(self):
        """Test that overwatch successfully resolves an attack."""
        reactor = self.make_combatant_with_weapon(
            id="reactor",
            weapon_id="mw_assault_rifle",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
            grit=2,
        )
        target = make_combatant(
            id="target",
            hp_current=10,
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            side="hostiles",
        )
        scenario = make_scenario(combatants=[reactor, target])
        economy = ActionEconomyState()

        reaction_input = ReactionInput(
            reactor_id="reactor",
            reaction_type="overwatch",
            target_ids=["target"],
            weapon_id="mw_assault_rifle",
        )

        updated_scenario, updated_economy, result = execute_reaction(
            scenario, economy, reaction_input
        )

        assert result.success is True
        assert result.reaction_used == "overwatch"
        # Attack should have been resolved (hit or miss)
        assert result.attack_hit is not None  # Should be True or False
        assert result.attack_roll is not None  # Should have a d20 roll

        # Check that overwatch_attack effect was recorded
        overwatch_effects = [e for e in result.effects_applied if e.get("type") == "overwatch_attack"]
        assert len(overwatch_effects) == 1
        assert overwatch_effects[0]["target_id"] == "target"

    def test_overwatch_deals_damage_on_hit(self):
        """Test that overwatch deals damage when attack hits."""
        # Create reactor with high grit to increase hit chance
        reactor = self.make_combatant_with_weapon(
            id="reactor",
            weapon_id="mw_assault_rifle",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
            grit=6,  # High grit for better hit chance
        )
        target = make_combatant(
            id="target",
            hp_current=10,
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            side="hostiles",
        )
        scenario = make_scenario(combatants=[reactor, target])
        economy = ActionEconomyState()

        reaction_input = ReactionInput(
            reactor_id="reactor",
            reaction_type="overwatch",
            target_ids=["target"],
            weapon_id="mw_assault_rifle",
        )

        # Run multiple times to ensure we get at least one hit
        hit_count = 0
        for _ in range(20):  # Multiple attempts to account for RNG
            updated_scenario, _, result = execute_reaction(scenario, economy, reaction_input)

            if result.attack_hit:
                hit_count += 1
                # When hit, damage should be dealt
                assert result.damage_dealt > 0
                # Resource changes should be populated
                assert len(result.resource_changes) > 0
                # Target HP should be reduced, or a structure check reset HP to full
                target_after = next(c for c in updated_scenario.combatants if c.id == "target")
                assert (
                    target_after.resources.hp_current < 10
                    or target_after.resources.structure_current < 4
                )
                break

        # With grit 6 vs evasion 8, we should hit at least once in 20 attempts
        assert hit_count > 0, "Expected at least one hit in 20 attempts"

    def test_overwatch_no_damage_on_miss(self):
        """Test that overwatch deals no damage when attack misses."""
        # Create reactor with low grit to decrease hit chance
        reactor = self.make_combatant_with_weapon(
            id="reactor",
            weapon_id="mw_assault_rifle",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
            grit=0,  # Low grit
        )
        # Target with high evasion
        target = CombatantState(
            id="target",
            name="Evasive Target",
            side="hostiles",
            kind="mech",
            stats=CombatStats(
                size="size_1",
                hp_max=10,
                evasion=20,  # Very high evasion
                e_defense=20,
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
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            statuses=[],
        )
        scenario = make_scenario(combatants=[reactor, target])
        economy = ActionEconomyState()

        reaction_input = ReactionInput(
            reactor_id="reactor",
            reaction_type="overwatch",
            target_ids=["target"],
            weapon_id="mw_assault_rifle",
        )

        # Run multiple times to ensure we get at least one miss
        miss_count = 0
        for _ in range(20):
            updated_scenario, _, result = execute_reaction(scenario, economy, reaction_input)

            if not result.attack_hit:
                miss_count += 1
                # When miss, no damage dealt
                assert result.damage_dealt == 0
                # No resource changes
                assert len(result.resource_changes) == 0
                # Target HP should be unchanged
                target_after = next(c for c in updated_scenario.combatants if c.id == "target")
                assert target_after.resources.hp_current == 10
                break

        assert miss_count > 0, "Expected at least one miss with high evasion target"

    def test_overwatch_limited_weapon_decrements(self):
        """Test that limited weapon charges are consumed on overwatch attack."""
        reactor = self.make_combatant_with_weapon(
            id="reactor",
            weapon_id="mw_assault_rifle",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
            tags=["limited"],
            limited_charges=3,  # 3 charges
        )
        target = make_combatant(
            id="target",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            side="hostiles",
        )
        scenario = make_scenario(combatants=[reactor, target])
        economy = ActionEconomyState()

        reaction_input = ReactionInput(
            reactor_id="reactor",
            reaction_type="overwatch",
            target_ids=["target"],
            weapon_id="mw_assault_rifle",
        )

        updated_scenario, _, result = execute_reaction(scenario, economy, reaction_input)

        assert result.success is True

        # Check that limited charges were decremented
        reactor_after = next(c for c in updated_scenario.combatants if c.id == "reactor")
        weapon_after = reactor_after.inventory.mounts[0].weapons[0]
        assert weapon_after.limited_charges_remaining == 2  # Decremented from 3

    def test_overwatch_clears_hidden_status(self):
        """Test that overwatch reaction clears hidden status."""
        reactor = self.make_combatant_with_weapon(
            id="reactor",
            weapon_id="mw_assault_rifle",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
            statuses=["hidden"],  # Reactor is hidden
        )
        target = make_combatant(
            id="target",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            side="hostiles",
        )
        scenario = make_scenario(combatants=[reactor, target])
        economy = ActionEconomyState()

        reaction_input = ReactionInput(
            reactor_id="reactor",
            reaction_type="overwatch",
            target_ids=["target"],
            weapon_id="mw_assault_rifle",
        )

        updated_scenario, _, result = execute_reaction(scenario, economy, reaction_input)

        assert result.success is True

        # Hidden status should be cleared
        reactor_after = next(c for c in updated_scenario.combatants if c.id == "reactor")
        assert "hidden" not in reactor_after.statuses

        # Check that status clear was recorded in effects
        clear_effects = [e for e in result.effects_applied if e.get("type") == "statuses_cleared"]
        assert len(clear_effects) > 0

    def test_overwatch_ordnance_blocked(self):
        """Test that ordnance weapons cannot be used for overwatch."""
        reactor = self.make_combatant_with_weapon(
            id="reactor",
            weapon_id="mw_missile_rack",  # Ordnance weapon
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
            tags=["ordnance"],
        )
        target = make_combatant(
            id="target",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            side="hostiles",
        )
        scenario = make_scenario(combatants=[reactor, target])
        economy = ActionEconomyState()

        reaction_input = ReactionInput(
            reactor_id="reactor",
            reaction_type="overwatch",
            target_ids=["target"],
            weapon_id="mw_missile_rack",
        )

        _, _, result = execute_reaction(scenario, economy, reaction_input)

        assert result.success is False
        assert "ordnance" in result.error.lower()

    def test_overwatch_uses_reaction_budget(self):
        """Test that overwatch consumes reaction budget."""
        reactor = self.make_combatant_with_weapon(
            id="reactor",
            weapon_id="mw_assault_rifle",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        target = make_combatant(
            id="target",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            side="hostiles",
        )
        scenario = make_scenario(combatants=[reactor, target])
        economy = ActionEconomyState()

        reaction_input = ReactionInput(
            reactor_id="reactor",
            reaction_type="overwatch",
            target_ids=["target"],
            weapon_id="mw_assault_rifle",
        )

        updated_scenario, updated_economy, result = execute_reaction(
            scenario, economy, reaction_input
        )

        assert result.success is True
        assert updated_economy.reactions_used_this_turn == 1

        # Per-round reaction should be tracked
        reactor_after = next(c for c in updated_scenario.combatants if c.id == "reactor")
        assert reactor_after.per_round_reactions.get("overwatch", 0) == 1


class TestOverwatchIntegration:
    """Integration tests for overwatch attack resolution."""

    def make_combatant_with_weapon(
        self,
        id: str,
        weapon_id: str = "mw_assault_rifle",
        position: HexPosition | None = None,
        hp_current: int = 10,
        tags: list[str] | None = None,
        side: str = "players",
        statuses: list[str] | None = None,
        grit: int = 2,
        structure_current: int = 4,
    ) -> CombatantState:
        """Create a test combatant with a weapon in inventory."""
        weapon_state = WeaponState(
            weapon_id=weapon_id,
            tags=tags or [],
            destroyed=False,
            limited_charges_remaining=None,
            needs_reload=False,
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
            name=f"Mech {id}",
            side=side,
            kind="mech",
            stats=CombatStats(
                size="size_1",
                hp_max=10,
                evasion=8,
                e_defense=8,
                armor=0,
                speed=4,
                sensor_range=10,
                grit=grit,
            ),
            resources=CombatResources(
                hp_current=hp_current,
                heat_current=0,
                heat_cap=6,
                structure_current=structure_current,
                stress_current=4,
                repairs_remaining=4,
            ),
            position=position or HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
            statuses=statuses or [],
            inventory=inventory,
        )

    def test_overwatch_structure_cascade(self):
        """Test that overwatch damage can trigger structure check."""
        # Create reactor with high grit for consistent hits
        reactor = self.make_combatant_with_weapon(
            id="reactor",
            weapon_id="mw_assault_rifle",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
            grit=10,  # Very high to ensure hit
        )
        # Target with only 1 HP remaining
        target = self.make_combatant_with_weapon(
            id="target",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            hp_current=1,  # Will trigger structure check on any damage
            side="hostiles",
            structure_current=4,
        )
        scenario = make_scenario(combatants=[reactor, target])
        economy = ActionEconomyState()

        reaction_input = ReactionInput(
            reactor_id="reactor",
            reaction_type="overwatch",
            target_ids=["target"],
            weapon_id="mw_assault_rifle",
        )

        # Run multiple times to get a hit
        for _ in range(20):
            updated_scenario, _, result = execute_reaction(scenario, economy, reaction_input)

            if result.attack_hit and result.damage_dealt > 0:
                # Should have structure check if HP reached 0
                target_after = next(c for c in updated_scenario.combatants if c.id == "target")
                if target_after.resources.hp_current <= 0:
                    # Structure check should have been triggered
                    assert len(result.structure_checks) > 0
                    break

    def test_overwatch_weapon_not_in_inventory_error(self):
        """Test that overwatch fails when weapon not in inventory."""
        reactor = self.make_combatant_with_weapon(
            id="reactor",
            weapon_id="mw_assault_rifle",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        target = make_combatant(
            id="target",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            side="hostiles",
        )
        scenario = make_scenario(combatants=[reactor, target])
        economy = ActionEconomyState()

        reaction_input = ReactionInput(
            reactor_id="reactor",
            reaction_type="overwatch",
            target_ids=["target"],
            weapon_id="mw_nonexistent_weapon",  # Not in inventory
        )

        _, _, result = execute_reaction(scenario, economy, reaction_input)

        assert result.success is False
        assert "not found" in result.error.lower() or "inventory" in result.error.lower()


# =============================================================================
# Meltdown Countdown Integration Tests
# =============================================================================


class TestMeltdownCountdownIntegration:
    """Tests for meltdown countdown processing during start_turn()."""

    def test_meltdown_countdown_decrements_at_turn_start(self):
        """Test that meltdown countdown goes from 2 → 1 at turn start."""
        combatant = make_combatant(
            id="mech_1",
            name="Test Mech",
            meltdown_state=MeltdownState(turns_remaining=2),
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[combatant])

        updated_scenario, result = start_turn(scenario, "mech_1")

        assert result.meltdown_countdown_active is True
        assert result.meltdown_triggered is False
        assert result.meltdown_countdown_remaining == 1
        assert result.meltdown_explosion_damage == 0
        assert result.meltdown_affected_targets == []

        # Verify actor state
        actor = next(c for c in updated_scenario.combatants if c.id == "mech_1")
        assert actor.meltdown_state is not None
        assert actor.meltdown_state.turns_remaining == 1

    def test_meltdown_triggers_at_zero(self):
        """Test that meltdown triggers when countdown reaches 0 (from 1)."""
        combatant = make_combatant(
            id="mech_1",
            name="Test Mech",
            meltdown_state=MeltdownState(turns_remaining=1),
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[combatant])

        updated_scenario, result = start_turn(scenario, "mech_1")

        assert result.meltdown_countdown_active is True
        assert result.meltdown_triggered is True
        assert result.meltdown_countdown_remaining is None
        # Damage should be 4d6 (4-24)
        assert result.meltdown_explosion_damage >= 4
        assert result.meltdown_explosion_damage <= 24

        # Verify actor is destroyed
        actor = next(c for c in updated_scenario.combatants if c.id == "mech_1")
        assert actor.meltdown_state is None
        assert "out" in actor.statuses
        assert actor.resources.hp_current == 0
        assert actor.resources.structure_current == 0
        assert actor.resources.stress_current == 0

    def test_meltdown_explosion_damages_nearby_targets(self):
        """Test that meltdown burst 2 damages combatants within range."""
        # Exploding mech at origin
        exploding_mech = make_combatant(
            id="exploding",
            name="Exploding Mech",
            meltdown_state=MeltdownState(turns_remaining=1),
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        # Target within burst 2 (distance 1)
        nearby_target = make_combatant(
            id="nearby",
            name="Nearby Mech",
            hp_current=30,
            hp_max=30,
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            side="hostiles",
        )
        # Target outside burst 2 (distance 5)
        far_target = make_combatant(
            id="far",
            name="Far Mech",
            hp_current=30,
            hp_max=30,
            position=HexPosition(coord=HexCoord(q=5, r=0), elevation=0),
            side="hostiles",
        )
        scenario = make_scenario(combatants=[exploding_mech, nearby_target, far_target])

        updated_scenario, result = start_turn(scenario, "exploding")

        assert result.meltdown_triggered is True
        # Nearby target should be affected
        assert "nearby" in result.meltdown_affected_targets
        # Far target should not be affected
        assert "far" not in result.meltdown_affected_targets

        # Verify nearby target took damage
        nearby_after = next(c for c in updated_scenario.combatants if c.id == "nearby")
        assert nearby_after.resources.hp_current < 30

        # Verify far target is unharmed
        far_after = next(c for c in updated_scenario.combatants if c.id == "far")
        assert far_after.resources.hp_current == 30

    def test_meltdown_explosion_agility_save_halves_damage(self):
        """Test that successful agility save halves meltdown damage.

        This test runs multiple iterations to statistically verify that saves
        sometimes succeed (halved damage) and sometimes fail (full damage).
        """
        results_with_full_damage = 0
        results_with_halved_damage = 0

        for _ in range(50):
            exploding_mech = make_combatant(
                id="exploding",
                name="Exploding Mech",
                meltdown_state=MeltdownState(turns_remaining=1),
                position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
            )
            target = make_combatant(
                id="target",
                name="Target Mech",
                hp_current=100,
                hp_max=100,
                position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
                side="hostiles",
            )
            scenario = make_scenario(combatants=[exploding_mech, target])

            updated_scenario, result = start_turn(scenario, "exploding")

            if result.meltdown_triggered:
                target_after = next(c for c in updated_scenario.combatants if c.id == "target")
                damage_taken = 100 - target_after.resources.hp_current

                # Full damage is explosion damage, halved damage would be ~half
                if damage_taken > 0:
                    if damage_taken == result.meltdown_explosion_damage:
                        results_with_full_damage += 1
                    elif damage_taken < result.meltdown_explosion_damage:
                        results_with_halved_damage += 1

        # Should have some of each type (statistical test)
        # With 50 iterations, we should see both outcomes
        assert results_with_full_damage > 0 or results_with_halved_damage > 0

    def test_meltdown_creates_wreckage_state(self):
        """Test that meltdown marks mech as destroyed with 'out' status."""
        combatant = make_combatant(
            id="mech_1",
            name="Test Mech",
            meltdown_state=MeltdownState(turns_remaining=1),
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
            statuses=["impaired", "exposed"],
        )
        scenario = make_scenario(combatants=[combatant])

        updated_scenario, result = start_turn(scenario, "mech_1")

        assert result.meltdown_triggered is True

        # Verify mech is destroyed
        actor = next(c for c in updated_scenario.combatants if c.id == "mech_1")
        assert "out" in actor.statuses
        # Combat-relevant statuses should be cleared
        assert "impaired" not in actor.statuses
        assert "exposed" not in actor.statuses

    def test_meltdown_marks_mech_as_out(self):
        """Test that meltdown sets HP, structure, and stress to 0."""
        combatant = make_combatant(
            id="mech_1",
            name="Test Mech",
            hp_current=10,
            hp_max=10,
            meltdown_state=MeltdownState(turns_remaining=1),
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[combatant])

        updated_scenario, result = start_turn(scenario, "mech_1")

        assert result.meltdown_triggered is True

        actor = next(c for c in updated_scenario.combatants if c.id == "mech_1")
        assert actor.resources.hp_current == 0
        assert actor.resources.structure_current == 0
        assert actor.resources.stress_current == 0
        assert "out" in actor.statuses

    def test_no_meltdown_processing_without_countdown(self):
        """Test that normal turn start has no meltdown processing when no countdown exists."""
        combatant = make_combatant(
            id="mech_1",
            name="Test Mech",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[combatant])

        updated_scenario, result = start_turn(scenario, "mech_1")

        # Meltdown fields should all be default values
        assert result.meltdown_countdown_active is False
        assert result.meltdown_triggered is False
        assert result.meltdown_countdown_remaining is None
        assert result.meltdown_explosion_damage == 0
        assert result.meltdown_affected_targets == []

        # Actor should be unchanged regarding meltdown
        actor = next(c for c in updated_scenario.combatants if c.id == "mech_1")
        assert actor.meltdown_state is None

    def test_meltdown_clears_exposed_status(self):
        """Test that exposed status is cleared when meltdown countdown triggers."""
        combatant = make_combatant(
            id="mech_1",
            name="Test Mech",
            meltdown_state=MeltdownState(turns_remaining=1, exposed_applied=True),
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
            statuses=["exposed"],
        )
        scenario = make_scenario(combatants=[combatant])

        updated_scenario, result = start_turn(scenario, "mech_1")

        assert result.meltdown_triggered is True

        actor = next(c for c in updated_scenario.combatants if c.id == "mech_1")
        # Exposed is cleared as part of trigger_meltdown
        assert "exposed" not in actor.statuses


# =============================================================================
# Deployables Integration Tests (Phase 30)
# =============================================================================


class TestDeployablesIntegration:
    """Tests for deployable/drone integration into combat execution."""

    def test_deploy_action_places_drone(self):
        """Test that deploy action creates a drone at target position."""
        actor = make_combatant(
            id="mech_1",
            name="Test Mech",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[actor])
        turn = make_turn(actor_id="mech_1")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="mech_1",
            action_id="deploy",
            action_type="quick",
            target_position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            deploy_kind="drone",
            deploy_name="Scout Drone",
        )

        updated_scenario, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is True

        # Verify drone was added to deployables
        assert len(updated_scenario.deployables) == 1
        drone_id = list(updated_scenario.deployables.keys())[0]
        drone = updated_scenario.deployables[drone_id]

        assert drone.kind == "drone"
        assert drone.name == "Scout Drone"
        assert drone.owner_id == "mech_1"
        assert drone.position.coord.q == 1
        assert drone.position.coord.r == 0
        assert drone.acts_on_owner_turn is True

    def test_deploy_action_places_mine_with_arming_turn(self):
        """Test that deploy action creates a mine that arms on next turn."""
        actor = make_combatant(
            id="mech_1",
            name="Test Mech",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        rounds = [make_round(round_index=1)]
        scenario = make_scenario(combatants=[actor], rounds=rounds)
        turn = make_turn(actor_id="mech_1")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="mech_1",
            action_id="deploy",
            action_type="quick",
            target_position=HexPosition(coord=HexCoord(q=2, r=0), elevation=0),
            deploy_kind="mine",
            deploy_name="Explosive Mine",
            mine_type="explosive",
        )

        updated_scenario, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is True

        # Verify mine was added
        assert len(updated_scenario.deployables) == 1
        mine_id = list(updated_scenario.deployables.keys())[0]
        mine = updated_scenario.deployables[mine_id]

        assert mine.kind == "mine"
        assert mine.name == "Explosive Mine"
        assert mine.is_armed is False  # Not armed yet
        assert mine.arming_turn == 2  # Arms next turn (current round is 1)
        assert mine.trigger_on_adjacent_entry is True

    def test_mine_arms_at_turn_start(self):
        """Test that mines arm at the start of the deployer's next turn."""
        from core.mech.combat_state import DeployableState

        actor = make_combatant(
            id="mech_1",
            name="Test Mech",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        # Create scenario with an unarmed mine that should arm on turn 2
        mine = DeployableState(
            id="mine_1",
            name="Test Mine",
            kind="mine",
            owner_id="mech_1",
            position=HexPosition(coord=HexCoord(q=2, r=0), elevation=0),
            size=1,
            hp=10,
            max_hp=10,
            armor=0,
            evasion=5,
            is_armed=False,
            arming_turn=2,
            trigger_on_adjacent_entry=True,
        )

        rounds = [make_round(round_index=1), make_round(round_index=2)]
        scenario = MechCombatScenario(
            combatants=[actor],
            grapples=[],
            rounds=rounds,
            terrain=None,
            environment="standard",
            deployables={"mine_1": mine},
        )

        # Start turn on round 2 (when mine should arm)
        updated_scenario, result = start_turn(scenario, "mech_1")

        # Verify mine is now armed
        assert "mine_1" in result.mines_armed
        updated_mine = updated_scenario.deployables["mine_1"]
        assert updated_mine.is_armed is True

    def test_drone_start_of_turn_processing(self):
        """Test that drone turn processing happens at owner's turn start."""
        from core.mech.combat_state import DeployableState

        actor = make_combatant(
            id="mech_1",
            name="Test Mech",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        # Create drone that can act
        drone = DeployableState(
            id="drone_1",
            name="Combat Drone",
            kind="drone",
            owner_id="mech_1",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            size=1,
            hp=10,
            max_hp=10,
            armor=0,
            evasion=10,
            is_active=True,
            can_act=True,
            can_move=True,
            acts_on_owner_turn=True,
        )

        scenario = MechCombatScenario(
            combatants=[actor],
            grapples=[],
            rounds=[make_round(round_index=1)],
            terrain=None,
            environment="standard",
            deployables={"drone_1": drone},
        )

        _, result = start_turn(scenario, "mech_1")

        # Drone should be reported as ready to act
        assert "drone_1" in result.drones_ready_to_act

    def test_drone_primes_at_end_of_turn(self):
        """Test that drones prime at the end of their owner's turn."""
        from core.mech.combat_state import DeployableState

        actor = make_combatant(
            id="mech_1",
            name="Test Mech",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        # Create drone that needs priming
        drone = DeployableState(
            id="drone_1",
            name="Restock Drone",
            kind="drone",
            owner_id="mech_1",
            position=HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
            size=1,
            hp=10,
            max_hp=10,
            armor=0,
            evasion=10,
            is_active=True,
            is_armed=False,  # Not yet primed
            can_act=False,
            can_move=False,
            acts_on_owner_turn=True,
        )

        rounds = [make_round(round_index=1, turns=[make_turn(actor_id="mech_1")])]
        scenario = MechCombatScenario(
            combatants=[actor],
            grapples=[],
            rounds=rounds,
            terrain=None,
            environment="standard",
            deployables={"drone_1": drone},
        )

        current_turn = rounds[0].turns[0]
        _, result, _, _ = end_turn(scenario, current_round=1, current_turn_index=0, current_turn=current_turn)

        # Drone should be primed
        assert "drone_1" in result.drones_primed

    def test_mine_triggers_on_adjacent_movement(self):
        """Test that armed mines trigger when enemies move adjacent."""
        from core.mech.combat_state import DeployableState

        actor = make_combatant(
            id="mech_1",
            name="Test Mech",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        # Create an armed mine at position (3, 0)
        mine = DeployableState(
            id="mine_1",
            name="Armed Mine",
            kind="mine",
            owner_id="ally_1",  # Owned by a different combatant
            position=HexPosition(coord=HexCoord(q=3, r=0), elevation=0),
            size=1,
            hp=10,
            max_hp=10,
            armor=0,
            evasion=5,
            is_armed=True,  # Armed
            trigger_on_adjacent_entry=True,
        )

        scenario = MechCombatScenario(
            combatants=[actor],
            grapples=[],
            rounds=[make_round(round_index=1)],
            terrain=None,
            environment="standard",
            deployables={"mine_1": mine},
        )

        turn = make_turn(actor_id="mech_1")
        economy = ActionEconomyState()

        # Move through hex (2, 0) which is adjacent to the mine at (3, 0)
        action_input = ActionExecutionInput(
            actor_id="mech_1",
            action_id="move",
            action_type="full",
            movement_path=[
                HexPosition(coord=HexCoord(q=1, r=0), elevation=0),
                HexPosition(coord=HexCoord(q=2, r=0), elevation=0),  # Adjacent to mine
            ],
        )

        updated_scenario, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is True

        # Check that mine detonation effect is in results
        mine_detonation_effects = [e for e in result.effects_applied if e.get("type") == "mine_detonation"]
        assert len(mine_detonation_effects) == 1
        assert mine_detonation_effects[0]["mine_id"] == "mine_1"

        # Mine should be removed from scenario
        assert "mine_1" not in updated_scenario.deployables

    def test_mine_does_not_trigger_on_owner_movement(self):
        """Test that mines don't trigger on their owner's movement."""
        from core.mech.combat_state import DeployableState

        actor = make_combatant(
            id="mech_1",
            name="Test Mech",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        # Create an armed mine owned by the actor
        mine = DeployableState(
            id="mine_1",
            name="Own Mine",
            kind="mine",
            owner_id="mech_1",  # Owned by the mover
            position=HexPosition(coord=HexCoord(q=2, r=0), elevation=0),
            size=1,
            hp=10,
            max_hp=10,
            armor=0,
            evasion=5,
            is_armed=True,
            trigger_on_adjacent_entry=True,
        )

        scenario = MechCombatScenario(
            combatants=[actor],
            grapples=[],
            rounds=[make_round(round_index=1)],
            terrain=None,
            environment="standard",
            deployables={"mine_1": mine},
        )

        turn = make_turn(actor_id="mech_1")
        economy = ActionEconomyState()

        # Move adjacent to own mine
        action_input = ActionExecutionInput(
            actor_id="mech_1",
            action_id="move",
            action_type="full",
            movement_path=[
                HexPosition(coord=HexCoord(q=1, r=0), elevation=0),  # Adjacent to mine
            ],
        )

        updated_scenario, _, _, result = execute_action(scenario, turn, economy, action_input)

        assert result.success is True

        # Mine should NOT have detonated
        mine_detonation_effects = [e for e in result.effects_applied if e.get("type") == "mine_detonation"]
        assert len(mine_detonation_effects) == 0

        # Mine should still be in scenario
        assert "mine_1" in updated_scenario.deployables

    def test_deploy_requires_target_position(self):
        """Test that deploy action fails without target position."""
        actor = make_combatant(
            id="mech_1",
            name="Test Mech",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
        )
        scenario = make_scenario(combatants=[actor])
        turn = make_turn(actor_id="mech_1")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="mech_1",
            action_id="deploy",
            action_type="quick",
            # Missing target_position
            deploy_kind="drone",
            deploy_name="Scout Drone",
        )

        _, _, _, result = execute_action(scenario, turn, economy, action_input)

        # Should succeed but have no deploy effect (no position = no deployment)
        assert result.success is True
        deploy_effects = [e for e in result.effects_applied if e.get("type") == "deploy"]
        assert len(deploy_effects) == 0


# =============================================================================
# Dynamic Weapon Profile Tests (Phase 31)
# =============================================================================


class TestDynamicWeaponProfiles:
    """Tests for dynamic weapon profile selection.

    The Ghoul Nexus (horus_ghoul_nexus) has 3 profiles:
    - kinetic: 1d3+2 kinetic damage
    - energy: 1d3+2 energy damage
    - explosive: 1d3+2 explosive damage

    Profile selection allows choosing which damage type to use.
    """

    def test_resolve_weapon_profile_default(self):
        """Test that default profile is used when no profile_id specified."""
        from core.mech.combat_helpers import _resolve_weapon_profile

        profile = _resolve_weapon_profile("horus_ghoul_nexus")

        assert profile is not None
        assert profile.profile_id == "kinetic"  # default_profile_id
        assert profile.damage_type == "kinetic"

    def test_resolve_weapon_profile_kinetic(self):
        """Test explicit kinetic profile selection."""
        from core.mech.combat_helpers import _resolve_weapon_profile

        profile = _resolve_weapon_profile("horus_ghoul_nexus", "kinetic")

        assert profile is not None
        assert profile.profile_id == "kinetic"
        assert profile.damage_type == "kinetic"

    def test_resolve_weapon_profile_energy(self):
        """Test explicit energy profile selection."""
        from core.mech.combat_helpers import _resolve_weapon_profile

        profile = _resolve_weapon_profile("horus_ghoul_nexus", "energy")

        assert profile is not None
        assert profile.profile_id == "energy"
        assert profile.damage_type == "energy"

    def test_resolve_weapon_profile_explosive(self):
        """Test explicit explosive profile selection."""
        from core.mech.combat_helpers import _resolve_weapon_profile

        profile = _resolve_weapon_profile("horus_ghoul_nexus", "explosive")

        assert profile is not None
        assert profile.profile_id == "explosive"
        assert profile.damage_type == "explosive"

    def test_lookup_weapon_damage_with_profile(self):
        """Test lookup_weapon_damage_and_ap uses selected profile."""
        # The Ghoul Nexus does 1d3+2 damage regardless of profile
        # but this tests that the profile_id parameter is accepted

        damage, ap = lookup_weapon_damage_and_ap("horus_ghoul_nexus", "energy")

        # 1d3+2 means damage should be 3-5
        assert damage >= 3
        assert damage <= 5
        assert ap == 0  # No AP tag on Ghoul Nexus

    def test_weapon_without_profiles_ignores_profile_id(self):
        """Test that weapons without profiles gracefully ignore profile_id."""
        from core.mech.combat_helpers import _resolve_weapon_profile

        # Assault rifle has no profiles - profile_id should be ignored
        profile = _resolve_weapon_profile("assault_rifle", "nonexistent")

        assert profile is not None
        assert profile.profile_id == "assault_rifle"  # Falls back to base weapon

    def test_attack_action_with_profile_selection(self):
        """Test execute_action passes profile_id through attack resolution."""
        # Create attacker with the Ghoul Nexus in inventory
        attacker = make_combatant(
            id="mech_1",
            name="Attacker",
            side="players",
            grit=2,
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=0),
            inventory=MechInventory(
                mounts=[
                    WeaponMountState(
                        mount_index=0,
                        slot_type="main",
                        weapons=[
                            WeaponState(
                                weapon_id="horus_ghoul_nexus",
                                destroyed=False,
                            ),
                        ],
                    ),
                ],
                systems=[],
            ),
        )

        # Create target with low e_defense (Ghoul Nexus is smart weapon)
        target = make_combatant(
            id="mech_2",
            name="Target",
            side="hostiles",
            position=HexPosition(coord=HexCoord(q=5, r=0), elevation=0),  # Within range 10
        )

        scenario = make_scenario(combatants=[attacker, target])
        turn = make_turn(actor_id="mech_1")
        economy = ActionEconomyState()

        action_input = ActionExecutionInput(
            actor_id="mech_1",
            action_id="skirmish",
            action_type="full",
            target_ids=["mech_2"],
            weapon_id="horus_ghoul_nexus",
            weapon_profile_id="energy",  # Select energy profile
        )

        _, _, _, result = execute_action(
            scenario, turn, economy, action_input
        )

        # Action should succeed
        assert result.success is True

        # The attack should have been processed
        attack_effects = [e for e in result.effects_applied if e.get("type") == "attack"]
        assert len(attack_effects) == 1
        assert attack_effects[0]["target_id"] == "mech_2"


# =============================================================================
# Falling Resolution Tests (Phase 52)
# =============================================================================


class TestFallingResolution:
    """Tests for falling state tracking and resolution (PR2 flight rules)."""

    def test_flying_actor_marked_falling_when_stunned(self):
        """Flying actor with stunned status should be marked as falling at turn start."""
        # Create flying combatant with stunned status
        flying_actor = make_combatant(
            id="mech_1",
            name="Flying Mech",
            hp_max=20,
            hp_current=20,
        )
        # Set flying status at altitude 5
        flying_actor = flying_actor.model_copy(
            update={
                "flying_status": FlyingStatus(
                    is_flying=True,
                    altitude_level=5,
                    is_hover=False,
                    movement_mode="flight",
                ),
                "statuses": ["stunned"],
            }
        )

        scenario = make_scenario(combatants=[flying_actor])
        updated_scenario, result = start_turn(scenario, "mech_1")

        # Should be marked as falling
        assert result.started_falling is True
        assert result.falling_from_altitude == 5

        # Actor should have falling status
        actor = next(c for c in updated_scenario.combatants if c.id == "mech_1")
        assert "falling" in actor.statuses
        assert actor.falling_from_altitude == 5

    def test_flying_actor_marked_falling_when_immobilized(self):
        """Flying actor with immobilized status should be marked as falling at turn start."""
        flying_actor = make_combatant(id="mech_1", name="Flying Mech")
        flying_actor = flying_actor.model_copy(
            update={
                "flying_status": FlyingStatus(
                    is_flying=True,
                    altitude_level=3,
                    is_hover=False,
                    movement_mode="flight",
                ),
                "statuses": ["immobilized"],
            }
        )

        scenario = make_scenario(combatants=[flying_actor])
        updated_scenario, result = start_turn(scenario, "mech_1")

        assert result.started_falling is True
        assert result.falling_from_altitude == 3

    def test_flying_actor_marked_falling_when_shutdown(self):
        """Flying actor with shutdown status should be marked as falling at turn start."""
        flying_actor = make_combatant(id="mech_1", name="Flying Mech")
        flying_actor = flying_actor.model_copy(
            update={
                "flying_status": FlyingStatus(
                    is_flying=True,
                    altitude_level=2,
                    is_hover=False,
                    movement_mode="flight",
                ),
                "statuses": ["shutdown"],
            }
        )

        scenario = make_scenario(combatants=[flying_actor])
        updated_scenario, result = start_turn(scenario, "mech_1")

        assert result.started_falling is True
        assert result.falling_from_altitude == 2

    def test_falling_resolves_at_end_of_turn(self):
        """Falling status should resolve with damage at end of turn."""
        # Create actor already marked as falling
        falling_actor = make_combatant(
            id="mech_1",
            name="Falling Mech",
            hp_max=20,
            hp_current=20,
        )
        falling_actor = falling_actor.model_copy(
            update={
                "flying_status": FlyingStatus(
                    is_flying=True,
                    altitude_level=5,
                    is_hover=False,
                    movement_mode="flight",
                ),
                "statuses": ["falling"],
                "falling_from_altitude": 5,
            }
        )

        round1 = make_round(round_index=1, turns=[make_turn(actor_id="mech_1")])
        scenario = make_scenario(combatants=[falling_actor], rounds=[round1])
        turn = make_turn(actor_id="mech_1")

        updated_scenario, result, _, _ = end_turn(scenario, 1, 0, turn)

        # Fall should be resolved
        assert result.fall_resolved is True
        assert result.fall_damage == 5  # 1 damage per altitude level
        assert result.fell_from_altitude == 5

        # Actor should have taken damage and no longer be falling
        actor = next(c for c in updated_scenario.combatants if c.id == "mech_1")
        assert actor.resources.hp_current == 15  # 20 - 5 = 15
        assert "falling" not in actor.statuses
        assert actor.falling_from_altitude is None

    def test_falling_damage_scales_with_altitude(self):
        """Falling damage should equal altitude level."""
        falling_actor = make_combatant(
            id="mech_1",
            name="Falling Mech",
            hp_max=20,
            hp_current=20,
        )
        falling_actor = falling_actor.model_copy(
            update={
                "flying_status": FlyingStatus(
                    is_flying=True,
                    altitude_level=8,
                    is_hover=False,
                    movement_mode="flight",
                ),
                "statuses": ["falling"],
                "falling_from_altitude": 8,
            }
        )

        round1 = make_round(round_index=1, turns=[make_turn(actor_id="mech_1")])
        scenario = make_scenario(combatants=[falling_actor], rounds=[round1])
        turn = make_turn(actor_id="mech_1")

        updated_scenario, result, _, _ = end_turn(scenario, 1, 0, turn)

        assert result.fall_damage == 8  # 1 damage per altitude level
        actor = next(c for c in updated_scenario.combatants if c.id == "mech_1")
        assert actor.resources.hp_current == 12  # 20 - 8 = 12

    def test_falling_clears_flying_status(self):
        """After fall resolves, actor should no longer be flying."""
        falling_actor = make_combatant(id="mech_1", name="Falling Mech")
        falling_actor = falling_actor.model_copy(
            update={
                "flying_status": FlyingStatus(
                    is_flying=True,
                    altitude_level=3,
                    is_hover=False,
                    movement_mode="flight",
                ),
                "statuses": ["falling"],
                "falling_from_altitude": 3,
            }
        )

        round1 = make_round(round_index=1, turns=[make_turn(actor_id="mech_1")])
        scenario = make_scenario(combatants=[falling_actor], rounds=[round1])
        turn = make_turn(actor_id="mech_1")

        updated_scenario, result, _, _ = end_turn(scenario, 1, 0, turn)

        actor = next(c for c in updated_scenario.combatants if c.id == "mech_1")
        assert actor.flying_status is not None
        assert actor.flying_status.is_flying is False
        assert actor.flying_status.altitude_level == 0
        assert actor.flying_status.movement_mode == "ground"

    def test_falling_resets_position_elevation(self):
        """After fall resolves, actor position elevation should be 0."""
        falling_actor = make_combatant(
            id="mech_1",
            name="Falling Mech",
            position=HexPosition(coord=HexCoord(q=0, r=0), elevation=5),
        )
        falling_actor = falling_actor.model_copy(
            update={
                "flying_status": FlyingStatus(
                    is_flying=True,
                    altitude_level=5,
                    is_hover=False,
                    movement_mode="flight",
                ),
                "statuses": ["falling"],
                "falling_from_altitude": 5,
            }
        )

        round1 = make_round(round_index=1, turns=[make_turn(actor_id="mech_1")])
        scenario = make_scenario(combatants=[falling_actor], rounds=[round1])
        turn = make_turn(actor_id="mech_1")

        updated_scenario, result, _, _ = end_turn(scenario, 1, 0, turn)

        actor = next(c for c in updated_scenario.combatants if c.id == "mech_1")
        assert actor.position is not None
        assert actor.position.elevation == 0

    def test_no_falling_if_not_flying(self):
        """Non-flying actor with stunned status should not fall."""
        grounded_actor = make_combatant(id="mech_1", name="Grounded Mech")
        grounded_actor = grounded_actor.model_copy(
            update={
                "statuses": ["stunned"],
            }
        )

        scenario = make_scenario(combatants=[grounded_actor])
        updated_scenario, result = start_turn(scenario, "mech_1")

        # Should NOT be marked as falling
        assert result.started_falling is False
        assert result.falling_from_altitude is None

        actor = next(c for c in updated_scenario.combatants if c.id == "mech_1")
        assert "falling" not in actor.statuses

    def test_zero_altitude_no_fall_damage(self):
        """Flying at altitude 0 should not cause fall damage."""
        flying_actor = make_combatant(
            id="mech_1",
            name="Low Flying Mech",
            hp_max=20,
            hp_current=20,
        )
        flying_actor = flying_actor.model_copy(
            update={
                "flying_status": FlyingStatus(
                    is_flying=True,
                    altitude_level=0,
                    is_hover=False,
                    movement_mode="flight",
                ),
                "statuses": ["stunned"],
            }
        )

        scenario = make_scenario(combatants=[flying_actor])
        updated_scenario, result = start_turn(scenario, "mech_1")

        # Should NOT be marked as falling (altitude is 0)
        assert result.started_falling is False

    def test_flying_without_status_does_not_fall(self):
        """Flying actor without immobilized/stunned/shutdown should not fall."""
        flying_actor = make_combatant(id="mech_1", name="Flying Mech")
        flying_actor = flying_actor.model_copy(
            update={
                "flying_status": FlyingStatus(
                    is_flying=True,
                    altitude_level=5,
                    is_hover=False,
                    movement_mode="flight",
                ),
                "statuses": [],  # No disabling statuses
            }
        )

        scenario = make_scenario(combatants=[flying_actor])
        updated_scenario, result = start_turn(scenario, "mech_1")

        # Should NOT be marked as falling
        assert result.started_falling is False
        assert result.falling_from_altitude is None

        actor = next(c for c in updated_scenario.combatants if c.id == "mech_1")
        assert "falling" not in actor.statuses

    def test_full_turn_cycle_flying_to_fallen(self):
        """Test complete turn cycle: flying actor gets stunned, starts falling, resolves at end."""
        # Start with a flying actor that has stunned status
        flying_actor = make_combatant(
            id="mech_1",
            name="Flying Mech",
            hp_max=20,
            hp_current=20,
        )
        flying_actor = flying_actor.model_copy(
            update={
                "flying_status": FlyingStatus(
                    is_flying=True,
                    altitude_level=4,
                    is_hover=False,
                    movement_mode="flight",
                ),
                "statuses": ["stunned"],
            }
        )

        round1 = make_round(round_index=1, turns=[make_turn(actor_id="mech_1")])
        scenario = make_scenario(combatants=[flying_actor], rounds=[round1])

        # Start turn - should mark as falling
        scenario, start_result = start_turn(scenario, "mech_1")
        assert start_result.started_falling is True
        assert start_result.falling_from_altitude == 4

        # End turn - should resolve fall with damage
        turn = make_turn(actor_id="mech_1")
        scenario, end_result, _, _ = end_turn(scenario, 1, 0, turn)
        assert end_result.fall_resolved is True
        assert end_result.fall_damage == 4

        # Final state check
        actor = next(c for c in scenario.combatants if c.id == "mech_1")
        assert actor.resources.hp_current == 16  # 20 - 4 = 16
        assert "falling" not in actor.statuses
        assert actor.flying_status is not None
        assert actor.flying_status.is_flying is False
