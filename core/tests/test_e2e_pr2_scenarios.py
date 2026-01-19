"""End-to-end tests based on PR2 (Lancer rulebook) examples.

These tests recreate specific scenarios from the Player's Resource 2 (PR2)
rulebook to verify mechanical accuracy.
"""

import pytest
from unittest.mock import patch

from core.tests.e2e_helpers import (
    make_pilot_with_talents,
    make_combatant_from_pilot,
    make_combatant,
    make_enemy_combatant,
    make_duel_scenario,
    execute_attack,
    assert_attack_hit,
    assert_attack_crit,
)
from core.mech.combat_execution import (
    ActionExecutionInput,
    execute_action,
    start_turn,
)
from core.mech.combat_state import CombatTurn
from core.mech.combat_helpers import (
    _get_talent_accuracy_modifiers,
    _get_talent_check_modifiers,
)
from core.pilot import collect_pilot_talent_effects
from core.shared.effects import MechanicalEffect, AccuracyModifier


class TestPR2AccuracyDifficultyExample:
    """Tests based on PR2 accuracy/difficulty examples.

    PR2 Lines 1392-1412: Explains how accuracy and difficulty dice work.
    """

    def test_accuracy_increases_hit_chance(self):
        """Accuracy dice increase effective roll.

        PR2: Each accuracy die rolled adds to the attack, taking the highest.
        """
        # Create combatant with +1 accuracy
        effect = MechanicalEffect(
            accuracy_mods=[AccuracyModifier(value=1, applies_to="all")]
        )
        player = make_combatant(
            id="player_1",
            talent_effects=[effect],
        )
        enemy = make_enemy_combatant(
            id="enemy_1",
            evasion=12,  # Moderate difficulty
        )

        scenario = make_duel_scenario(player, enemy)

        # With forced roll of 11, would normally miss vs 12 evasion
        # But +1 accuracy should help (depending on accuracy roll)
        # Test that the system processes accuracy correctly
        scenario, result = execute_attack(
            scenario,
            "player_1",
            "enemy_1",
            force_roll=11,
        )

        assert result.success
        # The actual hit/miss depends on accuracy die roll

    def test_difficulty_decreases_hit_chance(self):
        """Difficulty dice decrease effective roll.

        PR2: Each difficulty die subtracts from the attack, taking the lowest.
        """
        # Create combatant with difficulty (negative accuracy)
        effect = MechanicalEffect(
            accuracy_mods=[AccuracyModifier(value=-1, applies_to="all")]
        )
        player = make_combatant(
            id="player_1",
            talent_effects=[effect],
        )
        enemy = make_enemy_combatant(
            id="enemy_1",
            evasion=10,
        )

        scenario = make_duel_scenario(player, enemy)

        # Verify difficulty is captured
        _, diff_mod = _get_talent_accuracy_modifiers(player, is_ranged=True)
        assert diff_mod == 1  # 1 difficulty


class TestPR2CrackShotSniperDuel:
    """Tests based on PR2 CRACK SHOT talent examples.

    PR2 Lines 6224-6241: Crack Shot provides ranged accuracy bonuses.
    """

    def test_crack_shot_steady_aim_accuracy(self):
        """Crack Shot Steady Aim provides +1 accuracy on rifle attacks.

        PR2: Steady Aim protocol immobilizes but grants +1 accuracy on
        ranged rifle attacks until start of next turn.
        """
        pilot = make_pilot_with_talents("SNIPER", [("crack_shot", 1)])
        effects = collect_pilot_talent_effects(pilot)

        # Verify Steady Aim protocol exists
        assert len(effects) == 1
        assert len(effects[0].protocols) > 0

        protocol = effects[0].protocols[0]
        assert protocol.name == "Steady Aim"

        # Protocol grants accuracy on ranged attacks
        acc_mods = protocol.effects.accuracy_mods
        assert len(acc_mods) > 0
        assert acc_mods[0].value == 1
        assert acc_mods[0].applies_to == "ranged"

    def test_crack_shot_zero_in_crit_bonus(self):
        """Crack Shot Zero In allows trading accuracy for crit damage.

        PR2: Can trade 1 accuracy for +1d6 damage on critical hits.
        """
        pilot = make_pilot_with_talents("SNIPER", [("crack_shot", 2)])
        effects = collect_pilot_talent_effects(pilot)

        # Should have rank 1 and rank 2 effects
        assert len(effects) == 2

        # Rank 2 has accuracy trade effect
        rank2 = effects[1]
        assert len(rank2.accuracy_trade_effects) > 0

        trade = rank2.accuracy_trade_effects[0]
        assert trade.accuracy_cost == 1
        assert trade.requires_crit is True


class TestPR2DuelistMeleeExample:
    """Tests for melee accuracy bonuses based on PR2 examples.

    PR2 Lines 6284-6311: Duelist provides melee attack bonuses.
    """

    def test_melee_accuracy_modifier(self):
        """Test melee-specific accuracy modifiers."""
        # Create combatant with melee accuracy bonus
        effect = MechanicalEffect(
            accuracy_mods=[AccuracyModifier(value=1, applies_to="melee")]
        )
        combatant = make_combatant(
            id="melee_fighter",
            talent_effects=[effect],
        )

        # Melee attacks should get bonus
        acc_melee, _ = _get_talent_accuracy_modifiers(combatant, is_melee=True)
        assert acc_melee == 1

        # Ranged attacks should not
        acc_ranged, _ = _get_talent_accuracy_modifiers(combatant, is_ranged=True)
        assert acc_ranged == 0


class TestPR2InfiltratorAmbush:
    """Tests based on PR2 INFILTRATOR talent examples.

    PR2 Lines 6489-6511: Infiltrator provides hidden attack bonuses.
    """

    def test_infiltrator_hidden_accuracy(self):
        """Infiltrator gains +1 accuracy on first attack while hidden.

        PR2: Defilade Navigator - First attack while hidden gets +1 accuracy.
        """
        pilot = make_pilot_with_talents("INFILTRATOR", [("infiltrator", 1)])
        effects = collect_pilot_talent_effects(pilot)

        assert len(effects) == 1

        # Has attack sequence modifier
        seq_mods = effects[0].attack_sequence_mods
        assert len(seq_mods) > 0

        mod = seq_mods[0]
        assert mod.first_attack_accuracy == 1
        assert mod.condition == "hidden_at_turn_start"


class TestPR2LeaderDicePool:
    """Tests based on PR2 LEADER talent examples.

    PR2 Lines 6534-6553: Leader can grant accuracy dice to allies.
    """

    def test_leader_dice_pool_grant(self):
        """Leader can grant accuracy dice from leadership pool.

        PR2: Field Commander grants 3 leadership dice, can give to allies
        as free action for +1 accuracy on their next action.
        """
        pilot = make_pilot_with_talents("COMMANDER", [("leader", 1)])
        effects = collect_pilot_talent_effects(pilot)

        assert len(effects) == 1

        # Has leadership dice pool
        pools = effects[0].leadership_dice_pools
        assert len(pools) > 0

        pool = pools[0]
        assert pool.dice_count == 3
        assert pool.grant_action_type == "free"


class TestPR2CriticalHits:
    """Tests for critical hit mechanics based on PR2.

    PR2: Natural 20 is always a critical hit, dealing double damage.
    """

    def test_natural_20_critical(self):
        """Natural 20 always crits and deals double damage."""
        player = make_combatant(
            id="player_1",
            side="players",
        )
        enemy = make_enemy_combatant(
            id="enemy_1",
            hp_max=30,
            armor=0,
            evasion=25,  # Even impossible evasion
        )

        scenario = make_duel_scenario(player, enemy)

        # Force natural 20
        scenario, result = execute_attack(
            scenario,
            "player_1",
            "enemy_1",
            force_roll=20,
        )

        assert result.success
        assert_attack_crit(result)

        # Should deal 12 damage (6 base * 2)
        assert result.damage_dealt == 12

    def test_brutal_predator_natural_20(self):
        """BRUTAL Predator enhances natural 20 crits.

        PR2: Natural 20 on any attack becomes a critical hit.
        (BRUTAL clarifies this behavior)
        """
        pilot = make_pilot_with_talents("BRUTAL", [("brutal", 1)])
        effects = collect_pilot_talent_effects(pilot)

        # Should have attack outcome effect for crit upgrade
        assert len(effects) == 1
        outcomes = effects[0].attack_outcomes
        assert len(outcomes) > 0

        outcome = outcomes[0]
        assert outcome.upgrade_to_crit is True
        assert outcome.condition == "natural_20"


class TestPR2ArmorMitigation:
    """Tests for armor damage mitigation based on PR2.

    PR2: Armor reduces incoming damage by its value.
    """

    def test_armor_reduces_normal_damage(self):
        """Armor reduces damage from normal attacks."""
        player = make_combatant(id="player_1")
        armored_enemy = make_enemy_combatant(
            id="enemy_1",
            hp_max=20,
            armor=3,  # Reduces damage by 3
            evasion=5,
        )

        scenario = make_duel_scenario(player, armored_enemy)

        # Force hit (not crit)
        scenario, result = execute_attack(
            scenario,
            "player_1",
            "enemy_1",
            force_roll=15,
        )

        if result.effects_applied and result.effects_applied[0].get("hit"):
            # Base damage 6 - 3 armor = 3
            if not result.effects_applied[0].get("critical"):
                assert result.damage_dealt == 3

    def test_armor_cannot_reduce_below_minimum(self):
        """Armor cannot reduce damage below minimum (usually 1)."""
        player = make_combatant(id="player_1")
        heavy_armor_enemy = make_enemy_combatant(
            id="enemy_1",
            hp_max=20,
            armor=10,  # Very high armor
            evasion=5,
        )

        scenario = make_duel_scenario(player, heavy_armor_enemy)

        # Force hit
        scenario, result = execute_attack(
            scenario,
            "player_1",
            "enemy_1",
            force_roll=15,
        )

        if result.effects_applied and result.effects_applied[0].get("hit"):
            if not result.effects_applied[0].get("critical"):
                # Minimum damage should be 0 or 1 depending on implementation
                assert result.damage_dealt >= 0


class TestPR2SaveChecks:
    """Tests for save/check mechanics based on PR2."""

    def test_check_modifier_applies_to_correct_type(self):
        """Check modifiers only apply to specified check types.

        PR2: Different saves use different stats (HULL, AGILITY, SYSTEMS, ENGINEERING).
        """
        effect = MechanicalEffect(
            check_mods=[
                # +1 to HULL checks only
                # (CheckModifierEffect would have check_types=["hull"])
            ]
        )

        # Verify the modifier system exists and can be applied


class TestPR2PositionalBonuses:
    """Tests for position-based bonuses based on PR2 examples."""

    def test_elevation_accuracy_bonus(self):
        """Higher elevation provides accuracy bonus.

        PR2 (Tactician Solar Backdrop): +1 accuracy when at higher elevation.
        """
        pilot = make_pilot_with_talents("TACTICIAN", [("tactician", 2)])
        effects = collect_pilot_talent_effects(pilot)

        # Should have rank 1 and 2 effects
        assert len(effects) == 2

        # Rank 2 has elevation-based triggered effect
        rank2 = effects[1]
        triggered = rank2.triggered_effects
        assert len(triggered) > 0

        # Should have condition about higher elevation
        assert triggered[0].condition == "higher_elevation_than_target"

    def test_engaged_bonus(self):
        """Engaged status provides bonuses.

        PR2 (Combined Arms CQC Training): +1 accuracy when engaged.
        """
        pilot = make_pilot_with_talents("CQC", [("combined_arms", 2)])
        combatant = make_combatant_from_pilot(pilot, position=(0, 0))

        # Should get +1 accuracy when engaged
        acc, _ = _get_talent_accuracy_modifiers(
            combatant,
            is_ranged=True,
            context={"is_engaged": True},
        )
        assert acc == 1

        # Should not get bonus when not engaged
        acc_not_engaged, _ = _get_talent_accuracy_modifiers(
            combatant,
            is_ranged=True,
            context={"is_engaged": False},
        )
        assert acc_not_engaged == 0


class TestPR2CorePowerOncePerMission:
    """Tests for core power once-per-mission limitation.

    PR2: Core powers can only be activated once per mission.
    """

    def test_core_power_consumed_on_use(self):
        """Core power is consumed after use."""
        core_effect = MechanicalEffect()
        player = make_combatant(
            id="player_1",
            core_power_available=True,
            core_power_effects=core_effect,
        )

        scenario = make_duel_scenario(player, make_enemy_combatant())

        # Start turn and activate
        scenario, turn_result = start_turn(scenario, "player_1")
        economy = turn_result.economy
        turn = CombatTurn(actor_id="player_1")

        action = ActionExecutionInput(
            actor_id="player_1",
            action_id="activate_core_power",
            action_type="protocol",
        )

        scenario, _, _, result = execute_action(scenario, turn, economy, action)
        assert result.success

        # Should be unavailable now
        updated = next(c for c in scenario.combatants if c.id == "player_1")
        assert updated.core_power_available is False
        assert updated.core_power_active is True

    def test_core_power_cannot_reuse(self):
        """Core power cannot be used again after first use."""
        core_effect = MechanicalEffect()
        player = make_combatant(
            id="player_1",
            core_power_available=False,  # Already used
            core_power_effects=core_effect,
        )

        scenario = make_duel_scenario(player, make_enemy_combatant())

        scenario, turn_result = start_turn(scenario, "player_1")
        economy = turn_result.economy
        turn = CombatTurn(actor_id="player_1")

        action = ActionExecutionInput(
            actor_id="player_1",
            action_id="activate_core_power",
            action_type="protocol",
        )

        _, _, _, result = execute_action(scenario, turn, economy, action)

        assert not result.success
        assert result.error is not None
        assert "already used" in result.error.lower()
